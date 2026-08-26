# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import glob
import os
import time
from collections.abc import Generator, Iterable
from typing import cast

import torch
from torch import nn
import threading as _threading
import re as _re
import json as _json
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.torchao import torchao_version_at_least
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.ep_weight_filter import (
    compute_local_expert_ids,
)
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf,
    download_weights_from_hf,
    fastsafetensors_weights_iterator,
    filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference,
    get_quant_config,
    instanttensor_weights_iterator,
    maybe_download_from_modelscope,
    multi_thread_pt_weights_iterator,
    multi_thread_safetensors_weights_iterator,
    np_cache_weights_iterator,
    pt_weights_iterator,
    safetensors_weights_iterator,
)
from vllm.tracing import instrument
import pickle as _pickle
import os as _os
from vllm.transformers_utils.repo_utils import list_filtered_repo_files

logger = init_logger(__name__)


def _exp42_emit(phase: str, **kw):
    """结构化打点：launcher 通过 grep 'EXP42_JSON' 提取。"""
    logger.info("EXP42_JSON %s", _json.dumps({"phase": phase, **kw}))


def _layer_idx(name: str) -> int:
    """参数名 -> 层号：embed=-1，*layers.N*=N，其余（norm/lm_head/vision/mtp）= 10**9。
    用 search 而非锚定 match：原始 checkpoint key 是 model.language_model.layers.N.*，
    vLLM 融合后是 model.language_model.model.layers.N.*，两种命名空间层号都在
    .layers.N. 里。mtp/vision 明确排除（mtp.layers.0.* 会被误判为第 0 层）。"""
    if ".mtp." in name or name.startswith("mtp.") or "visual" in name or "vision" in name:
        return 10**9
    m = _re.search(r"\.layers\.(\d+)\.", name)
    if m:
        return int(m.group(1))
    if "embed_tokens" in name:
        return -1
    return 10**9


def _pin_stage(it):
    """逐 tensor pinned staging（v1 串行版）：
    pageable(mmappage cache) -> pinned 缓冲 -> GPU，消除 driver 内部 staging。
    注意：v1 中 CPU memcpy 与 GPU DMA 串行，有效带宽 ~9 GB/s 量级而非
    纯 DMA 的 22；v2 升级为 per-file 双缓冲流水线后逼近 22。"""
    for name, tensor in it:
        if tensor is not None and tensor.device.type == "cpu":
            pinned = torch.empty_like(tensor, pin_memory=True)
            pinned.copy_(tensor)
            yield name, pinned
        else:
            yield name, tensor


class DefaultModelLoader(BaseModelLoader):
    """Model loader that can load different file types from disk."""

    # default number of thread when enable multithread weight loading
    DEFAULT_NUM_THREADS = 8

    @dataclasses.dataclass
    class Source:
        """A source for weights."""

        model_or_path: str
        """The model ID or path."""

        revision: str | None
        """The optional model revision."""

        subfolder: str | None = None
        """The subfolder inside the model repo."""

        prefix: str = ""
        """A prefix to prepend to all weights."""

        fall_back_to_pt: bool = True
        """Whether .pt weights can be used."""

        allow_patterns_overrides: list[str] | None = None
        """If defined, weights will load exclusively using these patterns."""

    counter_before_loading_weights: float = 0.0
    counter_after_loading_weights: float = 0.0

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        self.local_expert_ids: set[int] | None = None

        extra_config = load_config.model_loader_extra_config
        allowed_keys = {"enable_multithread_load", "num_threads"}
        unexpected_keys = set(extra_config.keys()) - allowed_keys

        if unexpected_keys:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: "
                f"{unexpected_keys}"
            )

    def _prepare_weights(
        self,
        model_name_or_path: str,
        subfolder: str | None,
        revision: str | None,
        fall_back_to_pt: bool,
        allow_patterns_overrides: list[str] | None,
    ) -> tuple[str, list[str], bool]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        model_name_or_path = (
            maybe_download_from_modelscope(model_name_or_path, revision)
            or model_name_or_path
        )

        is_local = os.path.isdir(model_name_or_path)
        load_format = self.load_config.load_format
        use_safetensors = False
        index_file = SAFE_WEIGHTS_INDEX_NAME

        # First check for 'auto' format that mistral files format are present.
        # This is to load mistral models with official format by default.
        if load_format == "auto":
            load_format = (
                "mistral"
                if len(
                    list_filtered_repo_files(
                        model_name_or_path=model_name_or_path,
                        allow_patterns=["consolidated*.safetensors"],
                        revision=revision,
                    )
                )
                > 0
                else "hf"
            )

        # Some quantized models use .pt files for storing the weights.
        if load_format == "hf":
            allow_patterns = ["*.safetensors", "*.bin"]
        elif (
            load_format == "safetensors"
            or load_format == "fastsafetensors"
            or load_format == "instanttensor"
        ):
            use_safetensors = True
            allow_patterns = ["*.safetensors"]
        elif load_format == "mistral":
            use_safetensors = True
            allow_patterns = ["consolidated*.safetensors"]
            index_file = "consolidated.safetensors.index.json"
        elif load_format == "pt":
            allow_patterns = ["*.pt"]
        elif load_format == "npcache":
            allow_patterns = ["*.bin"]
        else:
            raise ValueError(f"Unknown load_format: {load_format}")

        if fall_back_to_pt:
            allow_patterns += ["*.pt"]

        if allow_patterns_overrides is not None:
            allow_patterns = allow_patterns_overrides

        if not is_local:
            hf_folder = download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                subfolder=subfolder,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        else:
            hf_folder = model_name_or_path

        if subfolder is not None:
            hf_folder = os.path.join(hf_folder, subfolder)

        hf_weights_files: list[str] = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                if pattern == "*.safetensors":
                    use_safetensors = True
                break

        if use_safetensors:
            # For models like Mistral-7B-Instruct-v0.3
            # there are both sharded safetensors files and a consolidated
            # safetensors file. Using both breaks.
            # Here, we download the `model.safetensors.index.json` and filter
            # any files not found in the index.
            if not is_local:
                download_safetensors_index_file_from_hf(
                    model_name_or_path,
                    index_file,
                    cache_dir=self.load_config.download_dir,
                    subfolder=subfolder,
                    revision=revision,
                )
            hf_weights_files = filter_duplicate_safetensors_files(
                hf_weights_files, hf_folder, index_file
            )
        else:
            hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files)

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`"
            )

        return hf_folder, hf_weights_files, use_safetensors

    def _get_weights_iterator(
        self, source: "Source"
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""

        extra_config = self.load_config.model_loader_extra_config
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path,
            source.subfolder,
            source.revision,
            source.fall_back_to_pt,
            source.allow_patterns_overrides,
        )
        if self.load_config.load_format == "npcache":
            # Currently np_cache only support *.bin checkpoints
            assert use_safetensors is False
            weights_iterator = np_cache_weights_iterator(
                source.model_or_path,
                self.load_config.download_dir,
                hf_folder,
                hf_weights_files,
                self.load_config.use_tqdm_on_load,
            )
        elif use_safetensors:
            if self.load_config.load_format == "fastsafetensors":
                weights_iterator = fastsafetensors_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                )
            elif self.load_config.load_format == "instanttensor":
                weights_iterator = instanttensor_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                )
            else:
                if extra_config.get("enable_multithread_load"):
                    weights_iterator = multi_thread_safetensors_weights_iterator(
                        hf_weights_files,
                        self.load_config.use_tqdm_on_load,
                        max_workers=extra_config.get(
                            "num_threads", self.DEFAULT_NUM_THREADS
                        ),
                    )
                else:
                    weights_iterator = safetensors_weights_iterator(
                        hf_weights_files,
                        self.load_config.use_tqdm_on_load,
                        self.load_config.safetensors_load_strategy,
                        local_expert_ids=self.local_expert_ids,
                    )
        else:
            if extra_config.get("enable_multithread_load"):
                weights_iterator = multi_thread_pt_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                    self.load_config.pt_load_map_location,
                    max_workers=extra_config.get(
                        "num_threads", self.DEFAULT_NUM_THREADS
                    ),
                )
            else:
                weights_iterator = pt_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                    self.load_config.pt_load_map_location,
                )

        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        # Apply the prefix.
        return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)

    def get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        primary_weights = DefaultModelLoader.Source(
            model_config.model,
            model_config.revision,
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
        )
        yield from self._get_weights_iterator(primary_weights)

        secondary_weights = cast(
            Iterable[DefaultModelLoader.Source],
            getattr(model, "secondary_weights", ()),
        )
        for source in secondary_weights:
            yield from self._get_weights_iterator(source)

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(
            model_name_or_path=model_config.model,
            subfolder=None,
            revision=model_config.revision,
            fall_back_to_pt=True,
            allow_patterns_overrides=None,
        )

    def _init_ep_weight_filter(self, model_config: ModelConfig) -> None:
        """Compute local expert ids for EP weight filtering.

        When expert parallelism is active, each rank only needs a subset of
        expert weights.  By computing the set upfront we can skip non-local
        expert tensors *before* reading them from disk.
        """
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config

        if not (
            model_config.is_moe
            and parallel_config.enable_expert_parallel
            and parallel_config.enable_ep_weight_filter
        ):
            return

        # When EPLB is enabled, redundant physical expert slots may map to
        # logical experts that belong to other ranks in the default partition.
        # The weight loader needs to see ALL logical expert weights so it can
        # populate these redundant slots.  Skip the filter entirely.
        if parallel_config.enable_eplb:
            return

        num_experts = model_config.get_num_experts()
        if num_experts <= 0:
            return

        # EP size/rank computation mirrors FusedMoEParallelConfig.make():
        #   ep_size = dp_size * pcp_size * tp_size (flattened)
        #   ep_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank
        from vllm.distributed import (
            get_dp_group,
            get_pcp_group,
            get_tensor_model_parallel_rank,
        )

        dp_size = parallel_config.data_parallel_size
        tp_size = parallel_config.tensor_parallel_size
        pcp_size = parallel_config.prefill_context_parallel_size
        dp_rank = get_dp_group().rank_in_group if dp_size > 1 else 0
        tp_rank = get_tensor_model_parallel_rank() if tp_size > 1 else 0
        pcp_rank = get_pcp_group().rank_in_group if pcp_size > 1 else 0
        ep_size = dp_size * pcp_size * tp_size
        ep_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank

        self.local_expert_ids = compute_local_expert_ids(
            num_experts,
            ep_size,
            ep_rank,
            placement=parallel_config.expert_placement_strategy,
        )
        if self.local_expert_ids is not None:
            logger.info_once(
                "EP weight filter: ep_size=%d, ep_rank=%d, loading %d/%d experts",
                ep_size,
                ep_rank,
                len(self.local_expert_ids),
                num_experts,
            )

    @instrument(span_name="Load weights")
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:


        # ── Hybrid path: 前 VLLM_HYBRID_LAYERS 层(+embed) IPC + 其余层磁盘并发 ──
        # （必须在纯 IPC 分支之前判断，否则 IPC 先 return、hybrid 不可达）
        _ipc_import_file = _os.environ.get("VLLM_IPC_REGISTRY", "")
        _hybrid_layers = _os.environ.get("VLLM_HYBRID_LAYERS", "")
        if _ipc_import_file and _os.path.exists(_ipc_import_file) and _hybrid_layers:
            _k = int(_hybrid_layers)
            _params = [(n, p) for n, p in model.named_parameters()]
            _ipc_params = [(n, p) for n, p in _params
                           if (_layer_idx(n) == -1 or 0 <= _layer_idx(n) < _k)]
            logger.info("Hybrid import: layers<%d via IPC (%.1f GB), rest via disk",
                        _k,
                        sum(p.numel() * p.element_size() for _, p in _ipc_params) / 1e9)

            self._init_ep_weight_filter(model_config)
            _ipc_elapsed = {"v": None}
            _disk_elapsed = {"v": None}
            _disk_loaded = {"v": None}
            _t0 = time.perf_counter()

            # 预解析全部 IPC handle（打开 mapped tensor）后再启动磁盘线程——
            # batch2 实测「边开 handle 边拷」时快分支被拖到 ~15 GB/s（与磁盘线程
            # 的 GIL/driver 调用竞争）；预解析后 copy 循环为纯 GPU-paced 异步发射
            with open(_ipc_import_file, "rb") as _f:
                _handles = _pickle.load(_f)
            _ipc_loaded = set()
            _ipc_missing = []
            _mapped = []
            for _name, _param in _ipc_params:
                if _name in _handles:
                    _func, _args = _handles[_name]
                    _args_list = list(_args)
                    _args_list[6] = 0
                    _mapped.append((_name, _param, _func(*_args_list)))
                    _ipc_loaded.add(_name)
                else:
                    _ipc_missing.append(_name)

            def _disk_branch():
                # 注意：get_all_weights() 吐出的是原始 checkpoint key（融合前，
                # 如 q_proj/k_proj/v_proj），而 model.named_parameters() 里注册的是
                # 融合后参数名（如 qkv_proj）——两者名字空间不同，不能直接拿融合后
                # 参数名集合去过滤原始 key（会把所有融合层的原始 key 全部错误剔除，
                # 静默漏加载）。这里改成对原始 key 直接判层号，层前缀在融合前后
                # 都保留，判断结果一致。
                _iter = self.get_all_weights(model_config, model)
                _iter = ((n, t) for n, t in _iter
                         if not (_layer_idx(n) == -1 or 0 <= _layer_idx(n) < _k))
                if _os.environ.get("VLLM_PIN_LOAD", ""):
                    _iter = _pin_stage(_iter)
                _stream = torch.cuda.Stream()
                with torch.cuda.stream(_stream):
                    _disk_loaded["v"] = model.load_weights(_iter)
                torch.cuda.synchronize()
                _disk_elapsed["v"] = time.perf_counter() - _t0

            _thread = _threading.Thread(target=_disk_branch, daemon=True)
            _thread.start()

            _stream_ipc = torch.cuda.Stream()
            with torch.cuda.stream(_stream_ipc):
                for _name, _param, _tensor in _mapped:
                    _param.data.copy_(_tensor)
                torch.cuda.synchronize()
            _ipc_elapsed["v"] = time.perf_counter() - _t0

            _thread.join()
            _elapsed = time.perf_counter() - _t0
            self.counter_before_loading_weights = _t0
            self.counter_after_loading_weights = _t0 + _elapsed

            # 严格校验：IPC + 磁盘两个分支合起来必须覆盖全部参数，否则不能算
            # 加载成功——不依赖事后跑生成对比才发现（那是最后一道防线，不是第一道）。
            _weights_to_load = {n for n, _ in _params}
            _loaded_union = _ipc_loaded | (_disk_loaded["v"] or set())
            _missing = _weights_to_load - _loaded_union
            if _ipc_missing or _missing:
                raise ValueError(
                    f"Hybrid import incomplete: {len(_ipc_missing)} params missing "
                    f"from IPC handles, {len(_missing)} params not loaded by either "
                    f"branch: {sorted(_missing)[:10]}..."
                )

            _total_gb = sum(p.numel() * p.element_size() for _, p in _params) / 1e9
            _exp42_emit("weight_load", mode="hybrid",
                        ipc_s=round(_ipc_elapsed["v"], 4),
                        disk_s=round(_disk_elapsed["v"], 4),
                        total_s=round(_elapsed, 4),
                        total_gb=round(_total_gb, 3))
            logger.info("Hybrid import: total %.2f s (ipc %.2f s / disk %.2f s)",
                        _elapsed, _ipc_elapsed["v"], _disk_elapsed["v"])
            return
        # ── IPC direct import path ──
        _ipc_import_file = _os.environ.get("VLLM_IPC_REGISTRY", "")
        if _ipc_import_file and _os.path.exists(_ipc_import_file):
            logger.info("IPC import: loading weights from peer GPU (registry: %s)", _ipc_import_file)
            with open(_ipc_import_file, "rb") as _f:
                _handles = _pickle.load(_f)
            _t0 = time.perf_counter()
            _ipc_tensors = []
            for _name, _param in model.named_parameters():
                if _name in _handles:
                    _func, _args = _handles[_name]
                    _args_list = list(_args)
                    _args_list[6] = 0
                    _ipc_tensors.append((_param, _func(*_args_list)))
            for _param, _tensor in _ipc_tensors:
                _param.data.copy_(_tensor)
            torch.cuda.synchronize()
            _elapsed = time.perf_counter() - _t0
            _total_gb = sum(p.numel() * p.element_size() for _, p in model.named_parameters()) / 1e9
            logger.info("IPC import: %.1f GB in %.2f seconds (%.1f GB/s)", _total_gb, _elapsed, _total_gb / _elapsed)
            self.counter_before_loading_weights = _t0
            self.counter_after_loading_weights = _t0 + _elapsed
            _exp42_emit("weight_load", mode="ipc", ipc_s=round(_elapsed, 4),
                        total_s=round(_elapsed, 4))
            logger.info("IPC import: Loading weights took %.2f seconds", _elapsed)
            return

        if model_config.quantization == "torchao":
            quant_config = get_quant_config(model_config, self.load_config)
            if (
                hasattr(quant_config, "is_checkpoint_torchao_serialized")
                and quant_config.is_checkpoint_torchao_serialized
                and torchao_version_at_least("0.15.0")
            ):
                self.load_config.safetensors_load_strategy = "torchao"

        self._init_ep_weight_filter(model_config)

        weights_to_load = {name for name, _ in model.named_parameters()}
        _weights_iter = self.get_all_weights(model_config, model)
        if _os.environ.get("VLLM_PIN_LOAD", ""):
            _weights_iter = _pin_stage(_weights_iter)
        loaded_weights = model.load_weights(_weights_iter)

        self.counter_after_loading_weights = time.perf_counter()
        _exp42_emit("weight_load", mode="disk",
                    total_s=round(self.counter_after_loading_weights
                                  - self.counter_before_loading_weights, 4),
                    pin=int(bool(_os.environ.get("VLLM_PIN_LOAD", ""))))
        logger.info_once(
            "Loading weights took %.2f seconds",
            self.counter_after_loading_weights - self.counter_before_loading_weights,
        )
        # We only enable strict check for non-quantized models
        # that have loaded weights tracking currently.
        if model_config.quantization is None and loaded_weights is not None:
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                raise ValueError(
                    "Following weights were not initialized from "
                    f"checkpoint: {weights_not_loaded}"
                )

        # ── IPC export (rank 0 only) ──
        _ipc_export_file = _os.environ.get("VLLM_IPC_EXPORT", "")
        if _ipc_export_file:
            _tp_rank = int(_os.environ.get("LOCAL_RANK", _os.environ.get("RANK", "0")))
            if _tp_rank == 0 or _os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] == _os.environ.get("VLLM_IPC_EXPORT_RANK0_DEVICE", "0"):
                from torch.multiprocessing.reductions import reduce_tensor
                logger.info("IPC export: exporting handles to %s", _ipc_export_file)
                _t0 = time.perf_counter()
                _handles = {}
                for _name, _param in model.named_parameters():
                    _handles[_name] = reduce_tensor(_param.data)
                torch.cuda.synchronize()
                with open(_ipc_export_file, "wb") as _f:
                    _pickle.dump(_handles, _f)
                _elapsed = time.perf_counter() - _t0
                logger.info("IPC export: %d parameters in %.2f seconds", len(_handles), _elapsed)
