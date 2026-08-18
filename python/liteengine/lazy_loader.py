from __future__ import annotations

"""
Lazy Model Loader Module - PyTorch + Safetensors
Python 3.14 Compatible
延迟加载本地模型，仅在首次对话时加载
"""
import gc
import glob
import json
import os
import shutil
import threading
import torch
import torch.nn as nn
from typing import Any
from liteengine.engine import ModelConfig

import os
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"


class _ExpertDiskStore:
    """专家权重磁盘存储 + LRU 内存缓存（利用 MoE 稀疏性按需换入换出）。

    所有专家权重保存在磁盘临时目录；推理时只把被路由激活的专家换入内存
    （上限 max_resident 个，LRU 淘汰），大幅降低常驻内存。
    """

    def __init__(self, work_dir: str, max_resident: int = 8):
        self.work_dir = work_dir
        self.max_resident = max(1, max_resident)
        self._disk: dict[int, str] = {}          # expert_idx -> 磁盘文件
        self._mem: dict[int, dict] = {}          # expert_idx -> state_dict（内存）
        self._lru: list[int] = []

    def put(self, idx: int, state_dict: dict) -> None:
        """保存专家权重到磁盘（并预缓存至 LRU）"""
        path = os.path.join(self.work_dir, f"expert_{idx}.pt")
        torch.save(state_dict, path)
        self._disk[idx] = path
        self._lru_remove(idx)
        if len(self._mem) < self.max_resident:
            self._mem[idx] = {k: v.detach().cpu() for k, v in state_dict.items()}
            self._lru.append(idx)

    def get(self, idx: int) -> dict:
        """按需取回专家权重（内存命中则 LRU 刷新，未命中则从磁盘换入）"""
        if idx not in self._disk:
            raise KeyError(f"expert {idx} not offloaded")
        if idx in self._mem:
            self._lru_remove(idx)
            self._lru.append(idx)
            return self._mem[idx]
        sd = torch.load(self._disk[idx], map_location="cpu", weights_only=True)
        self._evict_if_needed()
        self._mem[idx] = sd
        self._lru.append(idx)
        return sd

    def _evict_if_needed(self) -> None:
        while len(self._mem) >= self.max_resident and self._lru:
            old = self._lru.pop(0)
            self._mem.pop(old, None)

    def _lru_remove(self, idx: int) -> None:
        if idx in self._lru:
            self._lru.remove(idx)

    def clear(self) -> None:
        self._disk.clear()
        self._mem.clear()
        self._lru.clear()
        shutil.rmtree(self.work_dir, ignore_errors=True)


class _OffloadedExpert(nn.Module):
    """被卸载的单个专家：权重在磁盘，forward 时按需换入计算。

    支持标准 SwiGLU FFN（gate_proj/up_proj/down_proj 三个 Linear）。
    """

    def __init__(self, expert: nn.Module, store: _ExpertDiskStore, idx: int):
        super().__init__()
        self.idx = idx
        self.store = store
        self.act_fn = expert.act_fn if hasattr(expert, "act_fn") else nn.functional.silu
        # 把专家权重搬到磁盘
        sd = {k: v.detach().cpu() for k, v in expert.state_dict().items()}
        store.put(idx, sd)

    def forward(self, x):
        sd = self.store.get(self.idx)
        dev, dt = x.device, x.dtype
        gate = nn.functional.linear(x, sd["gate_proj.weight"].to(dev, dt))
        up = nn.functional.linear(x, sd["up_proj.weight"].to(dev, dt))
        hidden = self.act_fn(gate) * up
        out = nn.functional.linear(hidden, sd["down_proj.weight"].to(dev, dt))
        return out


class LazyModelLoader:
    _instance = None
    _model = None
    _tokenizer = None
    _current_model_name = None
    # 权重共享缓存：abs_path -> {"model": ..., "tokenizer": ..., "name": ...}
    # 多个模型条目指向同一目录时复用，避免重复加载（weight_sharing=true）
    _shared_cache: dict = {}

    def __new__(cls) -> LazyModelLoader:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _check_model_files(self, config: ModelConfig) -> bool:
        """检查模型目录是否有必要的代码文件"""
        model_dir = config.model_dir
        if not os.path.isdir(model_dir):
            return False
        has_modeling = any(
            f.startswith("modeling_") and f.endswith(".py")
            for f in os.listdir(model_dir)
        )
        return has_modeling

    def _extract_modeling_from_llmcompressor(self, config: ModelConfig) -> bool:
        """从 llmcompressor 包中提取建模代码到模型目录"""
        try:
            import llmcompressor
            llmcompressor_dir = os.path.dirname(llmcompressor.__file__)
            print(f"[LazyLoader] llmcompressor location: {llmcompressor_dir}")

            # 搜索所有 modeling_*.py 文件
            modeling_files = glob.glob(
                os.path.join(llmcompressor_dir, "**", "modeling_*.py"), 
                recursive=True
            )
            print(f"[LazyLoader] Found {len(modeling_files)} modeling files in llmcompressor")

            # 筛选与 qwen3 或 moe 相关的文件
            needed_files = []
            for f in modeling_files:
                basename = os.path.basename(f).lower()
                if "qwen3" in basename or "moe" in basename or "qwen" in basename:
                    needed_files.append(f)

            if not needed_files:
                print("[LazyLoader] No matching modeling files found in llmcompressor")
                return False

            # 复制文件到模型目录
            model_dir = config.model_dir
            os.makedirs(model_dir, exist_ok=True)
            for src in needed_files:
                dst = os.path.join(model_dir, os.path.basename(src))
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    print(f"[LazyLoader] Copied {os.path.basename(src)} to model directory")

            return True

        except Exception as e:
            print(f"[LazyLoader] Failed to extract modeling files: {e}")
            return False

    def _check_lfs_pointer_files(self, config: ModelConfig) -> list[str]:
        """检测模型目录中的 Git LFS 指针文件（未真正下载的 LFS 文件）。

        这类文件是 130 字节左右的纯文本（内容形如
        "version https://git-lfs.github.com/spec/v1\\noid sha256:..."），
        直接当 tokenizer.json / config.json 使用会导致解码乱码或加载失败。
        """
        pointers: list[str] = []
        if not os.path.isdir(config.model_dir):
            return pointers
        for fname in os.listdir(config.model_dir):
            if not fname.endswith((".json", ".safetensors", ".bin", ".model")):
                continue
            path = os.path.join(config.model_dir, fname)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    head = f.read(64)
            except Exception:
                continue
            if head.startswith("version https://git-lfs"):
                pointers.append(fname)
        return pointers

    def _validate_tokenizer_vocab(self, config: ModelConfig) -> None:
        """加载后校验 tokenizer 词表与 config.vocab_size 是否匹配。

        Qwen 系列多模态模型允许 tokenizer 词表略小于模型词表（差值 =
        视觉/音频等扩展 token），但相差过大说明 tokenizer 加载不完整，
        此时解码必然乱码，应尽早提示。
        """
        try:
            with open(os.path.join(config.model_dir, "config.json"), "r", encoding="utf-8") as f:
                vocab_size = json.load(f).get("vocab_size")
        except Exception:
            return
        if vocab_size is None or self._tokenizer is None:
            return
        tok_size = self._tokenizer.vocab_size
        if abs(tok_size - vocab_size) > 4096:
            print(
                f"[LazyLoader] Warning: tokenizer vocab size {tok_size} "
                f"mismatches config vocab_size {vocab_size}. "
                f"Decoding will produce garbled text (乱码)."
            )

    def _native_supported(self, config: ModelConfig) -> bool:
        """检查 config.json 的 model_type 是否被本地 transformers 原生支持。

        原生支持时无需 llmcompressor 提取建模代码，也无需 trust_remote_code
        下载远端代码，加载更可靠。
        """
        try:
            from transformers.models.auto.modeling_auto import (
                MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
                MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
                MODEL_MAPPING_NAMES,
            )
            with open(os.path.join(config.model_dir, "config.json"), "r", encoding="utf-8") as f:
                model_type = json.load(f).get("model_type", "")
            supported = (
                set(MODEL_MAPPING_NAMES)
                | set(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)
                | set(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES)
            )
            return model_type in supported
        except Exception:
            return False

    def _check_awq_quantization(self, config: ModelConfig) -> bool:
        """检测 AWQ 量化模型并确认 gptqmodel 依赖已安装。

        transformers 5.x 加载 AWQ 权重依赖 gptqmodel（代替旧版 autoawq），
        缺失时给出明确安装提示，避免加载中途才报错。
        """
        try:
            with open(os.path.join(config.model_dir, "config.json"), "r", encoding="utf-8") as f:
                qcfg = json.load(f).get("quantization_config", {})
            if qcfg.get("quant_method") != "awq":
                return True
        except Exception:
            return True
        try:
            import gptqmodel  # noqa: F401
            print(f"[LazyLoader] AWQ model detected, gptqmodel available")
            return True
        except ImportError:
            print("\n" + "="*70)
            print("[WARNING] 该模型是 AWQ 4bit 量化模型，加载需要 gptqmodel：")
            print("  请先安装:  pip install gptqmodel")
            print("  未安装时 transformers 将无法解析 AWQ 量化权重。")
            print("="*70)
            return False

    def _has_unpacked_experts(self, config: ModelConfig) -> bool:
        """判断检查点专家权重是否为"未打包逐专家"格式（experts.N.gate_proj）。

        transformers 5.15 的 Qwen3_5MoeExperts 建模是打包 3D 格式
        （experts.gate_up_proj），而部分量化工具（autoawq）会把专家展开成
        逐专家 nn.Linear 保存（experts.0.gate_proj.qweight）。两者键名不兼容，
        需要 monkeypatch 建模代码为未打包版本才能加载。
        """
        try:
            with open(os.path.join(config.model_dir, "config.json"), "r", encoding="utf-8") as f:
                arch = json.load(f).get("architectures", [])
            if not any("Qwen3_5Moe" in a for a in arch):
                return False
            # 通过权重索引判断：存在 experts.<N>.gate_proj 形式的键
            import re
            idx_path = os.path.join(config.model_dir, "model.safetensors.index.json")
            if not os.path.exists(idx_path):
                return False
            with open(idx_path, "r", encoding="utf-8") as f:
                weight_map = json.load(f).get("weight_map", {})
            unpacked = re.compile(r"experts\.\d+\.(gate_proj|up_proj|down_proj)")
            return any(unpacked.search(k) for k in weight_map)
        except Exception:
            return False

    def _patch_unpacked_experts(self, config: ModelConfig) -> bool:
        """monkeypatch transformers 的 Qwen3_5MoeExperts 为未打包逐专家版本。

        仅当检查点是未打包专家格式时启用；加载完成后由 _unload_model 恢复，
        避免影响打包格式的其他 qwen3_5_moe 模型（如 Qwen3.8-1.0B-A0.6B）。
        返回 True 表示已启用补丁。
        """
        if not self._has_unpacked_experts(config):
            return False
        import torch.nn as nn
        import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as modeling

        if getattr(modeling, "_atomcode_unpacked_experts_patched", False):
            return True

        ACT2FN = modeling.ACT2FN

        class Qwen3_5MoeExpertFFN(nn.Module):
            """单个专家：gate/up/down 三个 Linear（与打包版 gate_up_proj 等价）"""

            def __init__(self, mconfig):
                super().__init__()
                self.hidden_dim = mconfig.hidden_size
                self.intermediate_dim = mconfig.moe_intermediate_size
                self.gate_proj = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False)
                self.up_proj = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False)
                self.down_proj = nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False)
                self.act_fn = ACT2FN[mconfig.hidden_act]

            def forward(self, x):
                return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        class UnpackedQwen3_5MoeExperts(nn.ModuleList):
            """未打包专家集合：ModuleList[num_experts]，键名对齐 experts.N.gate_proj"""

            def __init__(self, mconfig):
                super().__init__([Qwen3_5MoeExpertFFN(mconfig) for _ in range(mconfig.num_experts)])
                self.num_experts = mconfig.num_experts
                self.hidden_dim = mconfig.hidden_size
                self.intermediate_dim = mconfig.moe_intermediate_size

            def forward(self, hidden_states, top_k_index, top_k_weights):
                final_hidden_states = torch.zeros_like(hidden_states)
                with torch.no_grad():
                    expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
                    expert_mask = expert_mask.permute(2, 1, 0)
                    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
                for row in expert_hit:
                    expert_idx = row[0].item()
                    if expert_idx == self.num_experts:
                        continue
                    top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                    current_state = hidden_states[token_idx]
                    current_hidden_states = self[expert_idx](current_state)
                    current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
                    final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
                return final_hidden_states

        self._orig_experts_class = modeling.Qwen3_5MoeExperts
        modeling.Qwen3_5MoeExperts = UnpackedQwen3_5MoeExperts
        modeling._atomcode_unpacked_experts_patched = True
        print("[LazyLoader] Patched Qwen3_5MoeExperts -> unpacked per-expert (AWQ checkpoint)")
        return True

    def _restore_experts(self) -> None:
        """恢复被 monkeypatch 的专家建模类（模型卸载时调用）"""
        import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as modeling
        if getattr(modeling, "_atomcode_unpacked_experts_patched", False):
            modeling.Qwen3_5MoeExperts = self._orig_experts_class
            modeling._atomcode_unpacked_experts_patched = False
            print("[LazyLoader] Restored original Qwen3_5MoeExperts")

    def _get_shard_skip_prefixes(self, config: ModelConfig) -> list[str]:
        """分片按需加载时跳过的权重前缀（键名前缀）。

        默认跳过纯文本对话用不到的权重：视觉编码器（model.visual.*）与
        MTP 预测头（mtp.*）。可在 [model] 块的 shard_skip_prefixes 自定义。
        """
        prefixes = config.shard_skip_prefixes
        if not prefixes:
            prefixes = ["model.visual.", "mtp."]
        return [str(p) for p in prefixes]

    def _filter_shard_index(self, config: ModelConfig) -> bool:
        """按需加载：临时过滤 model.safetensors.index.json 的 weight_map。

        移除被跳过前缀的权重键；若某个分片文件因此不再包含任何需要加载的
        键，from_pretrained 就不会打开该分片（配合 mmap 显著减少读盘）。
        调用方必须在加载结束后调用 _restore_shard_index 恢复原始 index。
        返回 True 表示已过滤（需要恢复）。
        """
        model_dir = config.model_dir
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            return False
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            if not weight_map:
                return False
            prefixes = self._get_shard_skip_prefixes(config)
            kept = {k: v for k, v in weight_map.items()
                    if not any(k.startswith(p) for p in prefixes)}
            if len(kept) == len(weight_map):
                return False  # 没有可跳过的键

            skipped_keys = set(weight_map) - set(kept)
            kept_shards = set(kept.values())
            dropped_shards = set(weight_map.values()) - kept_shards
            if not dropped_shards:
                # 所有分片都仍包含需要加载的键 → 过滤无实际收益，跳过
                return False

            backup_path = index_path + ".atomcode.bak"
            shutil.copy2(index_path, backup_path)
            self._index_backup_path = backup_path

            new_index = dict(index)
            new_index["weight_map"] = kept
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(new_index, f, indent=2)

            print(f"[LazyLoader] shard_lazy: skipped {len(skipped_keys)} weight keys "
                  f"({len(dropped_shards)} shard file(s) not loaded): {sorted(dropped_shards)}")
            return True
        except Exception as e:
            print(f"[LazyLoader] Warning: shard_lazy index filtering failed, load all shards: {e}")
            return False

    def _restore_shard_index(self) -> None:
        """恢复被临时过滤的 model.safetensors.index.json（分片按需加载用）"""
        backup = getattr(self, "_index_backup_path", None)
        if backup and os.path.exists(backup):
            try:
                index_path = backup[: -len(".atomcode.bak")]
                shutil.move(backup, index_path)
                print("[LazyLoader] Restored original safetensors index")
            except Exception as e:
                print(f"[LazyLoader] Warning: failed to restore safetensors index: {e}")
            finally:
                self._index_backup_path = None

    def _select_model_class(self, config: ModelConfig):
        """根据 config.json 的 architectures 选择正确的模型类。

        - ForConditionalGeneration（多模态 VLM，如 Qwen3_5MoeForConditionalGeneration）
          → AutoModelForImageTextToText
        - 其余纯文本架构 → AutoModelForCausalLM
        """
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
        try:
            with open(os.path.join(config.model_dir, "config.json"), "r", encoding="utf-8") as f:
                archs = json.load(f).get("architectures", [])
            if any("ForConditionalGeneration" in a for a in archs):
                return AutoModelForImageTextToText
        except Exception:
            pass
        return AutoModelForCausalLM

    def _apply_expert_offload(self, config: ModelConfig) -> bool:
        """MoE 专家按需卸载：把未打包专家的权重换出到磁盘，推理时按需换入。

        利用 MoE 路由稀疏性（每 token 只激活 top-k 专家），常驻内存只需保留
        少量专家（expert_offload_cache 个，LRU），大幅降低大 MoE 模型内存占用。
        返回 True 表示已启用。
        """
        if not config.expert_offload:
            return False
        model = self._model
        if model is None:
            return False

        import tempfile
        work_dir = os.path.join(
            tempfile.gettempdir(),
            f"atomcode_experts_{os.path.basename(os.path.abspath(config.model_dir))}",
        )
        os.makedirs(work_dir, exist_ok=True)
        store = _ExpertDiskStore(work_dir, max_resident=config.expert_offload_cache)

        replaced = 0
        try:
            for name, module in model.named_modules():
                # 未打包专家集合：nn.ModuleList，子项为标准 SwiGLU FFN
                if not isinstance(module, nn.ModuleList) or len(module) == 0:
                    continue
                first = module[0]
                if not (hasattr(first, "gate_proj") and hasattr(first, "up_proj")
                        and hasattr(first, "down_proj")):
                    continue
                for idx in range(len(module)):
                    expert = module[idx]
                    if not isinstance(expert, nn.Module):
                        continue
                    module[idx] = _OffloadedExpert(expert, store, idx)
                    replaced += 1
                print(f"[LazyLoader] expert_offload: offloaded {len(module)} experts in '{name}'")
        except Exception as e:
            print(f"[LazyLoader] Warning: expert_offload failed, keep experts in memory: {e}")
            store.clear()
            return False

        self._expert_store = store
        print(f"[LazyLoader] expert_offload enabled: {replaced} experts on disk, "
              f"LRU cache={config.expert_offload_cache}")
        return True

    def _cleanup_expert_offload(self) -> None:
        """清理专家卸载的磁盘临时文件与缓存"""
        store = getattr(self, "_expert_store", None)
        if store is not None:
            store.clear()
            self._expert_store = None
            print("[LazyLoader] Cleaned up expert offload disk store")

    def _check_memory_headroom(self, config: ModelConfig) -> None:
        """加载前预检：模型 safetensors 总大小 vs 可用物理内存。

        CPU 环境下若权重远超可用内存，加载会陷入磁盘换页（看起来像卡死）
        甚至 OOM，提前给出提示比中途失败更友好。
        """
        total_bytes = 0
        for f in glob.glob(os.path.join(config.model_dir, "*.safetensors")):
            total_bytes += os.path.getsize(f)
        if total_bytes == 0:
            return
        total_gb = total_bytes / 2**30
        avail_gb = 0.0
        try:
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            m = _MEMORYSTATUSEX()
            m.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                avail_gb = m.ullAvailPhys / 2**30
        except Exception:
            pass
        if avail_gb and total_gb > avail_gb * 0.8:
            print(f"\n[Warning] 模型权重 {total_gb:.1f}GB 接近/超过可用物理内存 "
                  f"{avail_gb:.1f}GB：CPU 加载将严重换页或 OOM，\n"
                  f"建议换小模型，或增加内存/显卡后再加载该模型。\n")

    def _load_model(self, config: ModelConfig) -> None:
        """使用 PyTorch + Safetensors 加载本地模型"""
        if self._current_model_name == config.name and self._model is not None:
            return

        self._unload_model()

        # 可选优化：权重共享 —— 同一模型目录只加载一次，其余条目复用
        if config.weight_sharing:
            abs_path = os.path.abspath(config.model_dir)
            entry = self._shared_cache.get(abs_path)
            if entry is not None and entry["model"] is not None:
                self._model = entry["model"]
                self._tokenizer = entry["tokenizer"]
                self._current_model_name = config.name
                print(f"[LazyLoader] Weight sharing: reused loaded model from cache ({config.model_dir})")
                return

        print(f"[LazyLoader] Loading local model: {config.model_dir}")
        print(f"[LazyLoader] dtype: {config.dtype}, device_map: {config.device_map}")

        # 检测 Git LFS 指针文件（未真正下载的 LFS 文件），提前报错避免乱码/加载失败
        pointers = self._check_lfs_pointer_files(config)
        if pointers:
            critical = [p for p in pointers if p in ("tokenizer.json", "config.json", "generation_config.json")]
            if critical:
                raise RuntimeError(
                    f"模型文件未完整下载：{', '.join(critical)} 仍是 Git LFS 指针文件。\n"
                    f"请在模型目录 {config.model_dir} 中执行 `git lfs pull` 下载真实文件后重试。"
                )
            print(f"[LazyLoader] Warning: LFS pointer files present (not downloaded): {pointers}")

        try:
            # 原生支持（transformers 内置该 model_type）时无需 llmcompressor
            native = self._native_supported(config)
            if native:
                print(f"[LazyLoader] Model type natively supported by transformers, skip llmcompressor")
            else:
                # 导入 llmcompressor 并注册自定义模型架构
                llm_compressor_available = False
                try:
                    import llmcompressor
                    llm_compressor_available = True

                    # 显式导入 transformers 子模块以注册自定义架构
                    try:
                        import llmcompressor.transformers  # noqa: F401
                        print("[LazyLoader] llmcompressor.transformers loaded, architectures registered")
                    except ImportError as e:
                        print(f"[LazyLoader] Warning: Could not import llmcompressor.transformers: {e}")

                    print("[LazyLoader] llmcompressor detected, custom models may be registered")
                except ImportError:
                    print("[LazyLoader] llmcompressor not found. If model loading fails, install it:")
                    print("[LazyLoader]   pip install llmcompressor")

                # 检查并补充缺失的建模代码文件
                if not self._check_model_files(config):
                    if llm_compressor_available:
                        print("[LazyLoader] No modeling_*.py files found, attempting to extract from llmcompressor...")
                        if not self._extract_modeling_from_llmcompressor(config):
                            print("[LazyLoader] Warning: Could not extract modeling files from llmcompressor")
                            print("[LazyLoader] Will attempt to load using registered architecture only")
                    else:
                        print(f"[LazyLoader] Warning: No modeling_*.py files found in {config.model_dir}")
                        print("[LazyLoader] The model requires custom modeling code. Please install llmcompressor")

            # AWQ 量化模型依赖检查（gptqmodel）
            self._check_awq_quantization(config)

            # 未打包专家格式的 AWQ 检查点需要 monkeypatch 建模（见 _patch_unpacked_experts）
            self._patch_unpacked_experts(config)

            from transformers import AutoTokenizer

            # 加载 tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                config.model_dir,
                trust_remote_code=True,
                local_files_only=True,
            )

            # 校验 tokenizer 词表与 config 是否匹配（避免解码乱码）
            self._validate_tokenizer_vocab(config)

            # 手动加载 chat_template.jinja
            if (not hasattr(self._tokenizer, "chat_template") or 
                self._tokenizer.chat_template is None):
                template_path = os.path.join(config.model_dir, "chat_template.jinja")
                if os.path.exists(template_path):
                    with open(template_path, "r", encoding="utf-8") as f:
                        self._tokenizer.chat_template = f.read()
                    print(f"[LazyLoader] Loaded chat_template.jinja manually")

            # 构建模型加载参数
            model_kwargs: dict[str, Any] = {
                "trust_remote_code": True,
                "local_files_only": True,
            }

            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            if config.dtype in dtype_map:
                model_kwargs["dtype"] = dtype_map[config.dtype]
            elif config.dtype != "auto":
                try:
                    model_kwargs["dtype"] = getattr(torch, config.dtype)
                except AttributeError:
                    model_kwargs["dtype"] = torch.float16

            if config.device_map != "auto":
                model_kwargs["device_map"] = config.device_map
            else:
                if torch.cuda.is_available():
                    model_kwargs["device_map"] = "auto"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    model_kwargs["device_map"] = "auto"
                else:
                    model_kwargs["device_map"] = "cpu"

            # 可选优化：mmap 内存映射加载
            # transformers 5.x 中 from_pretrained 的 disable_mmap=False 表示
            # 显式启用内存映射（safetensors 按需分页读入，常驻内存更低）；
            # 配合 low_cpu_mem_usage 可在 CPU 环境下显著降低峰值内存。
            if config.mmap:
                model_kwargs["disable_mmap"] = False
                model_kwargs["low_cpu_mem_usage"] = True
                print("[LazyLoader] mmap enabled (memory-mapped weight loading)")

            print(f"[LazyLoader] Attempting to load model with dtype={model_kwargs.get('dtype')}...")
            model_class = self._select_model_class(config)
            print(f"[LazyLoader] Using model class: {model_class.__name__}")

            # 强制离线：防止 transformers/gptqmodel 在本地加载时尝试联网检查
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

            # 加载前内存预检：模型权重接近/超过可用物理内存时给出醒目提示
            self._check_memory_headroom(config)

            print("[LazyLoader] 开始权重加载：AWQ 层替换 + 权重解包在 CPU 上可能耗时较长，"
                  "属正常现象而非卡死（可观察 CPU/磁盘活动）...")

            # 可选优化：分片按需加载（过滤 index.json，跳过用不到的分片）
            index_filtered = False
            if config.shard_lazy:
                index_filtered = self._filter_shard_index(config)

            try:
                self._model = model_class.from_pretrained(
                    config.model_dir,
                    **model_kwargs,
                )
            finally:
                if index_filtered:
                    self._restore_shard_index()

            self._current_model_name = config.model_dir
            print(f"[LazyLoader] Model loaded successfully: {config.model_dir}")

            # 可选优化：MoE 专家按需卸载（利用稀疏性，把专家换出到磁盘）
            self._apply_expert_offload(config)

            # 可选优化：权重共享 —— 把新加载的模型写入缓存供其他条目复用
            if config.weight_sharing:
                self._shared_cache[os.path.abspath(config.model_dir)] = {
                    "model": self._model,
                    "tokenizer": self._tokenizer,
                    "name": config.name,
                }

        except Exception as e:
            error_msg = str(e)
            print("\n" + "="*70)
            print("[FATAL] 模型架构加载失败")
            print("="*70)
            print(f"模型路径: {config.model_dir}")
            print(f"错误信息: {error_msg}")
            print("\n可能的原因:")
            print("1. 缺少 modeling_*.py 和 configuration_*.py 文件")
            print("2. llmcompressor 未正确注册自定义架构")
            print("3. 模型文件不完整或损坏")
            print("\n解决方案:")
            print("1. 运行以下脚本手动下载建模代码:")
            print("   python -c \"from llmcompressor.transformers.models import *\"")
            print("2. 或从原始模型仓库下载代码文件")
            print("="*70)
            raise RuntimeError(f"Failed to load model {config.model_dir}: {error_msg}")

    def _unload_model(self) -> None:
        """卸载当前模型以释放内存"""
        # 恢复被 monkeypatch 的专家建模类，避免影响其他模型
        self._restore_experts()
        # 恢复被临时过滤的 safetensors index（安全兜底）
        self._restore_shard_index()
        # 清理 MoE 专家卸载的磁盘临时文件与缓存
        self._cleanup_expert_offload()

        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._current_model_name = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def ensure_loaded(self, config: ModelConfig) -> None:
        """确保模型已加载（懒加载触发点）"""
        if self._model is None or self._current_model_name != config.name:
            self._load_model(config)

    def is_loaded(self) -> bool:
        return self._model is not None

    def get_model(self):
        return self._model

    def get_tokenizer(self):
        return self._tokenizer

    def switch_model(self, config: ModelConfig) -> None:
        """切换模型"""
        if self._current_model_name != config.name:
            self._load_model(config)

    def generate_stream(self, config: ModelConfig, prompt: str, **gen_kwargs):
        """
        流式生成文本，逐 token yield
        使用 transformers TextIteratorStreamer 实现
        """
        self.ensure_loaded(config)

        tokenizer = self.get_tokenizer()
        model = self.get_model()

        inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(model, "device"):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=60,
        )

        generate_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": gen_kwargs.get("max_tokens", 2048),
            "temperature": gen_kwargs.get("temperature", 0.7),
            "top_p": gen_kwargs.get("top_p", 0.9),
            "repetition_penalty": gen_kwargs.get("repetition_penalty", 1.1),
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
        }

        thread = threading.Thread(target=model.generate, kwargs=generate_kwargs)
        thread.start()

        generated_text = ""
        for token in streamer:
            yield token
            generated_text += token

        thread.join()
        return generated_text