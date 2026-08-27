"""ccut.models.backend_transformers — L1 兜底后端（transformers AutoModel）。

R8 定位（§4）：L1 架构**功能可用**（能跑通文本生成）但三大机制（零驻留专家
流 / KV 双层 / 层流式）不生效——transformers 全量加载权重进内存。
本机约束（§0.2）：transformers 5.15 已装，仅作运行时兜底（非参考实现），
CPU 上 35B 级别不可用（内存），L1 实际服务对象 = 中小模型
（<10B BF16 或量化 checkpoint）。

接口（与 L0 generic 对齐的最小面）::

    L1Backend(model_dir, config)
      .generate(token_ids, sampling_params, callback) -> token 列表

内部：``transformers.AutoModelForCausalLM.from_pretrained``（trust_remote_code
按 config 透传）+ ``torch.inference_mode`` + 逐 token 采样（复用 sampling.py
的概率变换，避免 transformers 采样器重复实现）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

__all__ = ["L1Backend", "L1Unavailable"]


class L1Unavailable(Exception):
    """transformers 不可用（未安装 / 架构不支持）——启动期显式报错。"""


class L1Backend:
    """transformers AutoModel 兜底后端（单请求，无连续批——L1 语义）。"""

    tier = "L1"

    def __init__(self, model_dir: str | Path, trust_remote_code: bool = False, dtype: str = "bf16"):
        import torch

        self._torch = torch
        model_dir = Path(model_dir)
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(f"{model_dir}/config.json 缺失")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise L1Unavailable(
                "transformers 未安装或导入失败——L1 兜底后端不可用"
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code
        )
        torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(
            dtype.casefold(), torch.float32
        )
        # transformers 5.x：low_cpu_mem_usage 默认 True；CPU 推理
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.model.to("cpu")
        self._cache: dict[str, Any] = {}

    @property
    def dtype(self) -> str:
        return str(self.model.dtype)

    def embed(self, token_ids: list[int]) -> Any:
        import torch

        return self.model.get_input_embeddings()(torch.tensor([token_ids], dtype=torch.long))

    def forward_step(
        self,
        input_ids: list[int],
        past: Any,
    ) -> tuple[Any, Any]:
        """单步前向（transformers 内部 KV cache 管理——L1 不做块池）。

        返回 ``(logits [1, 1, vocab], past)``。
        """
        import torch

        self._torch.manual_seed(0)  # 占位：inference_mode 内无副作用
        with self._torch.inference_mode():
            out = self.model(
                self._torch.tensor([input_ids], dtype=torch.long),
                past_key_values=past,
                use_cache=past is not None,
                return_dict=True,
            )
        return out.logits[:, -1, :], out.past_key_values

    def generate(
        self,
        token_ids: list[int],
        max_tokens: int,
        on_token: Callable[[int], None] | None = None,
        sampling: Callable[[Any, int], int] | None = None,
    ) -> list[int]:
        """逐 token 生成（L1：无投机、无连续批）。

        ``sampling(logits, position) -> token_id``：调用方注入采样策略
        （默认 greedy）。
        """
        import torch

        if sampling is None:
            sampling = lambda logits, pos: int(logits.argmax())
        past: Any = None
        current = list(token_ids)
        generated: list[int] = []
        for _ in range(max_tokens):
            logits, past = self.forward_step(current, past)
            tok = sampling(logits, len(generated))
            generated.append(tok)
            if on_token is not None:
                on_token(tok)
            current.append(tok)
            if tok == getattr(self.tokenizer, "eos_token_id", None):
                break
        return generated

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([int(token_id)])

    def close(self) -> None:
        self.model = None
        self._cache.clear()
