"""ccut.tokenization — token ↔ text 适配层。

- 优先用模型目录的 tokenizer.json（HuggingFace tokenizers 库，~50MB 加载即用）；
- 缺失时**显式报错**（而非回退到 ad-hoc BPE——后者会让用户对结果困惑）。
- detokenize 走 tokenizers 的 decode（处理 BPE 边界特殊 token），不做自实现。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

__all__ = ["Tokenization", "TokenizationUnavailable", "decode_incremental"]


class TokenizationUnavailable(Exception):
    """模型目录无 tokenizer.json / tokenizers 库未装——显式报错。"""


class Tokenization:
    """tokenizer 适配（HF tokenizers 后端）。"""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        if not (self.model_dir / "tokenizer.json").exists():
            raise TokenizationUnavailable(
                f"{self.model_dir}/tokenizer.json 缺失——Ornith 主测 checkpoint 应该有；"
                "若 L1 兜底后端也未提供，自定义 tokenize 接口请通过 --tokenize-fn 注入"
            )
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise TokenizationUnavailable("tokenizers 库未装") from exc
        self._tok = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        # BOS / EOS / PAD / special tokens
        try:
            from tokenizers import AddedToken
        except ImportError:
            AddedToken = None  # type: ignore
        # 标 eos token id（engine 收尾判定用）
        meta_path = self.model_dir / "tokenizer_config.json"
        if meta_path.exists():
            try:
                meta = json_loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        else:
            meta = {}
        self._eos_id = (
            meta.get("eos_token_id")
            or (self._tok.token_to_id(meta["eos_token"]) if "eos_token" in meta else None)
            or 0
        )
        self._bos_id = (
            meta.get("bos_token_id")
            or (self._tok.token_to_id(meta["bos_token"]) if "bos_token" in meta else None)
        )

    @property
    def eos_token_id(self) -> int:
        return self._eos_id

    @property
    def bos_token_id(self) -> int | None:
        return self._bos_id

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(list(token_ids), skip_special_tokens=skip_special_tokens)

    def decode_one(self, token_id: int) -> str:
        return self._tok.decode([int(token_id)], skip_special_tokens=True)


def json_loads(s: str) -> dict:
    import json

    return json.loads(s)


def decode_incremental(tok, prev_text: str, new_text: str) -> tuple[str, str]:
    """增量 detokenize 辅助：避免重复解码整序列。

    HF tokenizers 无「partial」API；该函数作为约定，调用方缓存完整文本 + 上次
    解码后偏移位置，下次只在新增 token 上重新 decode（多数 BPE 边界稳定）。
    """
    # 简化：用 `new_text` 替换 `prev_text` 后的全解码（足够 small for 200B 量级）
    return new_text, new_text
