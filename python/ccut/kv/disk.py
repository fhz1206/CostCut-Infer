"""ccut.kv.disk — L2 磁盘块存储（R1：追加式自描述记录，可远大于 RAM）。

设计（§3.1 R1-L2）：
- **追加式文件**（``kv_l2_dir/kv_blocks_NN.bin``，默认上限 64GB 可配）：
  每块一条记录::

      [magic 4B "KVB2"][layer u16][block_len u32][block_hash u64]
      [prev_hash u64][n_tokens u16][tokens u32×n][payload 块字节]

  记录头自描述 → 进程重启**扫头重建索引**（零额外元数据文件，崩溃安全：
  半写记录在扫头时校验 magic+长度丢弃，追加写不覆盖旧数据）。
- **索引**：``{block_hash: (file_id, offset, layer, n_tokens, prev_hash, token_ids)}``
  内存态（启动重建，条目数 = 块数，10 万块 ≈ 数 MB，可接受）。
- **淘汰**：索引超 ``kv_l2_max_bytes`` → 整文件轮转删除（最旧文件优先，
  记录内不拆——块是原子单位）；对应 L1 槽位自然复用。
- **压缩**：``kv_l2_compression=lz4`` 时 payload 走 lz4（若已装，否则显式
  回退 none 并告警一次）；记录头带 flag 位。
- **TTL**：``kv_cache_ttl_seconds>0`` 时索引带写入时间戳，扫描清理（会话级）。
"""

from __future__ import annotations

import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["DiskBlockRecord", "DiskLayer", "L2_STORE_MAGIC", "RECORD_HEADER_FMT"]

L2_STORE_MAGIC = b"KVB2"

# 记录头：magic(4s) | flags(u8) | layer(u16) | n_tokens(u16) | block_hash(u64) | prev_hash(u64) | payload_len(u32) | mtime(f8)
RECORD_HEADER_FMT = "<4sBHHQQId"
RECORD_HEADER_SIZE = struct.calcsize(RECORD_HEADER_FMT)


@dataclass(frozen=True)
class DiskBlockRecord:
    """L2 索引条目。"""

    file_id: int
    offset: int
    layer: int
    n_tokens: int
    block_hash: int
    prev_hash: int
    token_ids: tuple[int, ...]
    payload_len: int
    mtime: float
    compressed: bool = False


class DiskLayer:
    """L2 磁盘层（全局单例；按层索引，文件混层存储）。"""

    def __init__(self, dir_path: str | Path, max_bytes: int, compression: str = "none", ttl_seconds: int = 0):
        self.dir = Path(dir_path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.compression = compression.casefold()
        self.ttl_seconds = ttl_seconds
        self._index: dict[int, DiskBlockRecord] = {}
        self._files: list[Path] = []
        self._current: int = 0
        self._current_size: int = 0
        self._total_bytes: int = 0
        self._lz4 = None
        if self.compression == "lz4":
            try:
                from lz4.frame import LZ4FrameFile

                self._lz4 = LZ4FrameFile
            except ImportError:
                self._lz4 = None
        self._rescan()

    # -- 启动重建 -----------------------------------------------------------
    def _rescan(self) -> None:
        """扫头重建索引（崩溃安全：半写/坏记录丢弃）。"""
        self._index.clear()
        self._files = sorted(self.dir.glob("kv_blocks_*.bin"))
        for f in self._files:
            n = 0
            size = f.stat().st_size
            off = 0
            with open(f, "rb") as fh:
                while off + RECORD_HEADER_SIZE <= size:
                    fh.seek(off)
                    hdr = fh.read(RECORD_HEADER_SIZE)
                    if len(hdr) < RECORD_HEADER_SIZE:
                        break
                    try:
                        magic, flags, layer, n_tok, bh, ph, plen, mtime = struct.unpack(
                            RECORD_HEADER_FMT, hdr
                        )
                    except struct.error:
                        break
                    if magic != L2_STORE_MAGIC:
                        break  # 撕裂 → 后续全弃
                    if off + RECORD_HEADER_SIZE + plen > size:
                        break  # 半写记录
                    tok_raw = fh.read(n_tok * 4)
                    if len(tok_raw) < n_tok * 4:
                        break
                    tokens = tuple(struct.unpack(f"<{n_tok}I", tok_raw))
                    fh.read(plen)  # payload 不驻留（索引只需位置）
                    self._index[bh] = DiskBlockRecord(
                        file_id=self._file_id(f),
                        offset=off,
                        layer=layer,
                        n_tokens=n_tok,
                        block_hash=bh,
                        prev_hash=ph,
                        token_ids=tokens,
                        payload_len=plen,
                        mtime=mtime,
                        compressed=bool(flags & 1),
                    )
                    off += RECORD_HEADER_SIZE + n_tok * 4 + plen
                    n += 1
            self._total_bytes += size
        self._current = len(self._files)
        self._current_size = self._files[-1].stat().st_size if self._files else 0

    def _file_id(self, p: Path) -> int:
        try:
            return int(p.stem.split("_")[-1])
        except (IndexError, ValueError):
            return -1

    def _new_file(self) -> None:
        p = self.dir / f"kv_blocks_{self._current:02d}.bin"
        p.touch()  # 真正创建空文件（追加写目标）
        self._files.append(p)
        self._current += 1
        self._current_size = 0

    # -- 读写 ---------------------------------------------------------------
    def store(
        self,
        layer: int,
        block_hash: int,
        prev_hash: int,
        token_ids: list[int],
        payload: bytes,
    ) -> bool:
        """写入一块（同哈希已存在 → 覆盖旧记录索引位置，不重写数据：
        追加新记录并更新索引指向；旧记录变孤儿由轮转回收）。"""
        if block_hash in self._index:
            old = self._index[block_hash]
            if not old.compressed and self._lz4 is None:
                return False  # 已存在且无需压缩重写
        data = payload
        compressed = False
        if self._lz4 is not None:
            import io

            buf = io.BytesIO()
            with self._lz4(buf) as f:
                f.write(payload)
            if len(buf.getvalue()) < len(payload):
                data = buf.getvalue()
                compressed = True
        tokens = tuple(int(t) & 0xFFFFFFFF for t in token_ids)
        if not self._files:
            self._new_file()
        cur = self._files[-1]
        if self._current_size > self.max_bytes // 4:  # 单文件上限 = 总上限/4
            self._new_file()
            cur = self._files[-1]
        hdr = struct.pack(
            RECORD_HEADER_FMT,
            L2_STORE_MAGIC,
            1 if compressed else 0,
            layer & 0xFFFF,
            len(tokens) & 0xFFFF,
            block_hash & 0xFFFFFFFFFFFFFFFF,
            prev_hash & 0xFFFFFFFFFFFFFFFF,
            len(data) & 0xFFFFFFFF,
            time.time(),
        )
        token_bytes = struct.pack(f"<{len(tokens)}I", *tokens) if tokens else b""
        rec_size = RECORD_HEADER_SIZE + len(token_bytes) + len(data)
        if self._current_size + rec_size > self.max_bytes // 4:
            self._new_file()
            cur = self._files[-1]
        with open(cur, "r+b") as fh:
            fh.seek(0, os.SEEK_END)
            offset = fh.tell()
            fh.write(hdr + token_bytes + data)
        self._current_size += rec_size
        self._total_bytes += rec_size
        self._index[block_hash] = DiskBlockRecord(
            file_id=self._file_id(cur),
            offset=offset,
            layer=layer,
            n_tokens=len(tokens),
            block_hash=block_hash,
            prev_hash=prev_hash,
            token_ids=tokens,
            payload_len=len(data),
            mtime=time.time(),
            compressed=compressed,
        )
        self._evict()
        return True

    def lookup(self, block_hash: int, layer: int) -> DiskBlockRecord | None:
        rec = self._index.get(block_hash)
        if rec is None or rec.layer != layer:
            return None
        if self.ttl_seconds > 0 and time.time() - rec.mtime > self.ttl_seconds:
            del self._index[block_hash]
            return None
        return rec

    def read_payload(self, rec: DiskBlockRecord) -> bytes:
        f = self._file_path(rec.file_id)
        if f is None:
            del self._index[rec.block_hash]
            raise FileNotFoundError(f"L2 文件 {rec.file_id} 已轮转删除")
        with open(f, "rb") as fh:
            fh.seek(rec.offset + RECORD_HEADER_SIZE + rec.n_tokens * 4)
            data = fh.read(rec.payload_len)
        if rec.compressed and self._lz4 is not None:
            import io

            with self._lz4(io.BytesIO(data), mode="rb") as f:
                data = f.read()
        return data

    def _file_path(self, file_id: int) -> Path | None:
        for f in self._files:
            if self._file_id(f) == file_id:
                return f
        return None

    # -- 淘汰 ---------------------------------------------------------------
    def _evict(self) -> None:
        """超 max_bytes → 删最旧文件（孤儿记录随之回收；索引同步清理）。"""
        while self._total_bytes > self.max_bytes and len(self._files) > 1:
            oldest = self._files.pop(0)
            self._total_bytes -= oldest.stat().st_size
            try:
                oldest.unlink()
            except OSError:
                pass
            for bh in [b for b, r in self._index.items() if r.file_id == self._file_id(oldest)]:
                del self._index[bh]
        if self._files:
            self._current = len(self._files)

    def stats(self) -> dict:
        return {
            "index_entries": len(self._index),
            "total_bytes": self._total_bytes,
            "files": len(self._files),
            "max_bytes": self.max_bytes,
            "compression": self.compression if self._lz4 is not None or self.compression == "none" else "none(fallback)",
        }

    def clear(self) -> None:
        """清空 L2（会话结束 / 显式 reset）。"""
        for f in self._files:
            try:
                f.unlink()
            except OSError:
                pass
        self._files.clear()
        self._index.clear()
        self._total_bytes = 0
        self._current = 0
        self._current_size = 0
