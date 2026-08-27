"""ccut.kv — KV cache 双层存储（R1：L1 内存块池 + L2 磁盘）。

- blocks：L1 固定块池 + 前缀滚动哈希（token 序列分块，跨步复用）。
- disk：L2 追加式磁盘块存储（自描述记录头，启动扫头重建索引）。
- coordinator：L1/L2 统一调度（lookup 三级：L1 命中 / L2 命中→装回 / miss）。

层模型（§3.4-7 参数化）：
- 仅 full_attn 层有 KV 块池（Ornith：10 层 × 2kv × 256d × 2B/token = 2KB/token/层）；
- linear_attn 层是 GDN 递归状态（固定 2MB/请求，不进块池，由请求生命周期管理）；
- 每层独立 :class:`BlockPool`，前缀哈希索引全局（token 序列级）。
"""
