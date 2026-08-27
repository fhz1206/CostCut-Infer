"""ccut.models — 模型装配层（R8 架构账本 + 声明式族模板）。

- spec：ModelSpec（config.json → 架构参数 + 层模板序列，纯声明）。
- registry：架构账本查找（registry_table.json + 族模板匹配 + L0/L1/L2 归层）。
- families/：族模板 JSON（Ornith 主测家族 + 标准家族骨架）。
- generic：通用组装器（ModelSpec + blocks + quant → 层前向，无手写模型代码）。
- backend_transformers：L1 兜底后端（transformers AutoModel 运行）。
"""
