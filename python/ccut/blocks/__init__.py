"""ccut.blocks — 共享积木（跨家族复用的算子层，§3.4-1）。

纯数值实现（numpy + numba），**不持有权重**——权重段由 WeightReader（R10）
按 LayerQuantSpec 提供，method（quant/method.py）负责 dequant+matmul。
块只描述「数据流」：

- norm：RMSNorm（bf16 精确路径）。
- rope：RoPE（rope_scaling: none/linear/dynamic/yarn）。
- attn_gqa：GQA/MQA 注意力（full_attn 层）。
- attn_gdn：Gated DeltaNet 线性注意力（Ornith linear_attn 层）。
- attn_mla：DeepSeek-V3 MLA 压缩注意力（L0 家族用）。
- attn_kimi_linear：Kimi K2/K3 线性注意力（KDA）。
- moe：专家路由（gate → top-k → softmax）+ 专家前向数据流。
- mtp：多 token 预测模块（1 层 draft）。
- heads：lm_head / embed 投影 + logprobs。

每个块暴露 ``forward(inputs) -> outputs``，输入输出约定见各模块 docstring。
"""
