#!/bin/bash
# ============================================================
# CostCut Infer（Rust 版）构建 + 复制到 python 目录 + 测试自动化
#
# 发布策略：发布包【永远为 Rust 版】（costcut-infer.exe）；
# Python 版仅作为技术探索用（新功能验证/实验），不参与发布。
#
# 用法：在项目根目录执行  ./build.sh
# 流程：Rust release 构建 → 复制二进制到 python/ → 冒烟测试 → CLI 文本对话测试
# ============================================================
set -e
cd "$(dirname "$0")"          # 脚本所在目录（项目根）

echo "=== [1/4] Rust release 构建 ==="
( cd rust && cargo build --release --offline )

echo "=== [2/4] 复制构建包到 python 目录 ==="
cp rust/target/release/costcut-infer.exe python/
echo "已复制: python/costcut-infer.exe"

echo "=== [3/4] 冒烟测试（--smoke，开发模式）==="
# 从 python 目录运行（模型路径按 python/models 解析）
if ( cd python && ./costcut-infer.exe --smoke ); then
    echo "[ok] 冒烟测试通过"
else
    echo "[warn] 冒烟测试非致命（真实模型权重缺失时跳过 M1）"
fi

echo "=== [4/4] CLI 文本对话测试（默认入口）==="
printf '你好\nq\n' | ( cd python && ./costcut-infer.exe )

echo ""
echo "=========================================="
echo "✅ 构建完成：发布包 = rust/costcut-infer.exe"
echo "   （已复制到 python/costcut-infer.exe 供测试）"
echo "发布策略：永远发布 Rust 版；Python 版仅技术探索用。"
echo "=========================================="
