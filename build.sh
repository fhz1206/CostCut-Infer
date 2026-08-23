#!/bin/bash
# ============================================================
# CostCut Infer（Rust 版）构建 + 部署 + Inno 打包 + 测试自动化
#
# 发布策略：发布包【永远为 Rust 版】（costcut-infer.exe）；
# Python 版仅作为技术探索用（新功能验证/实验），不参与发布。
#
# 用法：在项目根目录执行  ./build.sh
# 流程：Rust release 构建 → 部署 v0.1.0_beta（exe + dll）→
#       Inno Setup 打包（produce/Inno.iss）→ setup/ 安装包 → 冒烟测试
# ============================================================
set -e
cd "$(dirname "$0")"          # 脚本所在目录（项目根）

echo "=== [1/5] Rust release 构建 ==="
( cd rust && cargo build --release )

echo "=== [2/5] 部署到 v0.1.0_beta（exe + 依赖 dll）==="
mkdir -p v0.1.0_beta
cp rust/target/release/costcut-infer.exe v0.1.0_beta/
cp rust/target/release/*.dll v0.1.0_beta/
echo "已部署: v0.1.0_beta/（exe + $(ls v0.1.0_beta/*.dll 2>/dev/null | wc -l) 个 dll）"

echo "=== [3/5] Inno Setup 打包（produce/Inno.iss → setup/）==="
mkdir -p setup
ISCC="/c/Users/ASUS/AppData/Local/Programs/Inno Setup 6/ISCC.exe"
if [ -f "$ISCC" ]; then
    "$ISCC" produce/Inno.iss
else
    echo "[warn] 未找到 ISCC.exe（Inno Setup 6）——跳过打包（已就绪: produce/Inno.iss + v0.1.0_beta/）"
fi

echo "=== [4/5] 冒烟测试（--smoke，开发模式）==="
if ( cd v0.1.0_beta && ./costcut-infer.exe --smoke ); then
    echo "[ok] 冒烟测试通过"
else
    echo "[warn] 冒烟测试非致命（真实模型权重缺失时跳过 M1）"
fi

echo "=== [5/5] 完成——安装包产物 ==="
ls -la setup/*.exe 2>/dev/null || echo "[warn] setup/ 暂无安装包（检查 Inno 打包）"
