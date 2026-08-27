"""tests.test_models — 架构账本 + ModelSpec 解析 + 族模板匹配。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ccut.models.registry import (
    load_family_templates,
    load_registry_table,
    resolve_architecture,
)
from ccut.models.spec import load_model_spec


def test_registry_table_loads():
    tab = load_registry_table()
    assert tab["total"] >= 1
    assert "architectures" in tab


def test_resolve_ornith_via_text_side():
    """VLM 包装器 (Qwen3_5MoeForConditionalGeneration) → text_side 走 L0。"""
    tab = load_registry_table()
    fams = load_family_templates()
    r = resolve_architecture("Qwen3_5MoeForConditionalGeneration", "auto", table=tab, families=fams)
    assert r.tier == "L0"
    assert r.plan is not None


def test_resolve_unknown_rejected():
    tab = load_registry_table()
    fams = load_family_templates()
    r = resolve_architecture("TotallyUnknownArch", "auto", table=tab, families=fams)
    assert r.tier == "L2"
    assert not r.accepted


def test_resolve_llama_strict_without_family():
    tab = load_registry_table()
    # 假设未落地 llama 族模板时 strict 应拒绝
    fams_strict = {k: v for k, v in load_family_templates().items() if k != "llama"}
    r = resolve_architecture("LlamaForCausalLM", "strict", table=tab, families=fams_strict)
    assert r.tier == "L2"


@pytest.mark.slow
def test_spec_parses_ornith():
    spec = load_model_spec("python/models/Ornith-1.5-35B-A3B-MTP-FP8")
    assert spec.num_hidden_layers == 40
    assert spec.num_full_attn_layers == 10
    assert spec.num_linear_attn_layers == 30
    assert spec.full_attn_layers()[:3] == [3, 7, 11]
    assert spec.moe is not None
    assert spec.moe.num_experts == 256
    assert spec.moe.top_k == 8
    assert spec.mtp is not None
    assert spec.mtp.num_layers == 1
    assert spec.gdn_num_key_heads == 16
    assert spec.gdn_num_value_heads == 32
