"""tests.test_config — Config 三源合并 / casefold / 取值域校验 / 列表打印。"""

from __future__ import annotations

import os

import pytest

from ccut.config import Config, ConfigError, render_params_table


def test_default_build():
    cfg = Config.build(toml_path=None)
    assert cfg.get("temperature", "sampling") == 1.0
    assert cfg.get("max_num_seqs", "engine") == 8
    assert cfg.get("kv_l1_bytes", "kv_cache") == 1_073_741_824


def test_cli_casefold():
    """大小写不敏感 + 键存在性校验。"""
    cfg = Config.build(argv=["--Temperature=0.95", "--kv-l1-bytes=512mb", "--kv-policy=disk_first"])
    assert cfg.get("temperature", "sampling") == pytest.approx(0.95)
    assert cfg.get("kv_l1_bytes", "kv_cache") == 512 * 1024 * 1024
    assert cfg.get("kv_policy", "kv_cache") == "disk_first"


def test_cli_bare_flag_is_bool():
    """裸 --flag 视为布尔（True）。"""
    cfg = Config.build(argv=["--ignore-eos", "--enable-mtp"])
    assert cfg.get("ignore_eos", "sampling") is True
    assert cfg.get("enable_mtp", "spec_decode") is True


def test_cli_double_equals_lenient():
    """--key==value → 剥掉多余 ``=``。"""
    cfg = Config.build(argv=["--top_k==20"])
    assert cfg.get("top_k", "sampling") == 20


def test_invalid_choice_rejected():
    with pytest.raises(ConfigError) as exc:
        Config.build(argv=["--kv-policy=bogus"])
    assert "kv_policy" in str(exc.value)


def test_out_of_range_rejected():
    with pytest.raises(ConfigError) as exc:
        Config.build(argv=["--kv-l1-bytes=100kb"])
    assert "kv_l1_bytes" in str(exc.value)


def test_toml_overrides_and_cli_priority():
    """toml < env < CLI 优先级。"""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "engine.toml"
        p.write_text(
            "[sampling]\ntemperature = 0.5\n",
            encoding="utf-8",
        )
        # 仅 toml
        c1 = Config.build(argv=[], toml_path=p)
        assert c1.get("temperature", "sampling") == pytest.approx(0.5)
        # CLI 覆盖
        c2 = Config.build(argv=["--temperature=0.9"], toml_path=p)
        assert c2.get("temperature", "sampling") == pytest.approx(0.9)


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("CCUT_TEMPERATURE", "0.42")
    cfg = Config.build(argv=[])
    assert cfg.get("temperature", "sampling") == pytest.approx(0.42)


def test_env_scoped_priority(monkeypatch):
    """CCUT_SAMPLING__TEMPERATURE 优先级 > CCUT_TEMPERATURE。"""
    monkeypatch.setenv("CCUT_TEMPERATURE", "0.1")
    monkeypatch.setenv("CCUT_SAMPLING__TEMPERATURE", "0.7")
    cfg = Config.build(argv=[])
    assert cfg.get("temperature", "sampling") == pytest.approx(0.7)


def test_list_params_table_contains_sections():
    text = render_params_table()
    assert "[model]" in text
    assert "[engine]" in text
    assert "[kv_cache]" in text
    assert "[quant]" in text
    assert "[resources]" in text
    assert "CCUT_<KEY>" in text  # 环境变量说明


def test_tensor_parallel_strict_single():
    """单机版：TP/PP ≠ 1 显式拒绝。"""
    with pytest.raises(ConfigError) as exc:
        Config.build(argv=["--tensor-parallel-size=2"])
    assert "tensor_parallel_size" in str(exc.value) or "单卡" in str(exc.value) or "单机" in str(exc.value)


def test_kv_water_levels_relationship():
    with pytest.raises(ConfigError):
        Config.build(argv=["--kv-evict-low-water=0.9", "--kv-evict-high-water=0.5"])


def test_resolve_bytes():
    from ccut.config import resolve_bytes

    assert resolve_bytes("1GB") == 1024**3
    assert resolve_bytes("512mb") == 512 * 1024**2
    assert resolve_bytes("1024") == 1024
    with pytest.raises(ConfigError):
        resolve_bytes("")
