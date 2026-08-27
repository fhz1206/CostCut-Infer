"""tests.test_spec_decode — 投机解码：Ngram proposer + verify + 接受率指标。"""

from __future__ import annotations

import numpy as np

from ccut.spec_decode import NgramProposer, SpecDecoder, resolve_proposer


def test_ngram_proposer_basic():
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]
    ng = NgramProposer(n=5, max_draft=1)
    ng.build(prompt)
    assert ng.propose([1, 2, 3, 4, 5, 6, 7, 8, 9]) == [10]


def test_spec_decoder_no_proposer():
    sd = SpecDecoder(proposer=None, seed=42)
    acc, bonus = sd.step([1, 2, 3], np.random.randn(2, 50).astype(np.float32))
    assert acc == [] and bonus is None
    assert sd.stats.drafts_proposed == 0


def test_resolve_proposer_chooses_ngram_when_mtp_missing():
    prompt = [1, 2, 3, 4, 5, 6, 7, 8]
    sp = resolve_proposer(
        {"enable_mtp": True, "enable_ngram": True, "mtp_draft_tokens": 1, "ngram_window": 5},
        mtp_layer=None,
        prompt=prompt,
    )
    # mtp_layer=None 应降级到 ngram
    assert isinstance(sp, NgramProposer)


def test_resolve_proposer_none_when_all_disabled():
    sp = resolve_proposer({"enable_mtp": False, "enable_ngram": False})
    assert sp is None


def test_spec_decoder_with_ngram_updates_stats():
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]
    ng = NgramProposer(n=5, max_draft=1)
    ng.build(prompt)
    sd = SpecDecoder(proposer=ng, seed=42)
    acc, _bonus = sd.step([1, 2, 3, 4, 5, 6, 7, 8, 9], np.random.randn(2, 50).astype(np.float32))
    assert len(acc) == 1
    assert sd.stats.drafts_proposed == 1
    assert sd.stats.drafts_accepted == 1
