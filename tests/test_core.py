"""core 层不变量：确定性 seed 与 EventSeq 契约。"""

from __future__ import annotations

import numpy as np

from core.seed import get_seed, rng, set_all
from core.types import EventSeq


def test_set_all_is_reproducible():
    set_all(7)
    a = rng(7).standard_normal(5)
    set_all(7)
    b = rng(7).standard_normal(5)
    assert np.array_equal(a, b)
    assert get_seed() == 7


def test_eventseq_sort_and_T():
    t = np.array([1.0, 2.0, 3.0])
    d = np.array([0, 1, 2])
    seq = EventSeq(times=t, dims=d, d=3, T=0.0)
    assert np.all(np.diff(seq.times) > 0)
    assert seq.T == 3.0
    assert seq.dims.dtype.kind in ("i", "u")
    assert len(seq) == 3


def test_eventseq_empty():
    seq = EventSeq(times=np.array([]), dims=np.array([]), d=2, T=5.0)
    assert len(seq) == 0
    assert seq.T == 5.0
