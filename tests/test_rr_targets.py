# -*- coding: utf-8 -*-
"""Testes dos targets por risco-retorno."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from real_market_utils import generate_rr_targets


def _df(highs, lows, closes):
    return pd.DataFrame({
        "open": closes,
        "high": highs,
        "low": lows,
        "close": closes,
        "atr": np.ones(len(closes), dtype=float),
    })


def test_rr_target_positive_when_long_wins_and_short_loses():
    data = _df(
        highs=[100.2, 102.2, 100.0],
        lows=[99.8, 99.5, 99.5],
        closes=[100.0, 101.5, 101.0],
    )

    targets = generate_rr_targets(
        data,
        horizon=1,
        reward_to_risk=2.0,
        stop_mode="atr",
        atr_mult=1.0,
    )

    assert targets[0] == 1.0


def test_rr_target_negative_when_short_wins_and_long_loses():
    data = _df(
        highs=[100.2, 100.5, 100.0],
        lows=[99.8, 97.8, 98.0],
        closes=[100.0, 98.5, 99.0],
    )

    targets = generate_rr_targets(
        data,
        horizon=1,
        reward_to_risk=2.0,
        stop_mode="atr",
        atr_mult=1.0,
    )

    assert targets[0] == -1.0


def test_rr_target_zero_when_both_directions_are_stopped():
    data = _df(
        highs=[100.2, 101.2, 100.0],
        lows=[99.8, 98.8, 99.0],
        closes=[100.0, 100.0, 100.0],
    )

    targets = generate_rr_targets(
        data,
        horizon=1,
        reward_to_risk=1.0,
        stop_mode="atr",
        atr_mult=1.0,
    )

    assert targets[0] == 0.0
