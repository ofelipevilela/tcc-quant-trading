# -*- coding: utf-8 -*-
"""Niveis de risco estruturais usados por target e backtest."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def compute_structure_stop_levels(
    df: pd.DataFrame,
    *,
    atr_mult: float = 1.0,
    stop_buffer_atr: float = 0.05,
    sweep_max_age: int = 60,
    min_risk_atr: float = 0.25,
    max_risk_atr: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula stops estruturais causais para longs e shorts.

    A prioridade segue a leitura operacional usada no projeto:
    1. ultimo sweep valido na direcao do trade;
    2. perna estrutural atual (`leg_low`/`leg_high`);
    3. fallback por ATR quando a estrutura estiver indisponivel ou distante.
    """
    required = {"close", "high", "low", "atr"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame sem colunas obrigatorias: {sorted(missing)}")

    close = df["close"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    atr = df["atr"].to_numpy(dtype=float)
    length = len(df)

    is_sweep = (
        df["is_sweep"].to_numpy(dtype=bool)
        if "is_sweep" in df.columns
        else np.zeros(length, dtype=bool)
    )
    sweep_dir = (
        df["sweep_direction"].to_numpy(dtype=np.int8)
        if "sweep_direction" in df.columns
        else np.zeros(length, dtype=np.int8)
    )
    leg_low = (
        df["leg_low"].to_numpy(dtype=float)
        if "leg_low" in df.columns
        else np.full(length, np.nan)
    )
    leg_high = (
        df["leg_high"].to_numpy(dtype=float)
        if "leg_high" in df.columns
        else np.full(length, np.nan)
    )

    long_stops = np.zeros(length, dtype=float)
    short_stops = np.zeros(length, dtype=float)
    last_bull_sweep_low = np.nan
    last_bull_sweep_idx = -10**9
    last_bear_sweep_high = np.nan
    last_bear_sweep_idx = -10**9

    def fallback_long(entry: float, current_atr: float) -> float:
        return entry - max(current_atr * atr_mult, 1e-6)

    def fallback_short(entry: float, current_atr: float) -> float:
        return entry + max(current_atr * atr_mult, 1e-6)

    def valid_risk(entry: float, stop: float, current_atr: float, direction: int) -> bool:
        if not np.isfinite(stop) or current_atr <= 0:
            return False
        risk = (entry - stop) if direction == 1 else (stop - entry)
        if risk <= 1e-8:
            return False
        risk_atr = risk / max(current_atr, 1e-8)
        return min_risk_atr <= risk_atr <= max_risk_atr

    for i in range(length):
        entry = close[i]
        current_atr = atr[i] if atr[i] > 0 else np.nanmedian(atr[max(0, i - 50):i + 1])
        if not np.isfinite(current_atr) or current_atr <= 0:
            current_atr = max(abs(entry) * 0.001, 1e-6)

        if is_sweep[i] and sweep_dir[i] == 1:
            last_bull_sweep_low = lows[i]
            last_bull_sweep_idx = i
        elif is_sweep[i] and sweep_dir[i] == -1:
            last_bear_sweep_high = highs[i]
            last_bear_sweep_idx = i

        long_stop = np.nan
        if (i - last_bull_sweep_idx) <= sweep_max_age:
            long_stop = last_bull_sweep_low - (current_atr * stop_buffer_atr)
        if not valid_risk(entry, long_stop, current_atr, direction=1):
            candidate = leg_low[i]
            long_stop = candidate - (current_atr * stop_buffer_atr)
        if not valid_risk(entry, long_stop, current_atr, direction=1):
            long_stop = fallback_long(entry, current_atr)

        short_stop = np.nan
        if (i - last_bear_sweep_idx) <= sweep_max_age:
            short_stop = last_bear_sweep_high + (current_atr * stop_buffer_atr)
        if not valid_risk(entry, short_stop, current_atr, direction=-1):
            candidate = leg_high[i]
            short_stop = candidate + (current_atr * stop_buffer_atr)
        if not valid_risk(entry, short_stop, current_atr, direction=-1):
            short_stop = fallback_short(entry, current_atr)

        long_stops[i] = long_stop
        short_stops[i] = short_stop

    return long_stops, short_stops
