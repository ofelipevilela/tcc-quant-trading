# -*- coding: utf-8 -*-
"""
Metricas de desempenho do backtest.

Mantem as metricas basicas usadas no projeto e adiciona medidas mais adequadas
para analise academica, como Sharpe, Sortino, Calmar, expectancy e SQN.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_PERIODS_PER_YEAR = 96 * 252  # aproximacao para M15


def _infer_periods_per_year(timestamps: Optional[Iterable]) -> int:
    """Infere uma aproximacao de barras por ano a partir dos timestamps."""
    if timestamps is None:
        return DEFAULT_PERIODS_PER_YEAR

    idx = pd.to_datetime(pd.Index(list(timestamps)))
    if len(idx) < 3:
        return DEFAULT_PERIODS_PER_YEAR

    diffs = idx.to_series().diff().dropna()
    median_delta = diffs.median()
    if pd.isna(median_delta) or median_delta.total_seconds() <= 0:
        return DEFAULT_PERIODS_PER_YEAR

    seconds_per_year = 365.25 * 24 * 60 * 60
    periods = int(seconds_per_year / median_delta.total_seconds())
    return max(periods, 1)


def _safe_float(value: float) -> float:
    """Converte valores numpy para float nativo."""
    if value is None:
        return 0.0
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return float(value)


def calculate_performance_metrics(
    equity_curve: list,
    trades: list,
    timestamps: Optional[Iterable] = None,
    periods_per_year: Optional[int] = None,
) -> dict:
    """
    Calcula metricas de performance a partir da curva de equidade e dos trades.

    Parametros:
      equity_curve: lista de valores de capital ao longo do backtest
      trades: lista de dicionarios de trades
      timestamps: index ou lista temporal da curva de equidade
      periods_per_year: barras por ano para anualizacao

    Retorno:
      Dicionario com metricas de retorno, risco e qualidade operacional.
    """
    metrics = {
        "total_return": 0.0,
        "annualized_return": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "total_trades": len(trades),
        "profit_factor": 0.0,
        "payoff_ratio": 0.0,
        "expectancy": 0.0,
        "avg_trade_return_pct": 0.0,
        "trade_return_std_pct": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "calmar_ratio": 0.0,
        "sqn": 0.0,
        "winning_trades": 0,
        "losing_trades": 0,
        "long_trades": 0,
        "short_trades": 0,
        "avg_bars_held": 0.0,
    }

    if not equity_curve:
        return metrics

    eq_array = np.asarray(equity_curve, dtype=float)
    initial_capital = float(eq_array[0])
    final_capital = float(eq_array[-1])
    metrics["total_return"] = ((final_capital - initial_capital) / initial_capital) * 100.0

    running_max = np.maximum.accumulate(eq_array)
    drawdowns = (running_max - eq_array) / np.maximum(running_max, 1e-12)
    max_drawdown_decimal = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
    metrics["max_drawdown"] = max_drawdown_decimal * 100.0

    effective_periods = periods_per_year or _infer_periods_per_year(timestamps)
    equity_returns = np.diff(eq_array) / np.maximum(eq_array[:-1], 1e-12)

    if len(equity_returns) > 1:
        mean_ret = float(np.mean(equity_returns))
        std_ret = float(np.std(equity_returns, ddof=1)) if len(equity_returns) > 1 else 0.0
        downside = equity_returns[equity_returns < 0]
        downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0

        if std_ret > 0:
            metrics["sharpe_ratio"] = mean_ret / std_ret * np.sqrt(effective_periods)
        if downside_std > 0:
            metrics["sortino_ratio"] = mean_ret / downside_std * np.sqrt(effective_periods)

    if timestamps is not None:
        idx = pd.to_datetime(pd.Index(list(timestamps)))
        if len(idx) >= 2:
            elapsed_years = max((idx[-1] - idx[0]).total_seconds() / (365.25 * 24 * 60 * 60), 1e-12)
            growth = max(final_capital / max(initial_capital, 1e-12), 1e-12)
            metrics["annualized_return"] = (growth ** (1.0 / elapsed_years) - 1.0) * 100.0
        else:
            metrics["annualized_return"] = metrics["total_return"]
    else:
        n_periods = max(len(eq_array) - 1, 1)
        growth = max(final_capital / max(initial_capital, 1e-12), 1e-12)
        metrics["annualized_return"] = (growth ** (effective_periods / n_periods) - 1.0) * 100.0

    if max_drawdown_decimal > 0:
        metrics["calmar_ratio"] = (metrics["annualized_return"] / 100.0) / max_drawdown_decimal

    if trades:
        pnls = np.asarray([trade.get("pnl", 0.0) for trade in trades], dtype=float)
        trade_returns = np.asarray([trade.get("return_pct", 0.0) for trade in trades], dtype=float)
        bars_held = np.asarray([trade.get("bars_held", 0.0) for trade in trades], dtype=float)

        winning = pnls[pnls > 0]
        losing = pnls[pnls <= 0]

        metrics["winning_trades"] = int(len(winning))
        metrics["losing_trades"] = int(len(losing))
        metrics["win_rate"] = (len(winning) / len(trades)) * 100.0
        metrics["long_trades"] = int(sum(1 for trade in trades if trade.get("type") == "LONG"))
        metrics["short_trades"] = int(sum(1 for trade in trades if trade.get("type") == "SHORT"))
        metrics["avg_bars_held"] = float(np.mean(bars_held)) if len(bars_held) else 0.0

        gross_profit = float(np.sum(winning)) if len(winning) else 0.0
        gross_loss = float(abs(np.sum(losing))) if len(losing) else 0.0
        metrics["profit_factor"] = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        avg_win = float(np.mean(winning)) if len(winning) else 0.0
        avg_loss = float(abs(np.mean(losing))) if len(losing) else 0.0
        metrics["payoff_ratio"] = (avg_win / avg_loss) if avg_loss > 0 else float("inf")
        metrics["expectancy"] = float(np.mean(pnls))
        metrics["avg_trade_return_pct"] = float(np.mean(trade_returns)) if len(trade_returns) else 0.0
        metrics["trade_return_std_pct"] = float(np.std(trade_returns, ddof=1)) if len(trade_returns) > 1 else 0.0

        if len(trade_returns) > 1:
            trade_std = float(np.std(trade_returns, ddof=1))
            if trade_std > 0:
                metrics["sqn"] = (float(np.mean(trade_returns)) / trade_std) * np.sqrt(len(trades))

    return {key: _safe_float(value) if not isinstance(value, (int, float)) else value for key, value in metrics.items()}
