# -*- coding: utf-8 -*-
"""
Comparacao walk-forward entre modos de leitura SMC.

Objetivo:
- medir o quanto o desempenho muda quando a traducao estrutural troca de
  `causal_raw` para `legacy_like`;
- manter o mesmo pipeline de treino, calibracao e backtest em ambos os modos;
- gerar artefatos consolidados para decidir se o modo legado merece voltar
  como opcao principal de pesquisa.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    print("MetaTrader5 nao instalado.")
    sys.exit(1)

import torch

from backtest.engine import simulate_trading_from_scores
from data.mt5_client import MT5Client
from real_market_utils import (
    build_loader,
    compute_score_statistics,
    compute_signal_metrics,
    create_rescaled_model,
    evaluate_thresholds_with_backtest,
    plot_training_history,
    predict_scores,
    prepare_market_data,
    select_activation_threshold,
    select_backtest_threshold,
    set_seed,
    train_model,
)
from run_walkforward import build_folds
from smc.feature_factory import (
    FEATURE_MODE_CAUSAL_BOS_ANCHORED,
    FEATURE_MODE_CAUSAL_RAW,
    FEATURE_MODE_CAUSAL_V2,
    FEATURE_MODE_CAUSAL_V3,
    FEATURE_MODE_LEGACY_LIKE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_walkforward_mode_comparison")


OUTPUT_ROOT = Path("outputs/walkforward_compare")
SUMMARY_CSV = OUTPUT_ROOT / "mode_comparison_summary.csv"
SUMMARY_JSON = OUTPUT_ROOT / "mode_comparison_summary.json"
AGGREGATE_PLOT = OUTPUT_ROOT / "mode_comparison_aggregate.png"
FOLD_PLOT = OUTPUT_ROOT / "mode_comparison_folds.png"
EQUITY_PLOT = OUTPUT_ROOT / "mode_comparison_oos_equity.png"


def _safe_corr(x_values: np.ndarray, y_values: np.ndarray) -> float:
    if len(x_values) < 3:
        return 0.0
    if np.std(x_values) <= 1e-8 or np.std(y_values) <= 1e-8:
        return 0.0
    return float(np.corrcoef(x_values, y_values)[0, 1])


def _plot_mode_threshold_heatmap(
    output_path: Path,
    threshold_records: List[Dict[str, float]],
    feature_mode: str,
) -> None:
    df = pd.DataFrame(threshold_records)
    if df.empty:
        return

    pivot = df.pivot(index="threshold", columns="fold", values="profit_factor").sort_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(int(value)) for value in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{value:.2f}" for value in pivot.index])
    ax.set_xlabel("Fold")
    ax.set_ylabel("Threshold")
    ax.set_title(f"Profit Factor na validacao por threshold ({feature_mode})")

    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.iloc[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)

    fig.colorbar(im, ax=ax, label="Profit Factor")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_mode_fold_metrics(output_path: Path, results_df: pd.DataFrame, feature_mode: str) -> None:
    if results_df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fold_labels = results_df["fold"].astype(int).astype(str).tolist()

    axes[0, 0].bar(fold_labels, results_df["test_profit_factor"], color="#2a9d8f")
    axes[0, 0].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Profit Factor")
    axes[0, 0].grid(alpha=0.2, axis="y")

    axes[0, 1].bar(fold_labels, results_df["test_win_rate"], color="#457b9d")
    axes[0, 1].axhline(50.0, color="gray", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Win Rate (%)")
    axes[0, 1].grid(alpha=0.2, axis="y")

    axes[1, 0].bar(fold_labels, results_df["test_max_drawdown"], color="#e76f51")
    axes[1, 0].set_title("Max Drawdown (%)")
    axes[1, 0].grid(alpha=0.2, axis="y")

    axes[1, 1].bar(fold_labels, results_df["test_total_trades"], color="#e9c46a")
    axes[1, 1].set_title("Trades no teste")
    axes[1, 1].grid(alpha=0.2, axis="y")

    for ax in axes.flat:
        ax.set_xlabel("Fold")

    fig.suptitle(f"Walk-forward por fold ({feature_mode})", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _build_stitched_equity(fold_results: List[Dict]) -> np.ndarray:
    stitched: List[float] = [100.0]

    for fold in sorted(fold_results, key=lambda item: int(item["fold"])):
        curve = np.asarray(fold["test_equity_curve"], dtype=float)
        if len(curve) == 0 or curve[0] <= 0:
            continue

        normalized = curve / curve[0]
        segment = stitched[-1] * normalized
        stitched.extend(segment[1:].tolist())

    return np.asarray(stitched, dtype=float)


def _plot_comparison_aggregate(aggregate_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    modes = aggregate_df["feature_mode"].tolist()

    axes[0, 0].bar(modes, aggregate_df["mean_test_profit_factor"], color=["#e76f51", "#2a9d8f"])
    axes[0, 0].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Profit Factor medio")
    axes[0, 0].grid(alpha=0.2, axis="y")

    axes[0, 1].bar(modes, aggregate_df["mean_test_total_return"], color=["#e76f51", "#2a9d8f"])
    axes[0, 1].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Retorno medio por fold (%)")
    axes[0, 1].grid(alpha=0.2, axis="y")

    axes[1, 0].bar(modes, aggregate_df["mean_test_win_rate"], color=["#e76f51", "#2a9d8f"])
    axes[1, 0].axhline(50.0, color="gray", linestyle="--", linewidth=1)
    axes[1, 0].set_title("Win rate medio (%)")
    axes[1, 0].grid(alpha=0.2, axis="y")

    axes[1, 1].bar(modes, aggregate_df["positive_pf_folds"], color=["#e76f51", "#2a9d8f"])
    axes[1, 1].set_title("Folds com PF > 1")
    axes[1, 1].grid(alpha=0.2, axis="y")

    for ax in axes.flat:
        ax.set_xlabel("Modo")

    fig.suptitle("Comparacao walk-forward entre modos SMC", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(AGGREGATE_PLOT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_comparison_folds(folds_df: pd.DataFrame) -> None:
    if folds_df.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    pivot_pf = folds_df.pivot(index="fold", columns="feature_mode", values="test_profit_factor").sort_index()
    pivot_ret = folds_df.pivot(index="fold", columns="feature_mode", values="test_total_return").sort_index()

    pivot_pf.plot(kind="bar", ax=axes[0], color=["#e76f51", "#2a9d8f"])
    axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_title("Profit Factor por fold")
    axes[0].set_ylabel("PF")
    axes[0].grid(alpha=0.2, axis="y")

    pivot_ret.plot(kind="bar", ax=axes[1], color=["#e76f51", "#2a9d8f"])
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("Retorno por fold (%)")
    axes[1].set_ylabel("Retorno (%)")
    axes[1].grid(alpha=0.2, axis="y")
    axes[1].set_xlabel("Fold")

    fig.suptitle("Comparacao fold a fold entre modos SMC", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FOLD_PLOT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_comparison_equity(mode_payloads: List[Dict]) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    for payload in mode_payloads:
        stitched = _build_stitched_equity(payload["folds"])
        if len(stitched) == 0:
            continue
        ax.plot(np.arange(len(stitched)), stitched, linewidth=2, label=payload["feature_mode"])

    ax.axhline(100.0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Equity OOS encadeada por modo (base = 100)")
    ax.set_xlabel("Barras acumuladas de teste")
    ax.set_ylabel("Capital normalizado")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(EQUITY_PLOT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _run_mode(
    *,
    df: pd.DataFrame,
    symbol: str,
    timeframe_label: str,
    feature_mode: str,
    n_bars: int,
    swing_window: int,
    horizon: int,
    atr_mult: float,
    min_activation: float,
    seed: int,
    training_cfg: Dict,
    threshold_cfg: Dict,
    backtest_cfg: Dict,
    fold_cfg: Dict,
) -> Dict:
    mode_output_dir = OUTPUT_ROOT / feature_mode
    mode_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Preparando modo %s...", feature_mode)
    prepared = prepare_market_data(
        df=df,
        swing_window=swing_window,
        horizon=horizon,
        atr_mult=atr_mult,
        min_activation=min_activation,
        feature_mode=feature_mode,
    )

    x_values = prepared.features.values
    y_values = prepared.targets
    n_samples = len(x_values)
    folds = build_folds(
        n_samples=n_samples,
        initial_train=fold_cfg["initial_train"],
        val_size=fold_cfg["val_size"],
        test_size=fold_cfg["test_size"],
        step_size=fold_cfg["step_size"],
    )
    if not folds:
        raise RuntimeError(f"Nao foi possivel construir folds para o modo {feature_mode}.")

    logger.info(
        "Modo %s | amostras apos trimming=%d | folds=%d",
        feature_mode,
        n_samples,
        len(folds),
    )

    fold_results: List[Dict] = []
    threshold_records: List[Dict[str, float]] = []

    for fold in folds:
        fold_id = int(fold["fold"])
        logger.info(
            "Modo %s | fold %d | treino=[%d:%d] validacao=[%d:%d] teste=[%d:%d]",
            feature_mode,
            fold_id,
            fold["train_start"],
            fold["train_end"],
            fold["val_start"],
            fold["val_end"],
            fold["test_start"],
            fold["test_end"],
        )

        set_seed(seed + fold_id)

        x_train = x_values[fold["train_start"]:fold["train_end"]]
        y_train = y_values[fold["train_start"]:fold["train_end"]]
        x_val = x_values[fold["val_start"]:fold["val_end"]]
        y_val = y_values[fold["val_start"]:fold["val_end"]]
        x_test = x_values[fold["test_start"]:fold["test_end"]]
        y_test = y_values[fold["test_start"]:fold["test_end"]]

        df_val = prepared.df_smc.iloc[fold["val_start"]:fold["val_end"]].copy()
        df_test = prepared.df_smc.iloc[fold["test_start"]:fold["test_end"]].copy()

        train_loader = build_loader(x_train, y_train, training_cfg["batch_size"], shuffle=True)
        val_loader = build_loader(x_val, y_val, training_cfg["batch_size"], shuffle=False)

        model = create_rescaled_model()
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=training_cfg["epochs"],
            learning_rate=training_cfg["learning_rate"],
            weight_decay=training_cfg["weight_decay"],
            sigma_min=training_cfg["sigma_min"],
            lambda_rule_usage=training_cfg["lambda_rule_usage"],
            patience=training_cfg["patience"],
        )

        fold_plot = mode_output_dir / f"fold_{fold_id:02d}_training_loss.png"
        plot_training_history(history, str(fold_plot))

        val_scores, val_rules = predict_scores(model, x_val)
        test_scores, test_rules = predict_scores(model, x_test)

        directional_threshold, directional_diagnostics = select_activation_threshold(
            val_scores,
            y_val,
            candidate_thresholds=threshold_cfg["candidates"],
            min_activations=threshold_cfg["min_activations_directional"],
        )
        backtest_diagnostics = evaluate_thresholds_with_backtest(
            df_slice=df_val,
            scores=val_scores,
            candidate_thresholds=threshold_cfg["candidates"],
            initial_capital=backtest_cfg["initial_capital"],
            risk_per_trade=backtest_cfg["risk_per_trade"],
            reward_to_risk=backtest_cfg["reward_to_risk"],
            stop_mode=backtest_cfg["stop_mode"],
            atr_stop_mult=backtest_cfg["atr_stop_mult"],
            atr_target_mult=backtest_cfg["atr_target_mult"],
            max_holding_bars=backtest_cfg["max_holding_bars"],
            min_trades=threshold_cfg["min_trades_backtest"],
        )
        threshold, selected_threshold_metric = select_backtest_threshold(
            backtest_diagnostics,
            fallback_threshold=directional_threshold,
        )

        for item in backtest_diagnostics:
            threshold_records.append({
                "fold": fold_id,
                "threshold": float(item["threshold"]),
                "profit_factor": float(item["profit_factor"]),
            })

        val_signal_metric = compute_signal_metrics(val_scores, y_val, threshold)
        test_signal_metric = compute_signal_metrics(test_scores, y_test, threshold)
        test_backtest_metric = evaluate_thresholds_with_backtest(
            df_slice=df_test,
            scores=test_scores,
            candidate_thresholds=[threshold],
            initial_capital=backtest_cfg["initial_capital"],
            risk_per_trade=backtest_cfg["risk_per_trade"],
            reward_to_risk=backtest_cfg["reward_to_risk"],
            stop_mode=backtest_cfg["stop_mode"],
            atr_stop_mult=backtest_cfg["atr_stop_mult"],
            atr_target_mult=backtest_cfg["atr_target_mult"],
            max_holding_bars=backtest_cfg["max_holding_bars"],
            min_trades=threshold_cfg["min_trades_backtest"],
        )[0]

        val_stats = compute_score_statistics(val_scores)
        test_stats = compute_score_statistics(test_scores)
        test_rule_means = test_rules.mean(axis=0)

        model_path = mode_output_dir / f"fold_{fold_id:02d}_model.pt"
        torch.save(model.state_dict(), model_path)

        test_equity_curve, _ = simulate_trading_from_scores(
            data=df_test,
            scores=test_scores,
            initial_capital=backtest_cfg["initial_capital"],
            risk_per_trade=backtest_cfg["risk_per_trade"],
            reward_to_risk=backtest_cfg["reward_to_risk"],
            activation_threshold=threshold,
            stop_mode=backtest_cfg["stop_mode"],
            atr_stop_mult=backtest_cfg["atr_stop_mult"],
            atr_target_mult=backtest_cfg["atr_target_mult"],
            max_holding_bars=backtest_cfg["max_holding_bars"],
        )

        result = {
            "feature_mode": feature_mode,
            "fold": fold_id,
            "train_samples": len(x_train),
            "val_samples": len(x_val),
            "test_samples": len(x_test),
            "train_start_time": str(prepared.df_smc.index[fold["train_start"]]),
            "train_end_time": str(prepared.df_smc.index[fold["train_end"] - 1]),
            "val_start_time": str(df_val.index[0]),
            "val_end_time": str(df_val.index[-1]),
            "test_start_time": str(df_test.index[0]),
            "test_end_time": str(df_test.index[-1]),
            "directional_threshold": directional_threshold,
            "selected_threshold": threshold,
            "selected_threshold_source": "backtest" if threshold != directional_threshold else "directional_fallback",
            "val_directional_accuracy": val_signal_metric["directional_accuracy"],
            "val_coverage": val_signal_metric["coverage"],
            "val_score_target_corr": val_signal_metric["score_target_corr"],
            "test_directional_accuracy": test_signal_metric["directional_accuracy"],
            "test_coverage": test_signal_metric["coverage"],
            "test_score_target_corr": test_signal_metric["score_target_corr"],
            "test_full_score_corr": _safe_corr(test_scores, y_test),
            "test_score_mean": test_stats["mean"],
            "test_score_std": test_stats["std"],
            "val_score_mean": val_stats["mean"],
            "val_score_std": val_stats["std"],
            "test_rule_dominance": float(np.max(test_rule_means)),
            "test_active_rules_95": int(np.sum(test_rule_means > 0.05)),
            "test_total_return": test_backtest_metric["total_return"],
            "test_annualized_return": test_backtest_metric["annualized_return"],
            "test_max_drawdown": test_backtest_metric["max_drawdown"],
            "test_win_rate": test_backtest_metric["win_rate"],
            "test_profit_factor": test_backtest_metric["profit_factor"],
            "test_payoff_ratio": test_backtest_metric["payoff_ratio"],
            "test_expectancy": test_backtest_metric["expectancy"],
            "test_avg_trade_return_pct": test_backtest_metric["avg_trade_return_pct"],
            "test_sharpe_ratio": test_backtest_metric["sharpe_ratio"],
            "test_sortino_ratio": test_backtest_metric["sortino_ratio"],
            "test_calmar_ratio": test_backtest_metric["calmar_ratio"],
            "test_sqn": test_backtest_metric["sqn"],
            "test_total_trades": test_backtest_metric["total_trades"],
            "test_equity_curve": [float(value) for value in test_equity_curve],
            "directional_diagnostics": directional_diagnostics,
            "backtest_threshold_diagnostics": backtest_diagnostics,
            "selected_threshold_metric": selected_threshold_metric,
            "model_path": str(model_path),
            "training_plot": str(fold_plot),
        }
        fold_results.append(result)

        logger.info(
            "Modo %s | fold %d concluido | threshold=%.2f | PF=%.2f | WR=%.2f%% | Trades=%d | DirAcc=%.2f%%",
            feature_mode,
            fold_id,
            threshold,
            test_backtest_metric["profit_factor"],
            test_backtest_metric["win_rate"],
            int(test_backtest_metric["total_trades"]),
            100.0 * test_signal_metric["directional_accuracy"],
        )

    results_df = pd.DataFrame(fold_results)
    csv_df = results_df.drop(
        columns=[
            "test_equity_curve",
            "directional_diagnostics",
            "backtest_threshold_diagnostics",
            "selected_threshold_metric",
        ]
    )
    csv_df.to_csv(mode_output_dir / "walkforward_summary.csv", index=False, encoding="utf-8-sig")

    _plot_mode_fold_metrics(mode_output_dir / "walkforward_fold_metrics.png", results_df, feature_mode)
    _plot_mode_threshold_heatmap(
        mode_output_dir / "walkforward_threshold_heatmap.png",
        threshold_records,
        feature_mode,
    )

    aggregate = {
        "feature_mode": feature_mode,
        "folds": len(fold_results),
        "mean_test_profit_factor": float(results_df["test_profit_factor"].mean()),
        "median_test_profit_factor": float(results_df["test_profit_factor"].median()),
        "mean_test_win_rate": float(results_df["test_win_rate"].mean()),
        "mean_test_max_drawdown": float(results_df["test_max_drawdown"].mean()),
        "mean_test_total_return": float(results_df["test_total_return"].mean()),
        "mean_test_sharpe_ratio": float(results_df["test_sharpe_ratio"].mean()),
        "mean_test_directional_accuracy": float(results_df["test_directional_accuracy"].mean()),
        "mean_test_total_trades": float(results_df["test_total_trades"].mean()),
        "positive_pf_folds": int((results_df["test_profit_factor"] > 1.0).sum()),
        "positive_return_folds": int((results_df["test_total_return"] > 0.0).sum()),
    }

    payload = {
        "symbol": symbol,
        "timeframe": timeframe_label,
        "n_bars": n_bars,
        "feature_mode": feature_mode,
        "note": (
            "O modo legacy_like reconstrui uma leitura estrutural otimista, "
            "util para auditoria comparativa, mas nao deve ser confundido "
            "automaticamente com causalidade operacional."
        ),
        "config": {
            "swing_window": swing_window,
            "horizon": horizon,
            "atr_mult": atr_mult,
            "min_activation": min_activation,
            "seed": seed,
            "training_cfg": training_cfg,
            "threshold_cfg": threshold_cfg,
            "backtest_cfg": backtest_cfg,
            "fold_cfg": fold_cfg,
        },
        "aggregate": aggregate,
        "folds": fold_results,
    }
    (mode_output_dir / "walkforward_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    symbol = "USTEC.r"
    timeframe = mt5.TIMEFRAME_M15
    timeframe_label = "M15"
    n_bars = 20000
    swing_window = 5
    horizon = 15
    atr_mult = 1.0
    min_activation = 0.10
    seed = 42
    modes = [FEATURE_MODE_CAUSAL_V3]

    training_cfg = {
        "epochs": 160,
        "batch_size": 256,
        "learning_rate": 0.002,
        "weight_decay": 1e-4,
        "sigma_min": 0.05,
        "lambda_rule_usage": 0.01,
        "patience": 24,
    }
    threshold_cfg = {
        "candidates": [0.03, 0.05, 0.07, 0.10],
        "min_activations_directional": 80,
        "min_trades_backtest": 60,
    }
    backtest_cfg = {
        "initial_capital": 100000.0,
        "risk_per_trade": 0.01,
        "reward_to_risk": 1.0,
        "stop_mode": "atr",
        "atr_stop_mult": atr_mult,
        "atr_target_mult": atr_mult,
        "max_holding_bars": horizon,
    }
    fold_cfg = {
        "initial_train": 8000,
        "val_size": 2000,
        "test_size": 2000,
        "step_size": 2000,
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    logger.info(
        "Iniciando comparacao walk-forward. Symbol=%s, modos=%s, swing_window=%d, horizon=%d, n_bars=%d",
        symbol,
        ", ".join(modes),
        swing_window,
        horizon,
        n_bars,
    )

    mt5_client = MT5Client()
    df = mt5_client.get_historical_data(symbol, timeframe, n_bars)
    if len(df) < 1000:
        logger.error("Sem dados suficientes para comparacao walk-forward.")
        return

    mode_payloads: List[Dict] = []
    aggregate_rows: List[Dict] = []
    fold_rows: List[Dict] = []

    for feature_mode in modes:
        payload = _run_mode(
            df=df,
            symbol=symbol,
            timeframe_label=timeframe_label,
            feature_mode=feature_mode,
            n_bars=n_bars,
            swing_window=swing_window,
            horizon=horizon,
            atr_mult=atr_mult,
            min_activation=min_activation,
            seed=seed,
            training_cfg=training_cfg,
            threshold_cfg=threshold_cfg,
            backtest_cfg=backtest_cfg,
            fold_cfg=fold_cfg,
        )
        mode_payloads.append(payload)
        aggregate_rows.append(payload["aggregate"])
        fold_rows.extend(payload["folds"])

    aggregate_df = pd.DataFrame(aggregate_rows).sort_values("mean_test_profit_factor", ascending=False)
    folds_df = pd.DataFrame(fold_rows)

    aggregate_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    _plot_comparison_aggregate(aggregate_df)
    _plot_comparison_folds(folds_df)
    _plot_comparison_equity(mode_payloads)

    winner_row = aggregate_df.iloc[0].to_dict()
    payload = {
        "symbol": symbol,
        "timeframe": timeframe_label,
        "n_bars": n_bars,
        "note": (
            "Esta comparacao mede a superioridade relativa entre leituras estruturais "
            "sob o mesmo pipeline de treino e backtest. Em particular, legacy_like "
            "pode continuar embutindo uma leitura otimista da estrutura, entao o "
            "resultado deve ser interpretado como evidencia comparativa de pesquisa."
        ),
        "winner_by_mean_profit_factor": winner_row,
        "aggregate": aggregate_rows,
        "folds": fold_rows,
    }
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "Comparacao concluida. Melhor modo por PF medio: %s (PF medio=%.2f, retorno medio=%.2f%%, WR medio=%.2f%%)",
        winner_row["feature_mode"],
        winner_row["mean_test_profit_factor"],
        winner_row["mean_test_total_return"],
        winner_row["mean_test_win_rate"],
    )


if __name__ == "__main__":
    main()
