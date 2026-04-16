# -*- coding: utf-8 -*-
"""
Orquestrador de experimentos quantitativos do TCC.

Este script executa o fluxo completo:
1. carrega OHLCV via MT5 ou CSV;
2. gera features SMC causais;
3. roda validacao walk-forward;
4. treina o ANFIS com AdamTrainer, scheduler e early stopping;
5. calcula metricas academicas e operacionais;
6. salva os artefatos em uma pasta timestampada.

O objetivo e produzir resultados prontos para analise, sem depender de prints
soltos no terminal.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from anfis.adam_trainer import AdamTrainer
from anfis.config import LINGUISTIC_SETS, UNIVERSES
from backtest.engine import simulate_trading_from_scores
from metrics_evaluator import (
    evaluate_predictions,
    evaluate_train_validation,
    plot_error_comparison,
)
from real_market_utils import (
    build_loader,
    compute_score_statistics,
    compute_signal_metrics,
    create_rescaled_model,
    evaluate_thresholds_with_backtest,
    plot_training_history,
    prepare_market_data,
    select_activation_threshold,
    select_backtest_threshold,
    set_seed,
)
from smc.feature_factory import FEATURE_MODE_CAUSAL_V3


EXPERIMENT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "data": {
        "source": "mt5",  # "mt5" ou "csv"
        "symbol": "USTEC.r",
        "timeframe": "M15",
        "n_bars": 50000,
        "csv_path": None,
    },
    "features": {
        "feature_mode": FEATURE_MODE_CAUSAL_V3,
        "swing_window": 5,
        "horizon": 40,
        "atr_mult": 1.0,
        "min_activation": 0.0,
    },
    "target": {
        "mode": "rr",
        "reward_to_risk": 1.0,
        "stop_mode": "structure",
        "stop_buffer_atr": 0.05,
        "sweep_max_age": 60,
        "min_risk_atr": 0.25,
        "max_risk_atr": 4.0,
    },
    "walk_forward": {
        "initial_train": 21000,
        "val_size": 3000,
        "test_size": 3000,
        "step_size": 3000,
        "max_folds": 4,
    },
    "training": {
        "epochs": 160,
        "batch_size": 256,
        "learning_rate": 0.002,
        "consequent_lr_mult": 1.0,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "lambda_mf": 0.0,
        "lambda_rule_usage": 0.01,
        "sigma_min": 0.05,
        "scheduler_patience": 10,
        "scheduler_factor": 0.5,
        "early_stop_patience": 24,
        "min_delta": 1e-5,
        "min_lr": 5e-5,
        "log_every": 10,
        "device": "cpu",
    },
    "threshold": {
        "candidates": [0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20],
        "min_activations_directional": 80,
        "min_trades_backtest": 30,
    },
    "backtest": {
        "initial_capital": 100000.0,
        "risk_per_trade": 0.01,
        "reward_to_risk": 1.0,
        "stop_mode": "structure",
        "max_holding_bars": 40,
    },
    "metrics": {
        "zero_tolerance": 0.0,
    },
    "outputs": {
        "root": "outputs",
        "prefix": "experiment_rr50k",
        "save_models": True,
    },
}


logger = logging.getLogger("run_experiment")


def build_folds(
    n_samples: int,
    initial_train: int,
    val_size: int,
    test_size: int,
    step_size: int,
    max_folds: int,
) -> List[Dict[str, int]]:
    """Cria folds temporais crescentes para validacao walk-forward."""
    folds: List[Dict[str, int]] = []
    train_end = initial_train
    fold_id = 1

    while (train_end + val_size + test_size) <= n_samples and fold_id <= max_folds:
        val_end = train_end + val_size
        test_end = val_end + test_size
        folds.append({
            "fold": fold_id,
            "train_start": 0,
            "train_end": train_end,
            "val_start": train_end,
            "val_end": val_end,
            "test_start": val_end,
            "test_end": test_end,
        })
        train_end += step_size
        fold_id += 1

    return folds


def create_output_dir(config: Dict[str, Any]) -> Path:
    """Cria a pasta timestampada do experimento."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(config["outputs"]["root"])
    output_dir = output_root / f"{config['outputs']['prefix']}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def setup_logging(output_dir: Path) -> None:
    """Configura logging para terminal e arquivo do experimento."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "experiment.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    root.addHandler(stream_handler)
    root.addHandler(file_handler)


def gaussian_membership(
    x_values: np.ndarray,
    center: float,
    sigma: float,
) -> np.ndarray:
    """Calcula uma MF gaussiana para visualizacao academica."""
    sigma_safe = max(float(sigma), 1e-8)
    return np.exp(-0.5 * ((x_values - float(center)) / sigma_safe) ** 2)


def plot_mf_evolution(
    initial_snapshot: Dict[str, Dict[str, Dict[str, float]]],
    final_snapshot: Dict[str, Dict[str, Dict[str, float]]],
    output_path: Path,
) -> None:
    """Salva um grafico antes/depois das funcoes de pertinencia do ANFIS."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes_flat = axes.flatten()

    for ax, var_name in zip(axes_flat, LINGUISTIC_SETS):
        lo, hi = UNIVERSES[var_name]
        x_values = np.linspace(lo, hi, 300)
        for set_name in LINGUISTIC_SETS[var_name]:
            initial = initial_snapshot[var_name][set_name]
            final = final_snapshot[var_name][set_name]
            ax.plot(
                x_values,
                gaussian_membership(
                    x_values,
                    initial["center"],
                    initial["sigma"],
                ),
                linestyle="--",
                linewidth=1.2,
                alpha=0.7,
                label=f"{set_name} inicial",
            )
            ax.plot(
                x_values,
                gaussian_membership(
                    x_values,
                    final["center"],
                    final["sigma"],
                ),
                linewidth=1.8,
                label=f"{set_name} final",
            )

        ax.set_title(var_name)
        ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    fig.suptitle("Evolucao das funcoes de pertinencia por fold", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize_mf_shift(
    initial_snapshot: Dict[str, Dict[str, Dict[str, float]]],
    final_snapshot: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, float]:
    """Resume o deslocamento medio de centros e sigmas das MFs."""
    center_shifts: List[float] = []
    sigma_shifts: List[float] = []

    for var_name, sets in initial_snapshot.items():
        for set_name, initial_params in sets.items():
            final_params = final_snapshot[var_name][set_name]
            center_shifts.append(
                abs(float(final_params["center"]) - float(initial_params["center"]))
            )
            sigma_shifts.append(
                abs(float(final_params["sigma"]) - float(initial_params["sigma"]))
            )

    return {
        "mf_mean_abs_center_shift": float(np.mean(center_shifts)),
        "mf_mean_abs_sigma_shift": float(np.mean(sigma_shifts)),
        "mf_max_abs_center_shift": float(np.max(center_shifts)),
        "mf_max_abs_sigma_shift": float(np.max(sigma_shifts)),
    }


def resolve_mt5_timeframe(timeframe_label: str) -> int:
    """Converte um label como M15 na constante correspondente do MetaTrader5."""
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise ImportError(
            "MetaTrader5 nao esta instalado. Use source='csv' ou instale o pacote."
        ) from exc

    attr_name = f"TIMEFRAME_{timeframe_label.upper()}"
    if not hasattr(mt5, attr_name):
        raise ValueError(f"Timeframe MT5 desconhecido: {timeframe_label}")
    return int(getattr(mt5, attr_name))


def load_from_mt5(data_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Carrega dados OHLCV do MetaTrader 5."""
    from data.mt5_client import MT5Client

    timeframe = resolve_mt5_timeframe(str(data_cfg["timeframe"]))
    client = MT5Client()
    return client.get_historical_data(
        symbol=str(data_cfg["symbol"]),
        timeframe=timeframe,
        n_bars=int(data_cfg["n_bars"]),
    )


def load_from_csv(csv_path: str | Path) -> pd.DataFrame:
    """Carrega OHLCV de um CSV local."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV nao encontrado: {path}")

    df = pd.read_csv(path)
    df.columns = [str(col).lower().strip() for col in df.columns]

    time_col = next(
        (col for col in ("time", "datetime", "timestamp", "date") if col in df.columns),
        None,
    )
    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "CSV precisa conter uma coluna time, datetime, timestamp ou date."
        )

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV sem colunas obrigatorias: {sorted(missing)}")

    return df.sort_index()


def load_market_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Carrega dados conforme a fonte definida no dicionario de configuracao."""
    data_cfg = config["data"]
    source = str(data_cfg["source"]).lower()
    if source == "mt5":
        df = load_from_mt5(data_cfg)
    elif source == "csv":
        csv_path = data_cfg.get("csv_path")
        if not csv_path:
            raise ValueError("source='csv' exige data.csv_path preenchido.")
        df = load_from_csv(csv_path)
    else:
        raise ValueError(f"Fonte de dados desconhecida: {source}")

    if df.empty:
        raise RuntimeError("Nenhum dado foi carregado.")
    return df


def predict_scores_device(
    model: torch.nn.Module,
    x_values: np.ndarray,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prediz scores garantindo que tensores e modelo estejam no mesmo device."""
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_values, dtype=torch.float32, device=device)
        scores, _, normalized = model(x_tensor)
    return scores.squeeze(1).cpu().numpy(), normalized.cpu().numpy()


def safe_corr(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Calcula correlacao linear com protecao contra series degeneradas."""
    if len(x_values) < 3:
        return 0.0
    if np.std(x_values) <= 1e-8 or np.std(y_values) <= 1e-8:
        return 0.0
    return float(np.corrcoef(x_values, y_values)[0, 1])


def flatten_metrics(prefix: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Adiciona prefixo a um dicionario de metricas."""
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def aggregate_numeric(rows: List[Dict[str, Any]], prefixes: List[str]) -> Dict[str, float]:
    """Calcula medias de colunas numericas selecionadas."""
    df = pd.DataFrame(rows)
    aggregate: Dict[str, float] = {}
    for column in df.columns:
        if not any(column.startswith(prefix) for prefix in prefixes):
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            aggregate[f"mean_{column}"] = float(df[column].mean())
    return aggregate


def sanitize_for_json(value: Any) -> Any:
    """Converte objetos NumPy, pandas e Path para JSON estavel."""
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    return value


def run_fold(
    *,
    fold: Dict[str, int],
    prepared,
    config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Treina e avalia um fold walk-forward."""
    fold_id = int(fold["fold"])
    seed = int(config["seed"]) + fold_id
    set_seed(seed)

    training_cfg = dict(config["training"])
    threshold_cfg = config["threshold"]
    backtest_cfg = dict(config["backtest"])
    metrics_cfg = config["metrics"]
    feature_cfg = config["features"]
    target_cfg = config.get("target", {})
    device = str(training_cfg.pop("device", "cpu"))

    # Mantem backtest alinhado ao alvo por barreira ATR.
    backtest_cfg["atr_stop_mult"] = float(feature_cfg["atr_mult"])
    backtest_cfg["atr_target_mult"] = float(feature_cfg["atr_mult"])
    backtest_cfg["max_holding_bars"] = int(feature_cfg["horizon"])
    backtest_cfg["reward_to_risk"] = float(
        target_cfg.get("reward_to_risk", backtest_cfg["reward_to_risk"])
    )
    backtest_cfg["stop_mode"] = str(target_cfg.get("stop_mode", backtest_cfg["stop_mode"]))
    backtest_cfg["stop_buffer_atr"] = float(target_cfg.get("stop_buffer_atr", 0.05))
    backtest_cfg["sweep_max_age"] = int(target_cfg.get("sweep_max_age", 60))
    backtest_cfg["min_risk_atr"] = float(target_cfg.get("min_risk_atr", 0.25))
    backtest_cfg["max_risk_atr"] = float(target_cfg.get("max_risk_atr", 4.0))

    x_values = prepared.features.values
    y_values = prepared.targets

    x_train = x_values[fold["train_start"]:fold["train_end"]]
    y_train = y_values[fold["train_start"]:fold["train_end"]]
    x_val = x_values[fold["val_start"]:fold["val_end"]]
    y_val = y_values[fold["val_start"]:fold["val_end"]]
    x_test = x_values[fold["test_start"]:fold["test_end"]]
    y_test = y_values[fold["test_start"]:fold["test_end"]]

    df_val = prepared.df_smc.iloc[fold["val_start"]:fold["val_end"]].copy()
    df_test = prepared.df_smc.iloc[fold["test_start"]:fold["test_end"]].copy()

    train_loader = build_loader(
        x_train,
        y_train,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
    )
    val_loader = build_loader(
        x_val,
        y_val,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
    )

    logger.info(
        "Fold %d | treino=%d | validacao=%d | teste=%d",
        fold_id,
        len(x_train),
        len(x_val),
        len(x_test),
    )

    model = create_rescaled_model()
    initial_mf_snapshot = model.get_mf_params()
    trainer = AdamTrainer(model, config=training_cfg, device=device)
    history = trainer.train(train_loader, val_loader)
    final_mf_snapshot = model.get_mf_params()
    mf_shift = summarize_mf_shift(initial_mf_snapshot, final_mf_snapshot)

    learning_plot = output_dir / f"fold_{fold_id:02d}_learning_curve.png"
    plot_training_history(history, str(learning_plot))

    mf_plot = output_dir / f"fold_{fold_id:02d}_mf_evolution.png"
    plot_mf_evolution(initial_mf_snapshot, final_mf_snapshot, mf_plot)

    mf_params_path = output_dir / f"fold_{fold_id:02d}_mf_params.json"
    mf_params_path.write_text(
        json.dumps(
            {
                "initial": initial_mf_snapshot,
                "final_best_validation": final_mf_snapshot,
                "shift_summary": mf_shift,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    train_scores, train_rules = predict_scores_device(model, x_train, device)
    val_scores, val_rules = predict_scores_device(model, x_val, device)
    test_scores, test_rules = predict_scores_device(model, x_test, device)

    zero_tolerance = float(metrics_cfg.get("zero_tolerance", 0.0))
    train_metrics = evaluate_predictions(
        y_train, train_scores, zero_tolerance=zero_tolerance
    )
    val_metrics = evaluate_predictions(
        y_val, val_scores, zero_tolerance=zero_tolerance
    )
    test_metrics = evaluate_predictions(
        y_test, test_scores, zero_tolerance=zero_tolerance
    )

    train_val_metrics = evaluate_train_validation(
        y_train,
        train_scores,
        y_val,
        val_scores,
        zero_tolerance=zero_tolerance,
    )
    error_plot = output_dir / f"fold_{fold_id:02d}_train_val_errors.png"
    plot_error_comparison(train_val_metrics, output_path=error_plot)

    directional_threshold, directional_diagnostics = select_activation_threshold(
        val_scores,
        y_val,
        candidate_thresholds=threshold_cfg["candidates"],
        min_activations=int(threshold_cfg["min_activations_directional"]),
    )
    backtest_diagnostics = evaluate_thresholds_with_backtest(
        df_slice=df_val,
        scores=val_scores,
        candidate_thresholds=threshold_cfg["candidates"],
        initial_capital=float(backtest_cfg["initial_capital"]),
        risk_per_trade=float(backtest_cfg["risk_per_trade"]),
        reward_to_risk=float(backtest_cfg["reward_to_risk"]),
        stop_mode=str(backtest_cfg["stop_mode"]),
        atr_stop_mult=float(backtest_cfg["atr_stop_mult"]),
        atr_target_mult=float(backtest_cfg["atr_target_mult"]),
        max_holding_bars=int(backtest_cfg["max_holding_bars"]),
        min_trades=int(threshold_cfg["min_trades_backtest"]),
        stop_buffer_atr=float(backtest_cfg["stop_buffer_atr"]),
        sweep_max_age=int(backtest_cfg["sweep_max_age"]),
        min_risk_atr=float(backtest_cfg["min_risk_atr"]),
        max_risk_atr=float(backtest_cfg["max_risk_atr"]),
    )
    selected_threshold, selected_threshold_metric = select_backtest_threshold(
        backtest_diagnostics,
        fallback_threshold=directional_threshold,
    )

    thresholded_test_signal = compute_signal_metrics(
        test_scores,
        y_test,
        selected_threshold,
    )
    test_backtest_metric = evaluate_thresholds_with_backtest(
        df_slice=df_test,
        scores=test_scores,
        candidate_thresholds=[selected_threshold],
        initial_capital=float(backtest_cfg["initial_capital"]),
        risk_per_trade=float(backtest_cfg["risk_per_trade"]),
        reward_to_risk=float(backtest_cfg["reward_to_risk"]),
        stop_mode=str(backtest_cfg["stop_mode"]),
        atr_stop_mult=float(backtest_cfg["atr_stop_mult"]),
        atr_target_mult=float(backtest_cfg["atr_target_mult"]),
        max_holding_bars=int(backtest_cfg["max_holding_bars"]),
        min_trades=int(threshold_cfg["min_trades_backtest"]),
        stop_buffer_atr=float(backtest_cfg["stop_buffer_atr"]),
        sweep_max_age=int(backtest_cfg["sweep_max_age"]),
        min_risk_atr=float(backtest_cfg["min_risk_atr"]),
        max_risk_atr=float(backtest_cfg["max_risk_atr"]),
    )[0]

    equity_curve, trades = simulate_trading_from_scores(
        data=df_test,
        scores=test_scores,
        initial_capital=float(backtest_cfg["initial_capital"]),
        risk_per_trade=float(backtest_cfg["risk_per_trade"]),
        reward_to_risk=float(backtest_cfg["reward_to_risk"]),
        activation_threshold=float(selected_threshold),
        stop_mode=str(backtest_cfg["stop_mode"]),
        atr_stop_mult=float(backtest_cfg["atr_stop_mult"]),
        atr_target_mult=float(backtest_cfg["atr_target_mult"]),
        max_holding_bars=int(backtest_cfg["max_holding_bars"]),
        stop_buffer_atr=float(backtest_cfg["stop_buffer_atr"]),
        sweep_max_age=int(backtest_cfg["sweep_max_age"]),
        min_risk_atr=float(backtest_cfg["min_risk_atr"]),
        max_risk_atr=float(backtest_cfg["max_risk_atr"]),
    )

    model_path = None
    if bool(config["outputs"].get("save_models", True)):
        model_path = output_dir / f"fold_{fold_id:02d}_model.pt"
        torch.save(model.state_dict(), model_path)

    test_rule_means = test_rules.mean(axis=0)
    row: Dict[str, Any] = {
        "fold": fold_id,
        "seed": seed,
        "train_samples": len(x_train),
        "val_samples": len(x_val),
        "test_samples": len(x_test),
        "train_start_time": str(prepared.df_smc.index[fold["train_start"]]),
        "train_end_time": str(prepared.df_smc.index[fold["train_end"] - 1]),
        "val_start_time": str(df_val.index[0]),
        "val_end_time": str(df_val.index[-1]),
        "test_start_time": str(df_test.index[0]),
        "test_end_time": str(df_test.index[-1]),
        "epochs_ran": len(history["val_loss"]),
        "best_val_loss": min(history["val_loss"]) if history["val_loss"] else None,
        "last_lr": history["lr_mf"][-1] if history.get("lr_mf") else None,
        "directional_threshold": directional_threshold,
        "selected_threshold": selected_threshold,
        "selected_threshold_source": (
            "backtest"
            if selected_threshold != directional_threshold
            else "directional_fallback"
        ),
        "learning_plot": str(learning_plot),
        "error_plot": str(error_plot),
        "mf_plot": str(mf_plot),
        "mf_params_path": str(mf_params_path),
        **mf_shift,
        "model_path": str(model_path) if model_path else None,
        "test_score_target_corr": safe_corr(test_scores, y_test),
        "test_score_mean": compute_score_statistics(test_scores)["mean"],
        "test_score_std": compute_score_statistics(test_scores)["std"],
        "test_rule_dominance": float(np.max(test_rule_means)),
        "test_active_rules_95": int(np.sum(test_rule_means > 0.05)),
        "thresholded_test_directional_accuracy": thresholded_test_signal[
            "directional_accuracy"
        ],
        "thresholded_test_coverage": thresholded_test_signal["coverage"],
        "thresholded_test_activated": thresholded_test_signal["activated"],
        "test_total_trades": int(test_backtest_metric["total_trades"]),
        "test_profit_factor": float(test_backtest_metric["profit_factor"]),
        "test_win_rate": float(test_backtest_metric["win_rate"]),
        "test_total_return": float(test_backtest_metric["total_return"]),
        "test_annualized_return": float(test_backtest_metric["annualized_return"]),
        "test_max_drawdown": float(test_backtest_metric["max_drawdown"]),
        "test_payoff_ratio": float(test_backtest_metric["payoff_ratio"]),
        "test_expectancy": float(test_backtest_metric["expectancy"]),
        "test_avg_trade_return_pct": float(test_backtest_metric["avg_trade_return_pct"]),
        "test_sharpe_ratio": float(test_backtest_metric["sharpe_ratio"]),
        "test_sortino_ratio": float(test_backtest_metric["sortino_ratio"]),
        "test_calmar_ratio": float(test_backtest_metric["calmar_ratio"]),
        "test_sqn": float(test_backtest_metric["sqn"]),
        "test_trades_path": None,
        "test_equity_path": None,
    }
    row.update(flatten_metrics("train", train_metrics))
    row.update(flatten_metrics("val", val_metrics))
    row.update(flatten_metrics("test", test_metrics))

    trades_path = output_dir / f"fold_{fold_id:02d}_trades.csv"
    equity_path = output_dir / f"fold_{fold_id:02d}_equity.csv"
    pd.DataFrame(trades).to_csv(trades_path, index=False, encoding="utf-8-sig")
    pd.DataFrame({"equity": equity_curve}).to_csv(
        equity_path,
        index=False,
        encoding="utf-8-sig",
    )
    row["test_trades_path"] = str(trades_path)
    row["test_equity_path"] = str(equity_path)

    diagnostics = {
        "directional_diagnostics": directional_diagnostics,
        "backtest_threshold_diagnostics": backtest_diagnostics,
        "selected_threshold_metric": selected_threshold_metric,
    }
    diagnostics_path = output_dir / f"fold_{fold_id:02d}_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(sanitize_for_json(diagnostics), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    row["diagnostics_path"] = str(diagnostics_path)

    logger.info(
        "Fold %d concluido | MAE=%.4f | R2=%.4f | DA=%.2f%% | PF=%.2f | Trades=%d",
        fold_id,
        float(test_metrics["mae"]),
        float(test_metrics["r2"]),
        100.0 * float(test_metrics["directional_accuracy"]),
        float(test_backtest_metric["profit_factor"]),
        int(test_backtest_metric["total_trades"]),
    )
    return row


def run_experiment(config: Dict[str, Any]) -> Path:
    """Executa o experimento completo e retorna a pasta de saida."""
    output_dir = create_output_dir(config)
    setup_logging(output_dir)
    logger.info("Iniciando experimento. Saida: %s", output_dir)

    set_seed(int(config["seed"]))
    df = load_market_data(config)
    logger.info("Dados carregados: %d barras.", len(df))

    feature_cfg = config["features"]
    target_cfg = config.get("target", {})
    prepared = prepare_market_data(
        df=df,
        swing_window=int(feature_cfg["swing_window"]),
        horizon=int(feature_cfg["horizon"]),
        atr_mult=float(feature_cfg["atr_mult"]),
        min_activation=float(feature_cfg["min_activation"]),
        feature_mode=str(feature_cfg["feature_mode"]),
        target_mode=str(target_cfg.get("mode", "atr_barrier")),
        reward_to_risk=float(target_cfg.get("reward_to_risk", config["backtest"]["reward_to_risk"])),
        stop_mode=str(target_cfg.get("stop_mode", config["backtest"]["stop_mode"])),
        stop_buffer_atr=float(target_cfg.get("stop_buffer_atr", 0.05)),
        sweep_max_age=int(target_cfg.get("sweep_max_age", 60)),
        min_risk_atr=float(target_cfg.get("min_risk_atr", 0.25)),
        max_risk_atr=float(target_cfg.get("max_risk_atr", 4.0)),
    )
    logger.info(
        "Dataset preparado: %d amostras apos target/trimming | feature_mode=%s",
        len(prepared.features),
        prepared.feature_mode,
    )

    fold_cfg = config["walk_forward"]
    folds = build_folds(
        n_samples=len(prepared.features),
        initial_train=int(fold_cfg["initial_train"]),
        val_size=int(fold_cfg["val_size"]),
        test_size=int(fold_cfg["test_size"]),
        step_size=int(fold_cfg["step_size"]),
        max_folds=int(fold_cfg["max_folds"]),
    )
    if len(folds) < int(fold_cfg["max_folds"]):
        logger.warning(
            "Foram criados apenas %d folds; esperado=%d.",
            len(folds),
            int(fold_cfg["max_folds"]),
        )
    if not folds:
        raise RuntimeError("Nao foi possivel construir folds walk-forward.")

    fold_rows = [
        run_fold(
            fold=fold,
            prepared=prepared,
            config=config,
            output_dir=output_dir,
        )
        for fold in folds
    ]

    fold_df = pd.DataFrame(fold_rows)
    fold_metrics_path = output_dir / "fold_metrics.csv"
    fold_df.to_csv(fold_metrics_path, index=False, encoding="utf-8-sig")

    aggregate = aggregate_numeric(
        fold_rows,
        prefixes=[
            "train_",
            "val_",
            "test_",
            "thresholded_",
            "mf_",
        ],
    )
    aggregate.update({
        "folds": len(fold_rows),
        "mean_test_profit_factor": float(fold_df["test_profit_factor"].mean()),
        "mean_test_win_rate": float(fold_df["test_win_rate"].mean()),
        "mean_test_total_return": float(fold_df["test_total_return"].mean()),
        "mean_test_annualized_return": float(fold_df["test_annualized_return"].mean()),
        "mean_test_max_drawdown": float(fold_df["test_max_drawdown"].mean()),
        "mean_test_total_trades": float(fold_df["test_total_trades"].mean()),
        "positive_pf_folds": int((fold_df["test_profit_factor"] > 1.0).sum()),
        "positive_return_folds": int((fold_df["test_total_return"] > 0.0).sum()),
        "annual_return_ge_12_folds": int((fold_df["test_annualized_return"] >= 12.0).sum()),
    })

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "config": config,
        "aggregate": aggregate,
        "folds": fold_rows,
        "artifacts": {
            "fold_metrics_csv": str(fold_metrics_path),
            "log": str(output_dir / "experiment.log"),
        },
    }

    results_path = output_dir / "config_and_results.json"
    results_path.write_text(
        json.dumps(sanitize_for_json(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Experimento concluido.")
    logger.info("Resumo salvo em: %s", results_path)
    logger.info(
        "Media OOS | MAE=%.4f | R2=%.4f | DA=%.2f%% | PF=%.2f | RetAnual=%.2f%%",
        aggregate.get("mean_test_mae", float("nan")),
        aggregate.get("mean_test_r2", float("nan")),
        100.0 * aggregate.get("mean_test_directional_accuracy", float("nan")),
        aggregate.get("mean_test_profit_factor", float("nan")),
        aggregate.get("mean_test_annualized_return", float("nan")),
    )
    return output_dir


def main() -> None:
    """Ponto de entrada do script."""
    run_experiment(EXPERIMENT_CONFIG)


if __name__ == "__main__":
    main()
