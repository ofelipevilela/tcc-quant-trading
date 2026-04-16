# -*- coding: utf-8 -*-
"""
Treinamento ANFIS em mercado real com target orientado a barreiras.

Pipeline principal:
1. baixa OHLCV do MT5;
2. traduz o histórico para features SMC causais;
3. gera targets coerentes com uma lógica de barreiras em ATR + horizonte;
4. treina o ANFIS com split temporal, consequentes reescalados e clamp das MFs;
5. calibra o threshold pela performance da carteira na validação;
6. salva pesos, metadados e curva de treinamento.

O modo estrutural pode ser sobrescrito via variável de ambiente:
`ANFIS_FEATURE_MODE=legacy_like|causal_raw|causal_enhanced`.
"""

import os
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import MetaTrader5 as mt5
except ImportError:
    print("MetaTrader5 nao instalado.")
    sys.exit(1)

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
    split_temporal_data,
    train_model,
)
from smc.feature_factory import FEATURE_MODE_LEGACY_LIKE, resolve_feature_mode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_real_market")


MODEL_PATH = Path("anfis_trained_mt5.pt")
META_PATH = Path("anfis_trained_mt5_meta.json")
PLOT_PATH = Path("outputs/plots/mt5_training_loss.png")


def main() -> None:
    symbol = "USTEC.r"
    timeframe = mt5.TIMEFRAME_M15
    n_bars = 20000
    swing_window = 5
    horizon = 15
    atr_mult = 1.0
    min_activation = 0.10
    feature_mode = resolve_feature_mode(
        os.getenv("ANFIS_FEATURE_MODE"),
        default=FEATURE_MODE_LEGACY_LIKE,
    )
    seed = 42

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
        "candidates": [0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.25],
        "min_activations_directional": 80,
        "min_trades_backtest": 30,
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

    set_seed(seed)
    logger.info(
        "Iniciando pipeline real. Symbol=%s, feature_mode=%s, swing_window=%d, horizon=%d, n_bars=%d",
        symbol,
        feature_mode,
        swing_window,
        horizon,
        n_bars,
    )

    try:
        mt5_client = MT5Client()
        df = mt5_client.get_historical_data(symbol, timeframe, n_bars)
    except Exception as exc:
        logger.error(str(exc))
        return

    if len(df) < 200:
        logger.error("Sem dados suficientes para treino.")
        return

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
    legacy_values = prepared.legacy_targets

    kept = int(len(y_values))
    total = int(len(prepared.mask))
    logger.info(
        "Smart Trimming: mantidas %d amostras de %d (|y| > %.2f). Taxa de perda: %.2f%%",
        kept,
        total,
        min_activation,
        100.0 * (1.0 - (kept / max(total, 1))),
    )
    logger.info(
        "Distribuicao target real: mean=%.4f std=%.4f min=%.4f max=%.4f",
        float(np.mean(y_values)),
        float(np.std(y_values)),
        float(np.min(y_values)),
        float(np.max(y_values)),
    )
    logger.info(
        "Distribuicao target legacy: mean=%.4f std=%.4f min=%.4f max=%.4f",
        float(np.mean(legacy_values)),
        float(np.std(legacy_values)),
        float(np.min(legacy_values)),
        float(np.max(legacy_values)),
    )

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_temporal_data(x_values, y_values)
    train_end = len(x_train)
    val_end = len(x_train) + len(x_val)
    df_val = prepared.df_smc.iloc[train_end:val_end].copy()
    df_test = prepared.df_smc.iloc[val_end:].copy()

    logger.info(
        "Split temporal: train=%d | val=%d | test=%d",
        len(x_train),
        len(x_val),
        len(x_test),
    )

    train_loader = build_loader(x_train, y_train, training_cfg["batch_size"], shuffle=True)
    val_loader = build_loader(x_val, y_val, training_cfg["batch_size"], shuffle=False)

    logger.info("Inicializando modelo ANFIS...")
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
    recommended_threshold, selected_backtest_metric = select_backtest_threshold(
        backtest_diagnostics,
        fallback_threshold=directional_threshold,
    )

    val_metric = compute_signal_metrics(val_scores, y_val, recommended_threshold)
    test_metric = compute_signal_metrics(test_scores, y_test, recommended_threshold)

    val_backtest_metric = next(
        item for item in backtest_diagnostics if item["threshold"] == recommended_threshold
    )
    test_backtest_diagnostics = evaluate_thresholds_with_backtest(
        df_slice=df_test,
        scores=test_scores,
        candidate_thresholds=[recommended_threshold],
        initial_capital=backtest_cfg["initial_capital"],
        risk_per_trade=backtest_cfg["risk_per_trade"],
        reward_to_risk=backtest_cfg["reward_to_risk"],
        stop_mode=backtest_cfg["stop_mode"],
        atr_stop_mult=backtest_cfg["atr_stop_mult"],
        atr_target_mult=backtest_cfg["atr_target_mult"],
        max_holding_bars=backtest_cfg["max_holding_bars"],
        min_trades=threshold_cfg["min_trades_backtest"],
    )
    test_backtest_metric = test_backtest_diagnostics[0]

    val_stats = compute_score_statistics(val_scores)
    test_stats = compute_score_statistics(test_scores)

    logger.info(
        "Scores validacao: mean=%.4f std=%.4f min=%.4f max=%.4f",
        val_stats["mean"],
        val_stats["std"],
        val_stats["min"],
        val_stats["max"],
    )
    logger.info(
        "Scores teste: mean=%.4f std=%.4f min=%.4f max=%.4f",
        test_stats["mean"],
        test_stats["std"],
        test_stats["min"],
        test_stats["max"],
    )
    logger.info(
        "Threshold direcional=%.2f | threshold carteira=%.2f",
        directional_threshold,
        recommended_threshold,
    )
    logger.info(
        "Validacao direcional: %.2f%% (%d ativacoes) | Teste direcional: %.2f%% (%d ativacoes)",
        100.0 * val_metric["directional_accuracy"],
        val_metric["activated"],
        100.0 * test_metric["directional_accuracy"],
        test_metric["activated"],
    )
    logger.info(
        "Validacao carteira: PF=%.2f | WR=%.2f%% | Trades=%d | DD=%.2f%%",
        val_backtest_metric["profit_factor"],
        val_backtest_metric["win_rate"],
        int(val_backtest_metric["total_trades"]),
        val_backtest_metric["max_drawdown"],
    )
    logger.info(
        "Teste carteira: PF=%.2f | WR=%.2f%% | Trades=%d | DD=%.2f%%",
        test_backtest_metric["profit_factor"],
        test_backtest_metric["win_rate"],
        int(test_backtest_metric["total_trades"]),
        test_backtest_metric["max_drawdown"],
    )

    top_rule_means = np.sort(test_rules.mean(axis=0))[-5:]
    logger.info(
        "Top 5 medias de ativacao de regras no teste: %s",
        ", ".join(f"{value:.4f}" for value in top_rule_means),
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    logger.info("Modelo salvo em '%s'.", MODEL_PATH)

    metadata = {
        "symbol": symbol,
        "timeframe": "M15",
        "n_bars": n_bars,
        "swing_window": swing_window,
        "seed": seed,
        "target_mode": "atr_barrier",
        "feature_mode": prepared.feature_mode,
        "horizon": horizon,
        "atr_mult": atr_mult,
        "min_activation": min_activation,
        "recommended_threshold": recommended_threshold,
        "directional_threshold": directional_threshold,
        "training_cfg": training_cfg,
        "threshold_cfg": threshold_cfg,
        "backtest_cfg": backtest_cfg,
        "validation_metric": val_metric,
        "test_metric": test_metric,
        "validation_backtest_metric": val_backtest_metric,
        "test_backtest_metric": test_backtest_metric,
        "threshold_diagnostics": directional_diagnostics,
        "threshold_backtest_diagnostics": backtest_diagnostics,
        "selected_backtest_metric": selected_backtest_metric,
        "score_distribution": {
            "val_mean": val_stats["mean"],
            "val_std": val_stats["std"],
            "test_mean": test_stats["mean"],
            "test_std": test_stats["std"],
        },
        "feature_distribution": {
            "trend_strength_mean": float(np.mean(x_values[:, 0])),
            "price_zone_mean": float(np.mean(x_values[:, 1])),
            "fvg_quality_mean": float(np.mean(x_values[:, 2])),
            "sweep_quality_mean": float(np.mean(x_values[:, 3])),
        },
        "mf_params": model.get_mf_params(),
        "consequents": [float(value) for value in model.consequents.detach().cpu().numpy()],
    }

    META_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Metadados salvos em '%s'.", META_PATH)

    plot_training_history(history, str(PLOT_PATH))
    logger.info("Curva de aprendizado salva em '%s'.", PLOT_PATH)


if __name__ == "__main__":
    main()
