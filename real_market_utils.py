# -*- coding: utf-8 -*-
"""
Utilidades compartilhadas do pipeline ANFIS em mercado real.

Este módulo concentra:
- geração de targets orientados a barreiras;
- preparação causal das features SMC;
- treino do ANFIS com regularização leve de uso de regras;
- métricas de direcionalidade e correlação;
- calibração de threshold por performance de carteira.

Foi criado para evitar divergência entre:
- treino principal (`train_real_market.py`);
- validação walk-forward (`run_walkforward.py`);
- diagnósticos e visualizações do TCC.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from anfis.anfis_model import ANFISModel
from anfis.rule_base import RuleBase
from backtest.engine import simulate_trading_from_scores
from backtest.performance import calculate_performance_metrics
from backtest.risk_levels import compute_structure_stop_levels
from smc.feature_factory import (
    FEATURE_MODE_CAUSAL_ENHANCED,
    FEATURE_MODE_CAUSAL_RAW,
    build_smc_features,
)


@dataclass
class PreparedMarketData:
    """Estrutura com dados preparados para treino e avaliação."""

    df_smc: pd.DataFrame
    features: pd.DataFrame
    targets: np.ndarray
    legacy_targets: np.ndarray
    mask: np.ndarray
    feature_mode: str
    swing_window: int
    horizon: int
    atr_mult: float
    min_activation: float
    target_mode: str = "atr_barrier"
    reward_to_risk: float = 1.0


def set_seed(seed: int) -> None:
    """Define seed para reproducibilidade do treinamento."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_market_targets(df: pd.DataFrame, horizon: int = 15) -> np.ndarray:
    """
    Target legado: retorno futuro ajustado por ATR e comprimido em [-1, 1].

    Mantido por compatibilidade e para comparação diagnóstica.
    """
    if "atr" not in df.columns or "close" not in df.columns:
        raise ValueError("DataFrame deve conter 'close' e 'atr'.")

    close = df["close"].values
    atr = df["atr"].values
    targets = np.zeros(len(close), dtype=np.float32)

    for i in range(len(close) - horizon):
        move = (close[i + horizon] - close[i]) / (atr[i] + 1e-8)
        targets[i] = np.tanh(move * 0.5)

    return targets


def generate_barrier_targets(
    df: pd.DataFrame,
    horizon: int = 15,
    atr_mult: float = 1.0,
    fallback_scale: float = 0.35,
) -> np.ndarray:
    """
    Gera target coerente com um esquema de barreiras baseado em ATR.

    Regras:
    - Se a barreira superior for atingida primeiro -> alvo positivo.
    - Se a barreira inferior for atingida primeiro -> alvo negativo.
    - Se nenhuma for tocada no horizonte, usa um retorno final suavizado.
    """
    required = {"close", "high", "low", "atr"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame deve conter {sorted(required)}. Faltando: {sorted(missing)}")

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = df["atr"].values

    targets = np.zeros(len(df), dtype=np.float32)

    for i in range(len(df) - horizon):
        if atr[i] <= 0:
            continue

        upper = close[i] + atr[i] * atr_mult
        lower = close[i] - atr[i] * atr_mult
        assigned = False

        for step in range(1, horizon + 1):
            j = i + step
            hit_up = high[j] >= upper
            hit_down = low[j] <= lower

            if hit_up and hit_down:
                signed_close = (close[j] - close[i]) / (atr[i] + 1e-8)
                targets[i] = np.tanh(signed_close * 0.5)
                assigned = True
                break

            if hit_up:
                speed_bonus = 1.0 - 0.3 * ((step - 1) / max(horizon - 1, 1))
                targets[i] = speed_bonus
                assigned = True
                break

            if hit_down:
                speed_bonus = 1.0 - 0.3 * ((step - 1) / max(horizon - 1, 1))
                targets[i] = -speed_bonus
                assigned = True
                break

        if not assigned:
            move = (close[i + horizon] - close[i]) / (atr[i] + 1e-8)
            targets[i] = np.tanh(move * 0.35) * fallback_scale

    return targets


def _directional_rr_outcome(
    *,
    entry: float,
    stop: float,
    target: float,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    start_idx: int,
    horizon: int,
    direction: int,
    reward_to_risk: float,
) -> float:
    """Retorna o resultado em R de uma direcao ate o horizonte configurado."""
    risk = (entry - stop) if direction == 1 else (stop - entry)
    if risk <= 1e-8:
        return 0.0

    for step in range(1, horizon + 1):
        j = start_idx + step
        if direction == 1:
            hit_stop = lows[j] <= stop
            hit_target = highs[j] >= target
        else:
            hit_stop = highs[j] >= stop
            hit_target = lows[j] <= target

        # Criterio conservador: se stop e alvo aparecem na mesma barra, stop vence.
        if hit_stop:
            return -1.0
        if hit_target:
            return float(reward_to_risk)

    final_move = (
        closes[start_idx + horizon] - entry
        if direction == 1
        else entry - closes[start_idx + horizon]
    )
    return float(np.clip(final_move / risk, -1.0, reward_to_risk))


def generate_rr_targets(
    df: pd.DataFrame,
    horizon: int = 10,
    reward_to_risk: float = 1.0,
    stop_mode: str = "structure",
    atr_mult: float = 1.0,
    stop_buffer_atr: float = 0.05,
    sweep_max_age: int = 60,
    min_risk_atr: float = 0.25,
    max_risk_atr: float = 4.0,
) -> np.ndarray:
    """
    Gera target por resultado relativo em R para trades long vs short.

    Para cada candle, simula de forma independente um trade long e um trade
    short com o mesmo RR. O target final e a diferenca normalizada:
    `(resultado_long_R - resultado_short_R) / (RR + 1)`.

    Assim, valores positivos indicam que o lado comprador teria vantagem sob
    aquele gerenciamento, valores negativos favorecem venda, e valores perto
    de zero representam contexto ambiguo ou lateral.
    """
    required = {"close", "high", "low", "atr"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame deve conter {sorted(required)}. Faltando: {sorted(missing)}")

    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    length = len(df)
    targets = np.zeros(length, dtype=np.float32)

    if stop_mode in {"structure", "sweep"}:
        long_stops, short_stops = compute_structure_stop_levels(
            df,
            atr_mult=atr_mult,
            stop_buffer_atr=stop_buffer_atr,
            sweep_max_age=sweep_max_age,
            min_risk_atr=min_risk_atr,
            max_risk_atr=max_risk_atr,
        )
    else:
        atr = df["atr"].to_numpy(dtype=float)
        risk_dist = np.maximum(atr * atr_mult, 1e-6)
        long_stops = close - risk_dist
        short_stops = close + risk_dist

    for i in range(length - horizon):
        entry = close[i]
        long_stop = float(long_stops[i])
        short_stop = float(short_stops[i])
        long_risk = max(entry - long_stop, 1e-8)
        short_risk = max(short_stop - entry, 1e-8)
        long_target = entry + (long_risk * reward_to_risk)
        short_target = entry - (short_risk * reward_to_risk)

        long_r = _directional_rr_outcome(
            entry=entry,
            stop=long_stop,
            target=long_target,
            highs=high,
            lows=low,
            closes=close,
            start_idx=i,
            horizon=horizon,
            direction=1,
            reward_to_risk=reward_to_risk,
        )
        short_r = _directional_rr_outcome(
            entry=entry,
            stop=short_stop,
            target=short_target,
            highs=high,
            lows=low,
            closes=close,
            start_idx=i,
            horizon=horizon,
            direction=-1,
            reward_to_risk=reward_to_risk,
        )
        targets[i] = np.clip((long_r - short_r) / (reward_to_risk + 1.0), -1.0, 1.0)

    return targets


def prepare_market_data(
    df: pd.DataFrame,
    swing_window: int,
    horizon: int,
    atr_mult: float,
    min_activation: float,
    feature_mode: str = FEATURE_MODE_CAUSAL_RAW,
    target_mode: str = "atr_barrier",
    reward_to_risk: float = 1.0,
    stop_mode: str = "atr",
    stop_buffer_atr: float = 0.05,
    sweep_max_age: int = 60,
    min_risk_atr: float = 0.25,
    max_risk_atr: float = 4.0,
) -> PreparedMarketData:
    """
    Constrói features causais SMC e targets reais a partir de OHLCV.

    O cálculo é feito sobre a série inteira porque os indicadores foram
    construídos de forma causal. Assim, quando depois cortamos por fold,
    os primeiros candles de cada janela preservam o contexto anterior.
    """
    df_smc, features = build_smc_features(
        df,
        swing_window=swing_window,
        feature_mode=feature_mode,
    )

    if target_mode == "rr":
        barrier_targets = generate_rr_targets(
            df_smc,
            horizon=horizon,
            reward_to_risk=reward_to_risk,
            stop_mode=stop_mode,
            atr_mult=atr_mult,
            stop_buffer_atr=stop_buffer_atr,
            sweep_max_age=sweep_max_age,
            min_risk_atr=min_risk_atr,
            max_risk_atr=max_risk_atr,
        )
    elif target_mode == "atr_barrier":
        barrier_targets = generate_barrier_targets(df_smc, horizon=horizon, atr_mult=atr_mult)
    else:
        raise ValueError(f"target_mode invalido: {target_mode}")
    legacy_targets = generate_market_targets(df_smc, horizon=horizon)

    trimmed_features = features.iloc[:-horizon].copy()
    trimmed_targets = barrier_targets[:-horizon]
    trimmed_legacy = legacy_targets[:-horizon]
    trimmed_df = df_smc.iloc[:-horizon].copy()

    mask = np.abs(trimmed_targets) > min_activation

    return PreparedMarketData(
        df_smc=trimmed_df.loc[mask].copy(),
        features=trimmed_features.loc[mask].copy(),
        targets=trimmed_targets[mask],
        legacy_targets=trimmed_legacy[mask],
        mask=mask,
        feature_mode=feature_mode,
        swing_window=swing_window,
        horizon=horizon,
        atr_mult=atr_mult,
        min_activation=min_activation,
        target_mode=target_mode,
        reward_to_risk=reward_to_risk,
    )


def split_temporal_data(
    x_values: np.ndarray,
    y_values: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Divide o dataset de forma cronológica em treino, validação e teste."""
    n_samples = len(x_values)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    train = (x_values[:train_end], y_values[:train_end])
    val = (x_values[train_end:val_end], y_values[train_end:val_end])
    test = (x_values[val_end:], y_values[val_end:])
    return train, val, test


def build_loader(
    x_values: np.ndarray,
    y_values: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Cria um DataLoader simples para treino ou avaliação."""
    dataset = TensorDataset(
        torch.tensor(x_values, dtype=torch.float32),
        torch.tensor(y_values, dtype=torch.float32).unsqueeze(1),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_training_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    normalized_strengths: torch.Tensor,
    lambda_rule_usage: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Loss de treino do regime real.

    Mantém MSE como termo principal, mas penaliza dominância excessiva de uma
    única regra para evitar o colapso observado no treino anterior.
    """
    loss_mse = F.mse_loss(predictions, targets)

    mean_usage = normalized_strengths.mean(dim=0)
    mean_usage = mean_usage / (mean_usage.sum() + 1e-8)
    loss_usage = mean_usage.pow(2).sum()

    loss_total = loss_mse + (lambda_rule_usage * loss_usage)
    return loss_total, {
        "mse": float(loss_mse.item()),
        "usage": float(loss_usage.item()),
        "total": float(loss_total.item()),
    }


def create_rescaled_model(consequent_scale: float = 80.0) -> ANFISModel:
    """Instancia o ANFIS e reescala os consequentes para o alvo real."""
    model = ANFISModel(RuleBase())
    with torch.no_grad():
        model.consequents.div_(consequent_scale)
    return model


def train_model(
    model: ANFISModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    sigma_min: float,
    lambda_rule_usage: float,
    patience: int,
) -> Dict[str, List[float]]:
    """Loop de treinamento com early stopping e scheduler."""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=5e-5,
    )

    history = {
        "train_loss": [],
        "train_mse": [],
        "train_usage": [],
        "val_loss": [],
        "val_mse": [],
        "val_usage": [],
        "lr": [],
    }

    best_state = None
    best_val_loss = float("inf")
    bad_epochs = 0

    for _epoch in range(epochs):
        model.train()
        train_total = []
        train_mse = []
        train_usage = []

        for x_batch, y_batch in train_loader:
            predictions, _, normalized = model(x_batch)
            loss, components = compute_training_loss(
                predictions,
                y_batch,
                normalized,
                lambda_rule_usage=lambda_rule_usage,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.clamp_mf_params(sigma_min=sigma_min)

            train_total.append(components["total"])
            train_mse.append(components["mse"])
            train_usage.append(components["usage"])

        model.eval()
        val_total = []
        val_mse = []
        val_usage = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                predictions, _, normalized = model(x_batch)
                loss, components = compute_training_loss(
                    predictions,
                    y_batch,
                    normalized,
                    lambda_rule_usage=lambda_rule_usage,
                )
                val_total.append(components["total"])
                val_mse.append(components["mse"])
                val_usage.append(components["usage"])

        avg_train_loss = float(np.mean(train_total))
        avg_val_loss = float(np.mean(val_total))
        avg_train_mse = float(np.mean(train_mse))
        avg_val_mse = float(np.mean(val_mse))
        avg_train_usage = float(np.mean(train_usage))
        avg_val_usage = float(np.mean(val_usage))

        history["train_loss"].append(avg_train_loss)
        history["train_mse"].append(avg_train_mse)
        history["train_usage"].append(avg_train_usage)
        history["val_loss"].append(avg_val_loss)
        history["val_mse"].append(avg_val_mse)
        history["val_usage"].append(avg_val_usage)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss - 1e-5:
            best_val_loss = avg_val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def predict_scores(model: ANFISModel, x_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Retorna scores e firing strengths normalizadas para um array numpy."""
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_values, dtype=torch.float32)
        scores, _, normalized = model(x_tensor)
    return scores.squeeze(1).cpu().numpy(), normalized.cpu().numpy()


def compute_signal_metrics(scores: np.ndarray, targets: np.ndarray, threshold: float) -> Dict[str, float]:
    """Calcula métricas de direcionalidade dado um threshold de ativação."""
    mask = np.abs(scores) >= threshold
    activated = int(mask.sum())
    if activated == 0:
        return {
            "threshold": float(threshold),
            "activated": 0,
            "coverage": 0.0,
            "directional_accuracy": 0.0,
            "avg_abs_target": 0.0,
            "score_target_corr": 0.0,
            "objective": float("-inf"),
        }

    masked_scores = scores[mask]
    masked_targets = targets[mask]
    directional_accuracy = float(np.mean(np.sign(masked_scores) == np.sign(masked_targets)))
    avg_abs_target = float(np.mean(np.abs(masked_targets)))
    coverage = float(activated / len(scores))
    if activated > 2 and np.std(masked_scores) > 1e-8 and np.std(masked_targets) > 1e-8:
        score_target_corr = float(np.corrcoef(masked_scores, masked_targets)[0, 1])
    else:
        score_target_corr = 0.0
    objective = float((directional_accuracy - 0.5) * np.sqrt(activated) * max(avg_abs_target, 1e-6))

    return {
        "threshold": float(threshold),
        "activated": activated,
        "coverage": coverage,
        "directional_accuracy": directional_accuracy,
        "avg_abs_target": avg_abs_target,
        "score_target_corr": score_target_corr,
        "objective": objective,
    }


def select_activation_threshold(
    scores: np.ndarray,
    targets: np.ndarray,
    candidate_thresholds: List[float],
    min_activations: int = 120,
) -> Tuple[float, List[Dict[str, float]]]:
    """Seleciona threshold com melhor compromisso entre edge e quantidade de sinais."""
    diagnostics = []
    best_metric = None

    for threshold in candidate_thresholds:
        metric = compute_signal_metrics(scores, targets, threshold)
        diagnostics.append(metric)

        if metric["activated"] < min_activations:
            continue

        if best_metric is None or metric["objective"] > best_metric["objective"]:
            best_metric = metric

    if best_metric is None:
        best_metric = max(diagnostics, key=lambda item: item["activated"])

    return float(best_metric["threshold"]), diagnostics


def compute_score_statistics(scores: np.ndarray) -> Dict[str, float]:
    """Resumo simples da distribuição dos scores."""
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }


def evaluate_thresholds_with_backtest(
    df_slice: pd.DataFrame,
    scores: np.ndarray,
    candidate_thresholds: Iterable[float],
    *,
    initial_capital: float,
    risk_per_trade: float,
    reward_to_risk: float,
    stop_mode: str,
    atr_stop_mult: float,
    atr_target_mult: float,
    max_holding_bars: Optional[int],
    min_trades: int,
    stop_buffer_atr: float = 0.05,
    sweep_max_age: int = 60,
    min_risk_atr: float = 0.25,
    max_risk_atr: float = 4.0,
) -> List[Dict[str, float]]:
    """
    Avalia thresholds com o mesmo motor de carteira usado no backtest final.

    O objetivo aqui é alinhar a calibração do limiar com a métrica que mais
    interessa na prática do trabalho: a consistência da carteira, e não apenas
    a acurácia direcional do score.
    """
    diagnostics: List[Dict[str, float]] = []

    for threshold in candidate_thresholds:
        equity_curve, trades = simulate_trading_from_scores(
            data=df_slice,
            scores=scores,
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade,
            reward_to_risk=reward_to_risk,
            activation_threshold=float(threshold),
            stop_mode=stop_mode,
            atr_stop_mult=atr_stop_mult,
            atr_target_mult=atr_target_mult,
            max_holding_bars=max_holding_bars,
            stop_buffer_atr=stop_buffer_atr,
            sweep_max_age=sweep_max_age,
            min_risk_atr=min_risk_atr,
            max_risk_atr=max_risk_atr,
        )
        metrics = calculate_performance_metrics(equity_curve, trades, timestamps=df_slice.index)
        diagnostics.append({
            "threshold": float(threshold),
            "total_trades": int(metrics["total_trades"]),
            "total_return": float(metrics["total_return"]),
            "annualized_return": float(metrics["annualized_return"]),
            "max_drawdown": float(metrics["max_drawdown"]),
            "win_rate": float(metrics["win_rate"]),
            "profit_factor": float(metrics["profit_factor"]),
            "payoff_ratio": float(metrics["payoff_ratio"]),
            "expectancy": float(metrics["expectancy"]),
            "avg_trade_return_pct": float(metrics["avg_trade_return_pct"]),
            "sharpe_ratio": float(metrics["sharpe_ratio"]),
            "sortino_ratio": float(metrics["sortino_ratio"]),
            "calmar_ratio": float(metrics["calmar_ratio"]),
            "sqn": float(metrics["sqn"]),
            "eligible": bool(metrics["total_trades"] >= min_trades),
        })

    return diagnostics


def select_backtest_threshold(
    diagnostics: List[Dict[str, float]],
    fallback_threshold: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Seleciona threshold por hierarquia simples de robustez operacional.

    Critério:
    1. considera apenas thresholds com número mínimo de trades;
    2. escolhe maior profit factor;
    3. desempata por Calmar, retorno total e número de trades.

    Se nenhum candidato atender o mínimo de trades, cai para o fallback vindo
    do critério direcional.
    """
    eligible = [item for item in diagnostics if item["eligible"]]
    positive_edge = [item for item in eligible if item["profit_factor"] > 1.0]

    if positive_edge:
        best = max(
            positive_edge,
            key=lambda item: (
                (item["profit_factor"] - 1.0) * np.sqrt(max(item["total_trades"], 1)),
                item["total_return"],
                item["calmar_ratio"],
                item["total_trades"],
            ),
        )
        return float(best["threshold"]), best

    if eligible:
        best = max(
            eligible,
            key=lambda item: (
                (item["profit_factor"] - 1.0) * np.sqrt(max(item["total_trades"], 1)),
                item["total_return"],
                item["calmar_ratio"],
                item["total_trades"],
            ),
        )
        return float(best["threshold"]), best

    fallback = next((item for item in diagnostics if item["threshold"] == fallback_threshold), None)
    if fallback is not None:
        return float(fallback_threshold), fallback

    best = max(
        diagnostics,
        key=lambda item: (
            min(item["profit_factor"], 9.99),
            item["total_return"],
            item["total_trades"],
        ),
    )
    return float(best["threshold"]), best


def plot_training_history(history: Dict[str, List[float]], output_path: str) -> None:
    """Salva gráfico de convergência do treino em mercado real."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train Total", color="#ff8c00", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val Total", color="#39a7ff", linewidth=2)
    axes[0].set_title("Loss Total")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(history["train_mse"], label="Train MSE", color="#f45d48", linewidth=2)
    axes[1].plot(history["val_mse"], label="Val MSE", color="#2a9d8f", linewidth=2)
    axes[1].set_title("MSE")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("MSE")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
