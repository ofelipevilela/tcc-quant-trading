# -*- coding: utf-8 -*-
"""
Motor de Backtesting SMC e ANFIS.

Recebe um histórico de dados OHLCV e um modelo ANFIS treinado,
realiza a inferência em lote e simula operações de trading com base nos
scores fuzzy resultantes e no gerenciamento de risco no nível do preço.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from anfis.anfis_model import ANFISModel
from anfis.rule_base import RuleBase
from backtest.risk_levels import compute_structure_stop_levels
from smc.feature_factory import FEATURE_MODE_CAUSAL_RAW, build_smc_features

logger = logging.getLogger(__name__)


def simulate_trading_from_scores(
    data: pd.DataFrame,
    scores: np.ndarray,
    *,
    initial_capital: float = 10000.0,
    risk_per_trade: float = 0.01,
    reward_to_risk: float = 2.0,
    activation_threshold: float = 0.20,
    stop_mode: str = "atr",
    atr_stop_mult: float = 1.0,
    atr_target_mult: float = 1.0,
    max_holding_bars: Optional[int] = 15,
    stop_buffer_atr: float = 0.05,
    sweep_max_age: int = 60,
    min_risk_atr: float = 0.25,
    max_risk_atr: float = 4.0,
) -> Tuple[List[float], List[Dict]]:
    """
    Simula a carteira a partir de scores já calculados.

    Esta função separa a etapa de geração de sinais da etapa de execução da
    carteira. Assim, o mesmo motor pode ser usado tanto no backtest final
    quanto na seleção de threshold durante o walk-forward.
    """
    df = data.copy()
    if len(df) != len(scores):
        raise ValueError("Numero de scores deve ser igual ao numero de barras do DataFrame.")

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    atrs = df["atr"].values if "atr" in df.columns else np.zeros(len(df), dtype=float)
    structural_long_stops = structural_short_stops = None
    if stop_mode in {"structure", "sweep"}:
        structural_long_stops, structural_short_stops = compute_structure_stop_levels(
            df,
            atr_mult=atr_stop_mult,
            stop_buffer_atr=stop_buffer_atr,
            sweep_max_age=sweep_max_age,
            min_risk_atr=min_risk_atr,
            max_risk_atr=max_risk_atr,
        )

    capital = initial_capital
    current_position = 0
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    entry_idx = None
    entry_bar_index = None

    equity_curve: List[float] = []
    trades: List[Dict] = []

    def calculate_position_size(entry: float, sl: float, current_capital: float) -> float:
        if abs(entry - sl) < 1e-6:
            return 0.0
        risk_amount = current_capital * risk_per_trade
        price_risk = abs(entry - sl)
        return risk_amount / price_risk

    def open_position(direction: int, price: float, idx: int, atr_value: float) -> Tuple[float, float]:
        if stop_mode == "atr":
            risk_dist = max(float(atr_value) * atr_stop_mult, 1e-6)
            if direction == 1:
                sl = price - risk_dist
                tp = price + (float(atr_value) * atr_target_mult)
            else:
                sl = price + risk_dist
                tp = price - (float(atr_value) * atr_target_mult)
        elif stop_mode in {"structure", "sweep"}:
            if structural_long_stops is None or structural_short_stops is None:
                raise RuntimeError("Stops estruturais nao foram inicializados.")
            if direction == 1:
                sl = float(structural_long_stops[idx])
                risk_dist = max(price - sl, 1e-6)
                tp = price + (risk_dist * reward_to_risk)
            else:
                sl = float(structural_short_stops[idx])
                risk_dist = max(sl - price, 1e-6)
                tp = price - (risk_dist * reward_to_risk)
        else:
            lookback = min(idx, 15)
            window = df.iloc[idx - lookback: idx + 1]
            if direction == 1:
                sl = window["low"].min()
                if abs(price - sl) < 1e-4:
                    sl = price * 0.99
                risk_dist = price - sl
                tp = price + (risk_dist * reward_to_risk)
            else:
                sl = window["high"].max()
                if abs(sl - price) < 1e-4:
                    sl = price * 1.01
                risk_dist = sl - price
                tp = price - (risk_dist * reward_to_risk)
        return sl, tp

    for i in range(len(df)):
        current_price = closes[i]
        current_date = df.index[i]

        if current_position != 0:
            closed = False
            exit_price = 0.0
            exit_reason = "OPEN"

            if current_position == 1:
                if lows[i] <= stop_loss:
                    exit_price = stop_loss
                    closed = True
                    exit_reason = "STOP"
                elif highs[i] >= take_profit:
                    exit_price = take_profit
                    closed = True
                    exit_reason = "TARGET"
            else:
                if highs[i] >= stop_loss:
                    exit_price = stop_loss
                    closed = True
                    exit_reason = "STOP"
                elif lows[i] <= take_profit:
                    exit_price = take_profit
                    closed = True
                    exit_reason = "TARGET"

            if (
                not closed
                and max_holding_bars is not None
                and entry_bar_index is not None
                and (i - entry_bar_index) >= max_holding_bars
            ):
                exit_price = current_price
                closed = True
                exit_reason = "TIME"

            if closed:
                pos_size = calculate_position_size(entry_price, stop_loss, capital)
                pnl_per_share = (
                    exit_price - entry_price
                    if current_position == 1
                    else entry_price - exit_price
                )
                realized_pnl = pnl_per_share * pos_size
                capital_before_trade = capital
                capital += realized_pnl

                trades.append({
                    "entry_date": entry_idx,
                    "exit_date": current_date,
                    "type": "LONG" if current_position == 1 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": realized_pnl,
                    "return_pct": (realized_pnl / capital_before_trade) * 100 if capital_before_trade else 0.0,
                    "exit_reason": exit_reason,
                    "bars_held": int(i - entry_bar_index) if entry_bar_index is not None else 0,
                })

                current_position = 0
                entry_price = 0.0
                stop_loss = 0.0
                take_profit = 0.0
                entry_idx = None
                entry_bar_index = None

        if current_position == 0:
            score = scores[i]
            if score >= activation_threshold:
                stop_loss, take_profit = open_position(1, current_price, i, atrs[i])
                current_position = 1
                entry_price = current_price
                entry_idx = current_date
                entry_bar_index = i
            elif score <= -activation_threshold:
                stop_loss, take_profit = open_position(-1, current_price, i, atrs[i])
                current_position = -1
                entry_price = current_price
                entry_idx = current_date
                entry_bar_index = i

        floating_pnl = 0.0
        if current_position == 1:
            floating_pnl = current_price - entry_price
        elif current_position == -1:
            floating_pnl = entry_price - current_price

        pos_size = calculate_position_size(entry_price, stop_loss, capital) if current_position != 0 else 0.0
        equity_curve.append(capital + (floating_pnl * pos_size))

    return equity_curve, trades


class BacktestEngine:
    """
    Simulador de backtest passo-a-passo processando sinais ANFIS
    e aplicando gestão de risco (target/stop loss).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        model_path: str,
        initial_capital: float = 10000.0,
        risk_per_trade: float = 0.01,
        reward_to_risk: float = 2.0,
        activation_threshold: float = 0.20,
        stop_mode: str = "atr",
        atr_stop_mult: float = 1.0,
        atr_target_mult: float = 1.0,
        max_holding_bars: Optional[int] = 15,
        smc_swing_window: int = 5,
        smc_feature_mode: str = FEATURE_MODE_CAUSAL_RAW,
    ):
        self.data = data.copy()
        self.model_path = model_path
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.reward_to_risk = reward_to_risk
        self.activation_threshold = activation_threshold
        self.stop_mode = stop_mode
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.max_holding_bars = max_holding_bars
        self.smc_swing_window = smc_swing_window
        self.smc_feature_mode = smc_feature_mode

        self.equity_curve: List[float] = []
        self.trades: List[Dict] = []
        self.capital = initial_capital

        self._prepare_model()

    def _prepare_model(self) -> None:
        """Prepara o modelo ANFIS carregando os pesos."""
        logger.info("Carregando modelo ANFIS de %s", self.model_path)
        self.model = ANFISModel(RuleBase())
        self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
        self.model.eval()

    def run(self) -> Tuple[List[float], List[Dict]]:
        """
        Executa a simulação completa do backtest.

        Passos:
        1. Gera os sinais SMC puros (vetorizado).
        2. Avalia todas as amostras via ANFIS no forward pass.
        3. Executa a carteira com a mesma lógica usada na calibração.
        """
        logger.info("Processando indicadores SMC...")
        df_smc, anfis_inputs = build_smc_features(
            self.data,
            swing_window=self.smc_swing_window,
            feature_mode=self.smc_feature_mode,
        )

        logger.info("Executando inferencia do ANFIS...")
        x_tensor = torch.tensor(anfis_inputs.values, dtype=torch.float32)

        with torch.no_grad():
            scores_tensor, _, _ = self.model(x_tensor)
            scores = scores_tensor.squeeze().numpy()

        df_smc["anfis_score"] = scores
        self.processed_data = df_smc

        logger.info(
            "Distribuicao dos scores: mean=%.4f std=%.4f min=%.4f max=%.4f",
            float(np.mean(scores)),
            float(np.std(scores)),
            float(np.min(scores)),
            float(np.max(scores)),
        )
        logger.info(
            "Ativacoes acima de |%.2f|: %d/%d | feature_mode=%s",
            self.activation_threshold,
            int(np.sum(np.abs(scores) >= self.activation_threshold)),
            len(scores),
            self.smc_feature_mode,
        )

        logger.info("Iniciando loop do motor de trade simulado...")
        self.equity_curve, self.trades = simulate_trading_from_scores(
            data=df_smc,
            scores=scores,
            initial_capital=self.initial_capital,
            risk_per_trade=self.risk_per_trade,
            reward_to_risk=self.reward_to_risk,
            activation_threshold=self.activation_threshold,
            stop_mode=self.stop_mode,
            atr_stop_mult=self.atr_stop_mult,
            atr_target_mult=self.atr_target_mult,
            max_holding_bars=self.max_holding_bars,
        )
        self.capital = self.equity_curve[-1] if self.equity_curve else self.initial_capital

        logger.info("Simulacao concluida.")
        return self.equity_curve, self.trades
