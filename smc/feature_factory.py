# -*- coding: utf-8 -*-
"""
Fabrica de modos de leitura SMC para o pipeline ANFIS.

Centraliza a construcao das features para evitar que:
- treino,
- backtest,
- walk-forward,
- scripts de diagnostico
usem traducoes estruturais diferentes sem perceber.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .indicators import DEFAULT_TREND_DECAY, DEFAULT_TREND_MAX, SMCIndicators


FEATURE_MODE_CAUSAL_RAW = "causal_raw"
FEATURE_MODE_CAUSAL_ENHANCED = "causal_enhanced"
FEATURE_MODE_LEGACY_LIKE = "legacy_like"
FEATURE_MODE_CAUSAL_BOS_ANCHORED = "causal_bos_anchored"
FEATURE_MODE_CAUSAL_V2 = "causal_v2"
FEATURE_MODE_CAUSAL_V3 = "causal_v3"

VALID_FEATURE_MODES = {
    FEATURE_MODE_CAUSAL_RAW,
    FEATURE_MODE_CAUSAL_ENHANCED,
    FEATURE_MODE_LEGACY_LIKE,
    FEATURE_MODE_CAUSAL_BOS_ANCHORED,
    FEATURE_MODE_CAUSAL_V2,
    FEATURE_MODE_CAUSAL_V3,
}


def resolve_feature_mode(
    value: Optional[str],
    *,
    default: str = FEATURE_MODE_LEGACY_LIKE,
) -> str:
    """Resolve e valida um modo de feature a partir de um valor externo."""
    feature_mode = (value or default).strip()
    if feature_mode not in VALID_FEATURE_MODES:
        valid = ", ".join(sorted(VALID_FEATURE_MODES))
        raise ValueError(f"feature_mode invalido: {feature_mode}. Opcoes: {valid}")
    return feature_mode


def _build_legacy_like_dataset(df: pd.DataFrame, swing_window: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reconstrucao aproximada da leitura estrutural anterior a revisao causal.

    Importante:
    - esse modo nao representa uma implementacao operacionalmente honesta;
    - ele existe para auditoria comparativa e para quantificar o efeito da
      leitura estrutural otimista usada antes da causalizacao.
    """
    smc = SMCIndicators(df.copy(), swing_window=swing_window)
    smc._compute_atr()
    smc.detect_swings()
    base = smc.df.copy()

    base["swing_high_price_legacy"] = base["pivot_swing_high_price"]
    base["swing_low_price_legacy"] = base["pivot_swing_low_price"]

    close = base["close"].values
    opens = base["open"].values
    highs = base["high"].values
    lows = base["low"].values
    atr = base["atr"].values
    sh_prices = base["swing_high_price_legacy"].values
    sl_prices = base["swing_low_price_legacy"].values
    length = len(base)

    is_bos = np.zeros(length, dtype=bool)
    bos_dir = np.zeros(length, dtype=np.int8)
    last_sh = np.nan
    last_sl = np.nan

    for i in range(length):
        if i > 0:
            if not np.isnan(sh_prices[i - 1]):
                last_sh = sh_prices[i - 1]
            if not np.isnan(sl_prices[i - 1]):
                last_sl = sl_prices[i - 1]

        if not np.isnan(last_sh) and close[i] > last_sh:
            is_bos[i] = True
            bos_dir[i] = 1
            last_sh = np.nan
        elif not np.isnan(last_sl) and close[i] < last_sl:
            is_bos[i] = True
            bos_dir[i] = -1
            last_sl = np.nan

    base["is_bos"] = is_bos
    base["bos_direction"] = bos_dir

    is_cisd = np.zeros(length, dtype=bool)
    cisd_dir = np.zeros(length, dtype=np.int8)
    sl_delivery_top = np.nan
    sh_delivery_bot = np.nan
    sl_consumed = True
    sh_consumed = True

    for i in range(length):
        if not np.isnan(base["pivot_swing_low_price"].iloc[i]):
            sl_delivery_top = max(opens[i], close[i])
            sl_consumed = False

        if not np.isnan(base["pivot_swing_high_price"].iloc[i]):
            sh_delivery_bot = min(opens[i], close[i])
            sh_consumed = False

        if (not sl_consumed) and (not np.isnan(sl_delivery_top)) and close[i] > sl_delivery_top:
            is_cisd[i] = True
            cisd_dir[i] = 1
            sl_consumed = True

        if (not sh_consumed) and (not np.isnan(sh_delivery_bot)) and close[i] < sh_delivery_bot:
            is_cisd[i] = True
            cisd_dir[i] = -1
            sh_consumed = True

    base["is_cisd"] = is_cisd
    base["cisd_direction"] = cisd_dir

    is_fvg = np.zeros(length, dtype=bool)
    fvg_dir = np.zeros(length, dtype=np.int8)
    fvg_quality = np.zeros(length, dtype=float)

    for i in range(2, length):
        if atr[i] <= 0:
            continue

        bull_gap = lows[i] - highs[i - 2]
        if bull_gap > 0:
            ratio = bull_gap / atr[i]
            if ratio >= 0.1:
                is_fvg[i] = True
                fvg_dir[i] = 1
                fvg_quality[i] = min(ratio, 4.0)
            continue

        bear_gap = lows[i - 2] - highs[i]
        if bear_gap > 0:
            ratio = bear_gap / atr[i]
            if ratio >= 0.1:
                is_fvg[i] = True
                fvg_dir[i] = -1
                fvg_quality[i] = min(ratio, 4.0)

    base["is_fvg"] = is_fvg
    base["fvg_direction"] = fvg_dir
    base["fvg_quality"] = fvg_quality

    is_sweep = np.zeros(length, dtype=bool)
    sweep_dir = np.zeros(length, dtype=np.int8)
    sweep_quality = np.zeros(length, dtype=float)
    last_sl = np.nan
    last_sh = np.nan

    for i in range(1, length):
        if not np.isnan(sl_prices[i - 1]):
            last_sl = sl_prices[i - 1]
        if not np.isnan(sh_prices[i - 1]):
            last_sh = sh_prices[i - 1]

        if atr[i] <= 0:
            continue

        body = max(abs(close[i] - opens[i]), 1e-10)

        if not np.isnan(last_sl) and lows[i] < last_sl and close[i] > last_sl:
            lower_wick = min(opens[i], close[i]) - lows[i]
            wick_ratio = lower_wick / body
            rejection_depth = abs(last_sl - lows[i]) / atr[i]
            quality = min(wick_ratio * rejection_depth, 3.0)
            if quality > 0.1:
                is_sweep[i] = True
                sweep_dir[i] = 1
                sweep_quality[i] = quality
                last_sl = np.nan
        elif not np.isnan(last_sh) and highs[i] > last_sh and close[i] < last_sh:
            upper_wick = highs[i] - max(opens[i], close[i])
            wick_ratio = upper_wick / body
            rejection_depth = abs(highs[i] - last_sh) / atr[i]
            quality = min(wick_ratio * rejection_depth, 3.0)
            if quality > 0.1:
                is_sweep[i] = True
                sweep_dir[i] = -1
                sweep_quality[i] = quality
                last_sh = np.nan

    base["is_sweep"] = is_sweep
    base["sweep_direction"] = sweep_dir
    base["sweep_quality"] = sweep_quality

    leg_high = np.full(length, np.nan)
    leg_low = np.full(length, np.nan)
    price_zone = np.full(length, 0.5)
    current_leg_h = np.nan
    current_leg_l = np.nan

    for i in range(length):
        if not np.isnan(sh_prices[i]):
            current_leg_h = sh_prices[i]
        if not np.isnan(sl_prices[i]):
            current_leg_l = sl_prices[i]

        leg_high[i] = current_leg_h
        leg_low[i] = current_leg_l

        if not np.isnan(current_leg_h) and not np.isnan(current_leg_l):
            span = current_leg_h - current_leg_l
            if span > 0:
                price_zone[i] = np.clip((close[i] - current_leg_l) / span, 0.0, 1.0)

    base["price_zone"] = price_zone
    base["leg_high"] = leg_high
    base["leg_low"] = leg_low

    trend = np.zeros(length, dtype=float)
    current_trend = 0.0
    current_direction = 0

    for i in range(length):
        current_trend *= DEFAULT_TREND_DECAY

        if bos_dir[i] != 0:
            if bos_dir[i] == current_direction:
                current_trend += bos_dir[i] * 25.0
            else:
                current_direction = int(bos_dir[i])
                current_trend = bos_dir[i] * 25.0

        if cisd_dir[i] != 0:
            if cisd_dir[i] == current_direction:
                current_trend += cisd_dir[i] * 15.0
            else:
                current_direction = int(cisd_dir[i])
                current_trend = cisd_dir[i] * 15.0

        if bos_dir[i] != 0 and bos_dir[i] == cisd_dir[i]:
            current_trend += bos_dir[i] * 10.0

        current_trend = np.clip(current_trend, -DEFAULT_TREND_MAX, DEFAULT_TREND_MAX)
        trend[i] = current_trend

    base["trend_strength"] = trend

    decay = 0.05
    for quality_col, direction_col, max_quality in (
        ("fvg_quality", "fvg_direction", 4.0),
        ("sweep_quality", "sweep_direction", 3.0),
    ):
        qualities = base[quality_col].values.copy()
        directions = base[direction_col].values.copy()
        last_quality = 0.0
        last_direction = 0

        for i in range(len(qualities)):
            if qualities[i] > 0:
                last_quality = qualities[i]
                last_direction = directions[i]
            else:
                last_quality = max(0.0, last_quality - decay)
                qualities[i] = last_quality
                directions[i] = last_direction if last_quality > 0 else 0

        base[quality_col] = np.clip(qualities, 0.0, max_quality)
        base[direction_col] = directions

    features = base[["trend_strength", "price_zone", "fvg_quality", "sweep_quality"]].copy()
    features["trend_strength"] = features["trend_strength"].clip(-100.0, 100.0)
    features["price_zone"] = features["price_zone"].clip(0.0, 1.0)
    features["fvg_quality"] = features["fvg_quality"].clip(0.0, 4.0)
    features["sweep_quality"] = features["sweep_quality"].clip(0.0, 3.0)
    features = features.fillna(0.0)
    return base, features


def build_smc_features(
    df: pd.DataFrame,
    *,
    swing_window: int = 5,
    feature_mode: str = FEATURE_MODE_CAUSAL_RAW,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Constroi DataFrame SMC + features do ANFIS para um modo especifico."""
    feature_mode = resolve_feature_mode(feature_mode, default=FEATURE_MODE_CAUSAL_RAW)

    if feature_mode == FEATURE_MODE_LEGACY_LIKE:
        return _build_legacy_like_dataset(df, swing_window=swing_window)

    if feature_mode == FEATURE_MODE_CAUSAL_BOS_ANCHORED:
        return _build_causal_bos_anchored_dataset(df, swing_window=swing_window)

    if feature_mode == FEATURE_MODE_CAUSAL_V2:
        return _build_causal_v2_dataset(df, swing_window=swing_window)

    if feature_mode == FEATURE_MODE_CAUSAL_V3:
        return _build_causal_v3_dataset(df, swing_window=swing_window)

    smc = SMCIndicators(df.copy(), swing_window=swing_window)
    df_smc = smc.compute_all()
    features = smc.get_anfis_inputs(use_enhanced=(feature_mode == FEATURE_MODE_CAUSAL_ENHANCED))
    return df_smc, features


def _build_causal_bos_anchored_dataset(df: pd.DataFrame, swing_window: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Leitura estrutural causal com leg ancorada no BOS.

    Diferenca em relacao ao causal_raw:
    - No causal_raw, a leg de referencia (leg_high/leg_low) so atualiza
      quando um swing e confirmado causalmente (i+n barras de delay).
    - Aqui, no momento do BOS, a leg ja e atualizada imediatamente:
        BOS bullish: leg_low  = ultimo swing_low confirmado,
                     leg_high = high[candle_bos] (atualiza com novos highs)
        BOS bearish: leg_high = ultimo swing_high confirmado,
                     leg_low  = low[candle_bos]  (atualiza com novos lows)

    Isso reflete o que um trader faz manualmente: ao ver o close acima do
    ultimo swing high confirmado, a nova leg e imediatamente desenhada
    usando a maxima daquele candle como topo provisorio.

    Todos os outros indicadores (BOS, CISD, FVG, sweep, trend_strength)
    sao identicos ao pipeline causal padrao — zero look-ahead.
    """
    smc = SMCIndicators(df.copy(), swing_window=swing_window)
    df_smc = smc.compute_all()
    base = df_smc.copy()

    close = base["close"].values
    highs = base["high"].values
    lows = base["low"].values
    # Swings confirmados causalmente (disponivel em i+n, 100% honestos)
    sh_prices = base["swing_high_price"].values
    sl_prices = base["swing_low_price"].values
    bos_dir = base["bos_direction"].values
    length = len(base)

    leg_high = np.full(length, np.nan)
    leg_low = np.full(length, np.nan)
    price_zone = np.full(length, 0.5)

    current_leg_h = np.nan
    current_leg_l = np.nan
    last_known_sh = np.nan  # ultimo swing high confirmado visto ate aqui
    last_known_sl = np.nan  # ultimo swing low confirmado visto ate aqui
    current_bias = 0        # +1 bullish, -1 bearish, 0 indefinido

    for i in range(length):
        # Atualiza os swings conhecidos (ja confirmados causalmente)
        if not np.isnan(sh_prices[i]):
            last_known_sh = sh_prices[i]
        if not np.isnan(sl_prices[i]):
            last_known_sl = sl_prices[i]

        if bos_dir[i] == 1:
            # BOS bullish: nova leg parte do ultimo swing low conhecido
            # O teto da leg e provisoriamente a maxima deste candle
            current_bias = 1
            if not np.isnan(last_known_sl):
                current_leg_l = last_known_sl
            current_leg_h = highs[i]

        elif bos_dir[i] == -1:
            # BOS bearish: nova leg parte do ultimo swing high conhecido
            # O fundo da leg e provisoriamente a minima deste candle
            current_bias = -1
            if not np.isnan(last_known_sh):
                current_leg_h = last_known_sh
            current_leg_l = lows[i]

        else:
            # Sem BOS: atualiza o extreme corrente na direcao da tendencia
            if current_bias == 1 and not np.isnan(current_leg_h):
                current_leg_h = max(current_leg_h, highs[i])
            elif current_bias == -1 and not np.isnan(current_leg_l):
                current_leg_l = min(current_leg_l, lows[i])

        leg_high[i] = current_leg_h
        leg_low[i] = current_leg_l

        if not np.isnan(current_leg_h) and not np.isnan(current_leg_l):
            span = current_leg_h - current_leg_l
            if span > 0:
                price_zone[i] = np.clip((close[i] - current_leg_l) / span, 0.0, 1.0)

    base["price_zone"] = price_zone
    base["leg_high"] = leg_high
    base["leg_low"] = leg_low

    features = base[["trend_strength", "price_zone", "fvg_quality", "sweep_quality"]].copy()
    features["trend_strength"] = features["trend_strength"].clip(-100.0, 100.0)
    features["price_zone"] = features["price_zone"].clip(0.0, 1.0)
    features["fvg_quality"] = features["fvg_quality"].clip(0.0, 4.0)
    features["sweep_quality"] = features["sweep_quality"].clip(0.0, 3.0)
    features = features.fillna(0.0)
    return base, features


def _build_causal_v2_dataset(df: pd.DataFrame, swing_window: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Leitura estrutural causal V2 — empilha tres melhorias sobre causal_bos_anchored:

    1. Leg lock pos-BOS
       Apos BOS bullish, usa running max como leg_high provisorio ate o primeiro
       swing_high causal ser confirmado. A partir desse swing, trava o teto da leg.
       Resultado: price_zone deixa de comprimir para 0.5 conforme o mercado sobe;
       o ANFIS ve Premium quando o preco se aproxima do teto da leg concluida.

    2. Filtro direcional FVG e Sweep
       Zeramos fvg_quality e sweep_quality quando a direcao do evento nao alinha
       com o sinal de trend_strength. Um FVG bearish em tendencia bullish nao e
       confirmacao — e ruido. Antes, esse ruido entrava com magnitude completa.

    3. EMA momentum continuo no trend_strength
       Adiciona clip((EMA9 - EMA21) / ATR * 25, -25, +25) ao trend_strength base.
       Previne que o sinal de tendencia colapse entre eventos discretos (BOS/CISD)
       em mercados claramente direcionais. Totalmente causal.
    """
    smc = SMCIndicators(df.copy(), swing_window=swing_window)
    df_smc = smc.compute_all()
    base = df_smc.copy()

    close = base["close"].values
    highs = base["high"].values
    lows = base["low"].values
    atr = base["atr"].values
    sh_prices = base["swing_high_price"].values
    sl_prices = base["swing_low_price"].values
    bos_dir = base["bos_direction"].values
    length = len(base)

    # =========================================================
    # Melhoria 3: EMA momentum — componente continuo de tendencia
    # =========================================================
    def _ema(arr: np.ndarray, period: int) -> np.ndarray:
        alpha = 2.0 / (period + 1)
        out = np.empty(len(arr), dtype=np.float64)
        out[0] = arr[0]
        for k in range(1, len(arr)):
            out[k] = alpha * arr[k] + (1.0 - alpha) * out[k - 1]
        return out

    ema9 = _ema(close, 9)
    ema21 = _ema(close, 21)
    ema_momentum = np.zeros(length, dtype=np.float64)
    valid_atr = atr > 0
    ema_momentum[valid_atr] = (ema9[valid_atr] - ema21[valid_atr]) / atr[valid_atr] * 25.0
    ema_momentum = np.clip(ema_momentum, -25.0, 25.0)
    base["trend_strength"] = np.clip(base["trend_strength"].values + ema_momentum, -100.0, 100.0)

    # =========================================================
    # Melhoria 1: Leg lock apos primeiro swing confirmado pos-BOS
    # =========================================================
    leg_high = np.full(length, np.nan)
    leg_low = np.full(length, np.nan)
    price_zone = np.full(length, 0.5)

    current_leg_h = np.nan
    current_leg_l = np.nan
    last_known_sh = np.nan
    last_known_sl = np.nan
    current_bias = 0
    leg_h_locked = False
    leg_l_locked = False

    for i in range(length):
        # Atualiza swings causalmente confirmados ate a barra atual
        if not np.isnan(sh_prices[i]):
            last_known_sh = sh_prices[i]
        if not np.isnan(sl_prices[i]):
            last_known_sl = sl_prices[i]

        if bos_dir[i] == 1:
            current_bias = 1
            if not np.isnan(last_known_sl):
                current_leg_l = last_known_sl
                leg_l_locked = True
            current_leg_h = highs[i]
            leg_h_locked = False  # aguarda primeiro swing_high pos-BOS

        elif bos_dir[i] == -1:
            current_bias = -1
            if not np.isnan(last_known_sh):
                current_leg_h = last_known_sh
                leg_h_locked = True
            current_leg_l = lows[i]
            leg_l_locked = False  # aguarda primeiro swing_low pos-BOS

        # Tenta travar o lado aberto (primeiro swing confirmado pos-BOS)
        if current_bias == 1 and not leg_h_locked:
            if (not np.isnan(sh_prices[i])
                    and not np.isnan(current_leg_l)
                    and sh_prices[i] > current_leg_l):
                # Primeiro swing_high confirmado pos-BOS bullish — trava o teto
                current_leg_h = sh_prices[i]
                leg_h_locked = True
            elif not np.isnan(current_leg_h):
                # Ainda provisorio: tracking running max
                current_leg_h = max(current_leg_h, highs[i])

        elif current_bias == -1 and not leg_l_locked:
            if (not np.isnan(sl_prices[i])
                    and not np.isnan(current_leg_h)
                    and sl_prices[i] < current_leg_h):
                # Primeiro swing_low confirmado pos-BOS bearish — trava o fundo
                current_leg_l = sl_prices[i]
                leg_l_locked = True
            elif not np.isnan(current_leg_l):
                current_leg_l = min(current_leg_l, lows[i])

        leg_high[i] = current_leg_h
        leg_low[i] = current_leg_l

        if not np.isnan(current_leg_h) and not np.isnan(current_leg_l):
            span = current_leg_h - current_leg_l
            if span > 0:
                price_zone[i] = np.clip((close[i] - current_leg_l) / span, 0.0, 1.0)

    base["price_zone"] = price_zone
    base["leg_high"] = leg_high
    base["leg_low"] = leg_low

    # =========================================================
    # Melhoria 2: filtro direcional FVG e Sweep
    # =========================================================
    trend_sign = np.sign(base["trend_strength"].values)
    fvg_qual = base["fvg_quality"].values.copy()
    fvg_dir_vals = base["fvg_direction"].values
    sweep_qual = base["sweep_quality"].values.copy()
    sweep_dir_vals = base["sweep_direction"].values

    # Zeramos qualidade quando direcao do evento nao alinha com tendencia
    fvg_qual = np.where(fvg_dir_vals * trend_sign > 0, fvg_qual, 0.0)
    sweep_qual = np.where(sweep_dir_vals * trend_sign > 0, sweep_qual, 0.0)

    features = pd.DataFrame({
        "trend_strength": np.clip(base["trend_strength"].values, -100.0, 100.0),
        "price_zone": np.clip(price_zone, 0.0, 1.0),
        "fvg_quality": np.clip(fvg_qual, 0.0, 4.0),
        "sweep_quality": np.clip(sweep_qual, 0.0, 3.0),
    }, index=base.index).fillna(0.0)

    return base, features


def _build_causal_v3_dataset(df: pd.DataFrame, swing_window: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Leitura estrutural causal V3 — redesenha trend_strength e substitui
    sweep_quality por setup_phase, inspirado na logica sequencial do VFI.

    Melhorias sobre v2:

    1. trend_strength V3: tres componentes com papel claro
       - BOS direction × recency decay (estrutura, ±40)
       - EMA50 vs EMA200 / ATR (bias HTF equivalente, ±40)
       - EMA9 vs EMA21 / ATR (momentum curto prazo, ±20)
       Os tres precisam concordar para sinal forte (sum max = ±100).

    2. setup_phase [0, 3] substitui sweep_quality
       Codifica a sequencia causal que o VFI (e o trader manual) usa:
         Phase 0: nenhum setup ativo
         Phase 1: Sweep detectado — aguardando confirmacao
         Phase 2: CISD confirmado apos Sweep
         Phase 3: CISD + FVG (displacement completo)
       Com decaimento intra-phase por recencia. Reutiliza range [0, 3]
       e as MFs Fraco/Forte de sweep_quality sem alteracao.

    3. Mantem leg lock e filtro direcional FVG de v2.
    """
    smc = SMCIndicators(df.copy(), swing_window=swing_window)
    df_smc = smc.compute_all()
    base = df_smc.copy()

    close = base["close"].values
    highs = base["high"].values
    lows = base["low"].values
    atr = base["atr"].values
    sh_prices = base["swing_high_price"].values
    sl_prices = base["swing_low_price"].values
    bos_dir = base["bos_direction"].values
    is_sweep_arr = base["is_sweep"].values.astype(bool)
    sweep_dir_arr = base["sweep_direction"].values.astype(np.int8)
    is_cisd_arr = base["is_cisd"].values.astype(bool)
    cisd_dir_arr = base["cisd_direction"].values.astype(np.int8)
    is_fvg_arr = base["is_fvg"].values.astype(bool)
    fvg_dir_event = base["fvg_direction"].values.astype(np.int8)
    fvg_qual_raw = base["fvg_quality"].values.copy()
    length = len(base)

    # =========================================================
    # Funcao auxiliar: EMA
    # =========================================================
    def _ema(arr: np.ndarray, period: int) -> np.ndarray:
        alpha = 2.0 / (period + 1)
        out = np.empty(len(arr), dtype=np.float64)
        out[0] = arr[0]
        for k in range(1, len(arr)):
            out[k] = alpha * arr[k] + (1.0 - alpha) * out[k - 1]
        return out

    # =========================================================
    # 1. Trend Strength V3: Structural + HTF Bias + Momentum
    # =========================================================
    ema9 = _ema(close, 9)
    ema21 = _ema(close, 21)
    ema50 = _ema(close, 50)
    ema200 = _ema(close, 200)
    valid_atr = atr > 0

    # Componente 1: direcao do ultimo BOS × decaimento de recencia
    # Decai linearmente em 50 barras (~12h no M15)
    bos_component = np.zeros(length, dtype=np.float64)
    last_bos_dir_val = 0
    bars_since_bos = 999
    for i in range(length):
        if bos_dir[i] != 0:
            last_bos_dir_val = int(bos_dir[i])
            bars_since_bos = 0
        else:
            bars_since_bos += 1
        recency = max(0.0, 1.0 - bars_since_bos / 50.0)
        bos_component[i] = last_bos_dir_val * recency * 40.0

    # Componente 2: bias HTF — EMA50 vs EMA200 (equivale a ~H4 no M15)
    htf_bias = np.zeros(length, dtype=np.float64)
    htf_bias[valid_atr] = (ema50[valid_atr] - ema200[valid_atr]) / atr[valid_atr] * 20.0
    htf_bias = np.clip(htf_bias, -40.0, 40.0)

    # Componente 3: momentum de curto prazo — EMA9 vs EMA21
    momentum = np.zeros(length, dtype=np.float64)
    momentum[valid_atr] = (ema9[valid_atr] - ema21[valid_atr]) / atr[valid_atr] * 15.0
    momentum = np.clip(momentum, -20.0, 20.0)

    trend_v3 = np.clip(bos_component + htf_bias + momentum, -100.0, 100.0)

    # =========================================================
    # 2. Price Zone com Leg Lock (de v2)
    # =========================================================
    leg_high = np.full(length, np.nan)
    leg_low = np.full(length, np.nan)
    price_zone = np.full(length, 0.5)

    current_leg_h = np.nan
    current_leg_l = np.nan
    last_known_sh = np.nan
    last_known_sl = np.nan
    current_bias = 0
    leg_h_locked = False
    leg_l_locked = False

    for i in range(length):
        if not np.isnan(sh_prices[i]):
            last_known_sh = sh_prices[i]
        if not np.isnan(sl_prices[i]):
            last_known_sl = sl_prices[i]

        if bos_dir[i] == 1:
            current_bias = 1
            if not np.isnan(last_known_sl):
                current_leg_l = last_known_sl
                leg_l_locked = True
            current_leg_h = highs[i]
            leg_h_locked = False

        elif bos_dir[i] == -1:
            current_bias = -1
            if not np.isnan(last_known_sh):
                current_leg_h = last_known_sh
                leg_h_locked = True
            current_leg_l = lows[i]
            leg_l_locked = False

        if current_bias == 1 and not leg_h_locked:
            if (not np.isnan(sh_prices[i])
                    and not np.isnan(current_leg_l)
                    and sh_prices[i] > current_leg_l):
                current_leg_h = sh_prices[i]
                leg_h_locked = True
            elif not np.isnan(current_leg_h):
                current_leg_h = max(current_leg_h, highs[i])
        elif current_bias == -1 and not leg_l_locked:
            if (not np.isnan(sl_prices[i])
                    and not np.isnan(current_leg_h)
                    and sl_prices[i] < current_leg_h):
                current_leg_l = sl_prices[i]
                leg_l_locked = True
            elif not np.isnan(current_leg_l):
                current_leg_l = min(current_leg_l, lows[i])

        leg_high[i] = current_leg_h
        leg_low[i] = current_leg_l

        if not np.isnan(current_leg_h) and not np.isnan(current_leg_l):
            span = current_leg_h - current_leg_l
            if span > 0:
                price_zone[i] = np.clip((close[i] - current_leg_l) / span, 0.0, 1.0)

    base["price_zone"] = price_zone
    base["leg_high"] = leg_high
    base["leg_low"] = leg_low

    # =========================================================
    # 3. FVG Quality com filtro direcional (de v2)
    # =========================================================
    trend_sign = np.sign(trend_v3)
    fvg_qual = base["fvg_quality"].values.copy()
    fvg_dir_prop = base["fvg_direction"].values.copy()
    fvg_qual = np.where(fvg_dir_prop * trend_sign > 0, fvg_qual, 0.0)

    # =========================================================
    # 4. Setup Phase [0, 3] — substitui sweep_quality
    #    Sequencia causal: Sweep → CISD → FVG (displacement)
    #    Inspirado no modelo de entrada do VFI (Pine)
    # =========================================================
    SETUP_TIMEOUT = 30          # barras — ~7.5h no M15
    FVG_DISP_MIN_QUAL = 0.3    # qualidade minima do FVG para contar como displacement

    setup_phase = np.zeros(length, dtype=np.float64)

    bull_sweep_bar = -999
    bear_sweep_bar = -999
    bull_cisd_bar = -999
    bear_cisd_bar = -999
    bull_fvg_bar = -999
    bear_fvg_bar = -999

    for i in range(length):
        # --- Track sweep events ---
        if is_sweep_arr[i] and sweep_dir_arr[i] == 1:
            bull_sweep_bar = i
            bull_cisd_bar = -999
            bull_fvg_bar = -999
        if is_sweep_arr[i] and sweep_dir_arr[i] == -1:
            bear_sweep_bar = i
            bear_cisd_bar = -999
            bear_fvg_bar = -999

        # --- Track CISD after sweep ---
        if is_cisd_arr[i] and cisd_dir_arr[i] == 1:
            if bull_sweep_bar >= 0 and (i - bull_sweep_bar) <= SETUP_TIMEOUT:
                if bull_cisd_bar < bull_sweep_bar:
                    bull_cisd_bar = i
        if is_cisd_arr[i] and cisd_dir_arr[i] == -1:
            if bear_sweep_bar >= 0 and (i - bear_sweep_bar) <= SETUP_TIMEOUT:
                if bear_cisd_bar < bear_sweep_bar:
                    bear_cisd_bar = i

        # --- Track FVG (displacement) after CISD ---
        if is_fvg_arr[i] and fvg_dir_event[i] == 1:
            if (bull_cisd_bar > bull_sweep_bar
                    and (i - bull_cisd_bar) <= SETUP_TIMEOUT
                    and bull_fvg_bar < bull_cisd_bar
                    and fvg_qual_raw[i] >= FVG_DISP_MIN_QUAL):
                bull_fvg_bar = i
        if is_fvg_arr[i] and fvg_dir_event[i] == -1:
            if (bear_cisd_bar > bear_sweep_bar
                    and (i - bear_cisd_bar) <= SETUP_TIMEOUT
                    and bear_fvg_bar < bear_cisd_bar
                    and fvg_qual_raw[i] >= FVG_DISP_MIN_QUAL):
                bear_fvg_bar = i

        # --- Computa phase bull com decay intra-phase ---
        bull_phase = 0.0
        if bull_sweep_bar >= 0 and (i - bull_sweep_bar) <= SETUP_TIMEOUT:
            d_s = (i - bull_sweep_bar) / SETUP_TIMEOUT
            bull_phase = max(0.0, 1.0 - d_s * 0.9)
            if bull_cisd_bar > bull_sweep_bar and (i - bull_cisd_bar) <= SETUP_TIMEOUT:
                d_c = (i - bull_cisd_bar) / SETUP_TIMEOUT
                bull_phase = max(bull_phase, 1.0 + max(0.0, 1.0 - d_c * 0.9))
                if bull_fvg_bar > bull_cisd_bar and (i - bull_fvg_bar) <= SETUP_TIMEOUT:
                    d_f = (i - bull_fvg_bar) / SETUP_TIMEOUT
                    bull_phase = max(bull_phase, 2.0 + max(0.0, 1.0 - d_f * 0.9))

        # --- Computa phase bear com decay intra-phase ---
        bear_phase = 0.0
        if bear_sweep_bar >= 0 and (i - bear_sweep_bar) <= SETUP_TIMEOUT:
            d_s = (i - bear_sweep_bar) / SETUP_TIMEOUT
            bear_phase = max(0.0, 1.0 - d_s * 0.9)
            if bear_cisd_bar > bear_sweep_bar and (i - bear_cisd_bar) <= SETUP_TIMEOUT:
                d_c = (i - bear_cisd_bar) / SETUP_TIMEOUT
                bear_phase = max(bear_phase, 1.0 + max(0.0, 1.0 - d_c * 0.9))
                if bear_fvg_bar > bear_cisd_bar and (i - bear_fvg_bar) <= SETUP_TIMEOUT:
                    d_f = (i - bear_fvg_bar) / SETUP_TIMEOUT
                    bear_phase = max(bear_phase, 2.0 + max(0.0, 1.0 - d_f * 0.9))

        # Setup ativo e o alinhado com a tendencia corrente
        if trend_v3[i] > 0:
            setup_phase[i] = bull_phase
        elif trend_v3[i] < 0:
            setup_phase[i] = bear_phase
        else:
            setup_phase[i] = max(bull_phase, bear_phase)

    # =========================================================
    # Monta DataFrame de features
    # =========================================================
    features = pd.DataFrame({
        "trend_strength": np.clip(trend_v3, -100.0, 100.0),
        "price_zone": np.clip(price_zone, 0.0, 1.0),
        "fvg_quality": np.clip(fvg_qual, 0.0, 4.0),
        "sweep_quality": np.clip(setup_phase, 0.0, 3.0),
    }, index=base.index).fillna(0.0)

    return base, features
