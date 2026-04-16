# -*- coding: utf-8 -*-
"""
Smart Money Concepts — Indicadores Vetorizados para ANFIS.

Transforma um DataFrame OHLCV bruto nos 4 inputs que o sistema ANFIS
espera (trend_strength, price_zone, fvg_quality, sweep_quality).

Implementação 100% Pandas + NumPy, sem bibliotecas externas de SMC.
Cada decisão computacional está documentada para fins acadêmicos (TCC).

Decisões computacionais
1. Swings por janela rolante (default n=5 candles de cada lado).
2. BOS = close cruza o último swing na direção da tendência.
3. CISD = close ultrapassa a região open-close do candle pivot.
4. Perna de referência = último swing low → swing high (ou vice-versa).
5. Price Zone = posição relativa do close dentro da perna [0, 1].
6. FVG = gap entre high[i-2] e low[i], normalizado por ATR(14).
7. Sweep = pavio cruza swing anterior + rejeição com fechamento dentro.
8. Trend Strength = BOS consecutivos * 25 + bônus CISD * 15, em [-100, 100].
9. NaN preenchido com 0.0 — ANFIS nunca recebe valores nulos.
10. Vetorizado ao máximo; laço explícito apenas na propagação de estado.

Autor: Felipe Vilela — TCC Eng. Computacional
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Constantes default
# ============================================================================
DEFAULT_SWING_WINDOW: int = 5
DEFAULT_ATR_PERIOD: int = 14
DEFAULT_FVG_MIN_ATR: float = 0.1   # FVGs menores que 10% do ATR são descartados
DEFAULT_FVG_MAX_QUALITY: float = 4.0
DEFAULT_SWEEP_MAX_QUALITY: float = 3.0
DEFAULT_TREND_MAX: float = 100.0
DEFAULT_TREND_DECAY: float = 0.92
DEFAULT_FVG_TREND_WEIGHT: float = 5.0
DEFAULT_SWEEP_TREND_WEIGHT: float = 7.0
DEFAULT_EVENT_ALIGNMENT_BONUS: float = 6.0
DEFAULT_CONTEXT_MATCH_BOOST: float = 0.35
DEFAULT_CONTEXT_MISMATCH_PENALTY: float = 0.45


class SMCIndicators:
    """
    Extrator de indicadores Smart Money Concepts a partir de OHLCV.

    Recebe um DataFrame com colunas (open, high, low, close, volume) e
    calcula todas as features necessárias para alimentar o ANFIS.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com colunas OHLCV. Case-insensitive.
    swing_window : int
        Número de candles de cada lado para detecção de swings.
    atr_period : int
        Período do ATR para normalização de FVG e sweep.

    Examples
    --------
    >>> smc = SMCIndicators(df_ohlcv)
    >>> result = smc.compute_all()
    >>> anfis_inputs = smc.get_anfis_inputs()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        swing_window: int = DEFAULT_SWING_WINDOW,
        atr_period: int = DEFAULT_ATR_PERIOD,
    ) -> None:
        self.swing_window = swing_window
        self.atr_period = atr_period

        # Normaliza nomes de colunas para minúsculo
        self.df = df.copy()
        self.df.columns = [c.lower().strip() for c in self.df.columns]
        self._validate()

        # Pré-calcula o ATR (usado por FVG e Sweep)
        self._compute_atr()

        logger.info(
            f"SMCIndicators inicializado: {len(self.df)} barras, "
            f"swing_window={swing_window}, atr_period={atr_period}"
        )

    # ========================================================================
    # Validação
    # ========================================================================
    def _validate(self) -> None:
        """Verifica presença das colunas obrigatórias."""
        required = ['open', 'high', 'low', 'close']
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Colunas faltando no DataFrame: {missing}")

    # ========================================================================
    # ATR (Average True Range)
    # ========================================================================
    def _compute_atr(self) -> None:
        """
        Calcula o Average True Range (Wilder, 1978).

        O ATR serve como régua de volatilidade para normalizar FVG e Sweep.
        Usa exponential moving average (EMA) como no padrão original.
        """
        h = self.df['high'].values
        l = self.df['low'].values
        c = self.df['close'].values

        # True Range: max(H-L, |H-C_prev|, |L-C_prev|)
        c_prev = np.roll(c, 1)
        c_prev[0] = c[0]

        tr = np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))

        # EMA do True Range
        atr = np.empty_like(tr)
        atr[:self.atr_period] = np.nan
        atr[self.atr_period - 1] = np.mean(tr[:self.atr_period])
        alpha = 1.0 / self.atr_period
        for i in range(self.atr_period, len(tr)):
            atr[i] = atr[i - 1] * (1 - alpha) + tr[i] * alpha

        # Preenche NaN iniciais com o primeiro valor válido
        first_valid = atr[self.atr_period - 1]
        atr[:self.atr_period - 1] = first_valid

        self.df['atr'] = atr

    # ========================================================================
    # 1. Detecção de Swings (Topos e Fundos Locais)
    # ========================================================================
    def detect_swings(self) -> pd.DataFrame:
        """
        Detecta Swing Highs e Swing Lows por janela rolante.

        Um Swing High no índice i exige que high[i] seja estritamente
        maior que high[j] para todo j em [i-n, i+n], j != i.
        Análogo para Swing Low com low[i] estritamente menor.

        Colunas criadas:
            is_swing_high : bool
            is_swing_low : bool
            swing_high_price : float (preço do topo, NaN onde não é swing)
            swing_low_price : float (preço do fundo, NaN onde não é swing)
        """
        n = self.swing_window
        highs = self.df['high'].values
        lows = self.df['low'].values
        length = len(highs)

        is_sh_pivot = np.zeros(length, dtype=bool)
        is_sl_pivot = np.zeros(length, dtype=bool)
        is_sh_confirmed = np.zeros(length, dtype=bool)
        is_sl_confirmed = np.zeros(length, dtype=bool)
        sh_confirmed_price = np.full(length, np.nan)
        sl_confirmed_price = np.full(length, np.nan)
        sh_confirmed_pivot_idx = np.full(length, -1, dtype=np.int32)
        sl_confirmed_pivot_idx = np.full(length, -1, dtype=np.int32)

        for i in range(n, length - n):
            window_h = highs[i - n: i + n + 1]
            window_l = lows[i - n: i + n + 1]

            # Swing High: high[i] é o máximo estrito da janela
            if highs[i] == np.max(window_h) and np.sum(window_h == highs[i]) == 1:
                is_sh_pivot[i] = True
                confirm_idx = i + n
                if confirm_idx < length:
                    is_sh_confirmed[confirm_idx] = True
                    sh_confirmed_price[confirm_idx] = highs[i]
                    sh_confirmed_pivot_idx[confirm_idx] = i

            # Swing Low: low[i] é o mínimo estrito da janela
            if lows[i] == np.min(window_l) and np.sum(window_l == lows[i]) == 1:
                is_sl_pivot[i] = True
                confirm_idx = i + n
                if confirm_idx < length:
                    is_sl_confirmed[confirm_idx] = True
                    sl_confirmed_price[confirm_idx] = lows[i]
                    sl_confirmed_pivot_idx[confirm_idx] = i

        self.df['is_swing_high_pivot'] = is_sh_pivot
        self.df['is_swing_low_pivot'] = is_sl_pivot
        self.df['pivot_swing_high_price'] = np.where(is_sh_pivot, highs, np.nan)
        self.df['pivot_swing_low_price'] = np.where(is_sl_pivot, lows, np.nan)

        # As colunas sem sufixo passam a representar swings causalmente disponíveis.
        self.df['is_swing_high'] = is_sh_confirmed
        self.df['is_swing_low'] = is_sl_confirmed
        self.df['swing_high_price'] = sh_confirmed_price
        self.df['swing_low_price'] = sl_confirmed_price
        self.df['swing_high_pivot_idx'] = sh_confirmed_pivot_idx
        self.df['swing_low_pivot_idx'] = sl_confirmed_pivot_idx

        n_sh = int(np.sum(is_sh_pivot))
        n_sl = int(np.sum(is_sl_pivot))
        logger.info(
            "Swings detectados: %d Swing Highs, %d Swing Lows (confirmacao causal=%d barras)",
            n_sh,
            n_sl,
            n,
        )

        return self.df

    # ========================================================================
    # 2. BOS (Break of Structure)
    # ========================================================================
    def detect_bos(self) -> pd.DataFrame:
        """
        Detecta Break of Structure (BOS).

        Bullish BOS: close[i] > último swing_high confirmado.
        Bearish BOS: close[i] < último swing_low confirmado.

        A cada BOS, o swing de referência é atualizado para evitar
        contagem repetida no mesmo nível.

        Colunas criadas:
            is_bos : bool
            bos_direction : int (+1 bullish, -1 bearish, 0 nenhum)
        """
        if 'is_swing_high' not in self.df.columns:
            self.detect_swings()

        close = self.df['close'].values
        sh_prices = self.df['swing_high_price'].values
        sl_prices = self.df['swing_low_price'].values
        length = len(close)

        is_bos = np.zeros(length, dtype=bool)
        bos_dir = np.zeros(length, dtype=np.int8)

        # Estado: último swing de referência
        last_sh = np.nan  # último swing high confirmado
        last_sl = np.nan  # último swing low confirmado

        for i in range(length):
            # Atualiza referências de swing (só swings ANTES do candle atual)
            if i > 0:
                if not np.isnan(sh_prices[i - 1]):
                    last_sh = sh_prices[i - 1]
                if not np.isnan(sl_prices[i - 1]):
                    last_sl = sl_prices[i - 1]

            # Bullish BOS: close rompe o último swing high
            if not np.isnan(last_sh) and close[i] > last_sh:
                is_bos[i] = True
                bos_dir[i] = 1
                last_sh = np.nan  # consumido — aguarda próximo swing high

            # Bearish BOS: close rompe o último swing low
            elif not np.isnan(last_sl) and close[i] < last_sl:
                is_bos[i] = True
                bos_dir[i] = -1
                last_sl = np.nan  # consumido — aguarda próximo swing low

        self.df['is_bos'] = is_bos
        self.df['bos_direction'] = bos_dir

        n_bull = int(np.sum(bos_dir == 1))
        n_bear = int(np.sum(bos_dir == -1))
        logger.info(f"BOS detectados: {n_bull} bullish, {n_bear} bearish")

        return self.df

    # ========================================================================
    # 3. CISD (Change in State of Delivery)
    # ========================================================================
    def detect_cisd(self) -> pd.DataFrame:
        """
        Detecta Change in State of Delivery (CISD).

        Identifica a região de delivery (faixa open-close) do candle que
        formou o último Swing pivot. Um CISD ocorre quando o close de um
        candle subsequente ultrapassa completamente essa região.

        Bullish CISD: close[i] > max(open, close) do último swing low pivot.
        Bearish CISD: close[i] < min(open, close) do último swing high pivot.

        Colunas criadas:
            is_cisd : bool
            cisd_direction : int (+1 bullish, -1 bearish, 0 nenhum)
        """
        if 'is_swing_high' not in self.df.columns:
            self.detect_swings()

        close = self.df['close'].values
        opens = self.df['open'].values
        sl_pivot_idx = self.df['swing_low_pivot_idx'].values
        sh_pivot_idx = self.df['swing_high_pivot_idx'].values
        length = len(close)

        is_cisd = np.zeros(length, dtype=bool)
        cisd_dir = np.zeros(length, dtype=np.int8)

        # Estado: região de delivery do último pivot
        # Para bullish CISD: região do swing low pivot
        sl_delivery_top = np.nan  # max(open, close) do candle do swing low
        # Para bearish CISD: região do swing high pivot
        sh_delivery_bot = np.nan  # min(open, close) do candle do swing high

        # Flags para evitar CISD repetido no mesmo pivot
        sl_cisd_consumed = True
        sh_cisd_consumed = True

        for i in range(length):
            # Atualiza delivery zones quando novos swings são confirmados
            if sl_pivot_idx[i] >= 0:
                pivot_idx = sl_pivot_idx[i]
                sl_delivery_top = max(opens[pivot_idx], close[pivot_idx])
                sl_cisd_consumed = False

            if sh_pivot_idx[i] >= 0:
                pivot_idx = sh_pivot_idx[i]
                sh_delivery_bot = min(opens[pivot_idx], close[pivot_idx])
                sh_cisd_consumed = False

            # Bullish CISD: close supera a região de delivery do fundo
            if (not sl_cisd_consumed
                    and not np.isnan(sl_delivery_top)
                    and close[i] > sl_delivery_top):
                is_cisd[i] = True
                cisd_dir[i] = 1
                sl_cisd_consumed = True

            # Bearish CISD: close cai abaixo da região de delivery do topo
            if (not sh_cisd_consumed
                    and not np.isnan(sh_delivery_bot)
                    and close[i] < sh_delivery_bot):
                is_cisd[i] = True
                cisd_dir[i] = -1
                sh_cisd_consumed = True

        self.df['is_cisd'] = is_cisd
        self.df['cisd_direction'] = cisd_dir

        n_bull = int(np.sum(cisd_dir == 1))
        n_bear = int(np.sum(cisd_dir == -1))
        logger.info(f"CISD detectados: {n_bull} bullish, {n_bear} bearish")

        return self.df

    # ========================================================================
    # 4. FVG (Fair Value Gap)
    # ========================================================================
    def detect_fvg(self) -> pd.DataFrame:
        """
        Detecta Fair Value Gaps e calcula qualidade normalizada por ATR.

        Bullish FVG: low[i] > high[i-2]  (gap de alta — candle 2 saltou candle 0)
        Bearish FVG: high[i] < low[i-2]  (gap de baixa)

        A qualidade é a largura do gap dividida pelo ATR(14), clippada em
        [0, fvg_max_quality]. FVGs menores que fvg_min_atr * ATR são descartados.

        Colunas criadas:
            is_fvg : bool
            fvg_direction : int (+1 bullish, -1 bearish, 0 nenhum)
            fvg_quality : float [0, 4]
        """
        h = self.df['high'].values
        l = self.df['low'].values
        atr = self.df['atr'].values
        length = len(h)

        is_fvg = np.zeros(length, dtype=bool)
        fvg_dir = np.zeros(length, dtype=np.int8)
        fvg_qual = np.zeros(length, dtype=np.float64)

        for i in range(2, length):
            current_atr = atr[i]
            if current_atr <= 0:
                continue

            # Bullish FVG: gap entre high do candle i-2 e low do candle i
            bull_gap = l[i] - h[i - 2]
            if bull_gap > 0:
                ratio = bull_gap / current_atr
                if ratio >= DEFAULT_FVG_MIN_ATR:
                    is_fvg[i] = True
                    fvg_dir[i] = 1
                    fvg_qual[i] = min(ratio, DEFAULT_FVG_MAX_QUALITY)
                continue  # se detectou bullish, não testa bearish no mesmo candle

            # Bearish FVG: gap entre low do candle i-2 e high do candle i
            bear_gap = l[i - 2] - h[i]
            if bear_gap > 0:
                ratio = bear_gap / current_atr
                if ratio >= DEFAULT_FVG_MIN_ATR:
                    is_fvg[i] = True
                    fvg_dir[i] = -1
                    fvg_qual[i] = min(ratio, DEFAULT_FVG_MAX_QUALITY)

        self.df['is_fvg'] = is_fvg
        self.df['fvg_direction'] = fvg_dir
        self.df['fvg_quality'] = fvg_qual
        self.df['fvg_quality_raw'] = fvg_qual.copy()

        n_bull = int(np.sum(fvg_dir == 1))
        n_bear = int(np.sum(fvg_dir == -1))
        logger.info(f"FVGs detectados: {n_bull} bullish, {n_bear} bearish")

        return self.df

    # ========================================================================
    # 5. Sweep de Liquidez (V-Shape Rejection)
    # ========================================================================
    def detect_sweeps(self) -> pd.DataFrame:
        """
        Detecta sweeps de liquidez com rejeição (V-shape).

        Bullish Sweep (buy-side liquidity grab):
            - low[i] < último swing_low  (rompeu com pavio)
            - close[i] > último swing_low (fechou ACIMA — rejeição)
            - Qualidade = wick_ratio * rejection_depth / ATR

        Bearish Sweep (sell-side liquidity grab):
            - high[i] > último swing_high (rompeu com pavio)
            - close[i] < último swing_high (fechou ABAIXO — rejeição)

        Colunas criadas:
            is_sweep : bool
            sweep_direction : int (+1 bullish, -1 bearish, 0 nenhum)
            sweep_quality : float [0, 3]
        """
        if 'is_swing_high' not in self.df.columns:
            self.detect_swings()

        h = self.df['high'].values
        l = self.df['low'].values
        o = self.df['open'].values
        c = self.df['close'].values
        atr = self.df['atr'].values
        sh_prices = self.df['swing_high_price'].values
        sl_prices = self.df['swing_low_price'].values
        length = len(c)

        is_sweep = np.zeros(length, dtype=bool)
        sweep_dir = np.zeros(length, dtype=np.int8)
        sweep_qual = np.zeros(length, dtype=np.float64)

        # Estado: últimos swings de referência para sweep
        last_sl = np.nan
        last_sh = np.nan

        for i in range(1, length):
            # Atualiza referências de swing (swings anteriores ao candle atual)
            if not np.isnan(sl_prices[i - 1]):
                last_sl = sl_prices[i - 1]
            if not np.isnan(sh_prices[i - 1]):
                last_sh = sh_prices[i - 1]

            current_atr = atr[i]
            if current_atr <= 0:
                continue

            body = abs(c[i] - o[i])
            if body < 1e-10:
                body = 1e-10  # evita divisão por zero

            # ----- Bullish Sweep (varredura de fundo) -----
            # Pavio rompe abaixo do swing low, mas close fecha acima
            if not np.isnan(last_sl) and l[i] < last_sl and c[i] > last_sl:
                lower_wick = min(o[i], c[i]) - l[i]
                wick_ratio = lower_wick / body
                rejection_depth = abs(last_sl - l[i]) / current_atr

                quality = min(wick_ratio * rejection_depth, DEFAULT_SWEEP_MAX_QUALITY)
                if quality > 0.1:  # threshold mínimo de qualidade
                    is_sweep[i] = True
                    sweep_dir[i] = 1
                    sweep_qual[i] = quality
                    last_sl = np.nan  # consumido

            # ----- Bearish Sweep (varredura de topo) -----
            # Pavio rompe acima do swing high, mas close fecha abaixo
            elif not np.isnan(last_sh) and h[i] > last_sh and c[i] < last_sh:
                upper_wick = h[i] - max(o[i], c[i])
                wick_ratio = upper_wick / body
                rejection_depth = abs(h[i] - last_sh) / current_atr

                quality = min(wick_ratio * rejection_depth, DEFAULT_SWEEP_MAX_QUALITY)
                if quality > 0.1:
                    is_sweep[i] = True
                    sweep_dir[i] = -1
                    sweep_qual[i] = quality
                    last_sh = np.nan  # consumido

        self.df['is_sweep'] = is_sweep
        self.df['sweep_direction'] = sweep_dir
        self.df['sweep_quality'] = sweep_qual
        self.df['sweep_quality_raw'] = sweep_qual.copy()

        n_bull = int(np.sum(sweep_dir == 1))
        n_bear = int(np.sum(sweep_dir == -1))
        logger.info(f"Sweeps detectados: {n_bull} bullish, {n_bear} bearish")

        return self.df

    # ========================================================================
    # 6. Price Zone (Premium / Discount)
    # ========================================================================
    def compute_price_zone(self) -> pd.DataFrame:
        """
        Calcula a posição relativa do close dentro da perna de referência.

        A perna de referência (dealing range) é definida pelo par
        {último swing low, último swing high} mais recente. A cada novo
        swing, a perna é atualizada.

        price_zone = (close - leg_low) / (leg_high - leg_low)

        Valores:
            0.0 = fundo da perna (deep discount)
            0.5 = equilibrium
            1.0 = topo da perna (deep premium)

        Clipado em [0, 1].

        Colunas criadas:
            price_zone : float [0, 1]
            leg_high : float (topo da perna de referência)
            leg_low : float (fundo da perna de referência)
        """
        if 'is_swing_high' not in self.df.columns:
            self.detect_swings()

        close = self.df['close'].values
        sh_prices = self.df['swing_high_price'].values
        sl_prices = self.df['swing_low_price'].values
        length = len(close)

        leg_high = np.full(length, np.nan)
        leg_low = np.full(length, np.nan)
        price_zone = np.full(length, 0.5)  # default = equilibrium

        # Estado: perna de referência atual
        current_leg_h = np.nan
        current_leg_l = np.nan

        for i in range(length):
            # Atualiza a perna quando novos swings são encontrados
            if not np.isnan(sh_prices[i]):
                current_leg_h = sh_prices[i]
            if not np.isnan(sl_prices[i]):
                current_leg_l = sl_prices[i]

            leg_high[i] = current_leg_h
            leg_low[i] = current_leg_l

            # Calcula a zona somente se ambos extremos estão definidos
            if not np.isnan(current_leg_h) and not np.isnan(current_leg_l):
                span = current_leg_h - current_leg_l
                if span > 0:
                    pz = (close[i] - current_leg_l) / span
                    price_zone[i] = np.clip(pz, 0.0, 1.0)

        self.df['price_zone'] = price_zone
        self.df['leg_high'] = leg_high
        self.df['leg_low'] = leg_low

        logger.info(
            f"Price Zone: mean={np.nanmean(price_zone):.3f}, "
            f"std={np.nanstd(price_zone):.3f}"
        )

        return self.df

    # ========================================================================
    # 7. Trend Strength
    # ========================================================================
    def compute_trend_strength(self) -> pd.DataFrame:
        """
        Calcula a força da tendência estrutural combinando BOS e CISD.

        Lógica:
            - Cada BOS na mesma direção soma 25 pontos.
            - Cada CISD na mesma direção soma 15 pontos (bônus).
            - BOS ou CISD na direção oposta RESETA o contador.
            - Resultado clippado em [-100, +100].

        Sinal positivo = tendência de alta, negativo = tendência de baixa.

        Coluna criada:
            trend_strength : float [-100, +100]
        """
        if 'bos_direction' not in self.df.columns:
            self.detect_bos()
        if 'cisd_direction' not in self.df.columns:
            self.detect_cisd()

        bos_dir = self.df['bos_direction'].values
        cisd_dir = self.df['cisd_direction'].values
        fvg_dir = self.df['fvg_direction'].values if 'fvg_direction' in self.df.columns else np.zeros(len(self.df))
        sweep_dir = self.df['sweep_direction'].values if 'sweep_direction' in self.df.columns else np.zeros(len(self.df))
        fvg_quality = self.df['fvg_quality_raw'].values if 'fvg_quality_raw' in self.df.columns else self.df['fvg_quality'].values
        sweep_quality = self.df['sweep_quality_raw'].values if 'sweep_quality_raw' in self.df.columns else self.df['sweep_quality'].values
        length = len(bos_dir)

        trend = np.zeros(length, dtype=np.float64)

        # Estado: acumulador de tendência
        current_trend = 0.0
        current_direction = 0  # +1 BULLISH ou -1 BEARISH

        for i in range(length):
            current_trend *= DEFAULT_TREND_DECAY

            # BOS detectado nesta barra
            if bos_dir[i] != 0:
                if bos_dir[i] == current_direction:
                    # Continuidade: acumula +25 na mesma direção
                    current_trend += bos_dir[i] * 25.0
                else:
                    # Reversão: reseta para a nova direção
                    current_direction = int(bos_dir[i])
                    current_trend = (current_trend * 0.35) + (bos_dir[i] * 25.0)

            # CISD detectado nesta barra (bônus se mesma direção)
            if cisd_dir[i] != 0:
                if cisd_dir[i] == current_direction:
                    current_trend += cisd_dir[i] * 15.0
                else:
                    # CISD contrário indica possível reversão
                    current_direction = int(cisd_dir[i])
                    current_trend = (current_trend * 0.35) + (cisd_dir[i] * 15.0)

            if bos_dir[i] != 0 and bos_dir[i] == cisd_dir[i]:
                current_trend += bos_dir[i] * 10.0

            event_impulse = 0.0
            if fvg_dir[i] != 0 and fvg_quality[i] > 0:
                event_impulse += fvg_dir[i] * min(float(fvg_quality[i]), 2.0) * DEFAULT_FVG_TREND_WEIGHT
            if sweep_dir[i] != 0 and sweep_quality[i] > 0:
                event_impulse += sweep_dir[i] * min(float(sweep_quality[i]), 2.0) * DEFAULT_SWEEP_TREND_WEIGHT

            if event_impulse != 0.0:
                event_direction = int(np.sign(event_impulse))
                if current_direction == 0 or event_direction == current_direction:
                    current_trend += event_impulse
                else:
                    current_trend = (current_trend * 0.50) + event_impulse

                if abs(current_trend) > 1e-6:
                    current_direction = int(np.sign(current_trend))
                else:
                    current_direction = event_direction

            if fvg_dir[i] != 0 and fvg_dir[i] == sweep_dir[i]:
                current_trend += fvg_dir[i] * DEFAULT_EVENT_ALIGNMENT_BONUS

            # Clipa e registra
            current_trend = np.clip(current_trend, -DEFAULT_TREND_MAX, DEFAULT_TREND_MAX)
            trend[i] = current_trend

        self.df['trend_strength'] = trend

        logger.info(
            f"Trend Strength: mean={np.mean(trend):.1f}, "
            f"max={np.max(trend):.1f}, min={np.min(trend):.1f}"
        )

        return self.df

    def _compute_context_bias(self) -> None:
        """
        Combina estrutura e localização do preço em um viés contextual.

        O objetivo não é criar um novo sinal final, e sim ajudar a traduzir
        FVG e sweep para uma intensidade mais coerente com o contexto.
        """
        trend = self.df['trend_strength'].values.astype(np.float64)
        zone = self.df['price_zone'].values.astype(np.float64)

        trend_component = np.tanh(trend / 35.0)
        zone_component = np.where(
            zone <= 0.50,
            np.clip((0.50 - zone) / 0.50, 0.0, 1.0),
            -np.clip((zone - 0.50) / 0.50, 0.0, 1.0),
        )

        smc_bias = np.clip((0.70 * trend_component) + (0.30 * zone_component), -1.0, 1.0)
        zone_fit = np.where(
            smc_bias >= 0,
            np.clip((0.65 - zone) / 0.65, 0.0, 1.0),
            np.clip((zone - 0.35) / 0.65, 0.0, 1.0),
        )

        self.df['smc_bias'] = smc_bias
        self.df['smc_zone_fit'] = zone_fit

    def _contextualize_event_qualities(self) -> None:
        """
        Ajusta FVG e sweep de acordo com alinhamento estrutural.

        Se a direção do evento combina com o viés estrutural e com a posição
        do preço na perna atual, a qualidade é levemente reforçada. Quando o
        evento entra em conflito com o contexto, sua qualidade é atenuada.
        """
        self._compute_context_bias()

        bias = self.df['smc_bias'].values.astype(np.float64)
        zone_fit = self.df['smc_zone_fit'].values.astype(np.float64)

        def apply_context(raw_quality: np.ndarray, direction: np.ndarray, max_quality: float) -> np.ndarray:
            adjusted = raw_quality.astype(np.float64).copy()
            for i in range(len(adjusted)):
                if adjusted[i] <= 0 or direction[i] == 0:
                    continue

                bias_strength = abs(bias[i])
                directional_bias = np.sign(bias[i])

                if directional_bias == 0:
                    alignment_factor = 1.0
                elif directional_bias == direction[i]:
                    alignment_factor = 1.0 + (DEFAULT_CONTEXT_MATCH_BOOST * bias_strength)
                else:
                    alignment_factor = max(
                        0.30,
                        1.0 - (DEFAULT_CONTEXT_MISMATCH_PENALTY * bias_strength),
                    )

                zone_factor = 0.80 + (0.40 * zone_fit[i])
                adjusted[i] = np.clip(adjusted[i] * alignment_factor * zone_factor, 0.0, max_quality)

            return adjusted

        if 'fvg_quality_raw' not in self.df.columns:
            self.df['fvg_quality_raw'] = self.df['fvg_quality'].values.copy()
        if 'sweep_quality_raw' not in self.df.columns:
            self.df['sweep_quality_raw'] = self.df['sweep_quality'].values.copy()

        self.df['fvg_quality_adj'] = apply_context(
            self.df['fvg_quality_raw'].values,
            self.df['fvg_direction'].values,
            DEFAULT_FVG_MAX_QUALITY,
        )
        self.df['sweep_quality_adj'] = apply_context(
            self.df['sweep_quality_raw'].values,
            self.df['sweep_direction'].values,
            DEFAULT_SWEEP_MAX_QUALITY,
        )

    # ========================================================================
    # 8. Propagação da última qualidade de FVG e Sweep
    # ========================================================================
    def _propagate_last_values(self) -> None:
        """
        Propaga o último valor de fvg_quality e sweep_quality para
        barras subsequentes (forward fill limitado).

        O ANFIS precisa de um valor em cada barra, não apenas nos candles
        onde o evento ocorreu. Usamos forward fill com decaimento:
        a qualidade diminui linearmente a cada barra após o evento,
        simulando a perda de relevância temporal do padrão.

        Decaimento: -0.05 por barra (em ~20 barras o sinal desaparece
        se não for renovado por novo FVG/sweep).
        """
        decay_per_bar = 0.05

        # --- FVG quality propagation ---
        fvg_q = self.df['fvg_quality'].values.copy()
        fvg_dir = self.df['fvg_direction'].values.copy()
        last_fvg = 0.0
        last_fvg_dir = 0
        for i in range(len(fvg_q)):
            if fvg_q[i] > 0:
                last_fvg = fvg_q[i]
                last_fvg_dir = fvg_dir[i]
            else:
                last_fvg = max(0.0, last_fvg - decay_per_bar)
                fvg_q[i] = last_fvg
                fvg_dir[i] = last_fvg_dir if last_fvg > 0 else 0
        self.df['fvg_quality'] = fvg_q
        self.df['fvg_direction'] = fvg_dir

        # --- Sweep quality propagation ---
        sw_q = self.df['sweep_quality'].values.copy()
        sw_dir = self.df['sweep_direction'].values.copy()
        last_sw = 0.0
        last_sw_dir = 0
        for i in range(len(sw_q)):
            if sw_q[i] > 0:
                last_sw = sw_q[i]
                last_sw_dir = sw_dir[i]
            else:
                last_sw = max(0.0, last_sw - decay_per_bar)
                sw_q[i] = last_sw
                sw_dir[i] = last_sw_dir if last_sw > 0 else 0
        self.df['sweep_quality'] = sw_q
        self.df['sweep_direction'] = sw_dir

    # ========================================================================
    # Pipeline Completo
    # ========================================================================
    def compute_all(self) -> pd.DataFrame:
        """
        Executa todo o pipeline de detecção SMC na ordem correta.

        Ordem de dependência:
            1. Swings (sem dependência)
            2. BOS (depende de Swings)
            3. CISD (depende de Swings)
            4. FVG (sem dependência de Swings, usa ATR)
            5. Sweeps (depende de Swings)
            6. Price Zone (depende de Swings)
            7. Trend Strength (depende de BOS + CISD)
            8. Propagação de valores (depende de FVG + Sweeps)

        Returns
        -------
        pd.DataFrame
            DataFrame original enriquecido com todas as colunas SMC.
        """
        logger.info("Executando pipeline SMC completo...")

        self.detect_swings()
        self.detect_bos()
        self.detect_cisd()
        self.detect_fvg()
        self.detect_sweeps()
        self.compute_price_zone()
        self.compute_trend_strength()
        self._propagate_last_values()
        self.df['fvg_quality_raw'] = self.df['fvg_quality'].values.copy()
        self.df['sweep_quality_raw'] = self.df['sweep_quality'].values.copy()
        self._contextualize_event_qualities()

        # Garante que nenhum NaN chegue ao ANFIS
        anfis_cols = ['trend_strength', 'price_zone', 'fvg_quality', 'sweep_quality']
        for col in anfis_cols:
            self.df[col] = self.df[col].fillna(0.0)

        logger.info("Pipeline SMC completo.")
        return self.df

    # ========================================================================
    # Saída para ANFIS
    # ========================================================================
    def get_anfis_inputs(self, use_enhanced: bool = True) -> pd.DataFrame:
        """
        Retorna apenas as 4 colunas que o ANFIS espera, sem NaN.

        Universos esperados (conforme anfis/config.py):
            trend_strength : [-100, +100]
            price_zone     : [0, 1]
            fvg_quality    : [0, 4]
            sweep_quality  : [0, 3]

        Returns
        -------
        pd.DataFrame
            DataFrame com 4 colunas numéricas, sem NaN.
        """
        if 'trend_strength' not in self.df.columns:
            self.compute_all()

        cols = ['trend_strength', 'price_zone', 'fvg_quality', 'sweep_quality']
        result = self.df[cols].copy()

        if use_enhanced:
            if 'fvg_quality_adj' in self.df.columns:
                result['fvg_quality'] = self.df['fvg_quality_adj'].values
            if 'sweep_quality_adj' in self.df.columns:
                result['sweep_quality'] = self.df['sweep_quality_adj'].values

        # Clip final para garantir universos
        result['trend_strength'] = result['trend_strength'].clip(-100.0, 100.0)
        result['price_zone'] = result['price_zone'].clip(0.0, 1.0)
        result['fvg_quality'] = result['fvg_quality'].clip(0.0, 4.0)
        result['sweep_quality'] = result['sweep_quality'].clip(0.0, 3.0)

        # Zero NaN residual
        result = result.fillna(0.0)

        return result

    # ========================================================================
    # Utilidade: resumo estatístico
    # ========================================================================
    def summary(self) -> dict:
        """
        Retorna um dicionário com contagens e estatísticas dos indicadores.

        Útil para validação rápida e logging.
        """
        if 'trend_strength' not in self.df.columns:
            self.compute_all()

        df = self.df
        return {
            'total_bars': len(df),
            'swing_highs': int(df['is_swing_high'].sum()),
            'swing_lows': int(df['is_swing_low'].sum()),
            'bos_bullish': int((df['bos_direction'] == 1).sum()),
            'bos_bearish': int((df['bos_direction'] == -1).sum()),
            'cisd_bullish': int((df['cisd_direction'] == 1).sum()),
            'cisd_bearish': int((df['cisd_direction'] == -1).sum()),
            'fvg_bullish': int((df['is_fvg'] & (df['fvg_direction'] == 1)).sum()),
            'fvg_bearish': int((df['is_fvg'] & (df['fvg_direction'] == -1)).sum()),
            'sweep_bullish': int((df['is_sweep'] & (df['sweep_direction'] == 1)).sum()),
            'sweep_bearish': int((df['is_sweep'] & (df['sweep_direction'] == -1)).sum()),
            'trend_strength_mean': float(df['trend_strength'].mean()),
            'price_zone_mean': float(df['price_zone'].mean()),
            'fvg_quality_mean': float(df['fvg_quality'].mean()),
            'sweep_quality_mean': float(df['sweep_quality'].mean()),
        }
