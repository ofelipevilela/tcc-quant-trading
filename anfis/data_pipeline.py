# -*- coding: utf-8 -*-
"""
Pipeline de Dados para o ANFIS.

Duas fontes possíveis:
1. Dados sintéticos: gera amostras aleatórias e calcula o sinal "ideal"
   via sistema Mamdani existente + ruído gaussiano controlado.
2. Dados de mercado: carrega OHLCV via yfinance e calcula as 4 variáveis
   fuzzy a partir dos dados de preço.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import INPUT_VARS, TRAINING, UNIVERSES

logger = logging.getLogger(__name__)

# Adicionar o diretório raiz ao path para importar o sistema Mamdani
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _get_mamdani_system():
    """
    Importa e instancia o sistema Mamdani existente.

    Returns
    -------
    SMCFuzzySystem
        Instância do sistema fuzzy Mamdani.
    """
    from fuzzy.fuzzy_system import SMCFuzzySystem
    return SMCFuzzySystem()


def _gaussian_mf(x: np.ndarray, center: float, sigma: float) -> np.ndarray:
    """Função de pertinência gaussiana vetorizada."""
    return np.exp(-0.5 * ((x - center) / max(sigma, 1e-8)) ** 2)


def _compute_synthetic_signal(
    trend: np.ndarray,
    zone: np.ndarray,
    fvg: np.ndarray,
    sweep: np.ndarray,
) -> np.ndarray:
    """
    Calcula o sinal SMC analiticamente via regras vetorizadas em NumPy.

    Replica a semântica do sistema Mamdani original sem usar scikit-fuzzy,
    aplicando as mesmas regras fuzzy como funções contínuas:

    - Tendência alta + zona discount + FVG forte + sweep forte → compra forte
    - Tendência baixa + zona premium + FVG forte + sweep forte → venda forte
    - Combinações intermediárias → sinais proporcionais
    - Conflitos (alta+premium, baixa+discount) → neutro

    A saída é uma média ponderada dos consequentes das regras, onde o peso
    de cada regra é o produto (AND fuzzy = T-norm produto) das pertinências
    dos antecedentes. Isso é matematicamente equivalente à inferência TSK
    zero-order que o ANFIS aprenderá a otimizar.

    Parameters
    ----------
    trend, zone, fvg, sweep : np.ndarray
        Arrays de inputs, cada um com shape [n_samples].

    Returns
    -------
    np.ndarray
        Sinal em [-100, +100], shape [n_samples].
    """
    n = len(trend)

    # --- Pertinências das variáveis linguísticas ---
    # Trend_Strength: Alta (center=70, σ=25), Baixa (center=-70, σ=25), Neutra (center=0, σ=20)
    mu_trend_alta = _gaussian_mf(trend, 70.0, 25.0)
    mu_trend_baixa = _gaussian_mf(trend, -70.0, 25.0)
    mu_trend_neutra = _gaussian_mf(trend, 0.0, 20.0)

    # Price_Zone: Premium (center=0.85, σ=0.12), Discount (center=0.15, σ=0.12), Equilibrio (center=0.5, σ=0.12)
    mu_zone_premium = _gaussian_mf(zone, 0.85, 0.12)
    mu_zone_discount = _gaussian_mf(zone, 0.15, 0.12)
    mu_zone_equil = _gaussian_mf(zone, 0.5, 0.12)

    # FVG_Quality: Alto (center=3.0, σ=0.6), Medio (center=1.5, σ=0.5), Baixo (center=0.3, σ=0.3)
    mu_fvg_alto = _gaussian_mf(fvg, 3.0, 0.6)
    mu_fvg_medio = _gaussian_mf(fvg, 1.5, 0.5)
    mu_fvg_baixo = _gaussian_mf(fvg, 0.3, 0.3)

    # Sweep_Quality: Forte (center=2.2, σ=0.5), Moderado (center=1.2, σ=0.4), Fraco (center=0.3, σ=0.25)
    mu_sweep_forte = _gaussian_mf(sweep, 2.2, 0.5)
    mu_sweep_mod = _gaussian_mf(sweep, 1.2, 0.4)
    mu_sweep_fraco = _gaussian_mf(sweep, 0.3, 0.25)

    # --- Regras SMC (T-norm produto) ---
    # Cada regra: peso = produto das pertinências, consequente = valor crisp

    rules = []  # lista de (peso [n], consequente float)

    # COMPRA_FORTE: Alta + Discount + FVG Alto + Sweep Forte
    rules.append((mu_trend_alta * mu_zone_discount * mu_fvg_alto * mu_sweep_forte, 80.0))
    # COMPRA_FORTE: Alta + Discount + FVG Alto + Sweep Moderado
    rules.append((mu_trend_alta * mu_zone_discount * mu_fvg_alto * mu_sweep_mod, 70.0))
    # COMPRA: Alta + Discount + FVG Medio + Sweep Forte
    rules.append((mu_trend_alta * mu_zone_discount * mu_fvg_medio * mu_sweep_forte, 55.0))
    # COMPRA: Alta + Discount + FVG Medio + Sweep Moderado
    rules.append((mu_trend_alta * mu_zone_discount * mu_fvg_medio * mu_sweep_mod, 40.0))
    # COMPRA: Alta + Equilibrio + FVG Alto + Sweep Forte
    rules.append((mu_trend_alta * mu_zone_equil * mu_fvg_alto * mu_sweep_forte, 45.0))

    # VENDA_FORTE: Baixa + Premium + FVG Alto + Sweep Forte
    rules.append((mu_trend_baixa * mu_zone_premium * mu_fvg_alto * mu_sweep_forte, -80.0))
    # VENDA_FORTE: Baixa + Premium + FVG Alto + Sweep Moderado
    rules.append((mu_trend_baixa * mu_zone_premium * mu_fvg_alto * mu_sweep_mod, -70.0))
    # VENDA: Baixa + Premium + FVG Medio + Sweep Forte
    rules.append((mu_trend_baixa * mu_zone_premium * mu_fvg_medio * mu_sweep_forte, -55.0))
    # VENDA: Baixa + Premium + FVG Medio + Sweep Moderado
    rules.append((mu_trend_baixa * mu_zone_premium * mu_fvg_medio * mu_sweep_mod, -40.0))
    # VENDA: Baixa + Equilibrio + FVG Alto + Sweep Forte
    rules.append((mu_trend_baixa * mu_zone_equil * mu_fvg_alto * mu_sweep_forte, -45.0))

    # NEUTRO: Neutra + qualquer zona + FVG Baixo
    rules.append((mu_trend_neutra * mu_fvg_baixo, 0.0))
    # NEUTRO: qualquer trend + Equilibrio + FVG Baixo + Sweep Fraco
    rules.append((mu_zone_equil * mu_fvg_baixo * mu_sweep_fraco, 0.0))

    # COMPRA fraca: Alta + Discount + FVG Baixo (trend compensa FVG)
    rules.append((mu_trend_alta * mu_zone_discount * mu_fvg_baixo, 20.0))
    # VENDA fraca: Baixa + Premium + FVG Baixo
    rules.append((mu_trend_baixa * mu_zone_premium * mu_fvg_baixo, -20.0))

    # Conflitos → Neutro
    rules.append((mu_trend_alta * mu_zone_premium, 5.0))   # alta + premium = conflito
    rules.append((mu_trend_baixa * mu_zone_discount, -5.0)) # baixa + discount = conflito

    # COMPRA moderada: Alta + Discount + Sweep Fraco (setup incompleto)
    rules.append((mu_trend_alta * mu_zone_discount * mu_sweep_fraco, 25.0))
    # VENDA moderada: Baixa + Premium + Sweep Fraco
    rules.append((mu_trend_baixa * mu_zone_premium * mu_sweep_fraco, -25.0))

    # --- Defuzzificação: média ponderada (TSK zero-order) ---
    numerator = np.zeros(n)
    denominator = np.zeros(n)

    for weight, consequent in rules:
        numerator += weight * consequent
        denominator += weight

    # Evitar divisão por zero
    denominator = np.maximum(denominator, 1e-8)
    signal = numerator / denominator

    return np.clip(signal, -100.0, 100.0)


def generate_synthetic_data(
    n_samples: int = 5000,
    noise_level: float = 0.1,
    seed: int = 42,
    csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Gera dataset sintético respeitando a semântica SMC.

    Duas estratégias possíveis:
    1. Se csv_path fornecido e existir, carrega dados pré-gerados do CSV.
    2. Senão, gera analiticamente via regras SMC vetorizadas em NumPy.
       Equivalente funcional ao Mamdani, ~1000x mais rápido.

    O sinal é calculado como média ponderada TSK zero-order das regras
    e depois perturbado com ruído gaussiano controlado para simular
    a incerteza inerente ao processo discricionário.

    Parameters
    ----------
    n_samples : int
        Número de amostras a gerar.
    noise_level : float
        Nível de ruído relativo (fração do range da saída).
    seed : int
        Semente para reprodutibilidade.
    csv_path : str, optional
        Caminho para CSV pré-gerado. Se existir, carrega direto.

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas: trend_strength, price_zone, fvg_quality,
        sweep_quality, signal.
    """
    # Tentar carregar de CSV se disponível
    if csv_path and Path(csv_path).exists():
        logger.info(f"Carregando dados pre-gerados de {csv_path}...")
        df = pd.read_csv(csv_path)
        if len(df) >= n_samples:
            df = df.head(n_samples)
        logger.info(f"Dataset carregado: {len(df)} amostras")
        _log_dataset_stats(df)
        return df

    rng = np.random.RandomState(seed)

    logger.info(f"Gerando {n_samples} amostras sinteticas (noise={noise_level})...")

    # Gerar inputs aleatórios nos universos de discurso
    trend = rng.uniform(*UNIVERSES['trend_strength'], n_samples)
    zone = rng.uniform(*UNIVERSES['price_zone'], n_samples)
    fvg = rng.uniform(*UNIVERSES['fvg_quality'], n_samples)
    sweep = rng.uniform(*UNIVERSES['sweep_quality'], n_samples)

    # Calcular sinais analiticamente (vetorizado, ~1000x mais rápido que Mamdani)
    signals = _compute_synthetic_signal(trend, zone, fvg, sweep)

    # Adicionar ruído gaussiano
    signal_range = UNIVERSES['trade_signal'][1] - UNIVERSES['trade_signal'][0]
    noise = rng.normal(0, noise_level * signal_range, n_samples)
    signals_noisy = np.clip(signals + noise, *UNIVERSES['trade_signal'])

    data = {
        'trend_strength': trend,
        'price_zone': zone,
        'fvg_quality': fvg,
        'sweep_quality': sweep,
        'signal': signals_noisy,
    }

    df = pd.DataFrame(data)

    # Estatísticas descritivas
    logger.info(f"Dataset sintetico gerado: {len(df)} amostras")
    _log_dataset_stats(df)

    return df


def _log_dataset_stats(df: pd.DataFrame) -> None:
    """Loga estatísticas do dataset gerado."""
    logger.info(f"  signal: mean={df['signal'].mean():.2f}, "
                f"std={df['signal'].std():.2f}, "
                f"min={df['signal'].min():.2f}, max={df['signal'].max():.2f}")

    from .config import SIGNAL_THRESHOLDS
    for cls_name, (lo, hi) in SIGNAL_THRESHOLDS.items():
        count = ((df['signal'] >= lo) & (df['signal'] <= hi)).sum()
        pct = count / len(df) * 100
        logger.info(f"  {cls_name:>13s}: {count:4d} ({pct:.1f}%)")


def load_market_data(
    symbol: str = 'EURUSD=X',
    period: str = '2y',
    interval: str = '1h',
    future_bars: int = 10,
    range_bars: int = 20,
) -> pd.DataFrame:
    """
    Carrega dados de mercado e calcula as 4 variáveis fuzzy.

    Tenta MetaTrader5 primeiro; fallback automático para yfinance.

    Parameters
    ----------
    symbol : str
        Símbolo do ativo.
    period : str
        Período de dados (yfinance format).
    interval : str
        Timeframe (yfinance format).
    future_bars : int
        Barras à frente para calcular o target (retorno futuro).
    range_bars : int
        Barras para calcular o range do Price_Zone.

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas: trend_strength, price_zone, fvg_quality,
        sweep_quality, signal.
    """
    logger.info(f"Carregando dados de mercado: {symbol} ({period}, {interval})")

    # Tentar yfinance (mais portável)
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df_raw = ticker.history(period=period, interval=interval)

        if df_raw.empty:
            raise ValueError(f"yfinance retornou dados vazios para {symbol}")

        logger.info(f"Dados carregados via yfinance: {len(df_raw)} barras")
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        raise

    # Renomear colunas para lowercase
    df_raw.columns = [c.lower() for c in df_raw.columns]

    # Calcular as 4 variáveis fuzzy
    df = pd.DataFrame(index=df_raw.index)

    # 1. Trend_Strength: slope da EMA(50) normalizado em [-100, +100]
    ema50 = df_raw['close'].ewm(span=50, adjust=False).mean()
    slope = ema50.diff(5) / ema50.shift(5) * 100  # variação percentual
    slope_norm = slope.clip(-5, 5) / 5 * 100  # normalizar para [-100, 100]
    df['trend_strength'] = slope_norm

    # 2. Price_Zone: posição do fechamento no range [low_N, high_N]
    rolling_high = df_raw['high'].rolling(range_bars).max()
    rolling_low = df_raw['low'].rolling(range_bars).min()
    range_size = rolling_high - rolling_low
    df['price_zone'] = (df_raw['close'] - rolling_low) / (range_size + 1e-8)
    df['price_zone'] = df['price_zone'].clip(0, 1)

    # 3. FVG_Quality: tamanho do FVG / ATR(14)
    atr = _compute_atr(df_raw, period=14)
    fvg_size = _detect_fvg_size(df_raw)
    df['fvg_quality'] = (fvg_size / (atr + 1e-8)).clip(0, 4)

    # 4. Sweep_Quality: shadow / body
    body = (df_raw['close'] - df_raw['open']).abs()
    upper_shadow = df_raw['high'] - df_raw[['close', 'open']].max(axis=1)
    lower_shadow = df_raw[['close', 'open']].min(axis=1) - df_raw['low']
    max_shadow = pd.concat([upper_shadow, lower_shadow], axis=1).max(axis=1)
    df['sweep_quality'] = (max_shadow / (body + 1e-8)).clip(0, 3)

    # Target: retorno futuro normalizado para [-100, 100]
    future_return = df_raw['close'].shift(-future_bars) / df_raw['close'] - 1
    ret_norm = future_return.clip(-0.05, 0.05) / 0.05 * 100
    df['signal'] = ret_norm

    # Limpar NaNs
    df = df.dropna().reset_index(drop=True)

    logger.info(f"Dataset de mercado processado: {len(df)} amostras de {symbol}")

    return df


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calcula Average True Range."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def _detect_fvg_size(df: pd.DataFrame) -> pd.Series:
    """
    Detecta Fair Value Gaps e retorna o tamanho.

    FVG bullish: low[i] > high[i-2] → gap = low[i] - high[i-2]
    FVG bearish: high[i] < low[i-2] → gap = low[i-2] - high[i]
    """
    fvg_bull = (df['low'] - df['high'].shift(2)).clip(lower=0)
    fvg_bear = (df['low'].shift(2) - df['high']).clip(lower=0)
    return fvg_bull + fvg_bear


def prepare_dataloaders(
    df: pd.DataFrame,
    config: Optional[Dict] = None,
    temporal_split: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Prepara DataLoaders do PyTorch a partir do DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com colunas: trend_strength, price_zone, fvg_quality,
        sweep_quality, signal.
    config : dict, optional
        Configurações de treinamento. Se None, usa TRAINING.
    temporal_split : bool
        Se True, mantém ordem temporal (sem embaralhamento).

    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader
    test_loader : DataLoader
    metadata : dict
        Informações sobre o split e estatísticas.
    """
    if config is None:
        config = TRAINING

    # Extrair arrays
    X = df[INPUT_VARS].values.astype(np.float32)
    y = df['signal'].values.astype(np.float32).reshape(-1, 1)

    n = len(X)
    train_size = int(n * config['train_split'])
    val_size = int(n * config['val_split'])

    if temporal_split:
        # Split temporal (sem embaralhamento)
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
        X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]
    else:
        # Shuffle (para dados sintéticos)
        rng = np.random.RandomState(config['random_seed'])
        indices = rng.permutation(n)
        X, y = X[indices], y[indices]

        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
        X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    batch_size = config['batch_size']

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=not temporal_split,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=batch_size,
        shuffle=False,
    )

    metadata = {
        'n_total': n,
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'batch_size': batch_size,
        'temporal_split': temporal_split,
        'stats': {
            var: {
                'mean': float(df[var].mean()),
                'std': float(df[var].std()),
                'min': float(df[var].min()),
                'max': float(df[var].max()),
            }
            for var in INPUT_VARS + ['signal']
        },
    }

    logger.info(
        f"DataLoaders criados: train={metadata['n_train']}, "
        f"val={metadata['n_val']}, test={metadata['n_test']}, "
        f"batch_size={batch_size}"
    )

    return train_loader, val_loader, test_loader, metadata
