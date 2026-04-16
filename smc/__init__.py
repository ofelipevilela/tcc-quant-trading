# -*- coding: utf-8 -*-
"""
Smart Money Concepts (SMC) module.

Extrai indicadores institucionais a partir de dados OHLCV:
- Swing Highs / Swing Lows
- Break of Structure (BOS)
- Change in State of Delivery (CISD)
- Fair Value Gaps (FVG)
- Sweeps de Liquidez
- Premium / Discount Zones
- Trend Strength

Todos os indicadores são vetorizados (Pandas + NumPy) e mapeados
diretamente para os universos de entrada do sistema ANFIS.
"""

from .feature_factory import (
    FEATURE_MODE_CAUSAL_ENHANCED,
    FEATURE_MODE_CAUSAL_RAW,
    FEATURE_MODE_LEGACY_LIKE,
    VALID_FEATURE_MODES,
    build_smc_features,
    resolve_feature_mode,
)
from .indicators import SMCIndicators

__all__ = [
    "SMCIndicators",
    "build_smc_features",
    "FEATURE_MODE_CAUSAL_RAW",
    "FEATURE_MODE_CAUSAL_ENHANCED",
    "FEATURE_MODE_LEGACY_LIKE",
    "VALID_FEATURE_MODES",
    "resolve_feature_mode",
]
