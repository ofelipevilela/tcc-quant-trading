# -*- coding: utf-8 -*-
"""
Configurações globais do sistema de trading híbrido SMC + Fuzzy Logic.

Este arquivo centraliza todos os parâmetros do sistema fuzzy,
incluindo universos de discurso e parâmetros das funções de pertinência.

Variáveis baseadas em Smart Money Concepts (SMC):
- Trend_Strength: Força da tendência (ADX/Slope EMA)
- Price_Zone: Localização no range Premium/Discount
- FVG_Quality: Qualidade do Fair Value Gap
- Sweep_Quality: Qualidade da captura de liquidez
- Trade_Score: Score final do setup (saída)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class FuzzyUniverseConfig:
    """Configuração do universo de discurso para uma variável fuzzy."""
    name: str
    min_val: float
    max_val: float
    resolution: int = 100  # Número de pontos no universo


@dataclass
class MembershipFunctionConfig:
    """Configuração de uma função de pertinência."""
    name: str
    mf_type: str  # 'trimf', 'trapmf', 'gaussmf', 'sigmf', 'smf', 'zmf'
    params: List[float]


@dataclass
class FuzzyVariableConfig:
    """Configuração completa de uma variável fuzzy."""
    universe: FuzzyUniverseConfig
    membership_functions: List[MembershipFunctionConfig]


# =============================================================================
# VARIÁVEIS DE ENTRADA (ANTECEDENTES) - Smart Money Concepts
# =============================================================================

# -----------------------------------------------------------------------------
# 1. TREND_STRENGTH - Força e Direção da Tendência (BIPOLAR)
# Input Crisp: Slope da EMA ou indicador direcional
#   - Valores negativos = Tendência de BAIXA (favorece vendas)
#   - Zero = Sem tendência (neutro)
#   - Valores positivos = Tendência de ALTA (favorece compras)
# Universo: -100 a +100
# Conjuntos: {Baixa, Neutra, Alta}
# Tipo de Curva: Gaussiana (gaussmf) para suavidade
# -----------------------------------------------------------------------------
TREND_STRENGTH_CONFIG = FuzzyVariableConfig(
    universe=FuzzyUniverseConfig(
        name="Trend_Strength",
        min_val=-100,   # Tendência de baixa forte
        max_val=100,    # Tendência de alta forte
        resolution=201  # Mais pontos para escala maior
    ),
    membership_functions=[
        # gaussmf: [mean, sigma]
        MembershipFunctionConfig("Baixa", "gaussmf", [-100, 30]),   # Centrada em -100 (bearish forte)
        MembershipFunctionConfig("Neutra", "gaussmf", [0, 25]),     # Centrada em 0 (sem tendência)
        MembershipFunctionConfig("Alta", "gaussmf", [100, 30]),     # Centrada em +100 (bullish forte)
    ]
)

# -----------------------------------------------------------------------------
# 2. PRICE_ZONE - Localização do Preço (Premium/Discount)
# Input Crisp: % do Range (0 a 1, onde 1 é o topo)
# Universo: 0 a 1
# Conjuntos: {Deep_Discount, Discount, Equilibrium, Premium, Deep_Premium}
# Tipo de Curva: Trapezoidal (trapmf) para zonas rígidas
# -----------------------------------------------------------------------------
PRICE_ZONE_CONFIG = FuzzyVariableConfig(
    universe=FuzzyUniverseConfig(
        name="Price_Zone",
        min_val=0,
        max_val=1,
        resolution=101
    ),
    membership_functions=[
        # trapmf: [a, b, c, d] - a e d são os pés, b e c são o topo
        MembershipFunctionConfig("Deep_Discount", "trapmf", [0, 0, 0.1, 0.2]),       # 0-20%
        MembershipFunctionConfig("Discount", "trapmf", [0.15, 0.25, 0.35, 0.45]),    # 15-45%
        MembershipFunctionConfig("Equilibrium", "trapmf", [0.4, 0.47, 0.53, 0.6]),   # 40-60%
        MembershipFunctionConfig("Premium", "trapmf", [0.55, 0.65, 0.75, 0.85]),     # 55-85%
        MembershipFunctionConfig("Deep_Premium", "trapmf", [0.8, 0.9, 1.0, 1.0]),    # 80-100%
    ]
)

# -----------------------------------------------------------------------------
# 3. FVG_QUALITY - Qualidade do Fair Value Gap
# Input Crisp: Tamanho do FVG relativo ao ATR (FVG_Size / ATR)
# Universo: 0 a 4
# Conjuntos: {Pequeno, Padrao, Grande}
# Tipo de Curva: Triangular para Pequeno/Padrao, Sigmoidal para Grande
# -----------------------------------------------------------------------------
FVG_QUALITY_CONFIG = FuzzyVariableConfig(
    universe=FuzzyUniverseConfig(
        name="FVG_Quality",
        min_val=0,
        max_val=4,
        resolution=101
    ),
    membership_functions=[
        # trimf: [a, b, c] - a e c são as bases, b é o pico
        MembershipFunctionConfig("Pequeno", "trimf", [0, 0, 1.0]),           # Pico em 0
        MembershipFunctionConfig("Padrao", "trimf", [0.5, 1.5, 2.5]),        # Pico em 1.5
        # smf (S-shaped): [a, b] - a é onde começa a subir, b é onde chega a 1
        MembershipFunctionConfig("Grande", "smf", [2.0, 3.0]),               # Começa a subir em 2, máx em 3+
    ]
)

# -----------------------------------------------------------------------------
# 4. SWEEP_QUALITY - Qualidade da Captura de Liquidez
# Input Crisp: Razão (Pavio / Corpo) da vela de sweep
# Universo: 0 a 3
# Conjuntos: {Fraco, Forte}
# Tipo de Curva: Sigmoidal
# -----------------------------------------------------------------------------
SWEEP_QUALITY_CONFIG = FuzzyVariableConfig(
    universe=FuzzyUniverseConfig(
        name="Sweep_Quality",
        min_val=0,
        max_val=3,
        resolution=101
    ),
    membership_functions=[
        # zmf (Z-shaped): [a, b] - começa em 1, cai para 0 entre a e b
        MembershipFunctionConfig("Fraco", "zmf", [0.5, 1.5]),    # Forte em 0, cai até 1.5
        # smf (S-shaped): [a, b] - começa em 0, sobe para 1 entre a e b
        MembershipFunctionConfig("Forte", "smf", [0.8, 1.8]),    # Fraco em 0, sobe a partir de 0.8
    ]
)

# =============================================================================
# VARIÁVEL DE SAÍDA (CONSEQUENTE)
# =============================================================================

# -----------------------------------------------------------------------------
# TRADE_SCORE - Score do Setup (Probabilidade de Sucesso)
# Universo: 0 a 100
# Conjuntos: {Fraco, Moderado, Forte, Muito_Forte}
# Defuzzificação: Centroide (centroid)
# -----------------------------------------------------------------------------
TRADE_SCORE_CONFIG = FuzzyVariableConfig(
    universe=FuzzyUniverseConfig(
        name="Trade_Score",
        min_val=0,
        max_val=100,
        resolution=101
    ),
    membership_functions=[
        # trapmf para definir zonas claras de score
        MembershipFunctionConfig("Fraco", "trapmf", [0, 0, 15, 30]),
        MembershipFunctionConfig("Moderado", "trapmf", [20, 35, 50, 65]),
        MembershipFunctionConfig("Forte", "trapmf", [55, 70, 80, 90]),
        MembershipFunctionConfig("Muito_Forte", "trapmf", [80, 90, 100, 100]),
    ]
)

# Dicionário com todas as configurações
ALL_FUZZY_CONFIGS: Dict[str, FuzzyVariableConfig] = {
    "trend_strength": TREND_STRENGTH_CONFIG,
    "price_zone": PRICE_ZONE_CONFIG,
    "fvg_quality": FVG_QUALITY_CONFIG,
    "sweep_quality": SWEEP_QUALITY_CONFIG,
    "trade_score": TRADE_SCORE_CONFIG,
}

# =============================================================================
# CONFIGURAÇÕES DE VISUALIZAÇÃO
# =============================================================================

@dataclass
class VisualizationConfig:
    """Configurações para visualização das funções de pertinência."""
    figure_size: Tuple[int, int] = (14, 12)
    dpi: int = 150
    style: str = "seaborn-v0_8-whitegrid"
    save_path: str = "outputs/membership_functions.png"
    
    # Cores para as MFs (cicla se necessário)
    colors: List[str] = field(default_factory=lambda: [
        "#E63946",  # Vermelho
        "#457B9D",  # Azul
        "#2A9D8F",  # Verde
        "#E9C46A",  # Amarelo
        "#9B59B6",  # Roxo
    ])

VISUALIZATION_CONFIG = VisualizationConfig()
