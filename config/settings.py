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
- Trade_Signal: Sinal final bidirecional do setup (saída)
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
# Conjuntos: {Discount, Equilibrium, Premium}
# Tipo de Curva: Z / Triangular / S
# -----------------------------------------------------------------------------
PRICE_ZONE_CONFIG = FuzzyVariableConfig(
    universe=FuzzyUniverseConfig(
        name="Price_Zone",
        min_val=0,
        max_val=1,
        resolution=101
    ),
    membership_functions=[
        # zmf: pertinência total até 0.3, decaindo até 0 em 0.5
        MembershipFunctionConfig("Discount", "zmf", [0.3, 0.5]),
        # trimf: equilíbrio centrado no meio do range
        MembershipFunctionConfig("Equilibrium", "trimf", [0.4, 0.5, 0.6]),
        # smf: começa a subir em 0.5, atinge 1 em 0.7
        MembershipFunctionConfig("Premium", "smf", [0.5, 0.7]),
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
# TRADE_SIGNAL - Sinal Bidirecional do Setup
# Universo: -100 a +100
# Conjuntos: {Venda_Forte, Venda, Neutro, Compra, Compra_Forte}
# Defuzzificação: Centroide (centroid)
# -----------------------------------------------------------------------------
TRADE_SIGNAL_CONFIG = FuzzyVariableConfig(
    universe=FuzzyUniverseConfig(
        name="Trade_Signal",
        min_val=-100,
        max_val=100,
        resolution=201
    ),
    membership_functions=[
        MembershipFunctionConfig("Venda_Forte", "trapmf", [-100, -100, -75, -50]),
        MembershipFunctionConfig("Venda", "trimf", [-75, -50, -10]),
        MembershipFunctionConfig("Neutro", "trimf", [-20, 0, 20]),
        MembershipFunctionConfig("Compra", "trimf", [10, 50, 75]),
        MembershipFunctionConfig("Compra_Forte", "trapmf", [50, 75, 100, 100]),
    ]
)

# Dicionário com todas as configurações
ALL_FUZZY_CONFIGS: Dict[str, FuzzyVariableConfig] = {
    "trend_strength": TREND_STRENGTH_CONFIG,
    "price_zone": PRICE_ZONE_CONFIG,
    "fvg_quality": FVG_QUALITY_CONFIG,
    "sweep_quality": SWEEP_QUALITY_CONFIG,
    "trade_signal": TRADE_SIGNAL_CONFIG,
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
