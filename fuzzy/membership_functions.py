# -*- coding: utf-8 -*-
"""
Definição das Funções de Pertinência (Membership Functions) para o sistema fuzzy SMC.

Este módulo cria as variáveis fuzzy (antecedentes e consequentes) usando
a biblioteca scikit-fuzzy, baseado nas configurações do Smart Money Concepts.

Variáveis:
- Trend_Strength: Força da tendência (ADX/Slope EMA) - Gaussiana
- Price_Zone: Premium/Discount - Trapezoidal
- FVG_Quality: Qualidade do Fair Value Gap - Triangular/Sigmoidal
- Sweep_Quality: Captura de Liquidez - Sigmoidal
- Trade_Score: Score final do setup (saída)
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from typing import Dict

from config.settings import (
    TREND_STRENGTH_CONFIG,
    PRICE_ZONE_CONFIG,
    FVG_QUALITY_CONFIG,
    SWEEP_QUALITY_CONFIG,
    TRADE_SCORE_CONFIG,
    FuzzyVariableConfig,
)


def _create_universe(config: FuzzyVariableConfig) -> np.ndarray:
    """
    Cria o universo de discurso para uma variável fuzzy.
    
    Args:
        config: Configuração da variável fuzzy
        
    Returns:
        Array numpy representando o universo de discurso
    """
    return np.linspace(
        config.universe.min_val,
        config.universe.max_val,
        config.universe.resolution
    )


def _apply_membership_function(
    variable: ctrl.Antecedent | ctrl.Consequent,
    config: FuzzyVariableConfig
) -> None:
    """
    Aplica as funções de pertinência a uma variável fuzzy.
    
    Suporta os seguintes tipos de MF:
    - trimf: Triangular [a, b, c]
    - trapmf: Trapezoidal [a, b, c, d]
    - gaussmf: Gaussiana [mean, sigma]
    - smf: S-shaped (sigmoidal crescente) [a, b]
    - zmf: Z-shaped (sigmoidal decrescente) [a, b]
    - sigmf: Sigmoidal [b, c] onde b=inclinação, c=centro
    
    Args:
        variable: Variável fuzzy (Antecedente ou Consequente)
        config: Configuração com as MFs a serem aplicadas
    """
    for mf in config.membership_functions:
        if mf.mf_type == "trimf":
            variable[mf.name] = fuzz.trimf(variable.universe, mf.params)
        elif mf.mf_type == "trapmf":
            variable[mf.name] = fuzz.trapmf(variable.universe, mf.params)
        elif mf.mf_type == "gaussmf":
            # gaussmf(x, mean, sigma)
            variable[mf.name] = fuzz.gaussmf(variable.universe, mf.params[0], mf.params[1])
        elif mf.mf_type == "smf":
            # smf(x, a, b) - S-shaped, sobe de 0 para 1 entre a e b
            variable[mf.name] = fuzz.smf(variable.universe, mf.params[0], mf.params[1])
        elif mf.mf_type == "zmf":
            # zmf(x, a, b) - Z-shaped, desce de 1 para 0 entre a e b
            variable[mf.name] = fuzz.zmf(variable.universe, mf.params[0], mf.params[1])
        elif mf.mf_type == "sigmf":
            # sigmf(x, b, c) - Sigmoidal, b=inclinação, c=centro
            variable[mf.name] = fuzz.sigmf(variable.universe, mf.params[0], mf.params[1])
        elif mf.mf_type == "gbellmf":
            variable[mf.name] = fuzz.gbellmf(variable.universe, *mf.params)
        else:
            raise ValueError(f"Tipo de MF não suportado: {mf.mf_type}")


def create_trend_strength_variable() -> ctrl.Antecedent:
    """
    Cria a variável fuzzy Trend_Strength (Força da Tendência).
    
    Input Crisp sugerido: ADX ou Slope da EMA.
    
    Conjuntos fuzzy (Gaussianas):
    - Baixa: Tendência fraca (ADX < 25)
    - Neutra: Tendência moderada (ADX ~50)
    - Alta: Tendência forte (ADX > 75)
    
    Returns:
        Antecedente fuzzy para Trend_Strength
    """
    universe = _create_universe(TREND_STRENGTH_CONFIG)
    trend = ctrl.Antecedent(universe, 'Trend_Strength')
    _apply_membership_function(trend, TREND_STRENGTH_CONFIG)
    return trend


def create_price_zone_variable() -> ctrl.Antecedent:
    """
    Cria a variável fuzzy Price_Zone (Localização Premium/Discount).
    
    Input Crisp: % do Range (0 a 1, onde 0=fundo, 1=topo).
    
    Conjuntos fuzzy (Trapezoidais para zonas rígidas):
    - Deep_Discount: 0-20% do range (zona de compra forte)
    - Discount: 20-45% (zona de compra)
    - Equilibrium: 40-60% (zona neutra)
    - Premium: 55-85% (zona de venda)
    - Deep_Premium: 80-100% (zona de venda forte)
    
    Returns:
        Antecedente fuzzy para Price_Zone
    """
    universe = _create_universe(PRICE_ZONE_CONFIG)
    zone = ctrl.Antecedent(universe, 'Price_Zone')
    _apply_membership_function(zone, PRICE_ZONE_CONFIG)
    return zone


def create_fvg_quality_variable() -> ctrl.Antecedent:
    """
    Cria a variável fuzzy FVG_Quality (Qualidade do Fair Value Gap).
    
    Input Crisp: Tamanho do FVG relativo ao ATR (FVG_Size / ATR).
    
    Interpretação:
    - FVG < 0.5 ATR: Muito pequeno (não significativo)
    - FVG ~1.5 ATR: Tamanho padrão
    - FVG > 2.5 ATR: Grande (altamente significativo)
    
    Conjuntos fuzzy:
    - Pequeno: Triangular, FVG não significativo
    - Padrao: Triangular, FVG de tamanho ideal
    - Grande: Sigmoidal (S-shaped), FVG muito significativo
    
    Returns:
        Antecedente fuzzy para FVG_Quality
    """
    universe = _create_universe(FVG_QUALITY_CONFIG)
    fvg = ctrl.Antecedent(universe, 'FVG_Quality')
    _apply_membership_function(fvg, FVG_QUALITY_CONFIG)
    return fvg


def create_sweep_quality_variable() -> ctrl.Antecedent:
    """
    Cria a variável fuzzy Sweep_Quality (Qualidade da Captura de Liquidez).
    
    Input Crisp: Razão (Pavio / Corpo) da vela de sweep.
    
    Interpretação:
    - Ratio < 0.5: Sweep fraco (pavio pequeno relativo ao corpo)
    - Ratio > 1.5: Sweep forte (pavio grande = rejeição clara)
    
    Conjuntos fuzzy (Sigmoidais):
    - Fraco: Z-shaped (alta pertinência em 0, decai até 1.5)
    - Forte: S-shaped (baixa em 0, sobe a partir de 0.8)
    
    Returns:
        Antecedente fuzzy para Sweep_Quality
    """
    universe = _create_universe(SWEEP_QUALITY_CONFIG)
    sweep = ctrl.Antecedent(universe, 'Sweep_Quality')
    _apply_membership_function(sweep, SWEEP_QUALITY_CONFIG)
    return sweep


def create_trade_score_variable() -> ctrl.Consequent:
    """
    Cria a variável fuzzy Trade_Score (saída do sistema).
    
    Score final do setup de trading variando de 0 a 100.
    Defuzzificação: Centroide (centroid).
    
    Conjuntos fuzzy (Trapezoidais):
    - Fraco: 0-30 (não operar)
    - Moderado: 20-65 (operar com cautela)
    - Forte: 55-90 (bom setup)
    - Muito_Forte: 80-100 (setup de alta confiança)
    
    Returns:
        Consequente fuzzy para Trade_Score
    """
    universe = _create_universe(TRADE_SCORE_CONFIG)
    score = ctrl.Consequent(universe, 'Trade_Score', defuzzify_method='centroid')
    _apply_membership_function(score, TRADE_SCORE_CONFIG)
    return score


def create_fuzzy_variables() -> Dict[str, ctrl.Antecedent | ctrl.Consequent]:
    """
    Cria todas as variáveis fuzzy do sistema SMC.
    
    Esta função é o ponto de entrada principal para obter
    todas as variáveis fuzzy configuradas para o sistema de trading.
    
    Returns:
        Dicionário com todas as variáveis fuzzy:
        - 'trend_strength': Antecedente Força da Tendência
        - 'price_zone': Antecedente Zona de Preço
        - 'fvg_quality': Antecedente Qualidade do FVG
        - 'sweep_quality': Antecedente Qualidade do Sweep
        - 'trade_score': Consequente Score do Trade
    
    Example:
        >>> variables = create_fuzzy_variables()
        >>> print(variables['trend_strength'].terms)
        >>> print(variables['trade_score'].defuzzify_method)
    """
    return {
        'trend_strength': create_trend_strength_variable(),
        'price_zone': create_price_zone_variable(),
        'fvg_quality': create_fvg_quality_variable(),
        'sweep_quality': create_sweep_quality_variable(),
        'trade_score': create_trade_score_variable(),
    }
