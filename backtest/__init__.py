# -*- coding: utf-8 -*-
"""
Módulo de Backtesting.

Integra indicadores SMC vetorizados, avaliação ANFIS e motor de execução simulada.
Permite extrair a curva de capital resultante da estratégia baseada em Smart Money Concepts.
"""

from .engine import BacktestEngine
from .performance import calculate_performance_metrics

__all__ = ['BacktestEngine', 'calculate_performance_metrics']
