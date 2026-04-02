# -*- coding: utf-8 -*-
"""
Módulo ANFIS (Adaptive-Network-Based Fuzzy Inference System).

Implementação em PyTorch da arquitetura de 5 camadas proposta por
Jang (1993) com modelo TSK zero-order e treinamento via Adam.

Referência:
    Jang, J.-S. R. (1993). ANFIS: Adaptive-Network-Based Fuzzy
    Inference System. IEEE Transactions on Systems, Man, and
    Cybernetics, 23(3), 665-685.
"""

from .config import UNIVERSES, LINGUISTIC_SETS, INITIAL_MF_PARAMS, TRAINING, SIGNAL_THRESHOLDS
from .membership_functions import GaussianMF, FuzzificationLayer
from .rule_base import RuleBase, RULES
from .anfis_model import ANFISModel

__all__ = [
    "UNIVERSES",
    "LINGUISTIC_SETS",
    "INITIAL_MF_PARAMS",
    "TRAINING",
    "SIGNAL_THRESHOLDS",
    "GaussianMF",
    "FuzzificationLayer",
    "RuleBase",
    "RULES",
    "ANFISModel",
]
