# -*- coding: utf-8 -*-
"""
Fuzzy Logic module for the hybrid SMC + Fuzzy trading system.

This module implements the Mamdani fuzzy inference system for
generating trading signals based on SMC indicators.
"""

from .membership_functions import create_fuzzy_variables
from .visualization import plot_membership_functions

__all__ = [
    "create_fuzzy_variables",
    "plot_membership_functions",
]
