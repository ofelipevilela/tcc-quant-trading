# -*- coding: utf-8 -*-
"""
Fuzzy Logic module for the hybrid SMC + Fuzzy trading system.

This module implements the Mamdani fuzzy inference system for
generating trading signals based on SMC indicators.
"""

from .membership_functions import create_fuzzy_variables
from .visualization import (
    plot_membership_functions,
    plot_with_examples,
    get_pertinence_values,
    print_pertinence_table,
)
from .fuzzy_system import SMCFuzzySystem, create_fuzzy_system

__all__ = [
    "create_fuzzy_variables",
    "plot_membership_functions",
    "plot_with_examples",
    "get_pertinence_values",
    "print_pertinence_table",
    "SMCFuzzySystem",
    "create_fuzzy_system",
]
