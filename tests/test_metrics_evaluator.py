# -*- coding: utf-8 -*-
"""Testes das metricas quantitativas do ANFIS."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from metrics_evaluator import (
    ANFISMetricsEvaluator,
    directional_accuracy,
    evaluate_predictions,
    evaluate_train_validation,
    weighted_hit_rate,
)


def test_directional_accuracy_ignores_zero_true_returns():
    y_true = np.array([0.0, 1.0, -2.0, 3.0, -4.0])
    y_pred = np.array([1.0, 0.5, 0.2, -1.0, -0.3])

    # Amostras direcionais: acerta 1.0 e -4.0; erra -2.0 e 3.0.
    assert directional_accuracy(y_true, y_pred) == 0.5


def test_weighted_hit_rate_weights_large_moves_more():
    y_true = np.array([1.0, -3.0, 2.0])
    y_pred = np.array([0.2, 0.1, 0.4])

    # Acertos: movimentos 1.0 e 2.0. Peso total: 1 + 3 + 2 = 6.
    assert weighted_hit_rate(y_true, y_pred) == 0.5


def test_evaluate_predictions_returns_expected_keys():
    y_true = pd.Series([0.1, -0.2, 0.3, -0.4])
    y_pred = pd.Series([0.1, -0.1, -0.2, -0.3])

    metrics = evaluate_predictions(y_true, y_pred)

    expected_keys = {
        "n_samples",
        "n_directional_samples",
        "mae",
        "rmse",
        "r2",
        "directional_accuracy",
        "weighted_hit_rate",
    }
    assert expected_keys.issubset(metrics.keys())
    assert metrics["n_samples"] == 4
    assert metrics["n_directional_samples"] == 4
    assert 0.0 <= metrics["directional_accuracy"] <= 1.0
    assert 0.0 <= metrics["weighted_hit_rate"] <= 1.0


def test_evaluate_train_validation_returns_two_rows():
    train_true = np.array([0.1, -0.2, 0.3])
    train_pred = np.array([0.1, -0.1, 0.2])
    val_true = np.array([0.2, -0.4, 0.0])
    val_pred = np.array([-0.1, -0.3, 0.1])

    metrics_df = evaluate_train_validation(train_true, train_pred, val_true, val_pred)

    assert list(metrics_df["split"]) == ["train", "validation"]
    assert set(["mae", "rmse", "r2"]).issubset(metrics_df.columns)


def test_evaluator_class_uses_zero_tolerance():
    evaluator = ANFISMetricsEvaluator(zero_tolerance=0.05)
    metrics = evaluator.evaluate([0.01, 0.2, -0.3], [1.0, 0.1, -0.1])

    assert metrics["n_samples"] == 3
    assert metrics["n_directional_samples"] == 2
    assert metrics["directional_accuracy"] == 1.0
