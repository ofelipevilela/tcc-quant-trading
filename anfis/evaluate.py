# -*- coding: utf-8 -*-
"""
Métricas de Avaliação do ANFIS.

Calcula métricas quantitativas de regressão e classificação para
avaliar o desempenho do ANFIS treinado e compará-lo com o Mamdani original.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from .config import SEMANTIC_ORDER, SIGNAL_THRESHOLDS

logger = logging.getLogger(__name__)


def _classify_array(values: np.ndarray) -> np.ndarray:
    """
    Classifica um array de sinais em classes linguísticas.

    Parameters
    ----------
    values : np.ndarray
        Array de valores de sinal.

    Returns
    -------
    np.ndarray
        Array de strings com nomes de classe.
    """
    classes = np.full(len(values), 'NEUTRO', dtype=object)

    for cls_name, (lo, hi) in SIGNAL_THRESHOLDS.items():
        mask = (values >= lo) & (values <= hi)
        classes[mask] = cls_name

    # Extremos
    classes[values < -100] = 'VENDA_FORTE'
    classes[values > 100] = 'COMPRA_FORTE'

    return classes


def compute_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    verbose: bool = True,
) -> Dict:
    """
    Calcula métricas completas de avaliação.

    Parameters
    ----------
    y_pred : np.ndarray
        Predições do modelo.
    y_true : np.ndarray
        Valores reais (targets).
    verbose : bool
        Se True, imprime as métricas no terminal.

    Returns
    -------
    dict
        Dicionário com todas as métricas calculadas.
    """
    y_pred = np.asarray(y_pred).flatten()
    y_true = np.asarray(y_true).flatten()

    # --- Métricas de regressão ---
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    # --- Acurácia direcional ---
    # sign(pred) == sign(true) — considerando 0 como neutro
    correct_direction = np.sign(y_pred) == np.sign(y_true)
    directional_accuracy = float(np.mean(correct_direction) * 100)

    # --- Information Coefficient (Spearman) ---
    ic_result = spearmanr(y_pred, y_true)
    ic = float(ic_result.correlation) if not np.isnan(ic_result.correlation) else 0.0

    # --- Classificação 5 classes ---
    pred_classes = _classify_array(y_pred)
    true_classes = _classify_array(y_true)

    # Confusion matrix
    labels = SEMANTIC_ORDER
    cm = confusion_matrix(true_classes, pred_classes, labels=labels)

    # F1 score
    f1_macro = float(f1_score(true_classes, pred_classes, labels=labels,
                              average='macro', zero_division=0))
    f1_weighted = float(f1_score(true_classes, pred_classes, labels=labels,
                                 average='weighted', zero_division=0))

    # Classification report
    cls_report = classification_report(
        true_classes, pred_classes, labels=labels,
        zero_division=0, output_dict=True,
    )

    # --- Análise de quantis ---
    quantile_analysis = _compute_quantile_analysis(y_pred, y_true)

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'directional_accuracy': directional_accuracy,
        'ic': ic,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm.tolist(),
        'classification_report': cls_report,
        'quantile_analysis': quantile_analysis,
    }

    if verbose:
        _print_metrics(metrics)

    return metrics


def _compute_quantile_analysis(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    n_quantiles: int = 5,
) -> Dict:
    """
    Divide predições em quantis e calcula retorno médio por quantil.

    Se a magnitude do sinal é preditiva, as barras devem crescer
    monotonicamente (do quintil mais baixo ao mais alto).

    Parameters
    ----------
    y_pred : np.ndarray
        Predições.
    y_true : np.ndarray
        Targets.
    n_quantiles : int
        Número de quantis.

    Returns
    -------
    dict
        Análise por quantil.
    """
    try:
        quantile_edges = np.percentile(y_pred, np.linspace(0, 100, n_quantiles + 1))
        # Assegurar valores únicos
        quantile_edges = np.unique(quantile_edges)
        n_actual = len(quantile_edges) - 1

        if n_actual < 2:
            return {'quantiles': [], 'mean_returns': [], 'is_monotonic': False}

        quantile_labels = np.digitize(y_pred, quantile_edges[1:-1])
        mean_returns = []
        quantile_names = []

        for q in range(n_actual):
            mask = quantile_labels == q
            if mask.any():
                mean_ret = float(y_true[mask].mean())
            else:
                mean_ret = 0.0
            mean_returns.append(mean_ret)
            quantile_names.append(f"Q{q + 1}")

        # Verificar monotonia
        is_monotonic = all(
            mean_returns[i] <= mean_returns[i + 1]
            for i in range(len(mean_returns) - 1)
        )

        return {
            'quantiles': quantile_names,
            'mean_returns': mean_returns,
            'is_monotonic': is_monotonic,
        }
    except Exception as e:
        logger.warning(f"Erro na análise de quantis: {e}")
        return {'quantiles': [], 'mean_returns': [], 'is_monotonic': False}


def _print_metrics(metrics: Dict) -> None:
    """Imprime métricas formatadas no terminal."""
    print("\n" + "=" * 50)
    print("MÉTRICAS DE AVALIAÇÃO")
    print("=" * 50)
    print(f"  MAE:                  {metrics['mae']:.4f}")
    print(f"  RMSE:                 {metrics['rmse']:.4f}")
    print(f"  R²:                   {metrics['r2']:.4f}")
    print(f"  Acurácia Direcional:  {metrics['directional_accuracy']:.1f}%")
    print(f"  IC (Spearman):        {metrics['ic']:.4f}")
    print(f"  F1 Macro:             {metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted:          {metrics['f1_weighted']:.4f}")

    qa = metrics['quantile_analysis']
    if qa.get('quantiles'):
        print(f"\n  Análise de Quantis (monotonia: {qa['is_monotonic']}):")
        for q, ret in zip(qa['quantiles'], qa['mean_returns']):
            print(f"    {q}: retorno médio = {ret:.2f}")

    print("=" * 50)


def compare_before_after(
    mamdani_preds: np.ndarray,
    anfis_preds: np.ndarray,
    y_true: np.ndarray,
    verbose: bool = True,
) -> Dict:
    """
    Tabela comparativa Mamdani vs ANFIS — tabela principal de resultados do TCC.

    Parameters
    ----------
    mamdani_preds : np.ndarray
        Saídas do sistema Mamdani original.
    anfis_preds : np.ndarray
        Saídas do ANFIS treinado.
    y_true : np.ndarray
        Targets.
    verbose : bool
        Se True, imprime tabela formatada.

    Returns
    -------
    dict
        Métricas de ambos os sistemas para serialização.
    """
    logger.info("Calculando métricas comparativas Mamdani vs ANFIS...")

    mamdani_metrics = compute_metrics(mamdani_preds, y_true, verbose=False)
    anfis_metrics = compute_metrics(anfis_preds, y_true, verbose=False)

    comparison = {
        'mamdani': mamdani_metrics,
        'anfis': anfis_metrics,
    }

    if verbose:
        _print_comparison_table(mamdani_metrics, anfis_metrics)

    return comparison


def _print_comparison_table(mamdani: Dict, anfis: Dict) -> None:
    """Imprime tabela comparativa formatada."""
    print("\n")
    print("=" * 59)
    print("  RESULTADOS FINAIS: MAMDANI vs ANFIS")
    print("=" * 59)
    print("┌─────────────────────┬──────────────┬──────────────┐")
    print("│ Métrica             │   Mamdani    │     ANFIS    │")
    print("├─────────────────────┼──────────────┼──────────────┤")

    rows = [
        ("MAE", f"{mamdani['mae']:.2f}", f"{anfis['mae']:.2f}"),
        ("RMSE", f"{mamdani['rmse']:.2f}", f"{anfis['rmse']:.2f}"),
        ("R²", f"{mamdani['r2']:.4f}", f"{anfis['r2']:.4f}"),
        ("Acurácia Direcional", f"{mamdani['directional_accuracy']:.1f}%",
         f"{anfis['directional_accuracy']:.1f}%"),
        ("IC (Spearman)", f"{mamdani['ic']:.3f}", f"{anfis['ic']:.3f}"),
        ("F1 Macro", f"{mamdani['f1_macro']:.3f}", f"{anfis['f1_macro']:.3f}"),
    ]

    for name, m_val, a_val in rows:
        print(f"│ {name:<19s} │ {m_val:>12s} │ {a_val:>12s} │")

    print("└─────────────────────┴──────────────┴──────────────┘")
    print()
