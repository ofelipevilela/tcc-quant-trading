# -*- coding: utf-8 -*-
"""
Metricas quantitativas para avaliacao do ANFIS em series financeiras.

Este modulo concentra metricas academicas de regressao e metricas direcionais
usadas para avaliar se o modelo neuro-fuzzy esta aprendendo informacao util do
target. As funcoes recebem arrays NumPy, listas ou pandas Series e retornam
estruturas simples para uso em notebooks, scripts de walk-forward e texto de
resultados do TCC.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ArrayLike = Sequence[float] | np.ndarray | pd.Series


@dataclass(frozen=True)
class PredictionMetrics:
    """
    Resultado consolidado da avaliacao de um vetor de predicoes.

    A classe existe para facilitar auditoria e tipagem durante a pesquisa, mas
    pode ser convertida diretamente para dicionario com ``to_dict()``.
    """

    n_samples: int
    n_directional_samples: int
    mae: float
    rmse: float
    r2: float
    directional_accuracy: float
    weighted_hit_rate: float

    def to_dict(self) -> Dict[str, float]:
        """Retorna as metricas em formato serializavel."""
        return {
            "n_samples": int(self.n_samples),
            "n_directional_samples": int(self.n_directional_samples),
            "mae": float(self.mae),
            "rmse": float(self.rmse),
            "r2": float(self.r2),
            "directional_accuracy": float(self.directional_accuracy),
            "weighted_hit_rate": float(self.weighted_hit_rate),
        }


def _to_1d_float_array(values: ArrayLike, name: str) -> np.ndarray:
    """
    Converte uma entrada numerica para ``np.ndarray`` unidimensional.

    Parameters
    ----------
    values:
        Serie, lista ou array contendo valores reais ou previstos.
    name:
        Nome usado nas mensagens de erro.

    Returns
    -------
    np.ndarray
        Array unidimensional de ``float``.
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} nao pode estar vazio.")
    return arr


def _prepare_arrays(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    drop_non_finite: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Valida e alinha os vetores de valores reais e previstos.

    Por padrao, observacoes com NaN ou infinito sao removidas dos dois vetores.
    Esse comportamento evita que uma unica barra invalida interrompa a avaliacao
    de uma janela experimental, mantendo o tamanho final registrado em
    ``n_samples``.
    """
    true_arr = _to_1d_float_array(y_true, "y_true")
    pred_arr = _to_1d_float_array(y_pred, "y_pred")

    if true_arr.shape[0] != pred_arr.shape[0]:
        raise ValueError(
            "y_true e y_pred devem ter o mesmo tamanho: "
            f"{true_arr.shape[0]} != {pred_arr.shape[0]}."
        )

    if drop_non_finite:
        finite_mask = np.isfinite(true_arr) & np.isfinite(pred_arr)
        true_arr = true_arr[finite_mask]
        pred_arr = pred_arr[finite_mask]

    if true_arr.size == 0:
        raise ValueError("Nenhuma observacao valida restou apos a limpeza.")

    return true_arr, pred_arr


def directional_accuracy(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    zero_tolerance: float = 0.0,
) -> float:
    """
    Calcula a acuracia direcional entre retorno real e retorno previsto.

    A metrica mede a proporcao de casos em que ``sign(y_pred)`` coincide com
    ``sign(y_true)``. Retornos reais com modulo menor ou igual a
    ``zero_tolerance`` sao ignorados, pois nao carregam direcao economica clara.

    Returns
    -------
    float
        Valor no intervalo ``[0, 1]``. Retorna ``np.nan`` se nao houver amostras
        direcionais validas.
    """
    true_arr, pred_arr = _prepare_arrays(y_true, y_pred)
    mask = np.abs(true_arr) > zero_tolerance
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.sign(true_arr[mask]) == np.sign(pred_arr[mask])))


def weighted_hit_rate(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    zero_tolerance: float = 0.0,
) -> float:
    """
    Calcula a taxa de acerto direcional ponderada pela magnitude do movimento.

    Diferente da acuracia direcional simples, esta metrica atribui mais peso a
    barras em que o retorno realizado foi maior em modulo. Assim, acertar a
    direcao de um movimento grande contribui mais do que acertar uma variacao
    marginal.

    Returns
    -------
    float
        Soma dos movimentos corretamente previstos dividida pela soma total dos
        movimentos avaliados. Retorna ``np.nan`` se nao houver denominador
        positivo.
    """
    true_arr, pred_arr = _prepare_arrays(y_true, y_pred)
    mask = np.abs(true_arr) > zero_tolerance
    if not np.any(mask):
        return float("nan")

    true_eval = true_arr[mask]
    pred_eval = pred_arr[mask]
    weights = np.abs(true_eval)
    correct = np.sign(true_eval) == np.sign(pred_eval)
    denominator = float(np.sum(weights))

    if denominator <= 0.0:
        return float("nan")
    return float(np.sum(weights[correct]) / denominator)


def evaluate_predictions(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    zero_tolerance: float = 0.0,
    as_dataframe: bool = False,
) -> Dict[str, float] | pd.DataFrame:
    """
    Avalia predicoes do ANFIS com metricas de regressao e direcionalidade.

    Parameters
    ----------
    y_true:
        Retornos reais do periodo avaliado.
    y_pred:
        Retornos previstos pelo ANFIS.
    zero_tolerance:
        Limite abaixo do qual o retorno real e tratado como sem direcao.
    as_dataframe:
        Se ``True``, retorna um DataFrame com uma linha. Caso contrario, retorna
        um dicionario simples.

    Returns
    -------
    dict or pandas.DataFrame
        Metrica principal de erro absoluto medio (MAE), RMSE, R2, acuracia
        direcional e hit rate ponderado.
    """
    true_arr, pred_arr = _prepare_arrays(y_true, y_pred)
    directional_mask = np.abs(true_arr) > zero_tolerance

    mae = float(mean_absolute_error(true_arr, pred_arr))
    rmse = float(np.sqrt(mean_squared_error(true_arr, pred_arr)))
    r2 = float(r2_score(true_arr, pred_arr))
    da = directional_accuracy(true_arr, pred_arr, zero_tolerance=zero_tolerance)
    whr = weighted_hit_rate(true_arr, pred_arr, zero_tolerance=zero_tolerance)

    metrics = PredictionMetrics(
        n_samples=int(true_arr.size),
        n_directional_samples=int(np.sum(directional_mask)),
        mae=mae,
        rmse=rmse,
        r2=r2,
        directional_accuracy=da,
        weighted_hit_rate=whr,
    ).to_dict()

    if as_dataframe:
        return pd.DataFrame([metrics])
    return metrics


def evaluate_train_validation(
    y_train_true: ArrayLike,
    y_train_pred: ArrayLike,
    y_val_true: ArrayLike,
    y_val_pred: ArrayLike,
    *,
    zero_tolerance: float = 0.0,
) -> pd.DataFrame:
    """
    Calcula uma tabela comparativa entre treino e validacao.

    Esta funcao e util para detectar sinais de sobreajuste: erro muito baixo no
    treino, erro alto na validacao e queda relevante de acuracia direcional fora
    da amostra.
    """
    rows = []
    for split_name, true_values, pred_values in (
        ("train", y_train_true, y_train_pred),
        ("validation", y_val_true, y_val_pred),
    ):
        metrics = evaluate_predictions(
            true_values,
            pred_values,
            zero_tolerance=zero_tolerance,
            as_dataframe=False,
        )
        metrics["split"] = split_name
        rows.append(metrics)

    columns = [
        "split",
        "n_samples",
        "n_directional_samples",
        "mae",
        "rmse",
        "r2",
        "directional_accuracy",
        "weighted_hit_rate",
    ]
    return pd.DataFrame(rows)[columns]


def plot_error_comparison(
    metrics_by_split: pd.DataFrame | Mapping[str, Mapping[str, float]],
    *,
    error_metrics: Iterable[str] = ("mae", "rmse"),
    output_path: Optional[str | Path] = None,
    title: str = "Erro de treino vs validacao",
) -> plt.Figure:
    """
    Plota barras comparando erros de treino e validacao.

    Parameters
    ----------
    metrics_by_split:
        DataFrame retornado por ``evaluate_train_validation`` ou dicionario no
        formato ``{"train": {...}, "validation": {...}}``.
    error_metrics:
        Metricas de erro que serao exibidas. Por padrao, MAE e RMSE.
    output_path:
        Caminho opcional para salvar o grafico.
    title:
        Titulo do grafico.

    Returns
    -------
    matplotlib.figure.Figure
        Figura criada para uso em notebooks ou salvamento posterior.
    """
    if isinstance(metrics_by_split, pd.DataFrame):
        if "split" not in metrics_by_split.columns:
            raise ValueError("DataFrame deve conter a coluna 'split'.")
        metrics_df = metrics_by_split.copy()
    else:
        rows = []
        for split, metrics in metrics_by_split.items():
            row = {"split": split}
            row.update(metrics)
            rows.append(row)
        metrics_df = pd.DataFrame(rows)

    selected_metrics = list(error_metrics)
    missing = [name for name in selected_metrics if name not in metrics_df.columns]
    if missing:
        raise ValueError(f"Metricas ausentes para plotagem: {missing}")

    plot_df = metrics_df.melt(
        id_vars="split",
        value_vars=selected_metrics,
        var_name="metric",
        value_name="value",
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(
        data=plot_df,
        x="metric",
        y="value",
        hue="split",
        palette={"train": "#2A9D8F", "validation": "#E76F51"},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Metrica")
    ax.set_ylabel("Erro")
    ax.legend(title="Janela")
    fig.tight_layout()

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")

    return fig


def plot_train_validation_errors(
    y_train_true: ArrayLike,
    y_train_pred: ArrayLike,
    y_val_true: ArrayLike,
    y_val_pred: ArrayLike,
    *,
    zero_tolerance: float = 0.0,
    output_path: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Calcula metricas de treino/validacao e gera grafico de erro.

    Returns
    -------
    tuple
        ``(metrics_df, figure)`` com a tabela de metricas e a figura gerada.
    """
    metrics_df = evaluate_train_validation(
        y_train_true,
        y_train_pred,
        y_val_true,
        y_val_pred,
        zero_tolerance=zero_tolerance,
    )
    fig = plot_error_comparison(metrics_df, output_path=output_path)
    return metrics_df, fig


class ANFISMetricsEvaluator:
    """
    Avaliador reutilizavel para experimentos ANFIS.

    Parameters
    ----------
    zero_tolerance:
        Limite usado para ignorar retornos realizados sem direcao economica
        clara no calculo de metricas direcionais.
    """

    def __init__(self, zero_tolerance: float = 0.0) -> None:
        self.zero_tolerance = float(zero_tolerance)

    def evaluate(self, y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, float]:
        """Calcula metricas para uma unica janela experimental."""
        return evaluate_predictions(
            y_true,
            y_pred,
            zero_tolerance=self.zero_tolerance,
            as_dataframe=False,
        )

    def evaluate_splits(
        self,
        y_train_true: ArrayLike,
        y_train_pred: ArrayLike,
        y_val_true: ArrayLike,
        y_val_pred: ArrayLike,
    ) -> pd.DataFrame:
        """Calcula tabela comparativa entre treino e validacao."""
        return evaluate_train_validation(
            y_train_true,
            y_train_pred,
            y_val_true,
            y_val_pred,
            zero_tolerance=self.zero_tolerance,
        )

    def plot_errors(
        self,
        metrics_by_split: pd.DataFrame | Mapping[str, Mapping[str, float]],
        *,
        output_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        """Gera grafico de MAE/RMSE para treino e validacao."""
        return plot_error_comparison(metrics_by_split, output_path=output_path)


__all__ = [
    "ANFISMetricsEvaluator",
    "PredictionMetrics",
    "directional_accuracy",
    "evaluate_predictions",
    "evaluate_train_validation",
    "plot_error_comparison",
    "plot_train_validation_errors",
    "weighted_hit_rate",
]
