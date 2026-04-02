# -*- coding: utf-8 -*-
"""
Visualizações para o TCC — Comparativo ANFIS.

Gera 6 plots acadêmicos com matplotlib:
1. MFs antes vs depois do treinamento
2. Curva de aprendizado (loss + IC)
3. Comparativo de consequentes (barras iniciais vs aprendidos)
4. Matriz de confusão (Mamdani vs ANFIS)
5. Análise de quantis (monotonia da magnitude)
6. Distribuição dos sinais gerados vs targets
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # Backend não-interativo para salvar sem display

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .config import (
    INPUT_VARS,
    LINGUISTIC_SETS,
    OUTPUT_DIR,
    SEMANTIC_ORDER,
    SIGNAL_THRESHOLDS,
    UNIVERSES,
)

logger = logging.getLogger(__name__)

# Estilo acadêmico
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
})

COLORS = {
    'VENDA_FORTE': '#c0392b',
    'VENDA': '#e74c3c',
    'NEUTRO': '#95a5a6',
    'COMPRA': '#27ae60',
    'COMPRA_FORTE': '#1e8449',
}

MF_COLORS = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A']


def _ensure_output_dir(output_dir: str = OUTPUT_DIR) -> Path:
    """Cria diretório de saída se necessário."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_mf_before_after(
    initial_params: Dict,
    trained_params: Dict,
    output_dir: str = OUTPUT_DIR,
) -> str:
    """
    Plot 1 — Funções de pertinência: antes vs depois do ANFIS.

    Parameters
    ----------
    initial_params : dict
        Parâmetros iniciais das MFs.
    trained_params : dict
        Parâmetros após treinamento.
    output_dir : str
        Diretório de saída.

    Returns
    -------
    str
        Caminho do arquivo salvo.
    """
    out_path = _ensure_output_dir(output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    fig.suptitle('Funções de Pertinência: Inicialização vs Treinadas (ANFIS)',
                 fontweight='bold')

    for idx, var_name in enumerate(INPUT_VARS):
        ax = axes[idx // 2][idx % 2]
        lo, hi = UNIVERSES[var_name]
        x = np.linspace(lo, hi, 300)
        sets = LINGUISTIC_SETS[var_name]

        for j, set_name in enumerate(sets):
            color = MF_COLORS[j % len(MF_COLORS)]

            # Inicial (pontilhado cinza)
            p_init = initial_params[var_name][set_name]
            y_init = np.exp(-0.5 * ((x - p_init['center']) / p_init['sigma']) ** 2)
            ax.plot(x, y_init, '--', color='gray', alpha=0.6, linewidth=1.2)

            # Treinada (sólido colorido)
            p_trained = trained_params[var_name][set_name]
            y_trained = np.exp(-0.5 * ((x - p_trained['center']) / p_trained['sigma']) ** 2)
            ax.plot(x, y_trained, '-', color=color, linewidth=2.0, label=set_name)

        ax.set_title(var_name.replace('_', ' ').title())
        ax.set_xlabel(f'[{lo}, {hi}]')
        ax.set_ylabel('μ(x)')
        ax.set_ylim(-0.05, 1.1)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = str(out_path / 'plot1_mf_before_after.png')
    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Plot 1 salvo: {filepath}")
    return filepath


def plot_learning_curve(
    history: Dict[str, List],
    output_dir: str = OUTPUT_DIR,
) -> str:
    """
    Plot 2 — Curva de aprendizado.

    Train loss e validation loss por época (escala log no eixo y).
    IC de validação por época no eixo secundário.
    Marcação do ponto de early stopping.

    Parameters
    ----------
    history : dict
        Histórico retornado pelo AdamTrainer.
    output_dir : str
        Diretório de saída.

    Returns
    -------
    str
        Caminho do arquivo salvo.
    """
    out_path = _ensure_output_dir(output_dir)

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=150)

    epochs = range(len(history['train_loss']))

    # Loss no eixo primário (escala log)
    ax1.semilogy(epochs, history['train_loss'], 'b-', alpha=0.7, label='Train Loss')
    ax1.semilogy(epochs, history['val_loss'], 'r-', alpha=0.7, label='Val Loss')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss (log scale)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)

    # IC no eixo secundário
    ax2 = ax1.twinx()
    ax2.plot(epochs, history['val_ic'], 'g--', alpha=0.6, label='Val IC (Spearman)')
    ax2.set_ylabel('IC (Spearman)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Marcar ponto de early stopping (última época)
    last_epoch = len(history['train_loss']) - 1
    ax1.axvline(x=last_epoch, color='orange', linestyle=':', alpha=0.5,
                label=f'Early Stop (ep. {last_epoch})')

    # Legendas combinadas
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    fig.suptitle('Curva de Aprendizado — ANFIS', fontweight='bold')
    plt.tight_layout()

    filepath = str(out_path / 'plot2_learning_curve.png')
    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Plot 2 salvo: {filepath}")
    return filepath


def plot_consequent_comparison(
    initial_values: List[float],
    trained_values: List[float],
    rule_descriptions: List[str],
    output_dir: str = OUTPUT_DIR,
) -> str:
    """
    Plot 3 — Comparativo de consequentes.

    Gráfico de barras: valor inicial vs valor aprendido de cada cᵢ.

    Parameters
    ----------
    initial_values : list of float
        Valores iniciais dos consequentes.
    trained_values : list of float
        Valores aprendidos dos consequentes.
    rule_descriptions : list of str
        Descrições curtas de cada regra.
    output_dir : str
        Diretório de saída.

    Returns
    -------
    str
        Caminho do arquivo salvo.
    """
    out_path = _ensure_output_dir(output_dir)

    n_rules = len(initial_values)
    y_pos = np.arange(n_rules)
    bar_height = 0.35

    fig, ax = plt.subplots(figsize=(10, max(6, n_rules * 0.35)), dpi=150)

    bars1 = ax.barh(y_pos - bar_height / 2, initial_values,
                     bar_height, label='Inicial', color='#95a5a6', alpha=0.7)
    bars2 = ax.barh(y_pos + bar_height / 2, trained_values,
                     bar_height, label='Treinado (ANFIS)', color='#3498db', alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'R{i}' for i in range(n_rules)], fontsize=8)
    ax.set_xlabel('Valor do Consequente (cᵢ)')
    ax.set_title('Consequentes TSK: Inicial vs Aprendido', fontweight='bold')
    ax.legend(loc='lower right')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')

    # Banda de cores por classe semântica
    for cls_name, (lo, hi) in SIGNAL_THRESHOLDS.items():
        color = COLORS.get(cls_name, '#cccccc')
        ax.axvspan(lo, hi, alpha=0.05, color=color)

    plt.tight_layout()
    filepath = str(out_path / 'plot3_consequent_comparison.png')
    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Plot 3 salvo: {filepath}")
    return filepath


def plot_confusion_matrices(
    cm_mamdani: np.ndarray,
    cm_anfis: np.ndarray,
    output_dir: str = OUTPUT_DIR,
) -> str:
    """
    Plot 4 — Matrizes de confusão lado a lado.

    Parameters
    ----------
    cm_mamdani : np.ndarray
        Confusion matrix do Mamdani.
    cm_anfis : np.ndarray
        Confusion matrix do ANFIS.
    output_dir : str
        Diretório de saída.

    Returns
    -------
    str
        Caminho do arquivo salvo.
    """
    out_path = _ensure_output_dir(output_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    labels_short = ['VF', 'V', 'N', 'C', 'CF']

    # Normalizar para percentuais
    cm_m_pct = cm_mamdani.astype(float)
    row_sums = cm_m_pct.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_m_pct = cm_m_pct / row_sums * 100

    cm_a_pct = cm_anfis.astype(float)
    row_sums = cm_a_pct.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_a_pct = cm_a_pct / row_sums * 100

    sns.heatmap(cm_m_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=labels_short, yticklabels=labels_short,
                ax=ax1, cbar_kws={'label': '%'})
    ax1.set_title('Mamdani (baseline)', fontweight='bold')
    ax1.set_xlabel('Predito')
    ax1.set_ylabel('Real')

    sns.heatmap(cm_a_pct, annot=True, fmt='.1f', cmap='Greens',
                xticklabels=labels_short, yticklabels=labels_short,
                ax=ax2, cbar_kws={'label': '%'})
    ax2.set_title('ANFIS (treinado)', fontweight='bold')
    ax2.set_xlabel('Predito')
    ax2.set_ylabel('Real')

    fig.suptitle('Matrizes de Confusão — 5 Classes', fontweight='bold', y=1.02)
    plt.tight_layout()

    filepath = str(out_path / 'plot4_confusion_matrices.png')
    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Plot 4 salvo: {filepath}")
    return filepath


def plot_quantile_analysis(
    quantile_data: Dict,
    output_dir: str = OUTPUT_DIR,
) -> str:
    """
    Plot 5 — Análise de quantis (validação da magnitude).

    Barras horizontais: retorno médio por quintil de sinal.

    Parameters
    ----------
    quantile_data : dict
        Dados de quantis do evaluate.py.
    output_dir : str
        Diretório de saída.

    Returns
    -------
    str
        Caminho do arquivo salvo.
    """
    out_path = _ensure_output_dir(output_dir)

    quantiles = quantile_data.get('quantiles', [])
    mean_returns = quantile_data.get('mean_returns', [])
    is_monotonic = quantile_data.get('is_monotonic', False)

    if not quantiles:
        logger.warning("Sem dados de quantis para plotar.")
        return ''

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    colors = ['#c0392b', '#e74c3c', '#95a5a6', '#27ae60', '#1e8449']
    n_q = len(quantiles)
    bar_colors = colors[:n_q] if n_q <= len(colors) else ['#3498db'] * n_q

    bars = ax.barh(quantiles, mean_returns, color=bar_colors, alpha=0.8, edgecolor='white')

    ax.set_xlabel('Retorno Médio do Target')
    ax.set_title(
        f'Análise de Quantis — Monotonia: {"✓" if is_monotonic else "✗"}',
        fontweight='bold',
    )
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')

    # Anotações nos barras
    for bar, val in zip(bars, mean_returns):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}', va='center', fontsize=9)

    plt.tight_layout()
    filepath = str(out_path / 'plot5_quantile_analysis.png')
    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Plot 5 salvo: {filepath}")
    return filepath


def plot_signal_distribution(
    anfis_preds: np.ndarray,
    targets: np.ndarray,
    output_dir: str = OUTPUT_DIR,
) -> str:
    """
    Plot 6 — Distribuição dos sinais gerados vs targets.

    Parameters
    ----------
    anfis_preds : np.ndarray
        Predições do ANFIS.
    targets : np.ndarray
        Targets reais.
    output_dir : str
        Diretório de saída.

    Returns
    -------
    str
        Caminho do arquivo salvo.
    """
    out_path = _ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    bins = np.linspace(-100, 100, 41)

    ax.hist(targets.flatten(), bins=bins, alpha=0.5, color='#3498db',
            label='Target', density=True, edgecolor='white')
    ax.hist(anfis_preds.flatten(), bins=bins, alpha=0.5, color='#e74c3c',
            label='ANFIS', density=True, edgecolor='white')

    # Bandas de classe
    for cls_name, (lo, hi) in SIGNAL_THRESHOLDS.items():
        color = COLORS.get(cls_name, '#cccccc')
        ax.axvspan(lo, hi, alpha=0.05, color=color)

    ax.set_xlabel('Sinal')
    ax.set_ylabel('Densidade')
    ax.set_title('Distribuição dos Sinais: ANFIS vs Target', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = str(out_path / 'plot6_signal_distribution.png')
    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Plot 6 salvo: {filepath}")
    return filepath


def generate_all_plots(
    initial_mf_params: Dict,
    trained_mf_params: Dict,
    history: Dict,
    initial_consequents: List[float],
    trained_consequents: List[float],
    rule_descriptions: List[str],
    cm_mamdani: np.ndarray,
    cm_anfis: np.ndarray,
    quantile_data: Dict,
    anfis_preds: np.ndarray,
    targets: np.ndarray,
    output_dir: str = OUTPUT_DIR,
) -> List[str]:
    """
    Gera todos os 6 plots do TCC.

    Returns
    -------
    list of str
        Caminhos dos arquivos salvos.
    """
    logger.info("Gerando todos os plots...")

    paths = [
        plot_mf_before_after(initial_mf_params, trained_mf_params, output_dir),
        plot_learning_curve(history, output_dir),
        plot_consequent_comparison(initial_consequents, trained_consequents,
                                   rule_descriptions, output_dir),
        plot_confusion_matrices(cm_mamdani, cm_anfis, output_dir),
        plot_quantile_analysis(quantile_data, output_dir),
        plot_signal_distribution(anfis_preds, targets, output_dir),
    ]

    logger.info(f"Todos os {len(paths)} plots gerados em {output_dir}/")
    return paths
