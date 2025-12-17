# -*- coding: utf-8 -*-
"""
Visualização das Funções de Pertinência (Membership Functions) - SMC System.

Este módulo gera gráficos profissionais das MFs para validação teórica
e apresentação do TCC.

Variáveis visualizadas:
- Trend_Strength: Gaussianas
- Price_Zone: Trapezoidais (5 zonas)
- FVG_Quality: Triangulares + Sigmoidal
- Sweep_Quality: Sigmoidais
- Trade_Score: Trapezoidais (saída)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Dict, List, Optional
from skfuzzy import control as ctrl

from config.settings import VISUALIZATION_CONFIG, ALL_FUZZY_CONFIGS


def _setup_plot_style() -> None:
    """Configura o estilo global dos gráficos."""
    available_styles = plt.style.available
    
    # Tentar usar seaborn style
    if 'seaborn-v0_8-whitegrid' in available_styles:
        plt.style.use('seaborn-v0_8-whitegrid')
    elif 'seaborn-whitegrid' in available_styles:
        plt.style.use('seaborn-whitegrid')
    else:
        plt.style.use('default')
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def _plot_single_variable(
    ax: Axes,
    variable: ctrl.Antecedent | ctrl.Consequent,
    colors: List[str],
    title: str
) -> None:
    """
    Plota as funções de pertinência de uma única variável.
    
    Args:
        ax: Eixo do matplotlib para plotar
        variable: Variável fuzzy (Antecedente ou Consequente)
        colors: Lista de cores para as MFs
        title: Título do subplot
    """
    for i, (term_name, term_mf) in enumerate(variable.terms.items()):
        color = colors[i % len(colors)]
        # Formatar nome para exibição
        display_name = term_name.replace('_', ' ').title()
        
        ax.plot(
            variable.universe,
            term_mf.mf,
            linewidth=2.5,
            label=display_name,
            color=color
        )
        ax.fill_between(
            variable.universe,
            term_mf.mf,
            alpha=0.15,
            color=color
        )
    
    ax.set_title(title, fontweight='bold', pad=10)
    ax.set_ylabel('Pertinência (μ)')
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_membership_functions(
    variables: Dict[str, ctrl.Antecedent | ctrl.Consequent],
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Gera visualização completa de todas as funções de pertinência SMC.
    
    Cria um gráfico com subplots para cada variável fuzzy, mostrando
    suas funções de pertinência de forma clara e profissional.
    
    Args:
        variables: Dicionário com as variáveis fuzzy
        save_path: Caminho para salvar o gráfico (None = não salva)
        show: Se True, exibe o gráfico interativamente
        
    Returns:
        Figura do matplotlib para manipulação adicional se necessário
        
    Example:
        >>> from fuzzy import create_fuzzy_variables, plot_membership_functions
        >>> variables = create_fuzzy_variables()
        >>> fig = plot_membership_functions(variables, save_path="outputs/mfs.png")
    """
    _setup_plot_style()
    
    colors = VISUALIZATION_CONFIG.colors
    
    # Configurações dos subplots - SMC Variables
    variable_info = [
        ('trend_strength', 'Trend Strength (Força da Tendência)', 'ADX / Slope EMA'),
        ('price_zone', 'Price Zone (Zona de Preço)', '% do Range (0=Fundo, 1=Topo)'),
        ('fvg_quality', 'FVG Quality (Qualidade do Fair Value Gap)', 'FVG Size / ATR'),
        ('sweep_quality', 'Sweep Quality (Captura de Liquidez)', 'Razão Pavio / Corpo'),
        ('trade_score', 'Trade Score (Score do Setup) - SAÍDA', 'Score (0-100)'),
    ]
    
    # Criar figura com subplots (3 linhas x 2 colunas)
    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=VISUALIZATION_CONFIG.figure_size,
        dpi=VISUALIZATION_CONFIG.dpi
    )
    axes = axes.flatten()
    
    # Plotar cada variável
    for idx, (var_key, title, xlabel) in enumerate(variable_info):
        if var_key in variables:
            _plot_single_variable(axes[idx], variables[var_key], colors, title)
            axes[idx].set_xlabel(xlabel)
    
    # Esconder o último subplot (não usado)
    axes[-1].set_visible(False)
    
    # Título principal
    fig.suptitle(
        'Funções de Pertinência - Sistema Híbrido SMC + Fuzzy Logic (Mamdani)',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    # Ajustar layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Adicionar nota de rodapé
    fig.text(
        0.5, 0.01,
        'TCC: Trading Quantitativo com Smart Money Concepts e Lógica Fuzzy | '
        'Variáveis: Trend, Price Zone, FVG, Sweep → Trade Score',
        ha='center',
        fontsize=9,
        style='italic',
        alpha=0.7
    )
    
    # Salvar se especificado
    if save_path:
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=VISUALIZATION_CONFIG.dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"✓ Gráfico salvo em: {save_path}")
    
    # Mostrar se especificado
    if show:
        plt.show()
    
    return fig


def plot_single_mf(
    variable: ctrl.Antecedent | ctrl.Consequent,
    title: str,
    xlabel: str,
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Plota as MFs de uma única variável fuzzy.
    
    Útil para análise isolada de uma variável específica.
    
    Args:
        variable: Variável fuzzy a ser plotada
        title: Título do gráfico
        xlabel: Label do eixo X
        save_path: Caminho para salvar (opcional)
        show: Se True, exibe o gráfico
        
    Returns:
        Figura do matplotlib
    """
    _setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=VISUALIZATION_CONFIG.dpi)
    _plot_single_variable(ax, variable, VISUALIZATION_CONFIG.colors, title)
    ax.set_xlabel(xlabel)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=VISUALIZATION_CONFIG.dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"✓ Gráfico salvo em: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def create_detailed_report(
    variables: Dict[str, ctrl.Antecedent | ctrl.Consequent],
    save_dir: str = "outputs"
) -> None:
    """
    Gera um relatório detalhado com gráficos individuais de cada variável.
    
    Args:
        variables: Dicionário com as variáveis fuzzy
        save_dir: Diretório para salvar os gráficos
    """
    os.makedirs(save_dir, exist_ok=True)
    
    variable_metadata = {
        'trend_strength': ('Trend Strength', 'ADX / Slope EMA'),
        'price_zone': ('Price Zone', '% do Range'),
        'fvg_quality': ('FVG Quality', 'FVG Size / ATR'),
        'sweep_quality': ('Sweep Quality', 'Pavio / Corpo'),
        'trade_score': ('Trade Score (Output)', 'Score'),
    }
    
    for var_key, var in variables.items():
        if var_key in variable_metadata:
            title, xlabel = variable_metadata[var_key]
            save_path = os.path.join(save_dir, f"mf_{var_key}.png")
            plot_single_mf(var, title, xlabel, save_path=save_path, show=False)
            print(f"  ✓ {var_key}")
