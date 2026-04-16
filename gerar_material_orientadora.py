# -*- coding: utf-8 -*-
"""
Gerador de material visual para apresentação à orientadora.

Sinopse do processo completo:
  Mamdani (base de regras + MFs manuais)
  → Motivação para o ANFIS (testar se as MFs estavam boas)
  → ANFIS treinado com dados reais de mercado (USTEC.r M15)
  → Walk-forward 4 folds (avaliação fora da amostra)
  → Resultado: DirAcc 53.25%, PF 0.99, 3/4 folds positivos

Saída: outputs/material_orientadora.png (figura grande A3-like)
       outputs/material_orientadora_mfs.png (MFs antes vs depois, separado)
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import torch

# ── ajusta path para imports do projeto ──────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from anfis.anfis_model import ANFISModel
from anfis.config import (
    INPUT_VARS, INITIAL_MF_PARAMS, LINGUISTIC_SETS, UNIVERSES,
)
from anfis.rule_base import RuleBase

# ── estilo global ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 13,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

FOLD_COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
OUT_DIR = ROOT / 'outputs'
OUT_DIR.mkdir(exist_ok=True)

# ── constantes das variáveis ──────────────────────────────────────────────────
VAR_LABELS = {
    'trend_strength': 'Trend Strength  [-100, 100]',
    'price_zone':     'Price Zone  [0, 1]',
    'fvg_quality':    'FVG Quality  [0, 4]',
    'sweep_quality':  'Setup Phase   [0, 3]',
}
SET_COLORS = {
    'Baixa':       '#E63946',
    'Neutra':      '#457B9D',
    'Alta':        '#2A9D8F',
    'Discount':    '#E63946',
    'Equilibrium': '#457B9D',
    'Premium':     '#2A9D8F',
    'Pequeno':     '#E63946',
    'Padrao':      '#457B9D',
    'Grande':      '#2A9D8F',
    'Fraco':       '#E63946',
    'Forte':       '#2A9D8F',
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Helpers
# ─────────────────────────────────────────────────────────────────────────────

def gaussian(x, center, sigma):
    sigma = max(abs(sigma), 1e-6)
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def load_fold_model(fold: int, mode: str = 'causal_v3') -> ANFISModel:
    path = ROOT / 'outputs' / 'walkforward_compare' / mode / f'fold_{fold:02d}_model.pt'
    rule_base = RuleBase()
    model = ANFISModel(rule_base)
    state = torch.load(str(path), map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def extract_mf_params_from_model(model: ANFISModel) -> dict:
    """Extrai center e sigma de cada MF do modelo treinado."""
    params = {}
    for var_name in INPUT_VARS:
        params[var_name] = {}
        sets = LINGUISTIC_SETS[var_name]
        for i, set_name in enumerate(sets):
            mf = model.fuzzification.mfs[var_name][i]
            params[var_name][set_name] = {
                'center': mf.center.item(),
                'sigma':  abs(mf.sigma.item()),
            }
    return params


def average_fold_params(fold_params_list: list) -> dict:
    """Média dos parâmetros treinados em todos os folds."""
    avg = {}
    for var_name in INPUT_VARS:
        avg[var_name] = {}
        for set_name in LINGUISTIC_SETS[var_name]:
            centers = [fp[var_name][set_name]['center'] for fp in fold_params_list]
            sigmas  = [fp[var_name][set_name]['sigma']  for fp in fold_params_list]
            avg[var_name][set_name] = {
                'center': np.mean(centers),
                'sigma':  np.mean(sigmas),
            }
    return avg


def load_walkforward_results(mode: str = 'causal_v3') -> dict:
    path = ROOT / 'outputs' / 'walkforward_compare' / mode / 'walkforward_summary.json'
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Figura 1 — MFs antes vs depois do treinamento
# ─────────────────────────────────────────────────────────────────────────────

def plot_mfs_before_after(initial_params: dict, trained_params: dict, save_path: Path):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    fig.suptitle(
        'Funções de Pertinência: Inicialização (Mamdani) vs. Após Treinamento ANFIS\n'
        '(média dos 4 folds — causal_v3, USTEC.r M15)',
        fontsize=12, fontweight='bold', y=1.01,
    )

    for col, var_name in enumerate(INPUT_VARS):
        universe = UNIVERSES[var_name]
        x = np.linspace(universe[0], universe[1], 300)
        sets = LINGUISTIC_SETS[var_name]

        for row, params_dict in enumerate([initial_params, trained_params]):
            ax = axes[row, col]
            title_prefix = 'INICIAL' if row == 0 else 'TREINADO'
            ax.set_title(f'{title_prefix}\n{VAR_LABELS[var_name]}', fontsize=9)
            ax.set_ylim(-0.05, 1.15)
            ax.set_xlabel(var_name)
            ax.set_ylabel('Pertinência μ')

            for set_name in sets:
                p = params_dict[var_name][set_name]
                y = gaussian(x, p['center'], p['sigma'])
                color = SET_COLORS.get(set_name, 'gray')
                ax.plot(x, y, color=color, linewidth=2, label=set_name)
                ax.axvline(p['center'], color=color, linewidth=0.8, linestyle=':', alpha=0.7)

            ax.legend(loc='upper right', fontsize=7)
            ax.set_xlim(universe)

            # marcação de mudança para o gráfico treinado
            if row == 1:
                for set_name in sets:
                    pi = initial_params[var_name][set_name]
                    pt = trained_params[var_name][set_name]
                    delta = abs(pt['center'] - pi['center'])
                    if delta > 1.0:
                        ax.annotate(
                            f'Δ={delta:.1f}',
                            xy=(pt['center'], 0.5),
                            fontsize=7, color='black',
                            ha='center',
                            bbox=dict(boxstyle='round,pad=0.2', fc='lightyellow', alpha=0.8),
                        )

    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] MFs salvas em: {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 3. Figura 2 — Painel principal para a orientadora
# ─────────────────────────────────────────────────────────────────────────────

def plot_main_panel(wf_data: dict, save_path: Path):
    folds_data = wf_data['folds']
    aggregate = wf_data['aggregate']

    # métricas por fold
    fold_ids    = [f['fold'] for f in folds_data]
    pf_vals     = [f['test_profit_factor'] for f in folds_data]
    wr_vals     = [f['test_win_rate'] for f in folds_data]
    dir_vals    = [f['test_directional_accuracy'] * 100 for f in folds_data]
    trade_vals  = [f['test_total_trades'] for f in folds_data]
    periods     = [
        f"{f['test_start_time'][:7]}→{f['test_end_time'][:7]}"
        for f in folds_data
    ]
    equity_curves = [f['test_equity_curve'] for f in folds_data]

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#FAFAFA')

    gs = gridspec.GridSpec(
        4, 4, figure=fig,
        hspace=0.55, wspace=0.40,
        left=0.06, right=0.97, top=0.91, bottom=0.06,
    )

    # ── Cabeçalho de contexto ─────────────────────────────────────────────
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_axis_off()
    ax_title.text(
        0.5, 0.72,
        'ANFIS aplicado a Smart Money Concepts — Avaliação Walk-Forward',
        ha='center', va='center', fontsize=14, fontweight='bold',
        transform=ax_title.transAxes,
    )
    ax_title.text(
        0.5, 0.35,
        'Ativo: USTEC.r (Nasdaq 100 Fracionado) | Timeframe: M15 | '
        '4 folds temporais fora da amostra | Jun/2025 – Mar/2026\n'
        'Modo causal_v3: trend_strength em 3 componentes + máquina de estados Sweep→CISD→FVG (setup_phase)',
        ha='center', va='center', fontsize=9.5, color='#444',
        transform=ax_title.transAxes,
    )

    # ── Linha do tempo do walk-forward ────────────────────────────────────
    ax_timeline = fig.add_subplot(gs[1, :])
    ax_timeline.set_title('Estrutura Walk-Forward — Expanding Window', fontweight='bold')
    ax_timeline.set_xlim(0, 20000)
    ax_timeline.set_ylim(0, len(folds_data) + 0.5)
    ax_timeline.set_xlabel('Barras (M15, passo = 2000 barras ≈ 3 semanas)')
    ax_timeline.set_yticks([])
    ax_timeline.spines['left'].set_visible(False)

    for i, f in enumerate(folds_data):
        y = len(folds_data) - i
        train_end = f['train_samples']
        val_end   = train_end + f['val_samples']
        test_end  = val_end  + f['test_samples']
        pf = f['test_profit_factor']
        color_test = '#4CAF50' if pf >= 1.0 else '#E53935'

        ax_timeline.barh(y, train_end,          height=0.5, color='#90CAF9', label='Treino' if i == 0 else '')
        ax_timeline.barh(y, f['val_samples'],   height=0.5, color='#FFB74D', left=train_end, label='Validação' if i == 0 else '')
        ax_timeline.barh(y, f['test_samples'],  height=0.5, color=color_test, left=val_end,  label='Teste' if i == 0 else '', alpha=0.85)

        ax_timeline.text(
            val_end + f['test_samples'] / 2, y,
            f'Fold {i+1}\nPF={pf:.2f}',
            ha='center', va='center', fontsize=7.5, fontweight='bold', color='white',
        )
        ax_timeline.text(
            val_end + f['test_samples'] + 100, y,
            periods[i],
            ha='left', va='center', fontsize=7, color='#555',
        )

    ax_timeline.legend(loc='lower right', ncol=3, fontsize=8)
    ax_timeline.grid(axis='x', alpha=0.3)

    # ── Profit Factor por fold ─────────────────────────────────────────────
    ax_pf = fig.add_subplot(gs[2, 0])
    bar_colors = ['#4CAF50' if pf >= 1 else '#E53935' for pf in pf_vals]
    bars = ax_pf.bar(fold_ids, pf_vals, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax_pf.axhline(1.0, color='black', linewidth=1.2, linestyle='--', label='Breakeven (PF=1)')
    ax_pf.axhline(
        aggregate['mean_test_profit_factor'], color='navy',
        linewidth=1.2, linestyle=':', label=f'Média={aggregate["mean_test_profit_factor"]:.2f}',
    )
    for bar, val in zip(bars, pf_vals):
        ax_pf.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax_pf.set_title('Profit Factor por Fold', fontweight='bold')
    ax_pf.set_xlabel('Fold')
    ax_pf.set_ylabel('PF')
    ax_pf.set_xticks(fold_ids)
    ax_pf.legend(fontsize=7)
    ax_pf.set_ylim(0, max(pf_vals) * 1.25)

    # ── Acurácia Direcional ────────────────────────────────────────────────
    ax_dir = fig.add_subplot(gs[2, 1])
    bars2 = ax_dir.bar(fold_ids, dir_vals, color=FOLD_COLORS[:len(fold_ids)],
                       edgecolor='white', linewidth=0.5)
    ax_dir.axhline(50, color='red', linewidth=1.2, linestyle='--', label='Chance (50%)')
    ax_dir.axhline(
        aggregate['mean_test_directional_accuracy'] * 100, color='navy',
        linewidth=1.2, linestyle=':',
        label=f'Média={aggregate["mean_test_directional_accuracy"]*100:.1f}%',
    )
    for bar, val in zip(bars2, dir_vals):
        ax_dir.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax_dir.set_title('Acurácia Direcional por Fold', fontweight='bold')
    ax_dir.set_xlabel('Fold')
    ax_dir.set_ylabel('DirAcc (%)')
    ax_dir.set_xticks(fold_ids)
    ax_dir.set_ylim(40, 65)
    ax_dir.legend(fontsize=7)

    # ── Win Rate por fold ──────────────────────────────────────────────────
    ax_wr = fig.add_subplot(gs[2, 2])
    bars3 = ax_wr.bar(fold_ids, wr_vals, color=FOLD_COLORS[:len(fold_ids)],
                      edgecolor='white', linewidth=0.5)
    ax_wr.axhline(50, color='red', linewidth=1.2, linestyle='--', label='WR de chance')
    ax_wr.axhline(
        aggregate['mean_test_win_rate'], color='navy', linewidth=1.2, linestyle=':',
        label=f'Média={aggregate["mean_test_win_rate"]:.1f}%',
    )
    for bar, val in zip(bars3, wr_vals):
        ax_wr.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax_wr.set_title('Win Rate por Fold', fontweight='bold')
    ax_wr.set_xlabel('Fold')
    ax_wr.set_ylabel('WR (%)')
    ax_wr.set_xticks(fold_ids)
    ax_wr.set_ylim(40, 65)
    ax_wr.legend(fontsize=7)

    # ── Número de trades ──────────────────────────────────────────────────
    ax_trades = fig.add_subplot(gs[2, 3])
    bars4 = ax_trades.bar(fold_ids, trade_vals, color=FOLD_COLORS[:len(fold_ids)],
                          edgecolor='white', linewidth=0.5)
    ax_trades.axhline(
        aggregate['mean_test_total_trades'], color='navy', linewidth=1.2, linestyle=':',
        label=f'Média={aggregate["mean_test_total_trades"]:.0f}',
    )
    for bar, val in zip(bars4, trade_vals):
        ax_trades.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:d}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax_trades.set_title('Operações por Fold (teste)', fontweight='bold')
    ax_trades.set_xlabel('Fold')
    ax_trades.set_ylabel('Nº trades')
    ax_trades.set_xticks(fold_ids)
    ax_trades.legend(fontsize=7)

    # ── Curvas de equity por fold ─────────────────────────────────────────
    ax_equity = fig.add_subplot(gs[3, :2])
    ax_equity.set_title('Curvas de Capital — Período de Teste de Cada Fold', fontweight='bold')
    ax_equity.set_xlabel('Barras (período de teste, ≈ 2000 barras cada)')
    ax_equity.set_ylabel('Capital (R$)')
    ax_equity.axhline(100000, color='black', linewidth=0.8, linestyle='--', alpha=0.5, label='Capital inicial')

    for i, (eq, f) in enumerate(zip(equity_curves, folds_data)):
        pf = f['test_profit_factor']
        lw = 2 if pf >= 1 else 1
        ls = '-' if pf >= 1 else '--'
        ax_equity.plot(
            eq, color=FOLD_COLORS[i], linewidth=lw, linestyle=ls,
            label=f'Fold {i+1} (PF={pf:.2f})',
        )

    ax_equity.legend(fontsize=8)
    ax_equity.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f'R${x/1000:.0f}k')
    )

    # ── Comparativo de modos (tabela visual) ─────────────────────────────
    ax_compare = fig.add_subplot(gs[3, 2:])
    ax_compare.set_axis_off()
    ax_compare.set_title('Comparativo de Modos (Walk-Forward, 4 Folds, USTEC.r M15)',
                          fontweight='bold', loc='center')

    table_data = [
        ['Modo', 'PF médio', 'DirAcc', 'Folds PF≥1', 'Obs.'],
        ['legacy_like',       '1.44', '~61%', '4/4', '⚠ Look-ahead bias'],
        ['causal_bos_anchored', '1.12*', '~51%', '1/4', '* Fold outlier (1.82)'],
        ['causal_v3',         '0.99', '53.25%', '3/4', '✔ Melhor modo causal'],
        ['causal_v2',         '0.98', '48.12%', '1/4', ''],
        ['causal_raw',        '0.93', '~48%',   '1/4', ''],
    ]

    col_widths = [0.26, 0.14, 0.13, 0.15, 0.30]
    row_colors_map = {
        1: '#FFF9C4',  # legacy — amarelo (atenção)
        2: '#F5F5F5',
        3: '#C8E6C9',  # causal_v3 — verde
        4: '#F5F5F5',
        5: '#F5F5F5',
    }

    y_start = 0.97
    row_h = 0.155
    header = table_data[0]
    x_start = 0.01

    x_positions = [x_start]
    for w in col_widths[:-1]:
        x_positions.append(x_positions[-1] + w)

    for ci, (cell, x) in enumerate(zip(header, x_positions)):
        ax_compare.text(
            x + col_widths[ci]/2, y_start, cell,
            ha='center', va='top', fontsize=8.5, fontweight='bold',
            transform=ax_compare.transAxes,
            bbox=dict(boxstyle='round,pad=0.1', facecolor='#37474F', alpha=1),
            color='white',
        )

    for ri, row in enumerate(table_data[1:], start=1):
        y = y_start - ri * row_h
        bg = row_colors_map.get(ri, '#F5F5F5')
        ax_compare.add_patch(mpatches.FancyBboxPatch(
            (x_start - 0.005, y - row_h + 0.01), 0.995, row_h - 0.01,
            boxstyle='round,pad=0.005', facecolor=bg, edgecolor='#CCC',
            transform=ax_compare.transAxes, zorder=0,
        ))
        for ci, (cell, x) in enumerate(zip(row, x_positions)):
            fw = 'bold' if ri == 3 else 'normal'
            ax_compare.text(
                x + col_widths[ci]/2, y - 0.02, cell,
                ha='center', va='top', fontsize=8, fontweight=fw,
                transform=ax_compare.transAxes,
            )

    ax_compare.text(
        0.5, 0.01,
        'DirAcc = acurácia direcional (% do modelo acertar a direção do movimento)\n'
        'legacy_like usa preço de pivot com informação futura — limite superior teórico, não operacional.',
        ha='center', va='bottom', fontsize=7, color='#666', style='italic',
        transform=ax_compare.transAxes,
    )

    fig.savefig(str(save_path), dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'[OK] Painel principal salvo em: {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 4. Figura 3 — Fluxo do projeto (diagrama textual)
# ─────────────────────────────────────────────────────────────────────────────

def plot_project_flow(save_path: Path):
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)

    ax.text(5, 4.7, 'Fluxo do Projeto — TCC Engenharia Computacional',
            ha='center', va='top', fontsize=13, fontweight='bold')
    ax.text(5, 4.35, 'Sistema de Trading Algorítmico via Lógica Fuzzy e Neuro Fuzzy (ANFIS)',
            ha='center', va='top', fontsize=10, color='#444')

    steps = [
        {
            'x': 0.5, 'y': 3.2, 'w': 1.6, 'h': 1.2,
            'title': '① Mamdani',
            'body': '4 variáveis de entrada\n26 regras manuais\nSaída: Trade_Signal\n[-100, 100] via centroide',
            'color': '#BBDEFB', 'border': '#1565C0',
        },
        {
            'x': 2.5, 'y': 3.2, 'w': 1.6, 'h': 1.2,
            'title': '② Problema',
            'body': 'MFs definidas manualmente\nsem treino com mercado\nreal — como validar?\nNecessário: dados reais',
            'color': '#FFCCBC', 'border': '#BF360C',
        },
        {
            'x': 4.5, 'y': 3.2, 'w': 1.6, 'h': 1.2,
            'title': '③ ANFIS (TSK)',
            'body': '22 regras | 44 params\nGaussian MFs treináveis\nAdam | Backpropagation\nEarly stopping',
            'color': '#C8E6C9', 'border': '#1B5E20',
        },
        {
            'x': 6.5, 'y': 3.2, 'w': 1.6, 'h': 1.2,
            'title': '④ Treino Real',
            'body': 'USTEC.r M15\n20.000 barras\nTarget: barreira ATR\nHorizonte: 15 barras',
            'color': '#E1BEE7', 'border': '#4A148C',
        },
        {
            'x': 8.5, 'y': 3.2, 'w': 1.3, 'h': 1.2,
            'title': '⑤ Walk-Forward',
            'body': '4 folds\nExpanding window\nSem look-ahead\nFora da amostra',
            'color': '#F0F4C3', 'border': '#827717',
        },
    ]

    for s in steps:
        rect = mpatches.FancyBboxPatch(
            (s['x'], s['y']), s['w'], s['h'],
            boxstyle='round,pad=0.08',
            facecolor=s['color'], edgecolor=s['border'], linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(s['x'] + s['w']/2, s['y'] + s['h'] - 0.15,
                s['title'], ha='center', va='top',
                fontsize=9.5, fontweight='bold', color=s['border'])
        ax.text(s['x'] + s['w']/2, s['y'] + s['h'] - 0.38,
                s['body'], ha='center', va='top',
                fontsize=7.8, linespacing=1.4, color='#222')

    # setas
    arrow_ys = 3.8
    arrow_xs = [(2.1, 2.5), (4.1, 4.5), (6.1, 6.5), (8.1, 8.5)]
    for (x0, x1) in arrow_xs:
        ax.annotate('', xy=(x1, arrow_ys), xytext=(x0, arrow_ys),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=2))

    # caixa de resultados
    res_box_x, res_box_y = 3.0, 0.3
    res = mpatches.FancyBboxPatch(
        (res_box_x, res_box_y), 4.0, 2.3,
        boxstyle='round,pad=0.1',
        facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2.5,
    )
    ax.add_patch(res)
    ax.text(5.0, res_box_y + 2.1, '✔  Resultados — modo causal_v3 (sem look-ahead)',
            ha='center', va='top', fontsize=10, fontweight='bold', color='#1B5E20')

    metrics = [
        ('DirAcc',  '53.25%',  'Acurácia direcional — todos os 4 folds acima de 50%'),
        ('PF',      '0.99',    'Profit Factor médio — 3/4 folds positivos'),
        ('WR',      '50.31%',  'Win Rate médio nos períodos de teste'),
        ('Trades',  '191/fold','Média de operações por fold'),
    ]
    for i, (label, val, desc) in enumerate(metrics):
        y = res_box_y + 1.7 - i * 0.43
        ax.text(3.3, y, f'{label}:', ha='left', va='center', fontsize=8.5,
                fontweight='bold', color='#333')
        ax.text(4.3, y, val, ha='left', va='center', fontsize=10,
                fontweight='bold', color='#1B5E20')
        ax.text(5.4, y, f'—  {desc}', ha='left', va='center', fontsize=8, color='#555')

    # aviso look-ahead
    ax.text(
        5.0, 0.1,
        '⚠  look-ahead (legacy_like): PF=1.44, DirAcc≈61%, 4/4 folds — limite superior teórico com info futura de pivot',
        ha='center', va='bottom', fontsize=8, color='#8B4000', style='italic',
    )

    fig.savefig(str(save_path), dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'[OK] Fluxo do projeto salvo em: {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import logging
    logging.basicConfig(level=logging.WARNING)

    print('=' * 60)
    print('Gerando material visual para orientadora...')
    print('=' * 60)

    # 1. Carrega parâmetros iniciais (da config)
    initial_params = INITIAL_MF_PARAMS

    # 2. Carrega modelos treinados (4 folds, causal_v3) e extrai parâmetros
    print('\n[1/4] Carregando modelos treinados...')
    fold_params_list = []
    for fold in range(1, 5):
        try:
            model = load_fold_model(fold, 'causal_v3')
            fp = extract_mf_params_from_model(model)
            fold_params_list.append(fp)
            print(f'     Fold {fold}: OK')
        except Exception as e:
            print(f'     Fold {fold}: ERRO — {e}')

    if not fold_params_list:
        print('ERRO: nenhum modelo carregado. Verifique outputs/walkforward_compare/causal_v3/')
        sys.exit(1)

    trained_params = average_fold_params(fold_params_list)

    # 3. MFs antes vs depois
    print('\n[2/4] Gerando figura de MFs (antes vs depois)...')
    plot_mfs_before_after(
        initial_params, trained_params,
        OUT_DIR / 'material_orientadora_mfs.png',
    )

    # 4. Painel principal com walk-forward
    print('\n[3/4] Gerando painel principal (walk-forward)...')
    wf_data = load_walkforward_results('causal_v3')
    plot_main_panel(wf_data, OUT_DIR / 'material_orientadora_walkforward.png')

    # 5. Diagrama de fluxo do projeto
    print('\n[4/4] Gerando diagrama de fluxo do projeto...')
    plot_project_flow(OUT_DIR / 'material_orientadora_fluxo.png')

    print('\n' + '=' * 60)
    print('Material gerado em outputs/:')
    print('  material_orientadora_mfs.png         ← MFs: inicial vs treinado')
    print('  material_orientadora_walkforward.png  ← walk-forward + métricas')
    print('  material_orientadora_fluxo.png        ← fluxo do projeto')
    print('=' * 60)


if __name__ == '__main__':
    main()
