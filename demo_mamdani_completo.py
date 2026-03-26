#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualização Completa do Processo Mamdani.

Mostra em uma única figura:
- 4 painéis de entrada com pontos marcados
- 1 painel de regras ativadas
- 1 painel de agregação com centroide bidirecional

Execute: python demo_mamdani_completo.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from demo_mamdani_visual import build_trade_signal_mfs, classify_signal, get_rule_activations
from fuzzy.membership_functions import create_fuzzy_variables
from fuzzy.visualization import get_pertinence_values


def create_complete_mamdani_visualization(
    trend_strength: float,
    price_zone: float,
    fvg_quality: float,
    sweep_quality: float,
    scenario_name: str = "Cenário",
    save_path: str = None,
    show: bool = True,
):
    """Cria uma visualização completa do processo Mamdani bidirecional."""
    variables = create_fuzzy_variables()
    colors = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#7F8C8D"]

    fig = plt.figure(figsize=(16, 20), dpi=100)
    gs = fig.add_gridspec(6, 2, height_ratios=[1, 1, 1, 1, 1.2, 1.5], hspace=0.35, wspace=0.25)

    def plot_input_panel(ax, variable_key, crisp_value, title, xlabel):
        var = variables[variable_key]
        for index, (term_name, term_mf) in enumerate(var.terms.items()):
            color = colors[index % len(colors)]
            display_name = term_name.replace("_", " ").title()
            ax.plot(var.universe, term_mf.mf, linewidth=2, label=display_name, color=color)
            ax.fill_between(var.universe, term_mf.mf, alpha=0.15, color=color)

            membership = fuzz.interp_membership(var.universe, term_mf.mf, crisp_value)
            if membership > 0.01:
                ax.plot(crisp_value, membership, "o", color=color, markersize=10, markeredgecolor="black", markeredgewidth=1.5)
                ax.plot([crisp_value, crisp_value], [0, membership], "--", color=color, linewidth=1.5, alpha=0.7)

        ax.axvline(x=crisp_value, color="black", linestyle="--", linewidth=2, alpha=0.5)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("μ")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.15)

    plot_input_panel(fig.add_subplot(gs[0, 0]), "trend_strength", trend_strength, f"Trend Strength = {trend_strength:+.0f}", "Slope (-100 a +100)")
    plot_input_panel(fig.add_subplot(gs[0, 1]), "price_zone", price_zone, f"Price Zone = {price_zone:.2f}", "% do Range (0-1)")
    plot_input_panel(fig.add_subplot(gs[1, 0]), "fvg_quality", fvg_quality, f"FVG Quality = {fvg_quality:.1f}", "FVG Size / ATR")
    plot_input_panel(fig.add_subplot(gs[1, 1]), "sweep_quality", sweep_quality, f"Sweep Quality = {sweep_quality:.1f}", "Pavio / Corpo")

    trend_pert = get_pertinence_values(variables["trend_strength"], trend_strength)
    zone_pert = get_pertinence_values(variables["price_zone"], price_zone)
    fvg_pert = get_pertinence_values(variables["fvg_quality"], fvg_quality)
    sweep_pert = get_pertinence_values(variables["sweep_quality"], sweep_quality)

    signal_universe, signal_mfs = build_trade_signal_mfs()
    activated_rules = get_rule_activations(trend_pert, zone_pert, fvg_pert, sweep_pert)

    mf_colors = {
        "Venda_Forte": "#E63946",
        "Venda": "#F4A261",
        "Neutro": "#7F8C8D",
        "Compra": "#457B9D",
        "Compra_Forte": "#2A9D8F",
    }

    ax_rules = fig.add_subplot(gs[2:4, :])
    for label, mf in signal_mfs.items():
        ax_rules.fill_between(signal_universe, mf, alpha=0.08, color=mf_colors[label])
        ax_rules.plot(signal_universe, mf, linewidth=1, linestyle="--", alpha=0.45, color=mf_colors[label])

    aggregated = np.zeros_like(signal_universe)
    for consequent, activation, rule_name in activated_rules:
        cut = np.fmin(activation, signal_mfs[consequent])
        ax_rules.axhline(y=activation, color=mf_colors[consequent], linestyle=":", linewidth=1, alpha=0.7)
        ax_rules.fill_between(
            signal_universe,
            cut,
            alpha=0.4,
            color=mf_colors[consequent],
            label=f"{rule_name} -> {consequent} (α={activation:.2f})",
        )
        aggregated = np.fmax(aggregated, cut)

    ax_rules.set_title("Regras Ativadas e Implicação (Corte)", fontweight="bold", fontsize=12)
    ax_rules.set_xlabel("Trade Signal")
    ax_rules.set_ylabel("Pertinência (μ)")
    ax_rules.legend(loc="upper left", fontsize=8, ncol=2)
    ax_rules.grid(True, alpha=0.3)
    ax_rules.set_xlim(-100, 100)
    ax_rules.set_ylim(-0.05, 1.15)

    centroid = fuzz.defuzz(signal_universe, aggregated, "centroid") if np.sum(aggregated) > 0 else 0.0
    classification, direction, position_size = classify_signal(centroid)

    ax_agg = fig.add_subplot(gs[4:, :])
    ax_agg.fill_between(signal_universe, aggregated, alpha=0.6, color="purple", label="Área Agregada (B)")
    ax_agg.plot(signal_universe, aggregated, "purple", linewidth=2)

    centroid_y = np.interp(centroid, signal_universe, aggregated)
    ax_agg.axvline(x=centroid, color="red", linestyle="--", linewidth=3, label=f"Centroide: {centroid:+.1f}")
    ax_agg.plot(centroid, 0, "r^", markersize=20)
    ax_agg.plot(centroid, centroid_y, "ro", markersize=12, markeredgecolor="white", markeredgewidth=2)
    ax_agg.plot([centroid, centroid], [0, centroid_y], "r-", linewidth=2)

    result_text = f"SIGNAL: {centroid:+.1f}  |  {classification}  |  {direction}"
    if position_size > 0:
        result_text += f"  |  Position: {position_size:.0%}"
    else:
        result_text += "  |  Sem ação"

    ax_agg.text(
        0,
        0.95,
        result_text,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8),
        transform=ax_agg.get_xaxis_transform(),
    )

    ax_agg.set_title("Agregação (União) e Defuzzificação por Centroide", fontweight="bold", fontsize=12)
    ax_agg.set_xlabel("Trade Signal", fontsize=11)
    ax_agg.set_ylabel("Pertinência (μ)", fontsize=11)
    ax_agg.legend(loc="upper left", fontsize=10)
    ax_agg.grid(True, alpha=0.3)
    ax_agg.set_xlim(-100, 100)
    ax_agg.set_ylim(-0.05, 1.15)

    fig.suptitle(f"Processo Completo de Inferência Fuzzy Mamdani\n{scenario_name}", fontsize=16, fontweight="bold", y=0.995)
    fig.subplots_adjust(top=0.965, bottom=0.03, left=0.06, right=0.97)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"\n✓ Gráfico salvo em: {save_path}")

    if show:
        plt.show()

    return {
        "signal": centroid,
        "classification": classification,
        "direction": direction,
        "activated_rules": len(activated_rules),
    }


def main():
    print("=" * 70)
    print("  VISUALIZAÇÃO COMPLETA DO PROCESSO MAMDANI")
    print("=" * 70)

    output_dir = ROOT_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    print("\n\n>>> Cenário 1: COMPRA_FORTE")
    create_complete_mamdani_visualization(
        trend_strength=80,
        price_zone=0.15,
        fvg_quality=2.8,
        sweep_quality=2.0,
        scenario_name="COMPRA_FORTE",
        save_path=str(output_dir / "mamdani_completo_compra.png"),
        show=True,
    )

    print("\n\n>>> Cenário 2: VENDA_FORTE")
    create_complete_mamdani_visualization(
        trend_strength=-80,
        price_zone=0.85,
        fvg_quality=2.8,
        sweep_quality=2.0,
        scenario_name="VENDA_FORTE",
        save_path=str(output_dir / "mamdani_completo_venda.png"),
        show=True,
    )

    print("\n\n>>> Cenário 3: NEUTRO_CONFLITO")
    create_complete_mamdani_visualization(
        trend_strength=60,
        price_zone=0.80,
        fvg_quality=1.5,
        sweep_quality=1.0,
        scenario_name="NEUTRO_CONFLITO",
        save_path=str(output_dir / "mamdani_completo_neutro.png"),
        show=True,
    )

    print("\n" + "=" * 70)
    print("  ✓ Todas as visualizações geradas!")
    print("=" * 70)


if __name__ == "__main__":
    main()
