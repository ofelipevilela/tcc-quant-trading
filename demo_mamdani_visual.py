#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualização do Processo de Inferência Mamdani.

Este script gera gráficos mostrando:
1. pertinências de entrada;
2. regras ativadas e seus cortes;
3. agregação da saída bidirecional;
4. defuzzificação por centroide em [-100, 100].

Execute: python demo_mamdani_visual.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from fuzzy.membership_functions import create_fuzzy_variables
from fuzzy.visualization import get_pertinence_values


def build_trade_signal_mfs():
    """Cria as MFs bidirecionais da saída."""
    signal_universe = np.linspace(-100, 100, 401)
    signal_mfs = {
        "Venda_Forte": fuzz.trapmf(signal_universe, [-100, -100, -75, -50]),
        "Venda": fuzz.trimf(signal_universe, [-75, -50, -10]),
        "Neutro": fuzz.trimf(signal_universe, [-20, 0, 20]),
        "Compra": fuzz.trimf(signal_universe, [10, 50, 75]),
        "Compra_Forte": fuzz.trapmf(signal_universe, [50, 75, 100, 100]),
    }
    return signal_universe, signal_mfs


def classify_signal(signal_value: float):
    """Classifica o sinal bidirecional usando apenas a saída."""
    if signal_value >= 60:
        return "COMPRA_FORTE", "COMPRA", 1.0
    if signal_value >= 15:
        return "COMPRA", "COMPRA", 0.5
    if signal_value <= -60:
        return "VENDA_FORTE", "VENDA", 1.0
    if signal_value <= -15:
        return "VENDA", "VENDA", 0.5
    return "NEUTRO", "NEUTRO", 0.0


def get_rule_activations(trend_pert, zone_pert, fvg_pert, sweep_pert):
    """Replica a base de regras bidirecional usada pelo sistema principal."""
    rules = [
        ("Compra_Forte", min(trend_pert.get("Alta", 0), zone_pert.get("Discount", 0), fvg_pert.get("Grande", 0), sweep_pert.get("Forte", 0)), "Alta+Discount+Grande+Forte"),
        ("Compra", min(trend_pert.get("Alta", 0), zone_pert.get("Discount", 0), fvg_pert.get("Padrao", 0), sweep_pert.get("Forte", 0)), "Alta+Discount+Padrao+Forte"),
        ("Compra", min(trend_pert.get("Alta", 0), zone_pert.get("Discount", 0), fvg_pert.get("Pequeno", 0), sweep_pert.get("Forte", 0)), "Alta+Discount+Pequeno+Forte"),
        ("Compra", min(trend_pert.get("Neutra", 0), zone_pert.get("Discount", 0), fvg_pert.get("Grande", 0), sweep_pert.get("Forte", 0)), "Neutra+Discount+Grande+Forte"),
        ("Compra", min(trend_pert.get("Alta", 0), zone_pert.get("Equilibrium", 0), fvg_pert.get("Grande", 0), sweep_pert.get("Forte", 0)), "Alta+Equilibrium+Grande+Forte"),
        ("Compra", min(trend_pert.get("Alta", 0), zone_pert.get("Equilibrium", 0), fvg_pert.get("Padrao", 0), sweep_pert.get("Forte", 0)), "Alta+Equilibrium+Padrao+Forte"),
        ("Compra", min(trend_pert.get("Alta", 0), zone_pert.get("Discount", 0), fvg_pert.get("Grande", 0), sweep_pert.get("Fraco", 0)), "Alta+Discount+Grande+Fraco"),
        ("Venda_Forte", min(trend_pert.get("Baixa", 0), zone_pert.get("Premium", 0), fvg_pert.get("Grande", 0), sweep_pert.get("Forte", 0)), "Baixa+Premium+Grande+Forte"),
        ("Venda", min(trend_pert.get("Baixa", 0), zone_pert.get("Premium", 0), fvg_pert.get("Padrao", 0), sweep_pert.get("Forte", 0)), "Baixa+Premium+Padrao+Forte"),
        ("Venda", min(trend_pert.get("Baixa", 0), zone_pert.get("Premium", 0), fvg_pert.get("Pequeno", 0), sweep_pert.get("Forte", 0)), "Baixa+Premium+Pequeno+Forte"),
        ("Venda", min(trend_pert.get("Neutra", 0), zone_pert.get("Premium", 0), fvg_pert.get("Grande", 0), sweep_pert.get("Forte", 0)), "Neutra+Premium+Grande+Forte"),
        ("Venda", min(trend_pert.get("Baixa", 0), zone_pert.get("Equilibrium", 0), fvg_pert.get("Grande", 0), sweep_pert.get("Forte", 0)), "Baixa+Equilibrium+Grande+Forte"),
        ("Venda", min(trend_pert.get("Baixa", 0), zone_pert.get("Equilibrium", 0), fvg_pert.get("Padrao", 0), sweep_pert.get("Forte", 0)), "Baixa+Equilibrium+Padrao+Forte"),
        ("Venda", min(trend_pert.get("Baixa", 0), zone_pert.get("Premium", 0), fvg_pert.get("Grande", 0), sweep_pert.get("Fraco", 0)), "Baixa+Premium+Grande+Fraco"),
        ("Neutro", min(trend_pert.get("Neutra", 0), zone_pert.get("Equilibrium", 0), fvg_pert.get("Grande", 0), sweep_pert.get("Forte", 0)), "Neutra+Equilibrium+Grande+Forte"),
        ("Neutro", min(sweep_pert.get("Fraco", 0), fvg_pert.get("Pequeno", 0)), "SweepFraco+FVGPequeno"),
        ("Neutro", min(trend_pert.get("Alta", 0), zone_pert.get("Premium", 0)), "Conflito_AltaPremium"),
        ("Neutro", min(trend_pert.get("Baixa", 0), zone_pert.get("Discount", 0)), "Conflito_BaixaDiscount"),
    ]
    return [(consequent, strength, label) for consequent, strength, label in rules if strength > 0.01]


def visualize_mamdani_inference(
    trend_strength: float,
    price_zone: float,
    fvg_quality: float,
    sweep_quality: float,
    scenario_name: str = "Cenário",
    save_path: str = None,
    show: bool = True,
):
    """
    Visualiza o processo completo de inferência Mamdani.
    """
    variables = create_fuzzy_variables()
    signal_universe, signal_mfs = build_trade_signal_mfs()

    trend_pert = get_pertinence_values(variables["trend_strength"], trend_strength)
    zone_pert = get_pertinence_values(variables["price_zone"], price_zone)
    fvg_pert = get_pertinence_values(variables["fvg_quality"], fvg_quality)
    sweep_pert = get_pertinence_values(variables["sweep_quality"], sweep_quality)

    print(f"\n{'=' * 60}")
    print(f"  INFERÊNCIA MAMDANI - {scenario_name}")
    print(f"{'=' * 60}")
    print("\nEntradas:")
    print(f"  Trend Strength: {trend_strength:+.0f}")
    print(f"  Price Zone: {price_zone:.2f}")
    print(f"  FVG Quality: {fvg_quality:.1f}")
    print(f"  Sweep Quality: {sweep_quality:.1f}")

    print("\nPertinências calculadas:")
    print(f"  Trend: {trend_pert}")
    print(f"  Zone: {zone_pert}")
    print(f"  FVG: {fvg_pert}")
    print(f"  Sweep: {sweep_pert}")

    activated_rules = get_rule_activations(trend_pert, zone_pert, fvg_pert, sweep_pert)

    print(f"\nRegras ativadas ({len(activated_rules)}):")
    for consequent, strength, rule_name in activated_rules:
        print(f"  • {rule_name} -> {consequent} (força: {strength:.3f})")

    aggregated = np.zeros_like(signal_universe)
    rule_cuts = []

    for consequent, strength, rule_name in activated_rules:
        mf = signal_mfs[consequent]
        cut = np.fmin(strength, mf)
        rule_cuts.append((consequent, strength, cut, rule_name))
        aggregated = np.fmax(aggregated, cut)

    centroid = fuzz.defuzz(signal_universe, aggregated, "centroid") if np.sum(aggregated) > 0 else 0.0
    classification, direction, position_size = classify_signal(centroid)

    print(f"\n{'─' * 40}")
    print(f"  CENTROIDE (defuzzificação): {centroid:+.2f}")
    print(f"{'─' * 40}")
    print(f"  Classificação: {classification}")
    print(f"  Direção: {direction}")
    print(f"  Position Size: {position_size:.1f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=120)

    colors = {
        "Venda_Forte": "#E63946",
        "Venda": "#F4A261",
        "Neutro": "#7F8C8D",
        "Compra": "#457B9D",
        "Compra_Forte": "#2A9D8F",
    }

    ax1 = axes[0, 0]
    for label, mf in signal_mfs.items():
        ax1.plot(signal_universe, mf, linewidth=2, label=label)
        ax1.fill_between(signal_universe, mf, alpha=0.1)
    ax1.set_title("Funções de Pertinência - Trade Signal", fontweight="bold")
    ax1.set_xlabel("Signal")
    ax1.set_ylabel("Pertinência (μ)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    for consequent, strength, cut, rule_name in rule_cuts:
        ax2.fill_between(
            signal_universe,
            cut,
            alpha=0.4,
            color=colors[consequent],
            label=f"{rule_name} ({strength:.2f})",
        )
    ax2.set_title("Regras Ativadas (Implicação)", fontweight="bold")
    ax2.set_xlabel("Signal")
    ax2.set_ylabel("Pertinência (μ)")
    if rule_cuts:
        ax2.legend(loc="upper right", fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)

    ax3 = axes[1, 0]
    ax3.fill_between(signal_universe, aggregated, alpha=0.5, color="purple", label="Área Agregada")
    ax3.plot(signal_universe, aggregated, "purple", linewidth=2)
    ax3.axvline(x=centroid, color="red", linestyle="--", linewidth=2.5, label=f"Centroide: {centroid:+.1f}")
    ax3.plot(centroid, 0, "rv", markersize=15)

    centroid_y = np.interp(centroid, signal_universe, aggregated)
    ax3.plot([centroid, centroid], [0, centroid_y], "r--", linewidth=2)
    ax3.plot(centroid, centroid_y, "ro", markersize=10)

    ax3.set_title("Agregação e Defuzzificação (Centroide)", fontweight="bold")
    ax3.set_xlabel("Trade Signal")
    ax3.set_ylabel("Pertinência (μ)")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.1)

    ax4 = axes[1, 1]
    ax4.axis("off")
    summary_text = f"""
    CENÁRIO: {scenario_name}
    {'─' * 40}

    ENTRADAS:
    • Trend Strength: {trend_strength:+.0f}
    • Price Zone: {price_zone:.2f}
    • FVG Quality: {fvg_quality:.1f}
    • Sweep Quality: {sweep_quality:.1f}

    {'─' * 40}

    RESULTADO:
    • Signal Final: {centroid:+.1f}
    • Classificação: {classification}
    • Direção: {direction}
    • Regras Ativadas: {len(activated_rules)}

    {'─' * 40}

    AÇÃO: {direction} | Position {position_size:.0%}
    """
    ax4.text(
        0.1,
        0.95,
        summary_text,
        transform=ax4.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.suptitle(f"Processo de Inferência Fuzzy Mamdani - {scenario_name}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

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
    print("  VISUALIZAÇÃO DO PROCESSO MAMDANI")
    print("=" * 70)

    scenarios = [
        {
            "name": "COMPRA_FORTE",
            "trend_strength": 80,
            "price_zone": 0.15,
            "fvg_quality": 2.8,
            "sweep_quality": 2.0,
        },
        {
            "name": "VENDA_FORTE",
            "trend_strength": -80,
            "price_zone": 0.85,
            "fvg_quality": 2.8,
            "sweep_quality": 2.0,
        },
        {
            "name": "NEUTRO_CONFLITO",
            "trend_strength": 60,
            "price_zone": 0.80,
            "fvg_quality": 1.5,
            "sweep_quality": 1.2,
        },
    ]

    output_dir = ROOT_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    for scenario in scenarios:
        save_path = output_dir / f"mamdani_{scenario['name'].lower()}.png"
        visualize_mamdani_inference(
            trend_strength=scenario["trend_strength"],
            price_zone=scenario["price_zone"],
            fvg_quality=scenario["fvg_quality"],
            sweep_quality=scenario["sweep_quality"],
            scenario_name=scenario["name"],
            save_path=str(save_path),
            show=True,
        )

    print("\n" + "=" * 70)
    print("  ✓ Visualizações geradas!")
    print("=" * 70)


if __name__ == "__main__":
    main()
