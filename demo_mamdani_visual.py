#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualização do Processo de Inferência Mamdani.

Este script gera gráficos mostrando:
1. Ativação das regras fuzzy
2. Agregação das saídas (união das geometrias)
3. Defuzzificação por centroide

Execute: python demo_mamdani_visual.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# Adicionar o diretório raiz ao path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from fuzzy.membership_functions import create_fuzzy_variables
from fuzzy.visualization import get_pertinence_values
from config.settings import TRADE_SCORE_CONFIG


def visualize_mamdani_inference(
    trend_strength: float,
    price_zone: float,
    fvg_quality: float,
    sweep_quality: float,
    scenario_name: str = "Cenário",
    save_path: str = None,
    show: bool = True
):
    """
    Visualiza o processo completo de inferência Mamdani.
    
    Mostra:
    - Gráfico 1: Pertinências de entrada
    - Gráfico 2: Regras ativadas e seus cortes
    - Gráfico 3: Agregação e centroide
    """
    # Criar variáveis
    variables = create_fuzzy_variables()
    
    # Universo de saída
    score_universe = np.linspace(0, 100, 201)
    
    # Definir MFs de saída manualmente para visualização
    score_fraco = fuzz.trapmf(score_universe, [0, 0, 15, 30])
    score_moderado = fuzz.trapmf(score_universe, [20, 35, 50, 65])
    score_forte = fuzz.trapmf(score_universe, [55, 70, 80, 90])
    score_muito_forte = fuzz.trapmf(score_universe, [80, 90, 100, 100])
    
    # Calcular pertinências das entradas
    trend_pert = get_pertinence_values(variables['trend_strength'], trend_strength)
    zone_pert = get_pertinence_values(variables['price_zone'], price_zone)
    fvg_pert = get_pertinence_values(variables['fvg_quality'], fvg_quality)
    sweep_pert = get_pertinence_values(variables['sweep_quality'], sweep_quality)
    
    print(f"\n{'='*60}")
    print(f"  INFERÊNCIA MAMDANI - {scenario_name}")
    print(f"{'='*60}")
    print(f"\nEntradas:")
    print(f"  Trend Strength: {trend_strength:+.0f}")
    print(f"  Price Zone: {price_zone:.2f}")
    print(f"  FVG Quality: {fvg_quality:.1f}")
    print(f"  Sweep Quality: {sweep_quality:.1f}")
    
    print(f"\nPertinências calculadas:")
    print(f"  Trend: {trend_pert}")
    print(f"  Zone: {zone_pert}")
    print(f"  FVG: {fvg_pert}")
    print(f"  Sweep: {sweep_pert}")
    
    # =========================================================================
    # APLICAR REGRAS (simplificado - principais regras)
    # =========================================================================
    
    activated_rules = []
    
    # Regra: Alta + Deep_Discount + Grande + Forte → Muito_Forte
    activation = min(
        trend_pert.get('Alta', 0),
        zone_pert.get('Deep_Discount', 0),
        fvg_pert.get('Grande', 0),
        sweep_pert.get('Forte', 0)
    )
    if activation > 0.01:
        activated_rules.append(('Muito_Forte', activation, 'Alta+DeepDisc+Grande+Forte'))
    
    # Regra: Alta + Discount + Grande + Forte → Muito_Forte
    activation = min(
        trend_pert.get('Alta', 0),
        zone_pert.get('Discount', 0),
        fvg_pert.get('Grande', 0),
        sweep_pert.get('Forte', 0)
    )
    if activation > 0.01:
        activated_rules.append(('Muito_Forte', activation, 'Alta+Discount+Grande+Forte'))
    
    # Regra: Alta + Deep_Discount + Padrao + Forte → Forte
    activation = min(
        trend_pert.get('Alta', 0),
        zone_pert.get('Deep_Discount', 0),
        fvg_pert.get('Padrao', 0),
        sweep_pert.get('Forte', 0)
    )
    if activation > 0.01:
        activated_rules.append(('Forte', activation, 'Alta+DeepDisc+Padrao+Forte'))
    
    # Regra: Alta + Discount + Padrao + Forte → Forte
    activation = min(
        trend_pert.get('Alta', 0),
        zone_pert.get('Discount', 0),
        fvg_pert.get('Padrao', 0),
        sweep_pert.get('Forte', 0)
    )
    if activation > 0.01:
        activated_rules.append(('Forte', activation, 'Alta+Discount+Padrao+Forte'))
    
    # Regra: Neutra + Deep_Discount + Grande + Forte → Moderado
    activation = min(
        trend_pert.get('Neutra', 0),
        zone_pert.get('Deep_Discount', 0),
        fvg_pert.get('Grande', 0),
        sweep_pert.get('Forte', 0)
    )
    if activation > 0.01:
        activated_rules.append(('Moderado', activation, 'Neutra+DeepDisc+Grande+Forte'))
    
    # Regras de VENDA
    # Regra: Baixa + Deep_Premium + Grande + Forte → Muito_Forte
    activation = min(
        trend_pert.get('Baixa', 0),
        zone_pert.get('Deep_Premium', 0),
        fvg_pert.get('Grande', 0),
        sweep_pert.get('Forte', 0)
    )
    if activation > 0.01:
        activated_rules.append(('Muito_Forte', activation, 'Baixa+DeepPrem+Grande+Forte'))
    
    # Regra: Baixa + Premium + Grande + Forte → Muito_Forte
    activation = min(
        trend_pert.get('Baixa', 0),
        zone_pert.get('Premium', 0),
        fvg_pert.get('Grande', 0),
        sweep_pert.get('Forte', 0)
    )
    if activation > 0.01:
        activated_rules.append(('Muito_Forte', activation, 'Baixa+Premium+Grande+Forte'))
    
    # Regra: Baixa + Deep_Premium + Padrao + Forte → Forte
    activation = min(
        trend_pert.get('Baixa', 0),
        zone_pert.get('Deep_Premium', 0),
        fvg_pert.get('Padrao', 0),
        sweep_pert.get('Forte', 0)
    )
    if activation > 0.01:
        activated_rules.append(('Forte', activation, 'Baixa+DeepPrem+Padrao+Forte'))
    
    # Regra fallback: Equilibrium → Fraco
    activation = zone_pert.get('Equilibrium', 0)
    if activation > 0.01:
        activated_rules.append(('Fraco', activation, 'Equilibrium'))
    
    # Regra fallback: Fraco + Pequeno → Fraco
    activation = min(sweep_pert.get('Fraco', 0), fvg_pert.get('Pequeno', 0))
    if activation > 0.01:
        activated_rules.append(('Fraco', activation, 'SweepFraco+FVGPequeno'))
    
    print(f"\nRegras ativadas ({len(activated_rules)}):")
    for consequent, strength, rule_name in activated_rules:
        print(f"  • {rule_name} → {consequent} (força: {strength:.3f})")
    
    # =========================================================================
    # AGREGAÇÃO - Cortar e unir as MFs de saída
    # =========================================================================
    
    # Inicializar agregação como zeros
    aggregated = np.zeros_like(score_universe)
    
    # Armazenar cortes individuais para visualização
    rule_cuts = []
    
    for consequent, strength, rule_name in activated_rules:
        if consequent == 'Fraco':
            mf = score_fraco
        elif consequent == 'Moderado':
            mf = score_moderado
        elif consequent == 'Forte':
            mf = score_forte
        else:  # Muito_Forte
            mf = score_muito_forte
        
        # Cortar a MF pelo nível de ativação (implicação Mamdani)
        cut = np.fmin(strength, mf)
        rule_cuts.append((consequent, strength, cut, rule_name))
        
        # Agregar (união = máximo)
        aggregated = np.fmax(aggregated, cut)
    
    # =========================================================================
    # DEFUZZIFICAÇÃO - Calcular centroide
    # =========================================================================
    
    if np.sum(aggregated) > 0:
        centroid = fuzz.defuzz(score_universe, aggregated, 'centroid')
    else:
        centroid = 0.0
    
    print(f"\n{'─'*40}")
    print(f"  CENTROIDE (defuzzificação): {centroid:.2f}")
    print(f"{'─'*40}")
    
    # Classificar
    if centroid >= 80:
        classification = "MUITO_FORTE"
    elif centroid >= 60:
        classification = "FORTE"
    elif centroid >= 40:
        classification = "MODERADO"
    else:
        classification = "FRACO"
    
    # Direção
    if trend_strength > 0 and price_zone < 0.5:
        direction = "COMPRA"
    elif trend_strength < 0 and price_zone > 0.5:
        direction = "VENDA"
    else:
        direction = "INDEFINIDO"
    
    print(f"  Classificação: {classification}")
    print(f"  Direção: {direction}")
    
    # =========================================================================
    # VISUALIZAÇÃO
    # =========================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=120)
    
    colors = {
        'Fraco': '#E63946',
        'Moderado': '#457B9D',
        'Forte': '#2A9D8F',
        'Muito_Forte': '#F4A261'
    }
    
    # --- Subplot 1: MFs de Saída (Trade Score) ---
    ax1 = axes[0, 0]
    ax1.plot(score_universe, score_fraco, 'r-', linewidth=2, label='Fraco')
    ax1.plot(score_universe, score_moderado, 'b-', linewidth=2, label='Moderado')
    ax1.plot(score_universe, score_forte, 'g-', linewidth=2, label='Forte')
    ax1.plot(score_universe, score_muito_forte, 'orange', linewidth=2, label='Muito_Forte')
    ax1.fill_between(score_universe, score_fraco, alpha=0.1, color='red')
    ax1.fill_between(score_universe, score_moderado, alpha=0.1, color='blue')
    ax1.fill_between(score_universe, score_forte, alpha=0.1, color='green')
    ax1.fill_between(score_universe, score_muito_forte, alpha=0.1, color='orange')
    ax1.set_title('Funções de Pertinência - Trade Score', fontweight='bold')
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Pertinência (μ)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # --- Subplot 2: Regras Ativadas (cortes) ---
    ax2 = axes[0, 1]
    for consequent, strength, cut, rule_name in rule_cuts:
        ax2.fill_between(score_universe, cut, alpha=0.4, 
                        color=colors[consequent], label=f'{rule_name} ({strength:.2f})')
    ax2.set_title('Regras Ativadas (Implicação)', fontweight='bold')
    ax2.set_xlabel('Score')
    ax2.set_ylabel('Pertinência (μ)')
    if len(rule_cuts) <= 6:
        ax2.legend(loc='upper right', fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    # --- Subplot 3: Agregação e Centroide ---
    ax3 = axes[1, 0]
    ax3.fill_between(score_universe, aggregated, alpha=0.5, color='purple', label='Área Agregada')
    ax3.plot(score_universe, aggregated, 'purple', linewidth=2)
    
    # Marcar centroide
    ax3.axvline(x=centroid, color='red', linestyle='--', linewidth=2.5, label=f'Centroide: {centroid:.1f}')
    ax3.plot(centroid, 0, 'rv', markersize=15)  # Triângulo vermelho
    
    # Área do centroide
    centroid_y = np.interp(centroid, score_universe, aggregated)
    ax3.plot([centroid, centroid], [0, centroid_y], 'r--', linewidth=2)
    ax3.plot(centroid, centroid_y, 'ro', markersize=10)
    
    ax3.set_title('Agregação e Defuzzificação (Centroide)', fontweight='bold')
    ax3.set_xlabel('Trade Score')
    ax3.set_ylabel('Pertinência (μ)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.1)
    
    # --- Subplot 4: Resumo do Cenário ---
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    CENÁRIO: {scenario_name}
    {'─'*40}
    
    ENTRADAS:
    • Trend Strength: {trend_strength:+.0f}
    • Price Zone: {price_zone:.2f}
    • FVG Quality: {fvg_quality:.1f}
    • Sweep Quality: {sweep_quality:.1f}
    
    {'─'*40}
    
    RESULTADO:
    • Score Final: {centroid:.1f} / 100
    • Classificação: {classification}
    • Direção: {direction}
    • Regras Ativadas: {len(activated_rules)}
    
    {'─'*40}
    
    AÇÃO: {direction} ({classification})
    """
    
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Título geral
    fig.suptitle(f'Processo de Inferência Fuzzy Mamdani - {scenario_name}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n✓ Gráfico salvo em: {save_path}")
    
    if show:
        plt.show()
    
    return {
        'score': centroid,
        'classification': classification,
        'direction': direction,
        'activated_rules': len(activated_rules)
    }


def main():
    print("="*70)
    print("  VISUALIZAÇÃO DO PROCESSO MAMDANI")
    print("="*70)
    
    # Cenários de teste
    scenarios = [
        {
            'name': 'COMPRA MUITO FORTE',
            'trend_strength': 80,
            'price_zone': 0.12,
            'fvg_quality': 2.8,
            'sweep_quality': 2.0
        },
        {
            'name': 'VENDA FORTE',
            'trend_strength': -75,
            'price_zone': 0.85,
            'fvg_quality': 1.8,
            'sweep_quality': 1.8
        },
        {
            'name': 'NEUTRO (Equilibrium)',
            'trend_strength': 5,
            'price_zone': 0.50,
            'fvg_quality': 1.0,
            'sweep_quality': 1.0
        }
    ]
    
    output_dir = ROOT_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    for scenario in scenarios:
        save_path = output_dir / f"mamdani_{scenario['name'].replace(' ', '_').lower()}.png"
        
        visualize_mamdani_inference(
            trend_strength=scenario['trend_strength'],
            price_zone=scenario['price_zone'],
            fvg_quality=scenario['fvg_quality'],
            sweep_quality=scenario['sweep_quality'],
            scenario_name=scenario['name'],
            save_path=str(save_path),
            show=True
        )
    
    print("\n" + "="*70)
    print("  ✓ Visualizações geradas!")
    print("="*70)


if __name__ == "__main__":
    main()
