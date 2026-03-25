#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualização Completa do Processo Mamdani.

Mostra em uma única figura:
- 4 painéis de entrada com pontos marcados
- 1 painel de regras ativadas
- 1 painel de agregação com centroide

Execute: python demo_mamdani_completo.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from fuzzy.membership_functions import create_fuzzy_variables
from fuzzy.visualization import get_pertinence_values


def create_complete_mamdani_visualization(
    trend_strength: float,
    price_zone: float,
    fvg_quality: float,
    sweep_quality: float,
    scenario_name: str = "Cenário",
    save_path: str = None,
    show: bool = True
):
    """Cria visualização completa do processo Mamdani."""
    
    # Criar variáveis fuzzy
    variables = create_fuzzy_variables()
    
    # Cores
    colors = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#9B59B6']
    
    # Criar figura grande
    fig = plt.figure(figsize=(16, 20), dpi=100)
    
    # Layout: 6 linhas
    gs = fig.add_gridspec(6, 2, height_ratios=[1, 1, 1, 1, 1.2, 1.5], 
                          hspace=0.35, wspace=0.25)
    
    # =========================================================================
    # PAINEL 1: TREND STRENGTH
    # =========================================================================
    ax_trend = fig.add_subplot(gs[0, 0])
    var = variables['trend_strength']
    
    for i, (term_name, term_mf) in enumerate(var.terms.items()):
        color = colors[i % len(colors)]
        display_name = term_name.replace('_', ' ').title()
        ax_trend.plot(var.universe, term_mf.mf, linewidth=2, label=display_name, color=color)
        ax_trend.fill_between(var.universe, term_mf.mf, alpha=0.15, color=color)
        
        # Marcar pertinência
        membership = fuzz.interp_membership(var.universe, term_mf.mf, trend_strength)
        if membership > 0.01:
            ax_trend.plot(trend_strength, membership, 'o', color=color, markersize=10, 
                         markeredgecolor='black', markeredgewidth=1.5)
            ax_trend.plot([trend_strength, trend_strength], [0, membership], '--', 
                         color=color, linewidth=1.5, alpha=0.7)
    
    ax_trend.axvline(x=trend_strength, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax_trend.set_title(f'Trend Strength = {trend_strength:+.0f}', fontweight='bold', fontsize=11)
    ax_trend.set_xlabel('Slope (-100 a +100)')
    ax_trend.set_ylabel('μ')
    ax_trend.legend(loc='upper right', fontsize=8)
    ax_trend.grid(True, alpha=0.3)
    ax_trend.set_ylim(-0.05, 1.15)
    
    # =========================================================================
    # PAINEL 2: PRICE ZONE
    # =========================================================================
    ax_zone = fig.add_subplot(gs[0, 1])
    var = variables['price_zone']
    
    for i, (term_name, term_mf) in enumerate(var.terms.items()):
        color = colors[i % len(colors)]
        display_name = term_name.replace('_', ' ').title()
        ax_zone.plot(var.universe, term_mf.mf, linewidth=2, label=display_name, color=color)
        ax_zone.fill_between(var.universe, term_mf.mf, alpha=0.15, color=color)
        
        membership = fuzz.interp_membership(var.universe, term_mf.mf, price_zone)
        if membership > 0.01:
            ax_zone.plot(price_zone, membership, 'o', color=color, markersize=10,
                        markeredgecolor='black', markeredgewidth=1.5)
            ax_zone.plot([price_zone, price_zone], [0, membership], '--',
                        color=color, linewidth=1.5, alpha=0.7)
    
    ax_zone.axvline(x=price_zone, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax_zone.set_title(f'Price Zone = {price_zone:.2f}', fontweight='bold', fontsize=11)
    ax_zone.set_xlabel('% do Range (0-1)')
    ax_zone.set_ylabel('μ')
    ax_zone.legend(loc='upper right', fontsize=8)
    ax_zone.grid(True, alpha=0.3)
    ax_zone.set_ylim(-0.05, 1.15)
    
    # =========================================================================
    # PAINEL 3: FVG QUALITY
    # =========================================================================
    ax_fvg = fig.add_subplot(gs[1, 0])
    var = variables['fvg_quality']
    
    for i, (term_name, term_mf) in enumerate(var.terms.items()):
        color = colors[i % len(colors)]
        display_name = term_name.replace('_', ' ').title()
        ax_fvg.plot(var.universe, term_mf.mf, linewidth=2, label=display_name, color=color)
        ax_fvg.fill_between(var.universe, term_mf.mf, alpha=0.15, color=color)
        
        membership = fuzz.interp_membership(var.universe, term_mf.mf, fvg_quality)
        if membership > 0.01:
            ax_fvg.plot(fvg_quality, membership, 'o', color=color, markersize=10,
                       markeredgecolor='black', markeredgewidth=1.5)
            ax_fvg.plot([fvg_quality, fvg_quality], [0, membership], '--',
                       color=color, linewidth=1.5, alpha=0.7)
    
    ax_fvg.axvline(x=fvg_quality, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax_fvg.set_title(f'FVG Quality = {fvg_quality:.1f}', fontweight='bold', fontsize=11)
    ax_fvg.set_xlabel('FVG Size / ATR')
    ax_fvg.set_ylabel('μ')
    ax_fvg.legend(loc='upper right', fontsize=8)
    ax_fvg.grid(True, alpha=0.3)
    ax_fvg.set_ylim(-0.05, 1.15)
    
    # =========================================================================
    # PAINEL 4: SWEEP QUALITY
    # =========================================================================
    ax_sweep = fig.add_subplot(gs[1, 1])
    var = variables['sweep_quality']
    
    for i, (term_name, term_mf) in enumerate(var.terms.items()):
        color = colors[i % len(colors)]
        display_name = term_name.replace('_', ' ').title()
        ax_sweep.plot(var.universe, term_mf.mf, linewidth=2, label=display_name, color=color)
        ax_sweep.fill_between(var.universe, term_mf.mf, alpha=0.15, color=color)
        
        membership = fuzz.interp_membership(var.universe, term_mf.mf, sweep_quality)
        if membership > 0.01:
            ax_sweep.plot(sweep_quality, membership, 'o', color=color, markersize=10,
                         markeredgecolor='black', markeredgewidth=1.5)
            ax_sweep.plot([sweep_quality, sweep_quality], [0, membership], '--',
                         color=color, linewidth=1.5, alpha=0.7)
    
    ax_sweep.axvline(x=sweep_quality, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax_sweep.set_title(f'Sweep Quality = {sweep_quality:.1f}', fontweight='bold', fontsize=11)
    ax_sweep.set_xlabel('Pavio / Corpo')
    ax_sweep.set_ylabel('μ')
    ax_sweep.legend(loc='upper right', fontsize=8)
    ax_sweep.grid(True, alpha=0.3)
    ax_sweep.set_ylim(-0.05, 1.15)
    
    # =========================================================================
    # CALCULAR ATIVAÇÕES DAS REGRAS
    # =========================================================================
    trend_pert = get_pertinence_values(variables['trend_strength'], trend_strength)
    zone_pert = get_pertinence_values(variables['price_zone'], price_zone)
    fvg_pert = get_pertinence_values(variables['fvg_quality'], fvg_quality)
    sweep_pert = get_pertinence_values(variables['sweep_quality'], sweep_quality)
    
    # Universo de saída
    score_universe = np.linspace(0, 100, 201)
    score_fraco = fuzz.trapmf(score_universe, [0, 0, 15, 30])
    score_moderado = fuzz.trapmf(score_universe, [20, 35, 50, 65])
    score_forte = fuzz.trapmf(score_universe, [55, 70, 80, 90])
    score_muito_forte = fuzz.trapmf(score_universe, [80, 90, 100, 100])
    
    mf_map = {
        'Fraco': (score_fraco, '#E63946'),
        'Moderado': (score_moderado, '#457B9D'),
        'Forte': (score_forte, '#2A9D8F'),
        'Muito_Forte': (score_muito_forte, '#F4A261')
    }
    
    # Calcular ativações
    activated_rules = []
    
    # Regras principais (simplificado)
    rules_to_check = [
        ('Muito_Forte', min(trend_pert.get('Alta', 0), zone_pert.get('Deep_Discount', 0), 
                           fvg_pert.get('Grande', 0), sweep_pert.get('Forte', 0)), 'Alta+DeepDisc+Grande+Forte'),
        ('Muito_Forte', min(trend_pert.get('Alta', 0), zone_pert.get('Discount', 0),
                           fvg_pert.get('Grande', 0), sweep_pert.get('Forte', 0)), 'Alta+Discount+Grande+Forte'),
        ('Forte', min(trend_pert.get('Alta', 0), zone_pert.get('Deep_Discount', 0),
                     fvg_pert.get('Padrao', 0), sweep_pert.get('Forte', 0)), 'Alta+DeepDisc+Padrao+Forte'),
        ('Forte', min(trend_pert.get('Alta', 0), zone_pert.get('Discount', 0),
                     fvg_pert.get('Padrao', 0), sweep_pert.get('Forte', 0)), 'Alta+Discount+Padrao+Forte'),
        ('Moderado', min(trend_pert.get('Neutra', 0), zone_pert.get('Deep_Discount', 0),
                        fvg_pert.get('Grande', 0), sweep_pert.get('Forte', 0)), 'Neutra+DeepDisc+Grande+Forte'),
        ('Muito_Forte', min(trend_pert.get('Baixa', 0), zone_pert.get('Deep_Premium', 0),
                           fvg_pert.get('Grande', 0), sweep_pert.get('Forte', 0)), 'Baixa+DeepPrem+Grande+Forte'),
        ('Muito_Forte', min(trend_pert.get('Baixa', 0), zone_pert.get('Premium', 0),
                           fvg_pert.get('Grande', 0), sweep_pert.get('Forte', 0)), 'Baixa+Premium+Grande+Forte'),
        ('Forte', min(trend_pert.get('Baixa', 0), zone_pert.get('Deep_Premium', 0),
                     fvg_pert.get('Padrao', 0), sweep_pert.get('Forte', 0)), 'Baixa+DeepPrem+Padrao+Forte'),
        ('Fraco', zone_pert.get('Equilibrium', 0), 'Equilibrium'),
        ('Fraco', min(sweep_pert.get('Fraco', 0), fvg_pert.get('Pequeno', 0)), 'SweepFraco+FVGPequeno'),
    ]
    
    for consequent, activation, rule_name in rules_to_check:
        if activation > 0.01:
            activated_rules.append((consequent, activation, rule_name))
    
    # =========================================================================
    # PAINEL 5: REGRAS ATIVADAS (IMPLICAÇÃO)
    # =========================================================================
    ax_rules = fig.add_subplot(gs[2:4, :])
    
    # Plot MFs de saída em cinza claro
    ax_rules.fill_between(score_universe, score_fraco, alpha=0.1, color='gray')
    ax_rules.fill_between(score_universe, score_moderado, alpha=0.1, color='gray')
    ax_rules.fill_between(score_universe, score_forte, alpha=0.1, color='gray')
    ax_rules.fill_between(score_universe, score_muito_forte, alpha=0.1, color='gray')
    
    ax_rules.plot(score_universe, score_fraco, 'gray', linewidth=1, linestyle='--', alpha=0.5)
    ax_rules.plot(score_universe, score_moderado, 'gray', linewidth=1, linestyle='--', alpha=0.5)
    ax_rules.plot(score_universe, score_forte, 'gray', linewidth=1, linestyle='--', alpha=0.5)
    ax_rules.plot(score_universe, score_muito_forte, 'gray', linewidth=1, linestyle='--', alpha=0.5)
    
    # Agregação
    aggregated = np.zeros_like(score_universe)
    
    for consequent, activation, rule_name in activated_rules:
        mf, color = mf_map[consequent]
        cut = np.fmin(activation, mf)
        
        # Linha horizontal de corte
        ax_rules.axhline(y=activation, color=color, linestyle=':', linewidth=1, alpha=0.7)
        
        # Área cortada
        ax_rules.fill_between(score_universe, cut, alpha=0.4, color=color, 
                             label=f'{rule_name} → {consequent} (α={activation:.2f})')
        
        # Agregar
        aggregated = np.fmax(aggregated, cut)
    
    ax_rules.set_title('Regras Ativadas e Implicação (Corte)', fontweight='bold', fontsize=12)
    ax_rules.set_xlabel('Trade Score')
    ax_rules.set_ylabel('Pertinência (μ)')
    ax_rules.legend(loc='upper left', fontsize=8, ncol=2)
    ax_rules.grid(True, alpha=0.3)
    ax_rules.set_xlim(0, 100)
    ax_rules.set_ylim(-0.05, 1.15)
    
    # Adicionar labels dos conjuntos
    ax_rules.text(7, 1.05, 'Fraco', ha='center', fontsize=9, color='gray')
    ax_rules.text(42, 1.05, 'Moderado', ha='center', fontsize=9, color='gray')
    ax_rules.text(72, 1.05, 'Forte', ha='center', fontsize=9, color='gray')
    ax_rules.text(92, 1.05, 'Muito_Forte', ha='center', fontsize=9, color='gray')
    
    # =========================================================================
    # PAINEL 6: AGREGAÇÃO E CENTROIDE
    # =========================================================================
    ax_agg = fig.add_subplot(gs[4:, :])
    
    # Área agregada
    ax_agg.fill_between(score_universe, aggregated, alpha=0.6, color='purple', 
                        label='Área Agregada (B)')
    ax_agg.plot(score_universe, aggregated, 'purple', linewidth=2)
    
    # Calcular centroide
    if np.sum(aggregated) > 0:
        centroid = fuzz.defuzz(score_universe, aggregated, 'centroid')
    else:
        centroid = 0.0
    
    # Marcar centroide
    centroid_y = np.interp(centroid, score_universe, aggregated)
    ax_agg.axvline(x=centroid, color='red', linestyle='--', linewidth=3, 
                   label=f'Centroide: {centroid:.1f}')
    ax_agg.plot(centroid, 0, 'r^', markersize=20)
    ax_agg.plot(centroid, centroid_y, 'ro', markersize=12, markeredgecolor='white', markeredgewidth=2)
    
    # Linha do centroide até o ponto
    ax_agg.plot([centroid, centroid], [0, centroid_y], 'r-', linewidth=2)
    
    # Classificação
    if centroid >= 80:
        classification = "MUITO_FORTE"
        position = 1.0
    elif centroid >= 60:
        classification = "FORTE"
        position = 1.0
    elif centroid >= 40:
        classification = "MODERADO"
        position = 0.5
    else:
        classification = "FRACO"
        position = 0.0
    
    # Direção
    if trend_strength > 0 and price_zone < 0.5:
        direction = "COMPRA"
    elif trend_strength < 0 and price_zone > 0.5:
        direction = "VENDA"
    else:
        direction = "INDEFINIDO"
    
    # Caixa de resultado
    result_text = f'SCORE: {centroid:.1f}/100  |  {classification}  |  {direction}'
    if position > 0:
        result_text += f'  |  Position: {position:.0%}'
    
    ax_agg.text(50, 0.95, result_text, ha='center', va='top', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    ax_agg.set_title('Agregação (União) e Defuzzificação por Centroide', fontweight='bold', fontsize=12)
    ax_agg.set_xlabel('Trade Score', fontsize=11)
    ax_agg.set_ylabel('Pertinência (μ)', fontsize=11)
    ax_agg.legend(loc='upper left', fontsize=10)
    ax_agg.grid(True, alpha=0.3)
    ax_agg.set_xlim(0, 100)
    ax_agg.set_ylim(-0.05, 1.15)
    
    # Título geral
    fig.suptitle(f'Processo Completo de Inferência Fuzzy Mamdani\n{scenario_name}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
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
    print("  VISUALIZAÇÃO COMPLETA DO PROCESSO MAMDANI")
    print("="*70)
    
    output_dir = ROOT_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Cenário 1: COMPRA MUITO FORTE
    print("\n\n>>> Cenário 1: COMPRA MUITO FORTE")
    create_complete_mamdani_visualization(
        trend_strength=80,
        price_zone=0.12,
        fvg_quality=2.8,
        sweep_quality=2.0,
        scenario_name="COMPRA MUITO FORTE",
        save_path=str(output_dir / "mamdani_completo_compra.png"),
        show=True
    )
    
    # Cenário 2: VENDA FORTE
    print("\n\n>>> Cenário 2: VENDA FORTE")
    create_complete_mamdani_visualization(
        trend_strength=-75,
        price_zone=0.85,
        fvg_quality=1.8,
        sweep_quality=1.8,
        scenario_name="VENDA FORTE",
        save_path=str(output_dir / "mamdani_completo_venda.png"),
        show=True
    )
    
    # Cenário 3: NEUTRO
    print("\n\n>>> Cenário 3: NEUTRO (EQUILIBRIUM)")
    create_complete_mamdani_visualization(
        trend_strength=5,
        price_zone=0.50,
        fvg_quality=1.0,
        sweep_quality=1.0,
        scenario_name="NEUTRO (EQUILIBRIUM)",
        save_path=str(output_dir / "mamdani_completo_neutro.png"),
        show=True
    )
    
    print("\n" + "="*70)
    print("  ✓ Todas as visualizações geradas!")
    print("="*70)


if __name__ == "__main__":
    main()
