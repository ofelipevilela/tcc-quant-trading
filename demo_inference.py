#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demonstração do Sistema Fuzzy Completo com Inferência Mamdani.

Este script demonstra o fluxo completo:
1. Fuzzificação dos inputs
2. Aplicação das regras bidirecionais
3. Agregação e defuzzificação (centroide)
4. Determinação de direção pela própria saída fuzzy
5. Position sizing (Lote cheio / Meio lote)

Execute: python demo_inference.py
"""

import sys
from pathlib import Path

# Adicionar o diretório raiz ao path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from fuzzy.fuzzy_system import create_fuzzy_system


def main():
    print("="*70)
    print("  SISTEMA FUZZY SMC - INFERÊNCIA MAMDANI COMPLETA")
    print("="*70)
    
    # Criar sistema
    system = create_fuzzy_system()
    
    # Mostrar regras
    system.print_rules_summary()
    
    # Definir cenários de teste
    # NOTA: trend_strength é BIPOLAR: -100 (baixa) a +100 (alta)
    # Esperamos sinais positivos para compra e negativos para venda.
    scenarios = {
        "COMPRA FORTE": {
            "trend_strength": 80,    # Tendência ALTA (bullish)
            "price_zone": 0.15,      # Discount
            "fvg_quality": 2.8,      # FVG grande
            "sweep_quality": 2.0,    # Sweep forte
        },
        "COMPRA": {
            "trend_strength": 70,
            "price_zone": 0.25,      # Discount
            "fvg_quality": 1.8,      # FVG padrão
            "sweep_quality": 1.8,
        },
        "COMPRA DE RISCO": {
            "trend_strength": 15,    # Tendência neutra/fraca
            "price_zone": 0.20,      # Discount compensa
            "fvg_quality": 2.5,      # FVG grande
            "sweep_quality": 1.5,
        },
        "VENDA FORTE": {
            "trend_strength": -85,   # Tendência BAIXA (bearish)
            "price_zone": 0.90,      # Premium
            "fvg_quality": 2.8,
            "sweep_quality": 2.0,
        },
        "VENDA": {
            "trend_strength": -70,
            "price_zone": 0.80,      # Premium
            "fvg_quality": 1.6,      # FVG padrão
            "sweep_quality": 1.8,
        },
        "NEUTRO (Equilibrium)": {
            "trend_strength": 0,     # Tendência neutra
            "price_zone": 0.50,      # Equilibrium
            "fvg_quality": 1.5,
            "sweep_quality": 1.2,
        },
        "CONFLITO (Alta em Premium)": {
            "trend_strength": 65,
            "price_zone": 0.80,
            "fvg_quality": 1.6,
            "sweep_quality": 1.2,
        },
        "SEM CONFIRMACAO": {
            "trend_strength": 10,
            "price_zone": 0.48,
            "fvg_quality": 0.5,      # FVG pequeno
            "sweep_quality": 0.3,    # Sweep fraco
        },
    }
    
    # Testar cada cenário
    print("\n" + "="*70)
    print("  RESULTADOS DA INFERÊNCIA")
    print("="*70)
    
    results = []
    for nome, cenario in scenarios.items():
        result = system.evaluate_scenario(cenario)
        results.append((nome, result))
        
        print(f"\n{'─'*60}")
        print(f"📊 Cenário: {nome}")
        print(f"{'─'*60}")
        print(f"   Inputs:")
        print(f"     • Trend Strength: {cenario['trend_strength']:+d}")
        print(f"     • Price Zone: {cenario['price_zone']:.2f}")
        print(f"     • FVG Quality: {cenario['fvg_quality']:.1f}")
        print(f"     • Sweep Quality: {cenario['sweep_quality']:.1f}")
        print(f"\n   ➜ SIGNAL: {result['signal']:+.1f} ({result['classification']})")
        print(f"   ➜ DIREÇÃO: {result['direction']}")
        print(f"   ➜ AÇÃO: {result['action']}")
        print(f"   ➜ POSITION SIZE: {result['position_size']:.1f}")
    
    # Resumo final
    print("\n" + "="*70)
    print("  RESUMO DAS ENTRADAS VÁLIDAS")
    print("="*70)
    print(f"\n{'Cenário':<25} {'Signal':>8} {'Direção':<10} {'Position':>10}")
    print("─"*60)
    
    for nome, result in results:
        if result['position_size'] > 0:
            print(f"{nome:<25} {result['signal']:>+8.1f} {result['direction']:<10} {result['position_size']:>10.1f}")
    
    print("\n" + "="*70)
    print("  ✓ Inferência completa!")
    print("="*70)


if __name__ == "__main__":
    main()
