#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demonstração do Sistema Fuzzy SMC com Exemplos Anotados.

Este script gera visualizações com exemplos concretos mostrando
como valores crisp são mapeados para graus de pertinência.

Execute: python demo_examples.py
"""

import sys
from pathlib import Path

# Adicionar o diretório raiz ao path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from fuzzy import (
    create_fuzzy_variables,
    plot_with_examples,
    print_pertinence_table,
)


def main():
    print("  DEMONSTRAÇÃO - Sistema Fuzzy SMC com Exemplos Anotados")
    print()
    
    # Criar variáveis fuzzy
    print("Criando variáveis fuzzy...")
    variables = create_fuzzy_variables()
    print(f"  ✓ {len(variables)} variáveis criadas\n")
    
    # Definir cenários de exemplo para demonstração
    # NOTA: trend_strength agora é bipolar: -100 (baixa) a +100 (alta)
    scenarios = {
        "compra_forte": {
            "trend_strength": 75,    # Tendência ALTA (positivo = bullish)
            "price_zone": 0.15,      # Deep discount
            "fvg_quality": 2.5,      # FVG grande
            "sweep_quality": 2.0,    # Sweep forte
        },
        "neutro": {
            "trend_strength": 0,     # Tendência NEUTRA (zero = sem direção)
            "price_zone": 0.5,       # Equilibrium
            "fvg_quality": 1.0,      # FVG pequeno
            "sweep_quality": 0.5,    # Sweep fraco
        },
        "venda_forte": {
            "trend_strength": -80,   # Tendência BAIXA (negativo = bearish)
            "price_zone": 0.9,       # Deep premium
            "fvg_quality": 2.0,      # FVG padrão/grande
            "sweep_quality": 1.8,    # Sweep forte
        },
    }
    
    # Imprimir tabelas de pertinência para cada cenário
    for nome, cenario in scenarios.items():
        print(f"CENÁRIO: {nome.upper().replace('_', ' ')}")
        print_pertinence_table(variables, cenario)
    
    # Gerar gráfico com exemplos anotados (usando cenário de compra)
    print("\n" + "="*70)
    print("Gerando gráfico com exemplos anotados...")
    
    output_path = ROOT_DIR / "outputs" / "membership_functions_annotated.png"
    
    plot_with_examples(
        variables,
        examples=scenarios["compra_forte"],
        save_path=str(output_path),
        show=True
    )
    
    print("\n" + "="*70)
    print("  ✓ Demonstração concluída!")
    print(f"  ✓ Gráfico salvo em: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()
