#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCC Quant Trading - Sistema Híbrido SMC + Fuzzy Logic

Entry point para visualização das Funções de Pertinência.
Este script gera os gráficos das MFs para validação teórica do TCC.

Variáveis do Sistema:
- Trend_Strength: Força da tendência (ADX/Slope EMA) - Gaussiana
- Price_Zone: Premium/Discount zones - Z / Triangular / S
- FVG_Quality: Qualidade do Fair Value Gap - Triangular/Sigmoidal
- Sweep_Quality: Captura de Liquidez - Sigmoidal
- Trade_Signal: sinal final bidirecional (saída)

Autor: Felipe Vilela
Data: Dezembro 2024
"""

import sys
import os
from pathlib import Path

# Adicionar o diretório raiz ao path para imports
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# Configurar backend não-interativo se necessário
import matplotlib
if os.environ.get('DISPLAY') is None and sys.platform != 'win32':
    matplotlib.use('Agg')

from fuzzy import create_fuzzy_variables, plot_membership_functions


def main():
    """
    Função principal - gera visualização das Membership Functions SMC.
    """
    print("=" * 70)
    print("  TCC Quant Trading - Visualização das Funções de Pertinência")
    print("  Sistema Híbrido: Smart Money Concepts + Fuzzy Logic (Mamdani)")
    print("=" * 70)
    print()
    
    # Criar variáveis fuzzy
    print("➤ Criando variáveis fuzzy SMC...")
    variables = create_fuzzy_variables()
    
    print(f"  ✓ {len(variables)} variáveis criadas:")
    print()
    
    # Mostrar detalhes de cada variável
    for name, var in variables.items():
        terms = list(var.terms.keys())
        var_type = "Consequente" if var.__class__.__name__ == "Consequent" else "Antecedente"
        universe_range = f"[{var.universe.min():.1f}, {var.universe.max():.1f}]"
        print(f"    {var_type}: {name}")
        print(f"      • Universo: {universe_range}")
        print(f"      • Conjuntos: {terms}")
        print()
    
    # Gerar visualização
    print("➤ Gerando visualização das Membership Functions...")
    
    # Caminho para salvar o gráfico
    output_path = ROOT_DIR / "outputs" / "membership_functions.png"
    
    # Plotar e salvar
    fig = plot_membership_functions(
        variables,
        save_path=str(output_path),
        show=True  # Mostrar interativamente
    )
    
    print()
    print("=" * 70)
    print("  ✓ Visualização concluída!")
    print(f"  ✓ Gráfico salvo em: {output_path}")
    print("=" * 70)
    
    return fig


if __name__ == "__main__":
    main()
