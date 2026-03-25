# -*- coding: utf-8 -*-
"""
Sistema de Inferência Fuzzy Mamdani para Trading SMC.

Este módulo implementa as regras fuzzy que conectam os antecedentes
(entradas) ao consequente (Trade_Score), realizando a inferência
e defuzzificação por centroide.

Fluxo:
1. Fuzzificação: Valores crisp → Graus de pertinência
2. Aplicação das Regras: Combinação dos antecedentes (AND/OR)
3. Agregação: União das saídas de todas as regras
4. Defuzzificação: Centroide → Score final (0-100)
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from skfuzzy import control as ctrl

from .membership_functions import create_fuzzy_variables


class SMCFuzzySystem:
    """
    Sistema de Inferência Fuzzy Mamdani para avaliação de setups SMC.
    
    Avalia a qualidade de um setup de trading baseado em:
    - Trend_Strength: Força da tendência (ADX/Slope)
    - Price_Zone: Localização Premium/Discount
    - FVG_Quality: Qualidade do Fair Value Gap
    - Sweep_Quality: Qualidade da captura de liquidez
    
    Retorna:
    - Trade_Score: 0-100 indicando qualidade do setup
    """
    
    def __init__(self):
        """Inicializa o sistema fuzzy com variáveis e regras."""
        self.variables = create_fuzzy_variables()
        self.rules: List[ctrl.Rule] = []
        self.control_system: Optional[ctrl.ControlSystem] = None
        self.simulator: Optional[ctrl.ControlSystemSimulation] = None
        
        self._create_rules()
        self._build_system()
    
    def _create_rules(self) -> None:
        """
        Define as regras fuzzy do sistema Mamdani (Sparse Rule Base).
        
        Apenas regras que geram AÇÃO são modeladas:
        - Muito_Forte / Forte → Lote cheio
        - Moderado → Meio lote
        - Fraco → Sem ação (não precisa de regras específicas)
        
        Regras baseadas em Smart Money Concepts por especialista.
        """
        # Atalhos para as variáveis
        trend = self.variables['trend_strength']
        zone = self.variables['price_zone']
        fvg = self.variables['fvg_quality']
        sweep = self.variables['sweep_quality']
        score = self.variables['trade_score']
        
        # =================================================================
        # REGRAS DE COMPRA - MUITO FORTE (Lote cheio)
        # =================================================================
        
        # Regra 1: Alta + Deep_Discount + Grande + Forte
        self.rules.append(ctrl.Rule(
            trend['Alta'] & zone['Deep_Discount'] & fvg['Grande'] & sweep['Forte'],
            score['Muito_Forte'],
            label='Compra_MuitoForte_1'
        ))
        
        # Regra 2: Alta + Discount + Grande + Forte
        self.rules.append(ctrl.Rule(
            trend['Alta'] & zone['Discount'] & fvg['Grande'] & sweep['Forte'],
            score['Muito_Forte'],
            label='Compra_MuitoForte_2'
        ))
        
        # =================================================================
        # REGRAS DE COMPRA - FORTE (Lote cheio)
        # =================================================================
        
        # Regra 3: Alta + Deep_Discount + Padrão + Forte
        self.rules.append(ctrl.Rule(
            trend['Alta'] & zone['Deep_Discount'] & fvg['Padrao'] & sweep['Forte'],
            score['Forte'],
            label='Compra_Forte_1'
        ))
        
        # Regra 4: Alta + Discount + Padrão + Forte
        self.rules.append(ctrl.Rule(
            trend['Alta'] & zone['Discount'] & fvg['Padrao'] & sweep['Forte'],
            score['Forte'],
            label='Compra_Forte_2'
        ))
        
        # Regra 5: Alta + Deep_Discount + Pequeno + Forte
        self.rules.append(ctrl.Rule(
            trend['Alta'] & zone['Deep_Discount'] & fvg['Pequeno'] & sweep['Forte'],
            score['Forte'],
            label='Compra_Forte_3'
        ))
        
        # =================================================================
        # REGRAS DE COMPRA - MODERADO (Meio lote)
        # =================================================================
        
        # Regra 6: Neutra + Deep_Discount + Grande + Forte (Precificação compensa Tendência)
        self.rules.append(ctrl.Rule(
            trend['Neutra'] & zone['Deep_Discount'] & fvg['Grande'] & sweep['Forte'],
            score['Moderado'],
            label='Compra_Moderado_1'
        ))
        
        # Regra 7: Neutra + Discount + Grande + Forte (Precificação compensa Tendência)
        self.rules.append(ctrl.Rule(
            trend['Neutra'] & zone['Discount'] & fvg['Grande'] & sweep['Forte'],
            score['Moderado'],
            label='Compra_Moderado_2'
        ))
        
        # Regra 8: Neutra + Deep_Discount + Padrão + Forte (Precificação compensa Tendência)
        self.rules.append(ctrl.Rule(
            trend['Neutra'] & zone['Deep_Discount'] & fvg['Padrao'] & sweep['Forte'],
            score['Moderado'],
            label='Compra_Moderado_3'
        ))
        
        # Regra 9: Alta + Equilibrio + Grande + Forte
        self.rules.append(ctrl.Rule(
            trend['Alta'] & zone['Equilibrium'] & fvg['Grande'] & sweep['Forte'],
            score['Moderado'],
            label='Compra_Moderado_4'
        ))
        
        # Regra 10: Alta + Equilibrio + Padrão + Forte
        self.rules.append(ctrl.Rule(
            trend['Alta'] & zone['Equilibrium'] & fvg['Padrao'] & sweep['Forte'],
            score['Moderado'],
            label='Compra_Moderado_5'
        ))
        
        # Regra 11: Alta + Deep_Discount + Grande + Fraco (FVG compensa Sweep)
        self.rules.append(ctrl.Rule(
            trend['Alta'] & zone['Deep_Discount'] & fvg['Grande'] & sweep['Fraco'],
            score['Moderado'],
            label='Compra_Moderado_6'
        ))
        
        # Regra 12: Alta + Discount + Grande + Fraco (FVG compensa Sweep)
        self.rules.append(ctrl.Rule(
            trend['Alta'] & zone['Discount'] & fvg['Grande'] & sweep['Fraco'],
            score['Moderado'],
            label='Compra_Moderado_7'
        ))
        
        # =================================================================
        # REGRAS DE COMPRA - FRACO (Sem ação, mas precisa existir)
        # =================================================================
        
        # Regra 13: Neutra + Equilibrio + Grande + Forte (borda)
        self.rules.append(ctrl.Rule(
            trend['Neutra'] & zone['Equilibrium'] & fvg['Grande'] & sweep['Forte'],
            score['Fraco'],
            label='Default_SemConfirmacao'
        ))
        
        # =================================================================
        # REGRAS DE VENDA - MUITO FORTE (Lote cheio)
        # Nota: Vendas requerem tendência BAIXA
        # =================================================================
        
        # Regra 14: Baixa + Deep_Premium + Grande + Forte
        self.rules.append(ctrl.Rule(
            trend['Baixa'] & zone['Deep_Premium'] & fvg['Grande'] & sweep['Forte'],
            score['Muito_Forte'],
            label='Venda_MuitoForte_1'
        ))
        
        # Regra 15: Baixa + Premium + Grande + Forte
        self.rules.append(ctrl.Rule(
            trend['Baixa'] & zone['Premium'] & fvg['Grande'] & sweep['Forte'],
            score['Muito_Forte'],
            label='Venda_MuitoForte_2'
        ))
        
        # =================================================================
        # REGRAS DE VENDA - FORTE (Lote cheio)
        # =================================================================
        
        # Regra 16: Baixa + Deep_Premium + Padrão + Forte
        self.rules.append(ctrl.Rule(
            trend['Baixa'] & zone['Deep_Premium'] & fvg['Padrao'] & sweep['Forte'],
            score['Forte'],
            label='Venda_Forte_1'
        ))
        
        # Regra 17: Baixa + Premium + Padrão + Forte
        self.rules.append(ctrl.Rule(
            trend['Baixa'] & zone['Premium'] & fvg['Padrao'] & sweep['Forte'],
            score['Forte'],
            label='Venda_Forte_2'
        ))
        
        # Regra 18: Baixa + Deep_Premium + Pequeno + Forte
        self.rules.append(ctrl.Rule(
            trend['Baixa'] & zone['Deep_Premium'] & fvg['Pequeno'] & sweep['Forte'],
            score['Forte'],
            label='Venda_Forte_3'
        ))
        
        # =================================================================
        # REGRAS DE VENDA - MODERADO (Meio lote)
        # =================================================================
        
        # Regra 19: Neutra + Deep_Premium + Grande + Forte (Precificação compensa Tendência)
        self.rules.append(ctrl.Rule(
            trend['Neutra'] & zone['Deep_Premium'] & fvg['Grande'] & sweep['Forte'],
            score['Moderado'],
            label='Venda_Moderado_1'
        ))
        
        # Regra 20: Neutra + Premium + Grande + Forte (Precificação compensa Tendência)
        self.rules.append(ctrl.Rule(
            trend['Neutra'] & zone['Premium'] & fvg['Grande'] & sweep['Forte'],
            score['Moderado'],
            label='Venda_Moderado_2'
        ))
        
        # Regra 21: Neutra + Deep_Premium + Padrão + Forte (Precificação compensa Tendência)
        self.rules.append(ctrl.Rule(
            trend['Neutra'] & zone['Deep_Premium'] & fvg['Padrao'] & sweep['Forte'],
            score['Moderado'],
            label='Venda_Moderado_3'
        ))
        
        # Regra 22: Baixa + Equilibrio + Grande + Forte
        self.rules.append(ctrl.Rule(
            trend['Baixa'] & zone['Equilibrium'] & fvg['Grande'] & sweep['Forte'],
            score['Moderado'],
            label='Venda_Moderado_4'
        ))
        
        # Regra 23: Baixa + Equilibrio + Padrão + Forte
        self.rules.append(ctrl.Rule(
            trend['Baixa'] & zone['Equilibrium'] & fvg['Padrao'] & sweep['Forte'],
            score['Moderado'],
            label='Venda_Moderado_5'
        ))
        
        # Regra 24: Baixa + Deep_Premium + Grande + Fraco (FVG compensa Sweep)
        self.rules.append(ctrl.Rule(
            trend['Baixa'] & zone['Deep_Premium'] & fvg['Grande'] & sweep['Fraco'],
            score['Moderado'],
            label='Venda_Moderado_6'
        ))
        
        # Regra 25: Baixa + Premium + Grande + Fraco (FVG compensa Sweep)
        self.rules.append(ctrl.Rule(
            trend['Baixa'] & zone['Premium'] & fvg['Grande'] & sweep['Fraco'],
            score['Moderado'],
            label='Venda_Moderado_7'
        ))  
        
        # =================================================================
        # REGRA DEFAULT - Casos não cobertos
        # =================================================================
        
        # Regra 26: Fallback para sweep fraco + fvg pequeno
        self.rules.append(ctrl.Rule(
            sweep['Fraco'] & fvg['Pequeno'],
            score['Fraco'],
            label='Default_SemConfirmacao'
        ))
        
        print(f"  ✓ {len(self.rules)} regras fuzzy criadas")
    
    def _build_system(self) -> None:
        """Constrói o sistema de controle fuzzy."""
        self.control_system = ctrl.ControlSystem(self.rules)
        self.simulator = ctrl.ControlSystemSimulation(self.control_system)
        print("  ✓ Sistema de inferência Mamdani construído")
    
    def compute(
        self,
        trend_strength: float,
        price_zone: float,
        fvg_quality: float,
        sweep_quality: float
    ) -> Dict[str, any]:
        """
        Calcula o Trade Score para um conjunto de inputs.
        
        Args:
            trend_strength: Força da tendência (0-100, ex: ADX)
            price_zone: Posição no range (0-1, onde 0=fundo, 1=topo)
            fvg_quality: Qualidade do FVG (0-4, FVG_size/ATR)
            sweep_quality: Qualidade do sweep (0-3, pavio/corpo)
            
        Returns:
            Dict com:
            - 'score': Valor defuzzificado (0-100)
            - 'classification': Classificação textual
            - 'inputs': Valores de entrada usados
            
        Example:
            >>> system = SMCFuzzySystem()
            >>> result = system.compute(75, 0.15, 2.5, 2.0)
            >>> print(f"Score: {result['score']:.1f} - {result['action']}")
            Score: 87.3 - ENTRADA (Lote cheio)
        """
        # Definir inputs
        self.simulator.input['Trend_Strength'] = trend_strength
        self.simulator.input['Price_Zone'] = price_zone
        self.simulator.input['FVG_Quality'] = fvg_quality
        self.simulator.input['Sweep_Quality'] = sweep_quality
        
        # Computar
        try:
            self.simulator.compute()
            score = self.simulator.output['Trade_Score']
        except Exception as e:
            print(f"⚠️ Erro na inferência: {e}")
            print("   Isso pode ocorrer se nenhuma regra foi ativada.")
            score = 0.0
        
        # =================================================================
        # DETERMINAR DIREÇÃO (BUY/SELL) baseado nos inputs
        # =================================================================
        # Trend: positivo = bullish (compra), negativo = bearish (venda)
        # Zone: < 0.5 = discount (compra), > 0.5 = premium (venda)
        
        if trend_strength > 20 and price_zone < 0.45:
            direction = "COMPRA"
        elif trend_strength < -20 and price_zone > 0.55:
            direction = "VENDA"
        elif trend_strength > 0 and price_zone < 0.5:
            direction = "COMPRA"  # Tendência fraca mas em discount
        elif trend_strength < 0 and price_zone > 0.5:
            direction = "VENDA"   # Tendência fraca mas em premium
        else:
            direction = "INDEFINIDO"  # Zona de conflito ou equilibrium
        
        # =================================================================
        # CLASSIFICAÇÃO E POSITION SIZING
        # =================================================================
        if score >= 80:
            classification = "MUITO_FORTE"
            position_size = 1.0  # Lote cheio
        elif score >= 60:
            classification = "FORTE"
            position_size = 1.0  # Lote cheio
        elif score >= 40:
            classification = "MODERADO"
            position_size = 0.5  # Meio lote
        else:
            classification = "FRACO"
            position_size = 0.0  # Sem ação
            direction = "SEM AÇÃO"
        
        # Montar ação final com direção
        if position_size > 0:
            action = f"{direction} ({classification} - {'Lote cheio' if position_size == 1.0 else 'Meio lote'})"
        else:
            action = "SEM AÇÃO"
        
        return {
            'score': float(score),
            'classification': classification,
            'direction': direction,
            'action': action,
            'position_size': position_size,
            'inputs': {
                'trend_strength': trend_strength,
                'price_zone': price_zone,
                'fvg_quality': fvg_quality,
                'sweep_quality': sweep_quality,
            }
        }
    
    def evaluate_scenario(self, scenario: Dict[str, float]) -> Dict[str, any]:
        """
        Avalia um cenário completo.
        
        Args:
            scenario: Dict com as 4 variáveis de entrada
            
        Returns:
            Resultado da inferência
        """
        return self.compute(
            trend_strength=scenario.get('trend_strength', 50),
            price_zone=scenario.get('price_zone', 0.5),
            fvg_quality=scenario.get('fvg_quality', 1.5),
            sweep_quality=scenario.get('sweep_quality', 1.0),
        )
    
    def print_rules_summary(self) -> None:
        """Imprime um resumo das regras do sistema."""
        print("\n" + "="*60)
        print("REGRAS DO SISTEMA FUZZY MAMDANI")
        print("="*60)
        
        for i, rule in enumerate(self.rules, 1):
            print(f"{i:2d}. {rule.label}")
        
        print("="*60)
        print(f"Total: {len(self.rules)} regras")


def create_fuzzy_system() -> SMCFuzzySystem:
    """
    Factory function para criar o sistema fuzzy.
    
    Returns:
        Instância configurada do SMCFuzzySystem
        
    Example:
        >>> from fuzzy.fuzzy_system import create_fuzzy_system
        >>> system = create_fuzzy_system()
        >>> result = system.compute(75, 0.15, 2.5, 2.0)
    """
    print("\n➤ Inicializando Sistema Fuzzy SMC...")
    system = SMCFuzzySystem()
    print("  ✓ Sistema pronto para inferência\n")
    return system
