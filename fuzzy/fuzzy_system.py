# -*- coding: utf-8 -*-
"""
Sistema de Inferência Fuzzy Mamdani para Trading SMC.

Este módulo implementa uma saída bidirecional para o sistema fuzzy:
- valores negativos representam viés vendedor;
- valores positivos representam viés comprador;
- valores próximos de zero representam neutralidade ou conflito.

Fluxo:
1. Fuzzificação: Valores crisp -> Graus de pertinência
2. Aplicação das Regras: Combinação dos antecedentes (AND/OR)
3. Agregação: União das saídas de todas as regras
4. Defuzzificação: Centroide -> sinal final em [-100, 100]
"""

from typing import Any, Dict, List, Optional

from skfuzzy import control as ctrl

from .membership_functions import create_fuzzy_variables


class SMCFuzzySystem:
    """
    Sistema de Inferência Fuzzy Mamdani para avaliação de setups SMC.

    Avalia a qualidade de um setup baseado em:
    - Trend_Strength: força da tendência
    - Price_Zone: localização no range premium/discount
    - FVG_Quality: qualidade do fair value gap
    - Sweep_Quality: qualidade da captura de liquidez

    Retorna:
    - Trade_Signal: sinal bidirecional em [-100, 100]
    """

    def __init__(self):
        """Inicializa o sistema fuzzy com variáveis e regras."""
        self.variables = create_fuzzy_variables()
        self.rules: List[ctrl.Rule] = []
        self.control_system: Optional[ctrl.ControlSystem] = None

        self._create_rules()
        self._build_system()

    def _create_rules(self) -> None:
        """
        Define a base de regras fuzzy Mamdani.

        A refatoração reduz a dimensionalidade de Price_Zone para
        {Discount, Equilibrium, Premium} e torna a saída bidirecional,
        encapsulando direção e magnitude no consequente.
        """
        trend = self.variables["trend_strength"]
        zone = self.variables["price_zone"]
        fvg = self.variables["fvg_quality"]
        sweep = self.variables["sweep_quality"]
        signal = self.variables["trade_signal"]

        # =================================================================
        # REGRAS DE COMPRA
        # =================================================================
        self.rules.extend([
            ctrl.Rule(
                trend["Alta"] & zone["Discount"] & fvg["Grande"] & sweep["Forte"],
                signal["Compra_Forte"],
                label="Compra_Forte_1",
            ),
            ctrl.Rule(
                trend["Alta"] & zone["Discount"] & fvg["Padrao"] & sweep["Forte"],
                signal["Compra"],
                label="Compra_1",
            ),
            ctrl.Rule(
                trend["Alta"] & zone["Discount"] & fvg["Pequeno"] & sweep["Forte"],
                signal["Compra"],
                label="Compra_2",
            ),
            ctrl.Rule(
                trend["Neutra"] & zone["Discount"] & fvg["Grande"] & sweep["Forte"],
                signal["Compra"],
                label="Compra_3",
            ),
            ctrl.Rule(
                trend["Alta"] & zone["Equilibrium"] & fvg["Grande"] & sweep["Forte"],
                signal["Compra"],
                label="Compra_4",
            ),
            ctrl.Rule(
                trend["Alta"] & zone["Equilibrium"] & fvg["Padrao"] & sweep["Forte"],
                signal["Compra"],
                label="Compra_5",
            ),
            ctrl.Rule(
                trend["Alta"] & zone["Discount"] & fvg["Grande"] & sweep["Fraco"],
                signal["Compra"],
                label="Compra_6",
            ),
        ])

        # =================================================================
        # REGRAS DE VENDA
        # =================================================================
        self.rules.extend([
            ctrl.Rule(
                trend["Baixa"] & zone["Premium"] & fvg["Grande"] & sweep["Forte"],
                signal["Venda_Forte"],
                label="Venda_Forte_1",
            ),
            ctrl.Rule(
                trend["Baixa"] & zone["Premium"] & fvg["Padrao"] & sweep["Forte"],
                signal["Venda"],
                label="Venda_1",
            ),
            ctrl.Rule(
                trend["Baixa"] & zone["Premium"] & fvg["Pequeno"] & sweep["Forte"],
                signal["Venda"],
                label="Venda_2",
            ),
            ctrl.Rule(
                trend["Neutra"] & zone["Premium"] & fvg["Grande"] & sweep["Forte"],
                signal["Venda"],
                label="Venda_3",
            ),
            ctrl.Rule(
                trend["Baixa"] & zone["Equilibrium"] & fvg["Grande"] & sweep["Forte"],
                signal["Venda"],
                label="Venda_4",
            ),
            ctrl.Rule(
                trend["Baixa"] & zone["Equilibrium"] & fvg["Padrao"] & sweep["Forte"],
                signal["Venda"],
                label="Venda_5",
            ),
            ctrl.Rule(
                trend["Baixa"] & zone["Premium"] & fvg["Grande"] & sweep["Fraco"],
                signal["Venda"],
                label="Venda_6",
            ),
        ])

        # =================================================================
        # REGRAS DE SEGURANÇA / CONFLITO
        # =================================================================
        self.rules.extend([
            ctrl.Rule(
                trend["Neutra"] & zone["Equilibrium"] & fvg["Grande"] & sweep["Forte"],
                signal["Neutro"],
                label="Neutro_1",
            ),
            ctrl.Rule(
                sweep["Fraco"] & fvg["Pequeno"],
                signal["Neutro"],
                label="Neutro_2",
            ),
            ctrl.Rule(
                trend["Alta"] & zone["Premium"],
                signal["Neutro"],
                label="Conflito_AltaPremium",
            ),
            ctrl.Rule(
                trend["Baixa"] & zone["Discount"],
                signal["Neutro"],
                label="Conflito_BaixaDiscount",
            ),
        ])

        print(f"  ✓ {len(self.rules)} regras fuzzy criadas")

    def _build_system(self) -> None:
        """Constrói o sistema de controle fuzzy."""
        self.control_system = ctrl.ControlSystem(self.rules)
        print("  ✓ Sistema de inferência Mamdani construído")

    def _classify_signal(self, signal_value: float) -> Dict[str, Any]:
        """
        Traduz o sinal defuzzificado em direção, classe e tamanho de posição.

        O sistema usa apenas a saída para tomar a decisão final, preservando
        o encapsulamento da inferência fuzzy.
        """
        magnitude = abs(signal_value)

        if signal_value >= 60:
            return {
                "classification": "COMPRA_FORTE",
                "direction": "COMPRA",
                "position_size": 1.0,
                "action": "COMPRA_FORTE (Lote cheio)",
                "magnitude": magnitude,
            }
        if signal_value >= 15:
            return {
                "classification": "COMPRA",
                "direction": "COMPRA",
                "position_size": 0.5,
                "action": "COMPRA (Meio lote)",
                "magnitude": magnitude,
            }
        if signal_value <= -60:
            return {
                "classification": "VENDA_FORTE",
                "direction": "VENDA",
                "position_size": 1.0,
                "action": "VENDA_FORTE (Lote cheio)",
                "magnitude": magnitude,
            }
        if signal_value <= -15:
            return {
                "classification": "VENDA",
                "direction": "VENDA",
                "position_size": 0.5,
                "action": "VENDA (Meio lote)",
                "magnitude": magnitude,
            }
        return {
            "classification": "NEUTRO",
            "direction": "NEUTRO",
            "position_size": 0.0,
            "action": "SEM AÇÃO (Neutro)",
            "magnitude": magnitude,
        }

    def compute(
        self,
        trend_strength: float,
        price_zone: float,
        fvg_quality: float,
        sweep_quality: float,
    ) -> Dict[str, Any]:
        """
        Calcula o sinal bidirecional para um conjunto de inputs.

        Args:
            trend_strength: força da tendência em [-100, 100]
            price_zone: posição no range em [0, 1]
            fvg_quality: qualidade do FVG em [0, 4]
            sweep_quality: qualidade do sweep em [0, 3]

        Returns:
            Dict com o sinal defuzzificado e a decisão final do sistema.
        """
        simulator = ctrl.ControlSystemSimulation(self.control_system)

        simulator.input["Trend_Strength"] = trend_strength
        simulator.input["Price_Zone"] = price_zone
        simulator.input["FVG_Quality"] = fvg_quality
        simulator.input["Sweep_Quality"] = sweep_quality

        try:
            simulator.compute()
            signal_value = float(simulator.output["Trade_Signal"])
        except Exception as exc:
            print(f"⚠️ Erro na inferência: {exc}")
            print("   Isso pode ocorrer se nenhuma regra foi ativada.")
            signal_value = 0.0

        decision = self._classify_signal(signal_value)

        return {
            "signal": signal_value,
            "classification": decision["classification"],
            "direction": decision["direction"],
            "action": decision["action"],
            "position_size": decision["position_size"],
            "magnitude": decision["magnitude"],
            "inputs": {
                "trend_strength": trend_strength,
                "price_zone": price_zone,
                "fvg_quality": fvg_quality,
                "sweep_quality": sweep_quality,
            },
        }

    def evaluate_scenario(self, scenario: Dict[str, float]) -> Dict[str, Any]:
        """
        Avalia um cenário completo.

        Args:
            scenario: Dict com as 4 variáveis de entrada

        Returns:
            Resultado da inferência
        """
        return self.compute(
            trend_strength=scenario.get("trend_strength", 0.0),
            price_zone=scenario.get("price_zone", 0.5),
            fvg_quality=scenario.get("fvg_quality", 1.5),
            sweep_quality=scenario.get("sweep_quality", 1.0),
        )

    def print_rules_summary(self) -> None:
        """Imprime um resumo das regras do sistema."""
        print("\n" + "=" * 60)
        print("REGRAS DO SISTEMA FUZZY MAMDANI")
        print("=" * 60)

        for index, rule in enumerate(self.rules, 1):
            print(f"{index:2d}. {rule.label}")

        print("=" * 60)
        print(f"Total: {len(self.rules)} regras")


def create_fuzzy_system() -> SMCFuzzySystem:
    """
    Factory function para criar o sistema fuzzy.

    Returns:
        Instância configurada do SMCFuzzySystem
    """
    print("\n➤ Inicializando Sistema Fuzzy SMC...")
    system = SMCFuzzySystem()
    print("  ✓ Sistema pronto para inferência\n")
    return system
