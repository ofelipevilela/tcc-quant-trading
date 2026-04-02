# -*- coding: utf-8 -*-
"""
Base de Regras Fuzzy para o ANFIS — Smart Money Concepts.

Define as regras de inferência fuzzy de forma declarativa e legível.
Cada regra mapeia uma combinação de antecedentes (conjuntos linguísticos)
a um consequente crisp inicial que será otimizado pelo ANFIS.

A semântica segue os princípios do SMC:
- Compra em Discount com tendência Alta e confirmação (sweep Forte, FVG Grande)
- Venda em Premium com tendência Baixa e confirmação
- Conflitos de direção geram sinal Neutro
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from .config import LINGUISTIC_SETS, SEMANTIC_ORDER, SIGNAL_THRESHOLDS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FuzzyRule:
    """
    Representação de uma regra fuzzy com antecedentes e consequente.

    Attributes
    ----------
    trend_set : str
        Conjunto linguístico de Trend_Strength.
    price_set : str
        Conjunto linguístico de Price_Zone.
    fvg_set : str
        Conjunto linguístico de FVG_Quality.
    sweep_set : str
        Conjunto linguístico de Sweep_Quality.
    consequent_init : float
        Valor inicial do consequente crisp (TSK zero-order).
    description : str
        Descrição legível em português da regra.
    """
    trend_set: str
    price_set: str
    fvg_set: str
    sweep_set: str
    consequent_init: float
    description: str


# ============================================================================
# Definição declarativa das regras SMC
#
# Formato: (trend_set, price_set, fvg_set, sweep_set, consequent_init, descricao)
#
# Princípio SMC: compra-se em Discount com tendência Alta e confirmação
# de sweep Forte com FVG Grande. Vende-se em Premium com tendência Baixa
# e confirmação de sweep Forte. Conflitos geram sinal Neutro.
# ============================================================================

RULES: List[FuzzyRule] = [
    # === COMPRA FORTE ===
    FuzzyRule(
        'Alta', 'Discount', 'Grande', 'Forte', +80.0,
        'Tendência alta + discount + FVG grande + sweep forte',
    ),
    FuzzyRule(
        'Alta', 'Discount', 'Padrao', 'Forte', +65.0,
        'Tendência alta + discount + FVG padrão + sweep forte',
    ),
    FuzzyRule(
        'Alta', 'Discount', 'Grande', 'Fraco', +55.0,
        'Tendência alta + discount + FVG grande sem sweep ideal',
    ),
    FuzzyRule(
        'Alta', 'Equilibrium', 'Grande', 'Forte', +60.0,
        'Tendência alta + equilibrium + FVG grande + sweep forte',
    ),

    # === COMPRA ===
    FuzzyRule(
        'Alta', 'Equilibrium', 'Padrao', 'Forte', +45.0,
        'Tendência alta + equilibrium + FVG padrão',
    ),
    FuzzyRule(
        'Alta', 'Discount', 'Pequeno', 'Forte', +40.0,
        'Tendência alta + discount + FVG pequeno + sweep forte',
    ),
    FuzzyRule(
        'Alta', 'Discount', 'Padrao', 'Fraco', +35.0,
        'Tendência alta + discount sem confirmação forte',
    ),
    FuzzyRule(
        'Neutra', 'Discount', 'Grande', 'Forte', +40.0,
        'Lateral mas em discount com setup forte',
    ),

    # === NEUTRO / CONFLITO ===
    FuzzyRule(
        'Alta', 'Premium', 'Grande', 'Forte', 0.0,
        'Conflito: tendência alta mas preço em premium',
    ),
    FuzzyRule(
        'Alta', 'Premium', 'Padrao', 'Forte', 0.0,
        'Conflito: tendência alta mas preço em premium',
    ),
    FuzzyRule(
        'Baixa', 'Discount', 'Grande', 'Forte', 0.0,
        'Conflito: tendência baixa mas preço em discount',
    ),
    FuzzyRule(
        'Neutra', 'Equilibrium', 'Padrao', 'Fraco', 0.0,
        'Mercado em equilíbrio sem setup claro',
    ),
    FuzzyRule(
        'Neutra', 'Equilibrium', 'Grande', 'Forte', 0.0,
        'Mercado em equilíbrio sem tendência definida',
    ),
    FuzzyRule(
        'Alta', 'Premium', 'Pequeno', 'Fraco', 0.0,
        'Alta tendência mas em premium sem setup',
    ),

    # === VENDA ===
    FuzzyRule(
        'Baixa', 'Premium', 'Padrao', 'Fraco', -35.0,
        'Tendência baixa + premium sem confirmação forte',
    ),
    FuzzyRule(
        'Baixa', 'Equilibrium', 'Padrao', 'Forte', -45.0,
        'Tendência baixa + equilibrium + FVG padrão',
    ),
    FuzzyRule(
        'Baixa', 'Premium', 'Pequeno', 'Forte', -40.0,
        'Tendência baixa + premium + FVG pequeno + sweep forte',
    ),
    FuzzyRule(
        'Baixa', 'Premium', 'Grande', 'Fraco', -55.0,
        'Tendência baixa + premium + FVG grande sem sweep ideal',
    ),

    # === VENDA FORTE ===
    FuzzyRule(
        'Baixa', 'Premium', 'Grande', 'Forte', -80.0,
        'Tendência baixa + premium + FVG grande + sweep forte',
    ),
    FuzzyRule(
        'Baixa', 'Premium', 'Padrao', 'Forte', -65.0,
        'Tendência baixa + premium + FVG padrão + sweep forte',
    ),
    FuzzyRule(
        'Baixa', 'Equilibrium', 'Grande', 'Forte', -60.0,
        'Tendência baixa + equilibrium + FVG grande + sweep forte',
    ),
    FuzzyRule(
        'Neutra', 'Premium', 'Grande', 'Forte', -40.0,
        'Lateral mas em premium com setup forte de venda',
    ),
]


def _classify_consequent(value: float) -> str:
    """
    Classifica o consequente inicial em uma categoria semântica.

    Parameters
    ----------
    value : float
        Valor do consequente.

    Returns
    -------
    str
        Nome da classe (VENDA_FORTE, VENDA, NEUTRO, COMPRA, COMPRA_FORTE).
    """
    for cls_name, (lo, hi) in SIGNAL_THRESHOLDS.items():
        if lo <= value <= hi:
            return cls_name
    # Caso extremo: clamp nos limites
    if value < -100:
        return 'VENDA_FORTE'
    return 'COMPRA_FORTE'


class RuleBase:
    """
    Gerenciador da base de regras fuzzy para o ANFIS.

    Valida as regras contra os conjuntos linguísticos definidos no config,
    gera índices tensoriais para o forward pass e disponibiliza descrições
    legíveis.

    Parameters
    ----------
    rules : list of FuzzyRule, optional
        Lista de regras a utilizar. Se None, usa RULES.

    Attributes
    ----------
    rules : list of FuzzyRule
        Regras validadas.
    n_rules : int
        Número de regras.
    antecedent_indices : torch.LongTensor
        Tensor [n_rules, 4] com os índices dos conjuntos para cada regra.
    """

    def __init__(self, rules: Optional[List[FuzzyRule]] = None) -> None:
        self.rules = rules if rules is not None else RULES
        self.n_rules = len(self.rules)

        self._validate_rules()
        self._antecedent_indices = self._build_antecedent_indices()
        self._log_summary()

    def _validate_rules(self) -> None:
        """
        Valida todas as regras contra os conjuntos definidos em config.

        Raises
        ------
        ValueError
            Se alguma regra referencia um conjunto linguístico inexistente.
        """
        var_set_map = {
            'trend_strength': ('trend_set', LINGUISTIC_SETS['trend_strength']),
            'price_zone':     ('price_set', LINGUISTIC_SETS['price_zone']),
            'fvg_quality':    ('fvg_set',   LINGUISTIC_SETS['fvg_quality']),
            'sweep_quality':  ('sweep_set', LINGUISTIC_SETS['sweep_quality']),
        }

        for i, rule in enumerate(self.rules):
            for var_name, (attr_name, valid_sets) in var_set_map.items():
                set_name = getattr(rule, attr_name)
                if set_name not in valid_sets:
                    raise ValueError(
                        f"Regra {i} referencia conjunto '{set_name}' para "
                        f"'{var_name}', mas os conjuntos válidos são "
                        f"{valid_sets}."
                    )

        logger.info(f"Todas as {self.n_rules} regras validadas com sucesso.")

    def _build_antecedent_indices(self) -> torch.LongTensor:
        """
        Converte nomes de conjuntos para índices tensoriais.

        Returns
        -------
        torch.LongTensor
            Tensor [n_rules, 4] onde cada linha contém os índices dos
            conjuntos para (trend, price, fvg, sweep).
        """
        indices = []

        for rule in self.rules:
            row = [
                LINGUISTIC_SETS['trend_strength'].index(rule.trend_set),
                LINGUISTIC_SETS['price_zone'].index(rule.price_set),
                LINGUISTIC_SETS['fvg_quality'].index(rule.fvg_set),
                LINGUISTIC_SETS['sweep_quality'].index(rule.sweep_set),
            ]
            indices.append(row)

        return torch.tensor(indices, dtype=torch.long)

    def get_antecedent_indices(self) -> torch.LongTensor:
        """
        Retorna tensor de índices dos antecedentes por regra.

        Returns
        -------
        torch.LongTensor
            Shape [n_rules, 4].
        """
        return self._antecedent_indices

    def get_initial_consequents(self) -> torch.Tensor:
        """
        Retorna tensor com os valores iniciais dos consequentes crisp.

        Returns
        -------
        torch.Tensor
            Shape [n_rules], dtype float32.
        """
        values = [rule.consequent_init for rule in self.rules]
        return torch.tensor(values, dtype=torch.float32)

    def describe_rule(self, i: int) -> str:
        """
        Retorna descrição legível da regra i.

        Parameters
        ----------
        i : int
            Índice da regra (0-based).

        Returns
        -------
        str
            Descrição formatada da regra.
        """
        rule = self.rules[i]
        cls = _classify_consequent(rule.consequent_init)
        return (
            f"Regra {i:02d} [{cls:>13s}]: "
            f"IF Trend={rule.trend_set} AND Zone={rule.price_set} "
            f"AND FVG={rule.fvg_set} AND Sweep={rule.sweep_set} "
            f"THEN c={rule.consequent_init:+.1f}  "
            f"({rule.description})"
        )

    def get_rules_by_class(self) -> Dict[str, List[int]]:
        """
        Agrupa índices das regras por classe de consequente.

        Returns
        -------
        dict
            {classe: [índices das regras]}.
        """
        by_class: Dict[str, List[int]] = {}
        for i, rule in enumerate(self.rules):
            cls = _classify_consequent(rule.consequent_init)
            by_class.setdefault(cls, []).append(i)
        return by_class

    def _log_summary(self) -> None:
        """Loga resumo das regras carregadas."""
        by_class = self.get_rules_by_class()

        logger.info(f"RuleBase carregada: {self.n_rules} regras")
        for cls_name in SEMANTIC_ORDER:
            count = len(by_class.get(cls_name, []))
            logger.info(f"  {cls_name:>13s}: {count} regras")

    def print_all_rules(self) -> None:
        """Imprime todas as regras de forma legível no terminal."""
        print("\n" + "=" * 80)
        print("BASE DE REGRAS ANFIS — Smart Money Concepts")
        print("=" * 80)

        by_class = self.get_rules_by_class()
        for cls_name in SEMANTIC_ORDER:
            rule_indices = by_class.get(cls_name, [])
            if rule_indices:
                print(f"\n  [{cls_name}] ({len(rule_indices)} regras)")
                for idx in rule_indices:
                    print(f"    {self.describe_rule(idx)}")

        print("\n" + "=" * 80)
        print(f"Total: {self.n_rules} regras")
        print("=" * 80)
