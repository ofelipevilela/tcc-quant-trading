# -*- coding: utf-8 -*-
"""
Modelo ANFIS — Arquitetura de 5 Camadas (Jang, 1993).

Implementação fiel ao paper original como nn.Module do PyTorch,
utilizando o modelo Takagi-Sugeno-Kang (TSK) de ordem zero.

Camadas:
    Layer 1: Fuzzificação (MFs gaussianas com parâmetros treináveis)
    Layer 2: Firing Strength (produto dos graus de pertinência por regra)
    Layer 3: Normalized Firing Strength (normalização das forças)
    Layer 4: Consequentes TSK zero-order (constantes crisp aprendidas)
    Layer 5: Defuzzificação (média ponderada — soma de w̄ᵢ·cᵢ)

Referência:
    Jang, J.-S. R. (1993). ANFIS: Adaptive-Network-Based Fuzzy
    Inference System. IEEE Transactions on Systems, Man, and
    Cybernetics, 23(3), 665-685.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .config import (
    INPUT_VARS,
    LINGUISTIC_SETS,
    N_INPUTS,
    SIGNAL_THRESHOLDS,
    UNIVERSES,
)
from .membership_functions import FuzzificationLayer
from .rule_base import RuleBase

logger = logging.getLogger(__name__)


class ANFISModel(nn.Module):
    """
    Adaptive-Network-Based Fuzzy Inference System (ANFIS).

    Arquitetura de 5 camadas que mapeia 4 variáveis de entrada
    (Trend_Strength, Price_Zone, FVG_Quality, Sweep_Quality) para
    1 sinal de saída (Trade_Signal) via inferência TSK zero-order.

    Parameters
    ----------
    rule_base : RuleBase
        Base de regras fuzzy validada.

    Attributes
    ----------
    fuzzification : FuzzificationLayer
        Layer 1 — MFs gaussianas treináveis.
    consequents : nn.Parameter
        Layer 4 — Constantes crisp por regra.
    antecedent_indices : torch.LongTensor
        Mapa regra → índices de conjuntos [n_rules, 4].
    n_rules : int
        Número de regras.
    """

    def __init__(self, rule_base: RuleBase) -> None:
        super().__init__()

        self.n_rules = rule_base.n_rules

        # --- Layer 1: Fuzzificação ---
        self.fuzzification = FuzzificationLayer()

        # Registrar índices dos antecedentes como buffer (não-treinável)
        antecedent_idx = rule_base.get_antecedent_indices()
        self.register_buffer('antecedent_indices', antecedent_idx)

        # --- Layer 4: Consequentes TSK zero-order ---
        initial_consequents = rule_base.get_initial_consequents()
        self.consequents = nn.Parameter(initial_consequents.clone())

        # Épsilon para estabilidade numérica na normalização
        self._eps = 1e-8

        # Guardar referencia à rule_base para descrição
        self._rule_base = rule_base

        logger.info(
            f"ANFISModel inicializado: {self.n_rules} regras, "
            f"{self._count_parameters()} parâmetros treináveis"
        )

    def forward(
        self,
        x: torch.Tensor,
        normalize: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass completo pelas 5 camadas do ANFIS.

        Parameters
        ----------
        x : torch.Tensor
            Input shape [batch, 4]. Cada coluna corresponde a uma variável
            na ordem: trend_strength, price_zone, fvg_quality, sweep_quality.
        normalize : bool
            Se True, normaliza os inputs para seus universos de discurso
            antes da fuzzificação.

        Returns
        -------
        output : torch.Tensor
            Sinal de saída, shape [batch, 1].
        firing_strengths : torch.Tensor
            Forças brutas de ativação, shape [batch, n_rules].
        normalized_strengths : torch.Tensor
            Forças normalizadas, shape [batch, n_rules].
        """
        if normalize:
            x = self._normalize_inputs(x)

        # ========================================
        # LAYER 1 — Fuzzificação
        # ========================================
        # mu: [batch, n_inputs, max_sets]
        mu = self.fuzzification(x)

        # ========================================
        # LAYER 2 — Firing Strength (produto)
        # ========================================
        # Para cada regra, multiplicar os graus de pertinência
        # dos conjuntos especificados pelos antecedentes.
        # antecedent_indices: [n_rules, 4] — índices dos conjuntos
        #
        # w_i = μ_trend(x1, set_t) × μ_price(x2, set_p)
        #     × μ_fvg(x3, set_f) × μ_sweep(x4, set_s)

        batch_size = x.shape[0]
        firing_strengths = torch.ones(
            batch_size, self.n_rules, device=x.device
        )

        for var_idx in range(N_INPUTS):
            # set_indices: [n_rules] — qual conjunto de cada variável cada regra usa
            set_indices = self.antecedent_indices[:, var_idx]  # [n_rules]

            # mu_var: [batch, max_sets] — pertinências da variável var_idx
            mu_var = mu[:, var_idx, :]  # [batch, max_sets]

            # Selecionar as pertinências relevantes para cada regra
            # mu_selected: [batch, n_rules]
            mu_selected = mu_var[:, set_indices]

            # Produto (T-norm)
            firing_strengths = firing_strengths * mu_selected

        # ========================================
        # LAYER 3 — Normalized Firing Strength
        # ========================================
        # w̄_i = w_i / (Σ w_j + ε)
        sum_w = firing_strengths.sum(dim=1, keepdim=True) + self._eps
        normalized_strengths = firing_strengths / sum_w

        # ========================================
        # LAYER 4 — Consequentes TSK zero-order
        # ========================================
        # f_i = w̄_i × c_i
        weighted = normalized_strengths * self.consequents.unsqueeze(0)

        # ========================================
        # LAYER 5 — Defuzzificação (soma)
        # ========================================
        # y = Σ (w̄_i × c_i)
        output = weighted.sum(dim=1, keepdim=True)  # [batch, 1]

        return output, firing_strengths, normalized_strengths

    def _normalize_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normaliza inputs crus para os universos de discurso.

        Parameters
        ----------
        x : torch.Tensor
            Input shape [batch, 4] com valores nos universos originais.

        Returns
        -------
        torch.Tensor
            Input normalizado (min-max) para [0, 1] por variável.
        """
        x_norm = x.clone()
        for i, var_name in enumerate(INPUT_VARS):
            lo, hi = UNIVERSES[var_name]
            x_norm[:, i] = (x[:, i] - lo) / (hi - lo + 1e-8)
        return x_norm

    def clamp_mf_params(self, sigma_min: Optional[float] = None) -> None:
        """
        Garante constraints físicas após cada update do otimizador.

        - sigma >= sigma_min (MF não colapsa a um ponto)
        - center dentro do universo de discurso da variável

        Operação in-place com torch.no_grad() para não poluir o grafo.

        Parameters
        ----------
        sigma_min : float, optional
            Sigma mínimo. Se None, usa 1e-3.
        """
        if sigma_min is None:
            from .config import TRAINING
            sigma_min = TRAINING.get('sigma_min', 1e-3)

        with torch.no_grad():
            for var_name in INPUT_VARS:
                mf_list = self.fuzzification.mfs[var_name]
                lo, hi = UNIVERSES[var_name]

                for mf in mf_list:
                    mf.sigma.clamp_(min=sigma_min)
                    mf.center.clamp_(min=lo, max=hi)

    def get_mf_parameters(self) -> List[nn.Parameter]:
        """
        Retorna lista de nn.Parameter dos centers e sigmas das MFs.

        Returns
        -------
        list of nn.Parameter
            Parâmetros de todas as funções de pertinência.
        """
        params = []
        for var_name in INPUT_VARS:
            mf_list = self.fuzzification.mfs[var_name]
            for mf in mf_list:
                params.append(mf.center)
                params.append(mf.sigma)
        return params

    def get_consequent_parameters(self) -> List[nn.Parameter]:
        """
        Retorna lista contendo o nn.Parameter das constantes crisp cᵢ.

        Returns
        -------
        list of nn.Parameter
            Lista com um único elemento: self.consequents.
        """
        return [self.consequents]

    def get_mf_params(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Retorna dict com parâmetros atuais das MFs para logging.

        Returns
        -------
        dict
            {var_name: {set_name: {center, sigma}}}.
        """
        return self.fuzzification.get_params_snapshot()

    def get_consequent_values(self) -> Dict[int, float]:
        """
        Retorna dict com valores atuais dos consequentes.

        Returns
        -------
        dict
            {rule_index: consequent_value}.
        """
        with torch.no_grad():
            values = self.consequents.cpu().numpy()
        return {i: float(v) for i, v in enumerate(values)}

    @staticmethod
    def classify_signal(value: float) -> str:
        """
        Mapeia um valor float de sinal para classe linguística.

        Parameters
        ----------
        value : float
            Valor do sinal em [-100, 100].

        Returns
        -------
        str
            Nome da classe (VENDA_FORTE, VENDA, NEUTRO, COMPRA, COMPRA_FORTE).
        """
        for cls_name, (lo, hi) in SIGNAL_THRESHOLDS.items():
            if lo <= value <= hi:
                return cls_name
        if value < -100:
            return 'VENDA_FORTE'
        return 'COMPRA_FORTE'

    def _count_parameters(self) -> int:
        """Conta o número total de parâmetros treináveis."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """
        Retorna resumo do modelo com contagem de parâmetros.

        Returns
        -------
        str
            Resumo formatado do modelo.
        """
        n_mf_params = sum(p.numel() for p in self.get_mf_parameters())
        n_consequent_params = self.consequents.numel()
        total = self._count_parameters()

        lines = [
            "",
            "=" * 60,
            "ANFIS Model Summary",
            "=" * 60,
            f"  Arquitetura: TSK zero-order (Jang, 1993)",
            f"  Variáveis de entrada: {N_INPUTS}",
            f"  Regras: {self.n_rules}",
            f"  Parâmetros MF (centers + sigmas): {n_mf_params}",
            f"  Parâmetros consequentes (cᵢ): {n_consequent_params}",
            f"  Total de parâmetros treináveis: {total}",
            "=" * 60,
            "",
            "Consequentes iniciais:",
        ]

        with torch.no_grad():
            for i in range(self.n_rules):
                desc = self._rule_base.describe_rule(i)
                lines.append(f"  {desc}")

        lines.append("")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ANFISModel(n_rules={self.n_rules}, "
            f"n_params={self._count_parameters()})"
        )
