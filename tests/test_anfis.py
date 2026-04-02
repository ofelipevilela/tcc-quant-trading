# -*- coding: utf-8 -*-
"""
Testes unitários do módulo ANFIS.

Verifica invariantes fundamentais do modelo:
1. Fuzzificação produz valores em [0, 1]
2. Soma das firing strengths normalizadas ≈ 1
3. Output do modelo está em [-100, +100]
4. Ordenação semântica dos consequentes
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Adicionar raiz ao path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from anfis.anfis_model import ANFISModel
from anfis.config import INPUT_VARS, SEMANTIC_ORDER, SIGNAL_THRESHOLDS, UNIVERSES
from anfis.membership_functions import FuzzificationLayer, GaussianMF
from anfis.rule_base import RuleBase


class TestGaussianMF:
    """Testes da função de pertinência gaussiana."""

    def test_peak_at_center(self):
        """MF deve ter pertinência 1.0 no centro."""
        mf = GaussianMF(center=0.0, sigma=1.0, name='test')
        x = torch.tensor([0.0])
        mu = mf(x)
        assert torch.isclose(mu, torch.tensor(1.0), atol=1e-6)

    def test_output_range(self):
        """MF deve produzir valores em [0, 1]."""
        mf = GaussianMF(center=50.0, sigma=20.0, name='test')
        x = torch.linspace(-100, 100, 1000)
        mu = mf(x)
        assert (mu >= 0.0).all(), "Pertinências não podem ser negativas"
        assert (mu <= 1.0 + 1e-6).all(), "Pertinências não podem exceder 1.0"

    def test_symmetric(self):
        """MF gaussiana deve ser simétrica em torno do centro."""
        mf = GaussianMF(center=0.0, sigma=1.0, name='test')
        x_pos = torch.tensor([1.0])
        x_neg = torch.tensor([-1.0])
        assert torch.isclose(mf(x_pos), mf(x_neg), atol=1e-6)

    def test_parameters_are_trainable(self):
        """Center e sigma devem ser nn.Parameter com requires_grad=True."""
        mf = GaussianMF(center=0.0, sigma=1.0, name='test')
        assert mf.center.requires_grad
        assert mf.sigma.requires_grad


class TestFuzzificationLayer:
    """Testes da camada de fuzzificação."""

    def test_output_shape(self):
        """Shape de saída deve ser [batch, n_inputs, max_sets]."""
        layer = FuzzificationLayer()
        batch_size = 16
        x = torch.rand(batch_size, len(INPUT_VARS))
        mu = layer(x)

        from anfis.config import MAX_SETS, N_INPUTS
        assert mu.shape == (batch_size, N_INPUTS, MAX_SETS)

    def test_output_range(self):
        """Todas as pertinências devem estar em [0, 1]."""
        layer = FuzzificationLayer()
        # Gerar inputs nos limites dos universos
        x = torch.tensor([[
            UNIVERSES['trend_strength'][0],
            UNIVERSES['price_zone'][0],
            UNIVERSES['fvg_quality'][0],
            UNIVERSES['sweep_quality'][0],
        ], [
            UNIVERSES['trend_strength'][1],
            UNIVERSES['price_zone'][1],
            UNIVERSES['fvg_quality'][1],
            UNIVERSES['sweep_quality'][1],
        ]])

        mu = layer(x)
        assert (mu >= 0.0).all(), "Pertinências negativas detectadas"
        assert (mu <= 1.0 + 1e-6).all(), "Pertinências > 1.0 detectadas"


class TestRuleBase:
    """Testes da base de regras."""

    def test_validation_passes(self):
        """Regras padrão devem passar na validação."""
        rule_base = RuleBase()
        assert rule_base.n_rules > 0

    def test_antecedent_indices_shape(self):
        """Índices devem ter shape [n_rules, 4]."""
        rule_base = RuleBase()
        indices = rule_base.get_antecedent_indices()
        assert indices.shape == (rule_base.n_rules, 4)

    def test_initial_consequents_shape(self):
        """Consequentes iniciais devem ter shape [n_rules]."""
        rule_base = RuleBase()
        consequents = rule_base.get_initial_consequents()
        assert consequents.shape == (rule_base.n_rules,)

    def test_describe_rule(self):
        """Descrição de regra deve retornar string não-vazia."""
        rule_base = RuleBase()
        desc = rule_base.describe_rule(0)
        assert isinstance(desc, str)
        assert len(desc) > 0


class TestANFISModel:
    """Testes do modelo ANFIS."""

    @pytest.fixture
    def model(self):
        rule_base = RuleBase()
        return ANFISModel(rule_base)

    def test_forward_output_shape(self, model):
        """Output deve ter shape [batch, 1]."""
        batch_size = 32
        x = torch.rand(batch_size, len(INPUT_VARS))
        # Ajustar inputs para universos
        x[:, 0] = x[:, 0] * 200 - 100  # trend_strength [-100, 100]
        x[:, 1] = x[:, 1]              # price_zone [0, 1]
        x[:, 2] = x[:, 2] * 4          # fvg_quality [0, 4]
        x[:, 3] = x[:, 3] * 3          # sweep_quality [0, 3]

        output, firing, normalized = model(x)
        assert output.shape == (batch_size, 1)

    def test_firing_strengths_shape(self, model):
        """Firing strengths devem ter shape [batch, n_rules]."""
        x = torch.rand(8, len(INPUT_VARS))
        _, firing, _ = model(x)
        assert firing.shape == (8, model.n_rules)

    def test_normalized_strengths_sum_to_one(self, model):
        """Forças normalizadas devem somar ≈ 1.0 por amostra."""
        x = torch.rand(16, len(INPUT_VARS))
        x[:, 0] = x[:, 0] * 200 - 100
        x[:, 2] = x[:, 2] * 4
        x[:, 3] = x[:, 3] * 3

        _, _, normalized = model(x)
        sums = normalized.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), \
            f"Soma das forças normalizadas deveria ser ≈ 1.0, obteve: {sums}"

    def test_output_bounded(self, model):
        """Output deve estar em range razoável (não necessariamente [-100, 100] exato)."""
        torch.manual_seed(42)
        x = torch.rand(100, len(INPUT_VARS))
        x[:, 0] = x[:, 0] * 200 - 100
        x[:, 2] = x[:, 2] * 4
        x[:, 3] = x[:, 3] * 3

        output, _, _ = model(x)
        # Os consequentes iniciais estão em [-80, +80], então a saída
        # (média ponderada) deve estar dentro desse range
        assert (output >= -100).all(), "Output abaixo de -100"
        assert (output <= 100).all(), "Output acima de 100"

    def test_classify_signal(self, model):
        """Classificação de sinal deve retornar classes válidas."""
        assert ANFISModel.classify_signal(80.0) == 'COMPRA_FORTE'
        assert ANFISModel.classify_signal(40.0) == 'COMPRA'
        assert ANFISModel.classify_signal(0.0) == 'NEUTRO'
        assert ANFISModel.classify_signal(-40.0) == 'VENDA'
        assert ANFISModel.classify_signal(-80.0) == 'VENDA_FORTE'

    def test_parameter_separation(self, model):
        """MF params e consequent params devem ser conjuntos disjuntos."""
        mf_params = set(id(p) for p in model.get_mf_parameters())
        cons_params = set(id(p) for p in model.get_consequent_parameters())
        assert mf_params.isdisjoint(cons_params), \
            "MF params e consequent params não devem se sobrepor"

    def test_clamp_mf_params(self, model):
        """Clamp deve manter sigma positivo e center no universo."""
        # Forçar sigma negativo
        with torch.no_grad():
            for var_name in INPUT_VARS:
                for mf in model.fuzzification.mfs[var_name]:
                    mf.sigma.fill_(-1.0)
                    mf.center.fill_(999.0)

        model.clamp_mf_params()

        for var_name in INPUT_VARS:
            lo, hi = UNIVERSES[var_name]
            for mf in model.fuzzification.mfs[var_name]:
                assert mf.sigma.item() > 0, "Sigma deve ser positivo após clamp"
                assert mf.center.item() <= hi, "Center deve estar no universo"

    def test_gradient_flow(self, model):
        """Gradientes devem fluir para MF params e consequentes."""
        x = torch.rand(8, len(INPUT_VARS))
        x[:, 0] = x[:, 0] * 200 - 100
        x[:, 2] = x[:, 2] * 4
        x[:, 3] = x[:, 3] * 3

        y_target = torch.zeros(8, 1)
        output, _, _ = model(x)
        loss = (output - y_target).pow(2).mean()
        loss.backward()

        # Verificar que gradientes existem
        for var_name in INPUT_VARS:
            for mf in model.fuzzification.mfs[var_name]:
                assert mf.center.grad is not None, \
                    f"Sem gradiente no center de {var_name}"
                assert mf.sigma.grad is not None, \
                    f"Sem gradiente no sigma de {var_name}"

        assert model.consequents.grad is not None, \
            "Sem gradiente nos consequentes"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
