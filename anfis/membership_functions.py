# -*- coding: utf-8 -*-
"""
Funções de Pertinência Diferenciáveis para o ANFIS.

Implementa MFs como módulos PyTorch com parâmetros treináveis (nn.Parameter),
permitindo ajuste via backpropagation. Todas as MFs são gaussianas para
garantir diferenciabilidade completa na Layer 1 do ANFIS.

Referência:
    Jang, J.-S. R. (1993). ANFIS: Adaptive-Network-Based Fuzzy
    Inference System. IEEE Transactions on Systems, Man, and
    Cybernetics, 23(3), 665-685.
"""

import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import (
    INITIAL_MF_PARAMS,
    INPUT_VARS,
    LINGUISTIC_SETS,
    MAX_SETS,
    N_INPUTS,
    UNIVERSES,
)

logger = logging.getLogger(__name__)


class GaussianMF(nn.Module):
    """
    Função de pertinência gaussiana diferenciável.

    Calcula μ(x) = exp(-(x - center)² / (2·σ²))

    Parameters
    ----------
    center : float
        Centro da gaussiana (valor de pertinência máxima).
    sigma : float
        Desvio padrão (largura da gaussiana).
    name : str, optional
        Nome do conjunto linguístico (ex: 'Alta', 'Discount').

    Attributes
    ----------
    center : nn.Parameter
        Centro da gaussiana, treinável via backprop.
    sigma : nn.Parameter
        Desvio padrão, treinável via backprop.

    Notes
    -----
    O sigma é clamped para valores positivos após cada atualização
    do otimizador para evitar degeneração da MF.
    """

    def __init__(self, center: float, sigma: float, name: str = '') -> None:
        super().__init__()
        self.name = name
        self.center = nn.Parameter(torch.tensor(center, dtype=torch.float32))
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula o grau de pertinência para cada elemento do batch.

        Parameters
        ----------
        x : torch.Tensor
            Tensor de entrada, shape arbitrário.

        Returns
        -------
        torch.Tensor
            Graus de pertinência em [0, 1], mesmo shape de x.
        """
        # Clamp sigma inline para segurança numérica durante forward
        sigma_safe = torch.clamp(self.sigma, min=1e-6)
        return torch.exp(-0.5 * ((x - self.center) / sigma_safe) ** 2)

    def plot_mf(
        self,
        ax: object,
        universe: Tuple[float, float],
        n_points: int = 200,
        **kwargs,
    ) -> None:
        """
        Plota a função de pertinência em um eixo matplotlib.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Eixo onde plotar.
        universe : tuple of float
            (min, max) do universo de discurso.
        n_points : int
            Número de pontos para a curva.
        **kwargs
            Argumentos adicionais para ax.plot().
        """
        x = np.linspace(universe[0], universe[1], n_points)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            y = self.forward(x_tensor).numpy()
        label = kwargs.pop('label', self.name)
        ax.plot(x, y, label=label, **kwargs)

    def __repr__(self) -> str:
        return (
            f"GaussianMF(name='{self.name}', "
            f"center={self.center.item():.4f}, "
            f"sigma={self.sigma.item():.4f})"
        )


class BellMF(nn.Module):
    """
    Função de pertinência generalized bell (alternativa à gaussiana).

    Calcula μ(x) = 1 / (1 + |(x - c) / a|^(2b))

    Parameters
    ----------
    a : float
        Largura da curva.
    b : float
        Inclinação das laterais.
    c : float
        Centro da curva.
    name : str, optional
        Nome do conjunto linguístico.
    """

    def __init__(self, a: float, b: float, c: float, name: str = '') -> None:
        super().__init__()
        self.name = name
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))
        self.c = nn.Parameter(torch.tensor(c, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula o grau de pertinência.

        Parameters
        ----------
        x : torch.Tensor
            Tensor de entrada.

        Returns
        -------
        torch.Tensor
            Graus de pertinência em [0, 1].
        """
        a_safe = torch.clamp(self.a.abs(), min=1e-6)
        return 1.0 / (1.0 + torch.abs((x - self.c) / a_safe) ** (2.0 * self.b))

    def plot_mf(
        self,
        ax: object,
        universe: Tuple[float, float],
        n_points: int = 200,
        **kwargs,
    ) -> None:
        """Plota a MF em um eixo matplotlib."""
        x = np.linspace(universe[0], universe[1], n_points)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            y = self.forward(x_tensor).numpy()
        label = kwargs.pop('label', self.name)
        ax.plot(x, y, label=label, **kwargs)

    def __repr__(self) -> str:
        return (
            f"BellMF(name='{self.name}', "
            f"a={self.a.item():.4f}, "
            f"b={self.b.item():.4f}, "
            f"c={self.c.item():.4f})"
        )


class FuzzificationLayer(nn.Module):
    """
    Layer 1 do ANFIS — Fuzzificação.

    Aplica todas as funções de pertinência a todas as variáveis de entrada,
    retornando um tensor de graus de pertinência.

    Parameters
    ----------
    mf_params : dict, optional
        Parâmetros iniciais das MFs. Se None, usa INITIAL_MF_PARAMS do config.

    Attributes
    ----------
    mfs : nn.ModuleDict
        Dicionário de ModuleLists contendo as MFs por variável.

    Notes
    -----
    Para variáveis com menos conjuntos que MAX_SETS (ex: sweep_quality
    tem 2 conjuntos enquanto as demais têm 3), o tensor de saída é
    preenchido com zeros nas posições excedentes.
    """

    def __init__(
        self,
        mf_params: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    ) -> None:
        super().__init__()

        if mf_params is None:
            mf_params = INITIAL_MF_PARAMS

        self.mfs = nn.ModuleDict()

        for var_name in INPUT_VARS:
            var_mfs = nn.ModuleList()
            sets = LINGUISTIC_SETS[var_name]

            for set_name in sets:
                params = mf_params[var_name][set_name]
                mf = GaussianMF(
                    center=params['center'],
                    sigma=params['sigma'],
                    name=f"{var_name}/{set_name}",
                )
                var_mfs.append(mf)

            self.mfs[var_name] = var_mfs

        n_mfs = sum(len(mf_list) for mf_list in self.mfs.values())
        logger.info(
            f"FuzzificationLayer inicializada: {N_INPUTS} variáveis, "
            f"{n_mfs} MFs gaussianas, MAX_SETS={MAX_SETS}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula graus de pertinência para todas as variáveis e conjuntos.

        Parameters
        ----------
        x : torch.Tensor
            Tensor de entrada com shape [batch, n_inputs].
            Cada coluna corresponde a uma variável em INPUT_VARS order.

        Returns
        -------
        torch.Tensor
            Tensor de pertinências com shape [batch, n_inputs, max_sets].
            Posições sem MF correspondente são preenchidas com 0.
        """
        batch_size = x.shape[0]
        mu = torch.zeros(batch_size, N_INPUTS, MAX_SETS, device=x.device)

        for i, var_name in enumerate(INPUT_VARS):
            x_var = x[:, i]  # [batch]
            mf_list = self.mfs[var_name]

            for j, mf in enumerate(mf_list):
                mu[:, i, j] = mf(x_var)

        return mu

    def get_params_snapshot(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Retorna snapshot dos parâmetros atuais das MFs.

        Returns
        -------
        dict
            Dicionário aninhado {var_name: {set_name: {center, sigma}}}.
        """
        snapshot = {}
        for var_name in INPUT_VARS:
            snapshot[var_name] = {}
            mf_list = self.mfs[var_name]
            sets = LINGUISTIC_SETS[var_name]

            for j, set_name in enumerate(sets):
                mf = mf_list[j]
                snapshot[var_name][set_name] = {
                    'center': mf.center.item(),
                    'sigma': mf.sigma.item(),
                }
        return snapshot

    def __repr__(self) -> str:
        lines = ["FuzzificationLayer("]
        for var_name in INPUT_VARS:
            mf_list = self.mfs[var_name]
            lines.append(f"  {var_name}:")
            for mf in mf_list:
                lines.append(f"    {mf}")
        lines.append(")")
        return "\n".join(lines)
