# -*- coding: utf-8 -*-
"""
Treinamento ANFIS via Adam Backpropagation Unificado.

O Hybrid Learning original de Jang (1993) alternava LSE (para consequentes)
e gradient descent (para antecedentes) por limitações computacionais da época:
o LSE resolve consequentes analiticamente em 1 passo, acelerando convergência
em hardware dos anos 90. Com o otimizador Adam (Kingma & Ba, 2015), que adapta
a taxa de aprendizado por parâmetro via estimativas de momento de primeira e
segunda ordem, todos os parâmetros (MF centers, sigmas e consequentes crisp)
convergem conjuntamente em poucas épocas, eliminando a necessidade do passo LSE
sem perda de qualidade — e frequentemente com ganho, pois Adam navega melhor
superfícies de erro não-convexas típicas de bases de regras fuzzy.

Referências:
    Jang, J.-S. R. (1993). ANFIS: Adaptive-Network-Based Fuzzy Inference
    System. IEEE Transactions on Systems, Man, and Cybernetics, 23(3), 665-685.

    Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic
    Optimization. ICLR 2015. arXiv:1412.6980.
"""

import copy
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from .anfis_model import ANFISModel
from .config import SEMANTIC_ORDER, SIGNAL_THRESHOLDS, TRAINING

logger = logging.getLogger(__name__)


def compute_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    model: ANFISModel,
    lambda_mf: float = 0.01,
    normalized_strengths: Optional[torch.Tensor] = None,
    lambda_rule_usage: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Loss composta de três termos para o ANFIS.

    Parameters
    ----------
    y_pred : torch.Tensor
        Predições do modelo, shape [batch, 1].
    y_true : torch.Tensor
        Targets, shape [batch, 1].
    model : ANFISModel
        Modelo ANFIS (para acessar parâmetros das MFs e consequentes).
    lambda_mf : float
        Peso das penalidades de regularização.

    Returns
    -------
    loss_total : torch.Tensor
        Loss total escalar.
    components : dict
        Dicionário com valores de cada componente para logging.

    Notes
    -----
    Componentes da loss:

    1. MSE — erro quadrático médio entre sinal predito e target.

    2. Penalidade de sobreposição das MFs — penaliza MFs que colapsam
       umas sobre as outras (perda de interpretabilidade linguística).
       Para cada variável, calcula a distância entre centros adjacentes
       e penaliza quando dist < soma dos sigmas (sobreposição excessiva).

    3. Penalidade de ordenação dos consequentes — preserva a semântica:
       regras de COMPRA_FORTE devem ter c > COMPRA > NEUTRO > VENDA > VENDA_FORTE.
       Violações são penalizadas com hinge loss.
    """
    # --- Termo 1: MSE ---
    loss_mse = F.mse_loss(y_pred, y_true)

    # --- Termo 2: Penalidade de sobreposição das MFs ---
    loss_overlap = torch.tensor(0.0, device=y_pred.device)

    from .config import INPUT_VARS, LINGUISTIC_SETS

    for var_name in INPUT_VARS:
        mf_list = model.fuzzification.mfs[var_name]
        n_sets = len(LINGUISTIC_SETS[var_name])

        if n_sets < 2:
            continue

        # Coletar centers e sigmas desta variável
        centers = torch.stack([mf_list[j].center for j in range(n_sets)])
        sigmas = torch.stack([mf_list[j].sigma for j in range(n_sets)])

        # Ordenar por center para verificar adjacência
        sorted_indices = torch.argsort(centers)
        centers_sorted = centers[sorted_indices]
        sigmas_sorted = sigmas[sorted_indices]

        # Penalizar sobreposição excessiva entre conjuntos adjacentes
        for k in range(n_sets - 1):
            dist = torch.abs(centers_sorted[k + 1] - centers_sorted[k])
            sigma_sum = sigmas_sorted[k] + sigmas_sorted[k + 1]
            # Penaliza quando a distância é menor que a soma dos sigmas
            overlap = torch.clamp(sigma_sum - dist, min=0.0)
            loss_overlap = loss_overlap + overlap

    # --- Termo 3: Penalidade de ordenação dos consequentes ---
    loss_order = torch.tensor(0.0, device=y_pred.device)

    consequents = model.consequents
    rule_base = model._rule_base

    # Agrupar consequentes por classe semântica
    class_means = {}
    by_class = rule_base.get_rules_by_class()

    for cls_name in SEMANTIC_ORDER:
        indices = by_class.get(cls_name, [])
        if indices:
            class_means[cls_name] = consequents[indices].mean()

    # Verificar ordenação: VENDA_FORTE < VENDA < NEUTRO < COMPRA < COMPRA_FORTE
    ordered_classes = [c for c in SEMANTIC_ORDER if c in class_means]

    for k in range(len(ordered_classes) - 1):
        c_lower = class_means[ordered_classes[k]]
        c_upper = class_means[ordered_classes[k + 1]]
        # Penaliza se c_lower >= c_upper (violação de ordenação)
        violation = torch.clamp(c_lower - c_upper + 1.0, min=0.0)
        loss_order = loss_order + violation

    # --- Termo 4: diversidade de uso das regras ---
    loss_rule_usage = torch.tensor(0.0, device=y_pred.device)
    if normalized_strengths is not None and lambda_rule_usage > 0:
        mean_usage = normalized_strengths.mean(dim=0)
        mean_usage = mean_usage / (mean_usage.sum() + 1e-8)
        loss_rule_usage = mean_usage.pow(2).sum()

    # --- Loss total ---
    loss_total = (
        loss_mse
        + lambda_mf * (loss_overlap + loss_order)
        + lambda_rule_usage * loss_rule_usage
    )

    components = {
        'mse': loss_mse.item(),
        'overlap': loss_overlap.item(),
        'order': loss_order.item(),
        'rule_usage': loss_rule_usage.item(),
        'total': loss_total.item(),
    }

    return loss_total, components


class AdamTrainer:
    """
    Treinador ANFIS via Adam com differential learning rates, scheduler e
    early stopping baseado em validação.

    Parameters
    ----------
    model : ANFISModel
        Modelo ANFIS a treinar.
    config : dict, optional
        Dicionário de configurações. Se None, usa TRAINING do config.
    device : str, optional
        Dispositivo de execução ('cpu' ou 'cuda').

    Attributes
    ----------
    model : ANFISModel
        Modelo ANFIS.
    optimizer : torch.optim.Adam
        Otimizador com grupos de parâmetros.
    scheduler : ReduceLROnPlateau
        Scheduler de learning rate monitorando ``val_loss``.
    """

    def __init__(
        self,
        model: ANFISModel,
        config: Optional[Dict] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = config if config is not None else TRAINING
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        # Grupos de parâmetros com learning rates distintos:
        # MF params (centers/sigmas) → lr base (mudanças lentas preservam semântica)
        # Consequentes crisp → lr × 5 (convergência mais rápida permitida)
        lr = self.config['learning_rate']
        lr_mult = self.config.get('consequent_lr_mult', 5.0)

        param_groups = [
            {
                'params': model.get_mf_parameters(),
                'lr': lr,
                'name': 'mf_params',
            },
            {
                'params': model.get_consequent_parameters(),
                'lr': lr * lr_mult,
                'name': 'consequents',
            },
        ]

        self.optimizer = torch.optim.Adam(
            param_groups,
            betas=(0.9, 0.999),    # valores padrão Adam
            eps=1e-8,
            weight_decay=self.config.get('weight_decay', 1e-4),
        )

        # Reduz lr pela metade se val_loss não melhorar
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config.get('scheduler_factor', 0.5),
            patience=self.config.get('scheduler_patience', 5),
            min_lr=self.config.get('min_lr', 1e-6),
        )

        logger.info(
            f"AdamTrainer inicializado: device={self.device}, "
            f"lr_mf={lr:.4f}, lr_consequents={lr * lr_mult:.4f}"
        )

    def _current_lrs(self) -> Tuple[float, float]:
        """Retorna os learning rates atuais dos grupos MF e consequentes."""
        return (
            float(self.optimizer.param_groups[0]['lr']),
            float(self.optimizer.param_groups[1]['lr']),
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, List]:
        """
        Executa o loop de treinamento completo.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader do conjunto de treino.
        val_loader : DataLoader
            DataLoader do conjunto de validação.

        Returns
        -------
        dict
            Histórico completo para plotagem:
            - train_loss, val_loss, train_mse, val_mse
            - val_ic (Information Coefficient)
            - lr_mf, lr_consequents
            - mf_snapshots (parâmetros MF a cada época)
        """
        history = {
            'train_loss': [], 'val_loss': [],
            'train_mse': [], 'val_mse': [],
            'train_rule_usage': [], 'val_rule_usage': [],
            'val_ic': [],
            'lr_mf': [], 'lr_consequents': [],
            'mf_snapshots': [],
        }

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_state = None
        lambda_mf = self.config.get('lambda_mf', 0.01)
        lambda_rule_usage = self.config.get('lambda_rule_usage', 0.0)
        max_epochs = self.config['epochs']
        early_stop_patience = self.config.get(
            'early_stop_patience',
            self.config.get('patience', 15),
        )
        min_delta = self.config.get('min_delta', 1e-6)
        patience = early_stop_patience
        log_every = self.config.get('log_every', 10)
        grad_clip = self.config.get('grad_clip', 1.0)

        logger.info(f"Iniciando treinamento: {max_epochs} épocas, patience={patience}")

        logger.info(
            "Scheduler ReduceLROnPlateau: patience=%d, factor=%.3f | "
            "EarlyStopping: patience=%d, min_delta=%.2e",
            self.config.get('scheduler_patience', 5),
            self.config.get('scheduler_factor', 0.5),
            patience,
            min_delta,
        )

        for epoch in range(max_epochs):

            # --- TREINO ---
            self.model.train()
            epoch_losses = []
            epoch_mse = []
            epoch_rule_usage = []

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                self.optimizer.zero_grad()
                y_pred, _, normalized = self.model(xb)
                loss, components = compute_loss(
                    y_pred,
                    yb,
                    self.model,
                    lambda_mf=lambda_mf,
                    normalized_strengths=normalized,
                    lambda_rule_usage=lambda_rule_usage,
                )

                loss.backward()

                # Gradient clipping: MF params têm gradientes
                # potencialmente grandes quando sigma → 0
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=grad_clip,
                )

                self.optimizer.step()

                # Forçar constraints após update
                self.model.clamp_mf_params(
                    sigma_min=self.config.get('sigma_min', None)
                )

                epoch_losses.append(loss.item())
                epoch_mse.append(components['mse'])
                epoch_rule_usage.append(components['rule_usage'])

            # --- VALIDAÇÃO ---
            self.model.eval()
            val_preds_list = []
            val_targets_list = []
            val_loss_accum = []
            val_mse_accum = []
            val_rule_usage_accum = []

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)

                    y_pred, _, normalized = self.model(xb)
                    loss, components = compute_loss(
                        y_pred,
                        yb,
                        self.model,
                        lambda_mf=lambda_mf,
                        normalized_strengths=normalized,
                        lambda_rule_usage=lambda_rule_usage,
                    )

                    val_loss_accum.append(loss.item())
                    val_mse_accum.append(components['mse'])
                    val_rule_usage_accum.append(components['rule_usage'])
                    val_preds_list.extend(y_pred.cpu().numpy().flatten())
                    val_targets_list.extend(yb.cpu().numpy().flatten())

            val_loss = float(np.mean(val_loss_accum))
            val_mse = float(np.mean(val_mse_accum))
            train_loss = float(np.mean(epoch_losses))
            train_mse = float(np.mean(epoch_mse))
            train_rule_usage = float(np.mean(epoch_rule_usage))
            val_rule_usage = float(np.mean(val_rule_usage_accum))

            # Information Coefficient (Spearman rank correlation)
            if len(val_preds_list) > 2:
                ic_result = spearmanr(val_preds_list, val_targets_list)
                val_ic = float(ic_result.correlation) if not np.isnan(ic_result.correlation) else 0.0
            else:
                val_ic = 0.0

            lr_before = self._current_lrs()
            self.scheduler.step(val_loss)
            lr_after = self._current_lrs()
            if lr_after != lr_before:
                logger.info(
                    "ReduceLROnPlateau acionado na época %03d | "
                    "val_loss=%.6f | lr_mf %.2e -> %.2e | "
                    "lr_consequents %.2e -> %.2e",
                    epoch,
                    val_loss,
                    lr_before[0],
                    lr_after[0],
                    lr_before[1],
                    lr_after[1],
                )

            # --- LOGGING ---
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_mse'].append(train_mse)
            history['val_mse'].append(val_mse)
            history['train_rule_usage'].append(train_rule_usage)
            history['val_rule_usage'].append(val_rule_usage)
            history['val_ic'].append(val_ic)
            history['lr_mf'].append(lr_after[0])
            history['lr_consequents'].append(lr_after[1])
            history['mf_snapshots'].append(self.model.get_mf_params())

            if epoch % log_every == 0:
                logger.info(
                    f"Epoch {epoch:03d} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"val_IC={val_ic:.4f} | "
                    f"lr_mf={lr_after[0]:.2e}"
                )

            # --- EARLY STOPPING + CHECKPOINT ---
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_state = copy.deepcopy(self.model.state_dict())
                logger.info(
                    "Novo melhor estado salvo em memória | época=%03d | val_loss=%.6f",
                    epoch,
                    best_val_loss,
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(
                        "Early stopping acionado na época %03d | "
                        "sem novo mínimo de val_loss por %d épocas | "
                        "melhor_val_loss=%.6f",
                        epoch,
                        patience,
                        best_val_loss,
                    )
                    break

        # Restaurar melhor estado
        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info("Pesos restaurados para o melhor estado observado em validação.")

        logger.info(f"Treinamento concluído. Melhor val_loss: {best_val_loss:.4f}")

        return history
