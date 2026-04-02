#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script principal de treinamento do ANFIS.

Uso:
    python -m anfis.train --mode synthetic --epochs 200
    python -m anfis.train --mode market --symbol EURUSD=X --epochs 300

Etapas:
    1. Carregar configuração e definir seed
    2. Carregar dados (sintético ou mercado)
    3. Inicializar RuleBase e validar
    4. Inicializar ANFISModel
    5. Treinar via AdamTrainer
    6. Avaliar no conjunto de teste
    7. Gerar todos os plots
    8. Salvar modelo e resultados
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Adicionar raiz ao path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from anfis.adam_trainer import AdamTrainer
from anfis.anfis_model import ANFISModel
from anfis.config import (
    BEST_MODEL_PATH,
    INITIAL_MF_PARAMS,
    MODEL_SAVE_PATH,
    OUTPUT_DIR,
    RESULTS_PATH,
    SIGNAL_THRESHOLDS,
    TRAINING,
)
from anfis.data_pipeline import (
    generate_synthetic_data,
    load_market_data,
    prepare_dataloaders,
)
from anfis.evaluate import compare_before_after, compute_metrics
from anfis.rule_base import RULES, RuleBase
from anfis.visualize_anfis import generate_all_plots

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Define seed para reproducibilidade completa."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Seed definida: {seed}")


def compute_mamdani_predictions(X: np.ndarray) -> np.ndarray:
    """
    Calcula predições do sistema Mamdani para comparação.

    Parameters
    ----------
    X : np.ndarray
        Inputs [n_samples, 4].

    Returns
    -------
    np.ndarray
        Predições Mamdani [n_samples].
    """
    from fuzzy.fuzzy_system import SMCFuzzySystem

    logger.info("Calculando predições do Mamdani (baseline)...")
    mamdani = SMCFuzzySystem()
    preds = []

    for i in range(len(X)):
        try:
            result = mamdani.compute(
                trend_strength=float(X[i, 0]),
                price_zone=float(X[i, 1]),
                fvg_quality=float(X[i, 2]),
                sweep_quality=float(X[i, 3]),
            )
            preds.append(result['signal'])
        except Exception:
            preds.append(0.0)

    return np.array(preds)


def main():
    """Pipeline principal de treinamento."""
    parser = argparse.ArgumentParser(description='Treinamento ANFIS — TCC')
    parser.add_argument('--mode', type=str, default='synthetic',
                        choices=['synthetic', 'market'],
                        help='Fonte de dados: synthetic ou market')
    parser.add_argument('--symbol', type=str, default='EURUSD=X',
                        help='Símbolo do ativo (modo market)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Número de épocas (sobrescreve config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (sobrescreve config)')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Número de amostras sintéticas')
    args = parser.parse_args()

    # ========================================
    # 1. Configuração e seed
    # ========================================
    config = dict(TRAINING)  # cópia
    if args.epochs:
        config['epochs'] = args.epochs
    if args.lr:
        config['learning_rate'] = args.lr
    if args.n_samples:
        config['n_synthetic'] = args.n_samples

    set_seed(config['random_seed'])

    print("\n" + "=" * 70)
    print("  ANFIS — Adaptive-Network-Based Fuzzy Inference System")
    print("  TCC: Quantificação de Estratégias Discricionárias de Trading")
    print("=" * 70)
    print()

    # ========================================
    # 2. Carregar dados
    # ========================================
    if args.mode == 'synthetic':
        df = generate_synthetic_data(
            n_samples=config['n_synthetic'],
            noise_level=config['noise_level'],
            seed=config['random_seed'],
        )
        temporal_split = False
    else:
        df = load_market_data(symbol=args.symbol)
        temporal_split = True

    train_loader, val_loader, test_loader, metadata = prepare_dataloaders(
        df, config, temporal_split=temporal_split,
    )

    # ========================================
    # 3. Inicializar RuleBase
    # ========================================
    rule_base = RuleBase()
    rule_base.print_all_rules()

    # ========================================
    # 4. Inicializar ANFISModel
    # ========================================
    model = ANFISModel(rule_base)
    print(model.summary())

    # Guardar parâmetros iniciais para comparação
    initial_mf_params = model.get_mf_params()
    initial_consequents = model.consequents.detach().cpu().numpy().tolist()

    # ========================================
    # 5. Treinar via AdamTrainer
    # ========================================
    logger.info(f"Dados: train={metadata['n_train']}, val={metadata['n_val']}, "
                f"test={metadata['n_test']}")

    trainer = AdamTrainer(model, config)
    history = trainer.train(train_loader, val_loader)

    # ========================================
    # 6. Avaliar no conjunto de teste
    # ========================================
    logger.info("Avaliando no conjunto de teste...")
    model.eval()

    test_preds = []
    test_targets = []
    test_inputs = []

    with torch.no_grad():
        device = next(model.parameters()).device
        for xb, yb in test_loader:
            xb = xb.to(device)
            y_pred, _, _ = model(xb)
            test_preds.extend(y_pred.cpu().numpy().flatten())
            test_targets.extend(yb.numpy().flatten())
            test_inputs.extend(xb.cpu().numpy())

    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    test_inputs = np.array(test_inputs)

    # Métricas ANFIS
    print("\n--- MÉTRICAS ANFIS (Teste) ---")
    anfis_metrics = compute_metrics(test_preds, test_targets, verbose=True)

    # Predições Mamdani nos mesmos inputs do teste (garante correspondência)
    mamdani_preds = compute_mamdani_predictions(test_inputs)

    # Comparação
    comparison = compare_before_after(mamdani_preds, test_preds, test_targets)

    # ========================================
    # 7. Gerar plots
    # ========================================
    trained_mf_params = model.get_mf_params()
    trained_consequents = model.consequents.detach().cpu().numpy().tolist()

    rule_descs = [rule_base.describe_rule(i) for i in range(rule_base.n_rules)]

    # Confusion matrices
    from sklearn.metrics import confusion_matrix
    from anfis.evaluate import _classify_array

    labels = list(SIGNAL_THRESHOLDS.keys())
    cm_mamdani = confusion_matrix(
        _classify_array(test_targets), _classify_array(mamdani_preds),
        labels=labels,
    )
    cm_anfis = confusion_matrix(
        _classify_array(test_targets), _classify_array(test_preds),
        labels=labels,
    )

    generate_all_plots(
        initial_mf_params=initial_mf_params,
        trained_mf_params=trained_mf_params,
        history=history,
        initial_consequents=initial_consequents,
        trained_consequents=trained_consequents,
        rule_descriptions=rule_descs,
        cm_mamdani=cm_mamdani,
        cm_anfis=cm_anfis,
        quantile_data=anfis_metrics['quantile_analysis'],
        anfis_preds=test_preds,
        targets=test_targets,
        output_dir=OUTPUT_DIR,
    )

    # ========================================
    # 8. Salvar modelo e resultados
    # ========================================
    # Modelo treinado
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logger.info(f"Modelo salvo: {MODEL_SAVE_PATH}")

    # Resultados em JSON
    results = {
        'mode': args.mode,
        'symbol': args.symbol if args.mode == 'market' else 'synthetic',
        'config': {k: v for k, v in config.items() if isinstance(v, (int, float, str))},
        'metadata': {k: v for k, v in metadata.items() if k != 'stats'},
        'anfis_metrics': {
            k: v for k, v in anfis_metrics.items()
            if k not in ('confusion_matrix', 'classification_report', 'quantile_analysis')
        },
        'mamdani_metrics': {
            k: v for k, v in comparison['mamdani'].items()
            if k not in ('confusion_matrix', 'classification_report', 'quantile_analysis')
        },
        'initial_consequents': initial_consequents,
        'trained_consequents': trained_consequents,
        'initial_mf_params': initial_mf_params,
        'trained_mf_params': trained_mf_params,
        'n_epochs_trained': len(history['train_loss']),
    }

    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Resultados salvos: {RESULTS_PATH}")

    print("\n" + "=" * 70)
    print("  ✓ Treinamento concluído!")
    print(f"  ✓ Modelo salvo: {MODEL_SAVE_PATH}")
    print(f"  ✓ Resultados: {RESULTS_PATH}")
    print(f"  ✓ Plots: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
