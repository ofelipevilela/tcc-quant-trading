#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pipeline ANFIS completo com saida para arquivo."""
import sys
import os
import time
import logging

# Redirecionar stdout/stderr para devnull (evitar bloqueio terminal)
devnull = open(os.devnull, 'w')
sys.stdout = devnull
sys.stderr = devnull

# Log para arquivo
lh = logging.FileHandler('train_log.txt', encoding='utf-8', mode='w')
lh.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S'))
root = logging.getLogger()
root.setLevel(logging.INFO)
root.addHandler(lh)
L = logging.getLogger('pipeline')

try:
    L.info("=" * 60)
    L.info("ANFIS Pipeline — TCC Quant Trading")
    L.info("=" * 60)

    L.info("ETAPA 1: Importacoes...")
    from anfis.data_pipeline import generate_synthetic_data, prepare_dataloaders
    from anfis.rule_base import RuleBase
    from anfis.anfis_model import ANFISModel
    from anfis.adam_trainer import AdamTrainer
    from anfis.evaluate import compute_metrics, _classify_array
    from anfis.config import (
        TRAINING, SIGNAL_THRESHOLDS, MODEL_SAVE_PATH,
        RESULTS_PATH, OUTPUT_DIR,
    )
    import numpy as np
    import torch
    L.info("OK: Importacoes bem-sucedidas")

    # Seed
    np.random.seed(TRAINING['random_seed'])
    torch.manual_seed(TRAINING['random_seed'])

    L.info("ETAPA 2: Geracao de dados sinteticos...")
    t0 = time.time()
    df = generate_synthetic_data(
        n_samples=TRAINING['n_synthetic'],
        noise_level=TRAINING['noise_level'],
        seed=TRAINING['random_seed'],
    )
    L.info(f"OK: {df.shape[0]} amostras em {time.time()-t0:.2f}s")

    L.info("ETAPA 3: DataLoaders...")
    config = dict(TRAINING)
    train_loader, val_loader, test_loader, metadata = prepare_dataloaders(
        df, config, temporal_split=False,
    )
    L.info(f"OK: train={metadata['n_train']}, val={metadata['n_val']}, test={metadata['n_test']}")

    L.info("ETAPA 4: RuleBase + ANFISModel...")
    rule_base = RuleBase()
    model = ANFISModel(rule_base)
    n_params = sum(p.numel() for p in model.parameters())
    L.info(f"OK: {model.n_rules} regras, {n_params} parametros treinaveis")

    initial_mf_params = model.get_mf_params()
    initial_consequents = model.consequents.detach().cpu().numpy().tolist()

    L.info("ETAPA 5: Treinamento Adam...")
    t0 = time.time()
    trainer = AdamTrainer(model, config)
    history = trainer.train(train_loader, val_loader)
    n_epochs = len(history['train_loss'])
    L.info(f"OK: {n_epochs} epocas em {time.time()-t0:.1f}s")
    L.info(f"  train_loss={history['train_loss'][-1]:.4f}")
    L.info(f"  val_loss={history['val_loss'][-1]:.4f}")
    L.info(f"  val_IC={history['val_ic'][-1]:.4f}")

    L.info("ETAPA 6: Avaliacao no teste...")
    model.eval()
    test_preds, test_targets, test_inputs = [], [], []
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

    metrics = compute_metrics(test_preds, test_targets, verbose=False)
    L.info(f"OK: Metricas ANFIS:")
    L.info(f"  MAE           = {metrics['mae']:.4f}")
    L.info(f"  RMSE          = {metrics['rmse']:.4f}")
    L.info(f"  R2            = {metrics['r2']:.4f}")
    L.info(f"  Dir. Accuracy = {metrics['directional_accuracy']:.1f}%")
    L.info(f"  IC (Spearman) = {metrics['ic']:.4f}")
    L.info(f"  F1 Macro      = {metrics['f1_macro']:.4f}")
    L.info(f"  F1 Weighted   = {metrics['f1_weighted']:.4f}")

    L.info("ETAPA 7: Predicoes Mamdani (baseline)...")
    # Calcular predicoes via regras analiticas (como baseline)
    from anfis.data_pipeline import _compute_synthetic_signal
    mamdani_preds = _compute_synthetic_signal(
        test_inputs[:, 0], test_inputs[:, 1],
        test_inputs[:, 2], test_inputs[:, 3],
    )
    mamdani_metrics = compute_metrics(mamdani_preds, test_targets, verbose=False)
    L.info(f"OK: Metricas Baseline (regras analiticas):")
    L.info(f"  MAE           = {mamdani_metrics['mae']:.4f}")
    L.info(f"  RMSE          = {mamdani_metrics['rmse']:.4f}")
    L.info(f"  R2            = {mamdani_metrics['r2']:.4f}")
    L.info(f"  Dir. Accuracy = {mamdani_metrics['directional_accuracy']:.1f}%")
    L.info(f"  IC (Spearman) = {mamdani_metrics['ic']:.4f}")

    L.info("ETAPA 8: Plots...")
    from anfis.visualize_anfis import generate_all_plots
    from sklearn.metrics import confusion_matrix

    trained_mf_params = model.get_mf_params()
    trained_consequents = model.consequents.detach().cpu().numpy().tolist()
    rule_descs = [rule_base.describe_rule(i) for i in range(rule_base.n_rules)]
    labels = list(SIGNAL_THRESHOLDS.keys())

    cm_mamdani = confusion_matrix(
        _classify_array(test_targets), _classify_array(mamdani_preds), labels=labels,
    )
    cm_anfis = confusion_matrix(
        _classify_array(test_targets), _classify_array(test_preds), labels=labels,
    )

    paths = generate_all_plots(
        initial_mf_params=initial_mf_params,
        trained_mf_params=trained_mf_params,
        history=history,
        initial_consequents=initial_consequents,
        trained_consequents=trained_consequents,
        rule_descriptions=rule_descs,
        cm_mamdani=cm_mamdani,
        cm_anfis=cm_anfis,
        quantile_data=metrics['quantile_analysis'],
        anfis_preds=test_preds,
        targets=test_targets,
        output_dir=OUTPUT_DIR,
    )
    L.info(f"OK: {len(paths)} plots salvos em {OUTPUT_DIR}")

    L.info("ETAPA 9: Salvamento...")
    model_dir = os.path.dirname(str(MODEL_SAVE_PATH))
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    L.info(f"OK: Modelo salvo em {MODEL_SAVE_PATH}")

    import json
    results = {
        'mode': 'synthetic',
        'n_samples': TRAINING['n_synthetic'],
        'n_epochs_trained': n_epochs,
        'anfis_metrics': {
            k: v for k, v in metrics.items()
            if k not in ('confusion_matrix', 'classification_report', 'quantile_analysis')
        },
        'baseline_metrics': {
            k: v for k, v in mamdani_metrics.items()
            if k not in ('confusion_matrix', 'classification_report', 'quantile_analysis')
        },
        'initial_consequents': initial_consequents,
        'trained_consequents': trained_consequents,
        'initial_mf_params': initial_mf_params,
        'trained_mf_params': trained_mf_params,
    }
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    L.info(f"OK: Resultados salvos em {RESULTS_PATH}")

    L.info("=" * 60)
    L.info("PIPELINE COMPLETO — TODAS AS ETAPAS OK!")
    L.info("=" * 60)

except Exception as e:
    L.error(f"ERRO FATAL: {e}")
    import traceback
    L.error(traceback.format_exc())
finally:
    devnull.close()
