# -*- coding: utf-8 -*-
"""
Configurações centralizadas do sistema ANFIS.

Todos os hiperparâmetros, universos de discurso, parâmetros iniciais
das funções de pertinência e thresholds de classificação são definidos
aqui. Nenhum número mágico nos demais módulos.

As MFs antecedentes são todas gaussianas para garantir diferenciabilidade
completa — requisito fundamental da arquitetura ANFIS (Jang, 1993).
"""

from typing import Dict, Tuple

# ============================================================================
# Universos de discurso por variável
# ============================================================================
UNIVERSES: Dict[str, Tuple[float, float]] = {
    'trend_strength': (-100.0, 100.0),
    'price_zone':     (0.0, 1.0),
    'fvg_quality':    (0.0, 4.0),
    'sweep_quality':  (0.0, 3.0),
    'trade_signal':   (-100.0, 100.0),
}

# ============================================================================
# Conjuntos linguísticos por variável de entrada
# ============================================================================
LINGUISTIC_SETS: Dict[str, list] = {
    'trend_strength': ['Baixa', 'Neutra', 'Alta'],
    'price_zone':     ['Discount', 'Equilibrium', 'Premium'],
    'fvg_quality':    ['Pequeno', 'Padrao', 'Grande'],
    'sweep_quality':  ['Fraco', 'Forte'],
}

# Número máximo de conjuntos por variável (para padding de tensores)
MAX_SETS: int = max(len(v) for v in LINGUISTIC_SETS.values())

# Nomes das variáveis de entrada na ordem do tensor
INPUT_VARS: list = ['trend_strength', 'price_zone', 'fvg_quality', 'sweep_quality']
N_INPUTS: int = len(INPUT_VARS)

# ============================================================================
# Parâmetros iniciais das MFs gaussianas
#
# Estes valores são a INICIALIZAÇÃO — o ANFIS os ajustará via backprop.
# Foram derivados como aproximações gaussianas das MFs originais do
# Mamdani (zmf, smf, trimf, gaussmf) para manter semântica equivalente.
# ============================================================================
INITIAL_MF_PARAMS: Dict[str, Dict[str, Dict[str, float]]] = {
    'trend_strength': {
        'Baixa':  {'center': -75.0, 'sigma': 20.0},
        'Neutra': {'center':   0.0, 'sigma': 25.0},
        'Alta':   {'center':  75.0, 'sigma': 20.0},
    },
    'price_zone': {
        'Discount':    {'center': 0.15, 'sigma': 0.12},
        'Equilibrium': {'center': 0.50, 'sigma': 0.12},
        'Premium':     {'center': 0.85, 'sigma': 0.12},
    },
    'fvg_quality': {
        'Pequeno': {'center': 0.5,  'sigma': 0.4},
        'Padrao':  {'center': 1.5,  'sigma': 0.5},
        'Grande':  {'center': 3.0,  'sigma': 0.6},
    },
    'sweep_quality': {
        'Fraco': {'center': 0.4, 'sigma': 0.4},
        'Forte': {'center': 2.0, 'sigma': 0.5},
    },
}

# ============================================================================
# Configurações de treinamento (Adam + Early Stopping)
# ============================================================================
TRAINING: Dict[str, object] = {
    'epochs':        200,
    'learning_rate': 0.01,       # lr base (aplicado aos MF params)
    'consequent_lr_mult': 5.0,   # multiplicador de lr para consequentes
    'batch_size':    64,
    'weight_decay':  1e-4,       # L2 regularization via Adam
    'grad_clip':     1.0,        # max_norm para gradient clipping
    'patience':      20,         # épocas sem melhora → early stopping
    'scheduler_patience': 15,    # épocas sem melhora → reduz lr
    'log_every':     10,         # épocas entre logs detalhados
    'train_split':   0.70,
    'val_split':     0.15,
    'test_split':    0.15,
    'random_seed':   42,
    'n_synthetic':   5000,       # amostras para dados sintéticos
    'noise_level':   0.1,        # nível de ruído nos dados sintéticos
    'lambda_mf':     0.01,       # peso da penalidade de sobreposição/ordenação
    'sigma_min':     1e-3,       # sigma mínimo das MFs (evita colapso)
}

# ============================================================================
# Thresholds de classificação do sinal final
# ============================================================================
SIGNAL_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    'VENDA_FORTE':  (-100.0, -60.0),
    'VENDA':        (-60.0,  -20.0),
    'NEUTRO':       (-20.0,   20.0),
    'COMPRA':       ( 20.0,   60.0),
    'COMPRA_FORTE': ( 60.0,  100.0),
}

# Ordem semântica esperada dos consequentes (para penalidade de ordenação)
SEMANTIC_ORDER: list = ['VENDA_FORTE', 'VENDA', 'NEUTRO', 'COMPRA', 'COMPRA_FORTE']

# ============================================================================
# Caminhos de saída
# ============================================================================
OUTPUT_DIR: str = 'outputs/plots'
MODEL_SAVE_PATH: str = 'anfis_trained.pt'
BEST_MODEL_PATH: str = 'anfis_best.pt'
RESULTS_PATH: str = 'results.json'
