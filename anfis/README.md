# ANFIS — Adaptive-Network-Based Fuzzy Inference System

Implementação em PyTorch da arquitetura ANFIS de 5 camadas (Jang, 1993) com modelo TSK zero-order, aplicada à quantificação de estratégias de trading baseadas em Smart Money Concepts.

## Referência

> Jang, J.-S. R. (1993). ANFIS: Adaptive-Network-Based Fuzzy Inference System.
> IEEE Transactions on Systems, Man, and Cybernetics, 23(3), 665-685.

## Pré-requisitos

```bash
pip install -r requirements.txt
```

Dependências principais: `torch>=2.0.0`, `scikit-fuzzy`, `seaborn`, `scikit-learn`, `scipy`.

## Execução

### Treinamento com dados sintéticos (padrão)

```bash
python -m anfis.train --mode synthetic --epochs 200
```

### Treinamento com dados de mercado

```bash
python -m anfis.train --mode market --symbol EURUSD=X --epochs 300
```

### Parâmetros opcionais

| Argumento    | Descrição                     | Padrão       |
|------------- |-------------------------------|--------------|
| `--mode`     | `synthetic` ou `market`       | `synthetic`  |
| `--symbol`   | Símbolo do ativo (modo market)| `EURUSD=X`   |
| `--epochs`   | Número de épocas              | `200`        |
| `--lr`       | Learning rate                 | `0.01`       |
| `--n_samples`| Amostras sintéticas           | `5000`       |

## Saídas

| Arquivo                    | Descrição                              |
|--------------------------- |----------------------------------------|
| `anfis_trained.pt`         | Modelo treinado (state_dict)           |
| `results.json`             | Métricas e parâmetros em JSON          |
| `outputs/plots/plot1_*.png`| MFs antes vs depois do treinamento     |
| `outputs/plots/plot2_*.png`| Curva de aprendizado                   |
| `outputs/plots/plot3_*.png`| Comparativo de consequentes            |
| `outputs/plots/plot4_*.png`| Matrizes de confusão                   |
| `outputs/plots/plot5_*.png`| Análise de quantis                     |
| `outputs/plots/plot6_*.png`| Distribuição dos sinais                |

## Testes

```bash
python -m pytest tests/test_anfis.py -v
```

## Arquitetura (5 camadas)

```
Layer 1: Fuzzificação      → MFs gaussianas com center/sigma treináveis
Layer 2: Firing Strength    → Produto (T-norm) dos antecedentes por regra
Layer 3: Normalização       → w̄ᵢ = wᵢ / Σwⱼ
Layer 4: Consequentes TSK   → fᵢ = w̄ᵢ × cᵢ  (cᵢ aprendido via Adam)
Layer 5: Defuzzificação     → y = Σ fᵢ
```

## Estrutura do módulo

```
anfis/
├── __init__.py               # Exports públicos
├── config.py                 # Hiperparâmetros centralizados
├── membership_functions.py   # MFs diferenciáveis (PyTorch)
├── rule_base.py              # 22 regras SMC declarativas
├── anfis_model.py            # Arquitetura nn.Module (5 camadas)
├── adam_trainer.py            # Treinamento via Adam
├── data_pipeline.py          # Dados sintéticos/mercado
├── evaluate.py               # Métricas de avaliação
├── visualize_anfis.py        # 6 plots para o TCC
└── train.py                  # Script principal
```
