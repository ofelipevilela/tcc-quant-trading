# TCC Quant Trading: Sistema Híbrido SMC + Fuzzy Logic

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Descrição

Sistema de trading quantitativo híbrido que combina **Smart Money Concepts (SMC)** com **Lógica Fuzzy (Mamdani)** para avaliação de setups de trading. O projeto faz parte de um TCC e visa evoluir para incluir **Neuro Fuzzy (ANFIS)**.

### Objetivos
- Avaliar a qualidade de setups de compra/venda usando inferência fuzzy
- Integrar indicadores institucionais (FVG, Sweep, Trend, Price Zone)
- Criar um sistema interpretável e transparente de tomada de decisão

---

## 🧠 Sistema Fuzzy

### Variáveis de Entrada (Antecedentes)

| Variável | Input Crisp | Universo | Conjuntos |
|----------|-------------|----------|-----------|
| **Trend_Strength** | ADX / Slope EMA | [0, 100] | Baixa, Neutra, Alta |
| **Price_Zone** | % do Range | [0, 1] | Deep_Discount → Deep_Premium |
| **FVG_Quality** | FVG_Size / ATR | [0, 4] | Pequeno, Padrao, Grande |
| **Sweep_Quality** | Pavio / Corpo | [0, 3] | Fraco, Forte |

### Variável de Saída (Consequente)

| Variável | Universo | Conjuntos | Defuzzificação |
|----------|----------|-----------|----------------|
| **Trade_Score** | [0, 100] | Fraco, Moderado, Forte, Muito_Forte | Centroide |

---

## 🚀 Quick Start

### 1. Instalar Dependências
```bash
pip install -r requirements.txt
.\.venv\Scripts\Activate.ps1

```

### 2. Gerar Visualização das Funções de Pertinência
```bash
python main.py
```

O gráfico será salvo em `outputs/membership_functions.png`.

---

## 📁 Estrutura do Projeto

```
tcc-quant-trading/
├── config/settings.py        # Configuração das variáveis fuzzy
├── fuzzy/
│   ├── membership_functions.py   # Definição das MFs
│   ├── fuzzy_system.py           # Sistema Mamdani (TODO)
│   └── visualization.py          # Visualização dos gráficos
├── smc/indicators.py         # Indicadores SMC (TODO)
├── outputs/                  # Gráficos gerados
├── main.py                   # Entry point
└── PROJETO.md                # Documentação detalhada
```

> 📄 Veja [PROJETO.md](PROJETO.md) para documentação completa de cada arquivo.

---

## 📈 Roadmap

- [x] Estrutura inicial do projeto
- [x] Definição das variáveis fuzzy SMC
- [x] Visualização das Membership Functions
- [ ] Implementação das regras fuzzy Mamdani
- [ ] Integração com biblioteca `smartmoneyconcepts`
- [ ] Pipeline de Backtesting
- [ ] Neuro Fuzzy (ANFIS)

---

## 👥 Autor

- **Felipe Vilela**

## 📄 Licença

Este projeto está licenciado sob a Licença MIT.
