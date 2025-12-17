# 📂 Estrutura do Projeto - TCC Quant Trading

Sistema híbrido de trading combinando **Smart Money Concepts (SMC)** com **Lógica Fuzzy (Mamdani)**.

---

## 🗂️ Estrutura de Diretórios

```
tcc-quant-trading/
├── config/                    # Configurações do sistema
│   ├── __init__.py
│   └── settings.py            # Parâmetros das variáveis fuzzy
├── fuzzy/                     # Módulo de Lógica Fuzzy
│   ├── __init__.py
│   ├── membership_functions.py   # Definição das MFs
│   ├── fuzzy_system.py           # Sistema Mamdani (TODO)
│   └── visualization.py          # Geração de gráficos
├── smc/                       # Smart Money Concepts (TODO)
│   ├── __init__.py
│   └── indicators.py          # Indicadores SMC
├── data/                      # Dados de mercado
├── notebooks/                 # Jupyter notebooks
├── outputs/                   # Gráficos gerados
├── tests/                     # Testes unitários
├── main.py                    # 🚀 Entry point principal
├── requirements.txt           # Dependências Python
└── README.md                  # Documentação geral
```

---

## 📄 Descrição dos Arquivos

### 🔧 Arquivos de Configuração

| Arquivo | Descrição |
|---------|-----------|
| `requirements.txt` | Lista de dependências Python necessárias para o projeto |
| `config/settings.py` | **Configuração central** das variáveis fuzzy: universos, conjuntos e parâmetros das funções de pertinência |

---

### 🧠 Módulo Fuzzy (`fuzzy/`)

| Arquivo | Descrição |
|---------|-----------|
| `fuzzy/__init__.py` | Exporta as funções públicas do módulo: `create_fuzzy_variables()` e `plot_membership_functions()` |
| `fuzzy/membership_functions.py` | **Define as 5 variáveis fuzzy** do sistema usando `scikit-fuzzy`. Cria antecedentes (inputs) e consequente (output) |
| `fuzzy/visualization.py` | Funções para **gerar gráficos** das Membership Functions. Salva em PNG para apresentação |
| `fuzzy/fuzzy_system.py` | *TODO:* Implementará o sistema de inferência Mamdani com as regras fuzzy |

---

### 📊 Variáveis Fuzzy Implementadas

#### **Antecedentes (Entradas)**

| Variável | Input Crisp | Universo | Conjuntos | Curva |
|----------|-------------|----------|-----------|-------|
| `Trend_Strength` | ADX ou Slope EMA | [0, 100] | Baixa, Neutra, Alta | Gaussiana |
| `Price_Zone` | % do Range | [0, 1] | Deep_Discount, Discount, Equilibrium, Premium, Deep_Premium | Trapezoidal |
| `FVG_Quality` | FVG_Size / ATR | [0, 4] | Pequeno, Padrao, Grande | Triangular + Sigmoidal |
| `Sweep_Quality` | Pavio / Corpo | [0, 3] | Fraco, Forte | Sigmoidal (S/Z) |

#### **Consequente (Saída)**

| Variável | Universo | Conjuntos | Defuzzificação |
|----------|----------|-----------|----------------|
| `Trade_Score` | [0, 100] | Fraco, Moderado, Forte, Muito_Forte | Centroide |

---

### 🎯 Módulo SMC (`smc/`) - *Em Desenvolvimento*

| Arquivo | Descrição |
|---------|-----------|
| `smc/__init__.py` | Placeholder para o módulo de Smart Money Concepts |
| `smc/indicators.py` | *TODO:* Integrará com a biblioteca `smartmoneyconcepts` para detectar FVG, Order Blocks, etc. |

---

### 🚀 Arquivo Principal

| Arquivo | Descrição |
|---------|-----------|
| `main.py` | **Entry point** do projeto. Executa a criação das variáveis fuzzy e gera a visualização das MFs |

---

## ▶️ Como Executar

### 1. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 2. Gerar Visualização das MFs
```bash
python main.py
```

### 3. Resultado
- Gráfico exibido na tela
- Arquivo salvo em: `outputs/membership_functions.png`

---

## 📈 Próximos Passos

- [ ] Implementar regras fuzzy em `fuzzy/fuzzy_system.py`
- [ ] Integrar indicadores SMC em `smc/indicators.py`
- [ ] Criar pipeline de backtesting
- [ ] Implementar ANFIS (Neuro Fuzzy)

---

## 🛠️ Stack Tecnológica

| Tecnologia | Uso |
|------------|-----|
| Python 3.10+ | Linguagem principal |
| scikit-fuzzy | Lógica Fuzzy (Mamdani) |
| scipy | Dependência do scikit-fuzzy |
| matplotlib | Visualização de gráficos |
| pandas/numpy | Manipulação de dados |
| smartmoneyconcepts | *Futuro:* Indicadores SMC |
