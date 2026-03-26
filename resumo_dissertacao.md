# Resumo Técnico para Dissertação - TCC Quant Trading

Este documento sintetiza os principais componentes técnicos do sistema híbrido de trading desenvolvido, estruturado para facilitar a transposição para o texto da dissertação.

## 1. Visão Geral do Sistema
O objetivo é reduzir a subjetividade na análise de **Smart Money Concepts (SMC)** aplicando **Lógica Fuzzy (Mamdani)**. O sistema atua como um classificador de setups que gera um `Trade_Score` (0-100), determinando a direção da operação e o tamanho da posição.

## 2. Modelagem das Variáveis Lingüísticas (Antecedentes)

### 2.1 Trend Strength (Força da Tendência)
- **Universo**: $[-100, 100]$ (Escala bipolar)
- **Significado**: Slope da média móvel ou força direcional.
- **Conjuntos Fuzzy (Gaussiana)**: 
    - `Baixa`: $\mu(x) = \exp(-(x + 100)^2 / (2 \cdot 30^2))$
    - `Neutra`: $\mu(x) = \exp(-x^2 / (2 \cdot 25^2))$
    - `Alta`: $\mu(x) = \exp(-(x - 100)^2 / (2 \cdot 30^2))$

### 2.2 Price Zone (Zona de Preço)
- **Universo**: $[0, 1]$ (% do range atual)
- **Significado**: Localização do price action em relação ao equilíbrio (0 = fundo do range, 1 = topo).
- **Conjuntos Fuzzy (Trapezoidal)**:
    - `Deep Discount` [0, 0, 0.1, 0.2]
    - `Discount` [0.15, 0.25, 0.35, 0.45]
    - `Equilibrium` [0.4, 0.47, 0.53, 0.6]
    - `Premium` [0.55, 0.65, 0.75, 0.85]
    - `Deep Premium` [0.8, 0.9, 1.0, 1.0]

### 2.3 FVG Quality (Qualidade do FVG)
- **Universo**: $[0, 4]$ (Razão FVG_Size / ATR)
- **Conjuntos**: `Pequeno` (Triangular), `Padrao` (Triangular), `Grande` (Sigmoidal/S-shape).

### 2.4 Sweep Quality (Qualidade da Captura)
- **Universo**: $[0, 3]$ (Razão pavio / corpo)
- **Conjuntos**: `Fraco` (Z-shape), `Forte` (S-shape).

## 3. Base de Regras (Sparse Rule Base)
Foram implementadas **26 regras de especialista**, divididas em cenários de COMPRA, VENDA e NEUTRO. O sistema utiliza a filosofia de "Base Dispersa", focando apenas em regras que resultam em ação (`Score > 40`).

### Exemplo de Regras Críticas:
1. **Regra de Ouro (Compra)**: SE `Trend` é `Alta` E `Price_Zone` é `Deep_Discount` E `FVG` é `Grande` E `Sweep` é `Forte` ENTÃO `Score` é `Muito_Forte`.
2. **Regra de Compensação**: SE `Trend` é `Neutra` MAS `Price_Zone` é `Deep_Discount` (precificação extrema) ENTÃO o `Score` é elevado para `Moderado` (entrada de risco).

## 4. Inferência e Tomada de Decisão

### 4.1 Processo de Inferência Mamdani
1. **Fuzzificação**: Conversão de entradas numéricas (ex: trend = 80) em graus de pertinência em cada conjunto.
2. **Implicação**: Utilização do operador `MIN` (AND/Conjunction) para determinar o nível de ativação de cada regra.
3. **Agregação**: Operador `MAX` (OR/Disjunction) para unir as áreas das funções de pertinência de saída resultantes de todas as regras.
4. **Defuzzificação**: Método do **Centroide**, calculando o ponto de equilíbrio de massa da área agregada para obter o `Trade_Score` final.

### 4.2 Lógica de Execução (Position Sizing)
O sistema traduz o score numérico em ações discretas:
- **Score [80-100] (Muito Forte)**: Lote cheio (100% do risco).
- **Score [60-80] (Forte)**: Lote cheio.
- **Score [40-60] (Moderado)**: Meio lote (50% do risco).
- **Score [0-40] (Fraco)**: Sem ação (Aguardar confirmação).

## 5. Contribuições para a Dissertação
- **Hibridismo**: Integração de análise técnica subjetiva (SMC) com rigor matemático (Fuzzy).
- **Visualização**: Desenvolvimento de ferramentas para "abrir a caixa preta" do modelo fuzzy, permitindo auditoria visual das decisões do algoritmo.
- **Gerenciamento de Risco Dinâmico**: O dimensionamento de posição baseado na incerteza (pertinência) é um diferencial em relação a modelos binários (compra/venda).
