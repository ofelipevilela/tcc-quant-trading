# Etapa 3: Redesenho das Features Causais (causal_v3) e Resultados Finais

## 1. Objetivo desta etapa

Esta etapa partiu de um diagnóstico direto sobre o causal_v2: apesar de ser causalmente correto, o sistema ainda não conseguia extrair informação direcional confiável do mercado. A acurácia direcional média, que é o indicador mais limpo de se o modelo aprende o sinal correto, ficou em torno de 48% no walk-forward — abaixo dos 50% de um modelo aleatório.

O objetivo, portanto, não foi ajustar hiperparâmetros do ANFIS. Foi repensar **o que estava sendo entregue como feature ao modelo**: identificar por que as quatro variáveis de entrada carregavam pouca ou nenhuma informação direcional, corrigir as causas raiz, e testar se a nova formulação produzia um edge mensurável.

---

## 2. Diagnóstico do causal_v2 e análise do VFI

### 2.1 O que o causal_v2 tinha resolvido

O causal_v2 havia corrigido três problemas estruturais herdados de versões anteriores:

- **filtro direcional do FVG**: zeros FVG que apontavam contra a tendência, eliminando ruído de gaps contra-trend;
- **leg lock**: estabilizou a definição do range [low, high] da perna atual, evitando que a `price_zone` comprimisse inutilmente após cada novo swing;
- **componente EMA no trend_strength**: adicionou um termo de momentum (EMA9 vs EMA21 normalizado por ATR) ao acumulador histórico do BOS/CISD/FVG.

### 2.2 O problema que permanecia

Mesmo com essas correções, a DirAcc ficou em ~48%. Isso indica que as features não carregavam informação direcional suficiente para o ANFIS generalizar.

Para entender o motivo, foi feita uma análise comparativa com o VFI (*Volume Flow Intelligence*), um indicador Pine Script usado como referência da metodologia ICT/SMC. O VFI opera com duas lógicas fundamentais:

1. **Bias de tendência via HTF**: usa uma comparação simples de abertura vs fechamento numa escala de tempo superior (H1 sobre M15). Isso dá ao sinal uma ancoragem temporal mais estável do que desvios de EMA de período curto.

2. **Entrada apenas após sequência confirmada**: o VFI não dispara sinal até que a sequência completa Sweep → CISD → C4 (displacement/FVG) seja confirmada, nessa ordem. Enquanto a sequência não estiver completa, não há sinal — independente de qualquer outra condição.

Comparando com o ANFIS:

- `trend_strength` no causal_v2 acumulava BOS, CISD, FVG e sweep num único score com decay multiplicativo de 0.92 por barra. Entre um evento SMC e outro, esse score decaía rapidamente para perto de zero, dificultando que o modelo aprendesse a direção estrutural de forma estável.
- `sweep_quality` era calculada como a razão pavio/corpo da barra do sweep — uma métrica de qualidade de execução. Ela dizia **quão forte** foi o sweep, mas não dizia **se o setup estava confirmado ou em qual estágio estava**. O modelo recebia quatro features independentes, sem noção de sequência.

### 2.3 Dois gaps identificados

**Gap 1 — trend_strength como "kitchen-sink"**: o acumulador misturava todos os eventos SMC em um único número que variava muito entre barras. Sem separação clara entre estrutura de longo prazo (BOS), bias de tendência (EMA de maior período) e momentum (EMA de curto prazo), o sinal ficava instável e difícil de aprender.

**Gap 2 — sweep_quality sem contexto sequencial**: entregar ao ANFIS a razão pavio/corpo de um sweep é dar uma métrica de intensidade de evento sem contexto. Na prática operacional, o que importa é saber se (1) houve sweep, (2) esse sweep foi confirmado por CISD, e (3) um FVG de displacement se formou depois. O modelo precisava de uma variável que rastreasse isso, não a intensidade isolada de um evento.

---

## 3. Redesenho implementado no causal_v3

O causal_v3 redesenhou duas features. As outras duas — `price_zone` com leg lock e o filtro direcional do FVG — foram mantidas de v2.

### 3.1 trend_strength V3 — três componentes com peso separado

Em vez de um único acumulador decrescente, o `trend_strength` V3 é a soma de três termos independentes, cada um com papel e cap definido. A soma total fica na faixa `[-100, 100]`.

**Componente 1 — Estrutura BOS com decaimento de recência (±40)**

```
recency = max(0.0, 1.0 - bars_since_bos / 50.0)
bos_component = last_bos_dir * recency * 40.0
```

Capta a direção do último BOS confirmado, com peso que decai linearmente em 50 barras (aproximadamente 12h30 no M15). O decaimento é linear, não exponencial — isso evita que o componente caia para zero muito rápido entre eventos. Depois de 50 barras sem BOS, contribui zero.

**Componente 2 — Bias HTF via EMA50 vs EMA200 (±40)**

```
htf_bias = clip((EMA50 - EMA200) / ATR * 20.0, -40.0, 40.0)
```

O spread EMA50/EMA200 normalizado pelo ATR captura o regime de tendência de prazo mais longo. No M15, a EMA200 equivale aproximadamente a ~50h de preço médio. Esse componente nunca vai a zero entre eventos SMC — ele reflete a posição relativa das médias, que muda lentamente.

**Componente 3 — Momentum de curto prazo via EMA9 vs EMA21 (±20)**

```
momentum = clip((EMA9 - EMA21) / ATR * 15.0, -20.0, 20.0)
```

Captura alinhamento do momentum imediato. Cap menor que os outros dois porque é o mais ruidoso.

Os três componentes precisam concordar para produzir sinal forte. Por exemplo, para obter `trend_strength` próximo de +100, o último BOS precisa ser de alta (recente), as EMAs de longo prazo precisam estar em conformação altista, e o momentum de curto prazo precisa estar positivo.

### 3.2 setup_phase — máquina de estados Sweep → CISD → FVG

O `sweep_quality` foi substituído por `setup_phase`, uma variável no mesmo range `[0, 3]` (mantendo compatibilidade com as MFs Fraco/Forte já calibradas no `anfis/config.py`).

`setup_phase` é uma máquina de estados que rastreia em qual etapa do setup sequencial o mercado está:

| Valor | Significado |
|-------|-------------|
| 0.0 | Nenhum setup ativo (sem sweep recente) |
| 0–1 | Sweep detectado — aguardando CISD (decai por recência) |
| 1–2 | CISD confirmado após Sweep — aguardando FVG |
| 2–3 | Setup completo: Sweep + CISD + FVG de displacement |

Parâmetros operacionais:
- `SETUP_TIMEOUT = 30` barras (~7h30 no M15) — tempo máximo entre eventos consecutivos da sequência; depois disso, o estado reseta.
- `FVG_DISP_MIN_QUAL = 0.3` — qualidade mínima do FVG para contar como displacement válido.

Dentro de cada fase, o valor decai por recência da barra que o ativou:

```
phase_val = N + max(0.0, 1.0 - d / SETUP_TIMEOUT * 0.9)
```

onde N é o piso da fase (0, 1 ou 2) e `d` é a distância em barras desde o evento que iniciou a fase. O coeficiente 0.9 garante que o valor nunca chegue exatamente a zero dentro do timeout.

O modelo rastreia setups bull e bear em paralelo. O `setup_phase` reportado ao ANFIS é o do lado alinhado com `trend_v3`: bull se trend > 0, bear se trend < 0.

**Diferença conceitual em relação ao sweep_quality original**: antes, o modelo sabia "o sweep foi forte". Agora, o modelo sabe "estamos na fase 2 do setup, o CISD confirmou, aguardando FVG" — informação que é estruturalmente mais próxima do raciocínio operacional manual.

---

## 4. Resultados do walk-forward (causal_v3)

O walk-forward foi executado com as mesmas configurações das etapas anteriores para comparação direta:

- Ativo: USTEC.r, timeframe M15
- Total de barras: 20.000 (~junho/2025 a março/2026)
- Estrutura expanding window: train inicial = 8.000 barras, val = 2.000, test = 2.000, step = 2.000
- 4 folds temporais

### Resultados por fold

| Fold | PF | Win Rate | Dir. Acc. | Trades |
|------|----|----------|-----------|--------|
| 1 | 1.00 | 50.29% | 52.96% | 173 |
| 2 | 1.03 | 51.72% | 55.80% | 58 |
| 3 | 1.04 | 51.27% | 53.43% | 314 |
| 4 | 0.91 | 47.95% | 50.79% | 219 |
| **Média** | **0.99** | **50.31%** | **53.25%** | 191 |

### Comparação com modos anteriores

| Modo | PF médio | DirAcc médio | Folds PF≥1 | WR médio |
|------|----------|--------------|------------|----------|
| `legacy_like` | 1.44 | ~61% | 4/4 | 59.4% |
| `causal_bos_anchored` | 1.12* | ~51% | 1/4 | 51.4% |
| **`causal_v3`** | **0.99** | **53.25%** | **3/4** | **50.31%** |
| `causal_v2` | 0.98 | 48.12% | 1/4 | 49.60% |
| `causal_raw` | 0.93 | ~48% | 1/4 | 48.0% |

*causal_bos_anchored com PF 1.12 foi puxado por um fold outlier (1.82); os outros três folds ficaram entre 0.87 e 0.91.

---

## 5. Análise dos resultados

### 5.1 O avanço principal: DirAcc consistentemente acima de 50%

O resultado mais significativo da Etapa 3 não é o PF médio, mas a **acurácia direcional**: todos os quatro folds ficaram acima de 50%, com média de 53.25%. Isso não havia acontecido em nenhum modo causal anterior.

A DirAcc mede se o modelo está acertando a direção correta mais do que errando, **independente da magnitude do sinal ou do threshold de ativação**. Um modelo que acerta sistematicamente a direção — mesmo acertando apenas 53% — demonstra que as features contêm informação direcional real. Modelos anteriores ficavam abaixo de 50%, o que indica que o ANFIS estava, na prática, invertendo a direção ou operando no ruído.

A evolução de 48% (causal_v2) para 53.25% (causal_v3) representa um salto qualitativo: saímos de um regime em que o modelo não extraía edge para um regime em que ele consegue capturar alguma estrutura direcional de forma consistente.

### 5.2 Folds positivos: de 1/4 para 3/4

Com causal_v2, apenas um fold ficou com PF > 1. Com causal_v3, três ficaram. O fold 4 (o mais recente temporalmente) ficou com PF = 0.91, o que pode refletir condições específicas daquele período de mercado — não necessariamente uma falha estrutural do modelo.

### 5.3 Comparação com legacy_like

O `legacy_like` (PF 1.44, DirAcc ~61%, 4/4 folds positivos) permanece significativamente melhor. Isso não é surpresa: o `legacy_like` usa preços de pivot calculados com informação futura (os extremos reais confirmados dos swings), o que é um viés de look-ahead. Os resultados dele representam um limite superior — o melhor que o modelo conseguiria com informação perfeita sobre os swings.

A diferença entre 1.44 (legacy) e 0.99 (causal_v3) quantifica o custo de ser causalmente correto. Uma parte dessa diferença é irrecuperável: são os swings que o sistema só conhece depois, quando a barra já fechou. Outra parte pode ser reduzida com melhorias futuras nas features.

### 5.4 PF médio ainda abaixo de 1

Um PF médio de 0.99 significa que o sistema ainda não é lucrativo de forma consistente. No entanto, considerando o contexto — quatro folds rigorosamente fora do período de treino, sem look-ahead — o resultado indica que o modelo está na fronteira da rentabilidade e que as features carregam edge real.

---

## 6. Limitações observadas

**Volume de trades por fold**: o fold 2 teve apenas 58 operações. Com fonds muito baixo de trades, as métricas são estatisticamente frágeis — o PF de 1.03 nesse fold pode ser ruído.

**Fold 4 negativo**: o último período testado (mais recente cronologicamente) foi o único com PF < 1. Pode indicar que o modelo não generalizou bem para condições de mercado mais recentes, ou que aquele período específico foi estruturalmente diferente (maior ruído, menos liquidez, eventos externos). Sem mais dados, não é possível distinguir.

**Setup_phase com poucos trades no estado 3**: a maioria das barras fica nos estados 0 e 1. O estado 3 (sequência completa) é raro por construção — exige três eventos em sequência dentro de um timeout de 30 barras. Isso pode estar limitando o número de sinais que chegam ao ANFIS com informação de sequência completa.

**Calibração do threshold**: o critério usado (`(pf-1) × √trades`) premiou folds com mais trades quando o PF era marginal. Isso pode ter contribuído para o threshold ser mais permissivo em alguns folds.

---

## 7. Arquivos produzidos

```
outputs/walkforward_compare/causal_v3/
├── fold_01_model.pt          # pesos do ANFIS treinado no fold 1
├── fold_02_model.pt
├── fold_03_model.pt
├── fold_04_model.pt
├── fold_01_training_loss.png # curva de perda treino × validação
├── fold_02_training_loss.png
├── fold_03_training_loss.png
├── fold_04_training_loss.png
├── walkforward_fold_metrics.png  # gráfico de métricas por fold
├── walkforward_threshold_heatmap.png
├── walkforward_summary.csv
└── walkforward_summary.json

outputs/walkforward_compare/
├── mode_comparison_aggregate.png  # comparação entre todos os modos
├── mode_comparison_folds.png
├── mode_comparison_oos_equity.png
├── mode_comparison_summary.csv
└── mode_comparison_summary.json
```

---

## 8. Código implementado

### Arquivos modificados

**`smc/feature_factory.py`**
- Constante `FEATURE_MODE_CAUSAL_V3 = "causal_v3"` adicionada
- Adicionado ao conjunto `VALID_FEATURE_MODES`
- Roteamento na função `build_smc_features()` para chamar `_build_causal_v3_dataset()`
- Função `_build_causal_v3_dataset()` implementada (~280 linhas)

**`run_walkforward_mode_comparison.py`**
- Importa `FEATURE_MODE_CAUSAL_V3`
- Executado com `modes = [FEATURE_MODE_CAUSAL_V3]` para rodar o walk-forward isolado

### Parâmetros do causal_v3

```python
# trend_strength
BOS_DECAY_WINDOW   = 50     # barras para decay linear do BOS recency
HTF_SCALE          = 20.0   # multiplicador (EMA50-EMA200)/ATR → ±40
HTF_CAP            = 40.0
MOMENTUM_SCALE     = 15.0   # multiplicador (EMA9-EMA21)/ATR → ±20
MOMENTUM_CAP       = 20.0

# setup_phase
SETUP_TIMEOUT      = 30     # barras máximas entre eventos da sequência
FVG_DISP_MIN_QUAL  = 0.3   # qualidade mínima do FVG para contar em fase 3
INTRA_PHASE_DECAY  = 0.9   # coeficiente de decaimento intra-fase
```

---

## 9. Estado final do projeto

O pipeline completo está funcional e documentado:

```
[MT5 OHLCV] → [SMC Indicators] → [causal_v3 features] → [ANFIS TSK] → [Walk-Forward] → [Backtest]
```

**Parâmetros fixos do ANFIS**: 4 inputs, 22 regras, 44 parâmetros treináveis, Gaussian MFs, Adam, patience=24.

**Modo de referência para a monografia**: `causal_v3` é o modo causalmente correto com melhor desempenho comprovado. `legacy_like` serve como limite superior (referência com viés de look-ahead declarado).

**Próximos passos possíveis** (para trabalho futuro, fora do escopo do TCC):
- Aumentar o histórico de dados para mais de 4 folds;
- Testar o impacto de `SETUP_TIMEOUT` diferente (ex: 20 ou 40 barras);
- Explorar se o estado 3 do setup_phase pode ser separado em uma variável binária adicional;
- Comparar com um baseline simples (ex: EMA crossover) para quantificar o ganho marginal do SMC.
