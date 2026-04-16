# Etapa 2: Walk-Forward, Refino da Leitura SMC e Consolidação dos Critérios de Avaliação

## 1. Objetivo desta etapa

Depois da Etapa 1, o pipeline deixou de colapsar e passou a gerar sinais operáveis. Mesmo assim, ainda havia um problema importante: os bons resultados estavam muito concentrados em validações e backtests pontuais. Faltava saber se o sistema conseguiria manter algum nível de consistência quando fosse reestimado em janelas diferentes do mercado.

Por isso, esta etapa teve quatro objetivos centrais:

- tornar a leitura SMC mais causal e menos dependente de interpretação implícita;
- revisar o critério de escolha do threshold de ativação do ANFIS;
- medir o comportamento do sistema em múltiplos folds temporais via walk-forward;
- gerar métricas e figuras mais adequadas para uso acadêmico na dissertação.

---

## 2. Problema encontrado no início da etapa

Logo no começo desta fase, ficou claro que a versão causal do SMC ainda não estava boa o suficiente.

Os sintomas principais foram:

- o walk-forward inicial com `swing_window=4` e seleção de threshold por maior profit factor local ficou fraco;
- a média do `profit factor` nos folds caiu para aproximadamente `0.88`;
- nenhum fold ficou com `profit factor > 1`;
- a acurácia direcional média ficou por volta de `47.11%`;
- a média de retorno dos folds de teste ficou negativa.

Em termos práticos, isso significava o seguinte: a versão causal estava mais honesta do ponto de vista temporal, mas a tradução SMC ainda não estava transmitindo informação direcional suficiente para o ANFIS aprender uma vantagem estável.

---

## 3. Hipótese principal desta etapa

A hipótese de trabalho passou a ser esta:

- a causalização do swing foi correta, mas a informação de direção dos eventos SMC ainda estava chegando “fraca demais” ao modelo;
- além disso, o critério de seleção do threshold estava premiando limiares com bom `profit factor` em poucas operações, o que aumentava o risco de superajuste na validação.

Ou seja, havia dois ajustes a fazer ao mesmo tempo:

1. melhorar a tradução estrutural do SMC;
2. tornar a calibração do threshold mais robusta.

---

## 4. Mudanças implementadas

### 4.1 Refatoração do pipeline real

Foi criado o módulo:

- `real_market_utils.py`

Esse arquivo concentrou a lógica compartilhada do pipeline real:

- geração de targets por barreira ATR;
- preparação das features SMC;
- treino do ANFIS;
- métricas de score;
- calibração de threshold por backtest;
- gráficos de treinamento.

Na prática, isso foi importante por dois motivos:

- evitou divergência entre o `train_real_market.py` e o `run_walkforward.py`;
- deixou o experimento mais reprodutível e mais fácil de documentar.

### 4.2 Separação entre score e carteira

No arquivo:

- `backtest/engine.py`

foi criada a função:

- `simulate_trading_from_scores(...)`

Essa função separa duas coisas que antes estavam acopladas:

- a geração dos scores pelo ANFIS;
- a simulação da carteira.

Isso permitiu usar exatamente o mesmo motor operacional para:

- o backtest final;
- a calibração de threshold na validação;
- os folds do walk-forward.

Essa mudança foi importante porque eliminou uma fonte de inconsistência metodológica.

### 4.3 Refino da leitura estrutural do SMC

No arquivo:

- `smc/indicators.py`

o `trend_strength` foi ajustado para absorver melhor a direção recente dos eventos SMC.

Antes, a força de tendência dependia principalmente de:

- BOS;
- CISD;
- decaimento temporal.

Depois do ajuste, ela passou a considerar também:

- direção e intensidade bruta de `FVG`;
- direção e intensidade bruta de `sweep`;
- um bônus adicional quando `FVG` e `sweep` aparecem alinhados.

Isso não mudou o número de inputs do ANFIS. A mudança foi feita dentro da própria tradução do conceito estrutural.

A ideia por trás disso foi simples:

- `FVG` e `sweep` não são apenas magnitudes isoladas;
- eles carregam informação de direção;
- se essa direção não for reaproveitada no estado estrutural, parte da semântica SMC se perde.

### 4.4 Revisão do critério de escolha do threshold

Inicialmente, o threshold vinha sendo escolhido por maior `profit factor` local na validação, respeitando apenas um número mínimo de trades.

Na prática, isso gerou um problema:

- thresholds muito altos podiam parecer bons em janelas pequenas;
- mas isso às vezes acontecia com poucas operações;
- quando o modelo ia para o teste, esse limiar não generalizava bem.

Por isso, a função `select_backtest_threshold(...)` foi revista.

O critério final passou a privilegiar:

- `profit factor` acima de 1;
- número de trades suficiente para reduzir instabilidade;
- vantagem operacional ponderada por `sqrt(total_trades)`.

Além disso, o número mínimo de trades exigido na validação foi aumentado para:

- `30`

Essa mudança foi decisiva para estabilizar a escolha dos limiares.

### 4.5 Ajuste do `swing_window`

Foi feito um search curto em um fold representativo para comparar combinações simples de:

- `swing_window`
- nível do Smart Trimming

Os testes mais relevantes foram:

- `swing_window=4`, `min_activation=0.10`
- `swing_window=4`, `min_activation=0.20`
- `swing_window=5`, `min_activation=0.10`

O melhor compromisso encontrado nessa rodada foi:

- `swing_window=5`
- `min_activation=0.10`

O trimming mais agressivo (`0.20`) até melhorou alguns cenários pontuais, mas tornou a seleção do threshold mais instável e reduziu demais a cobertura.

---

## 5. Experimentos exploratórios que não foram adotados

Esta parte é importante para não confundir tentativa com resultado consolidado.

### 5.1 Contextualização ajustada de `FVG` e `sweep`

Foi mantida no código a possibilidade de usar:

- `fvg_quality_adj`
- `sweep_quality_adj`

mas, nos testes feitos até aqui, a versão `causal_enhanced` não ficou melhor do que a `causal_raw`.

Por isso, a configuração adotada nesta etapa permaneceu:

- `feature_mode = causal_raw`

### 5.2 Smart Trimming mais forte

Também foi testado um trimming mais rígido com:

- `min_activation = 0.20`

Esse ajuste reduziu ruído em alguns cenários, mas deixou a calibração do limiar mais sensível e não trouxe uma melhora consistente quando comparado ao custo de perder cobertura.

Por isso, a configuração consolidada continuou sendo:

- `min_activation = 0.10`

---

## 6. Configuração final adotada nesta etapa

Ao final desta rodada, a configuração consolidada ficou assim:

- `symbol = USTEC.r`
- `timeframe = M15`
- `n_bars = 20000`
- `swing_window = 5`
- `feature_mode = causal_raw`
- `target_mode = atr_barrier`
- `horizon = 15`
- `atr_mult = 1.0`
- `min_activation = 0.10`
- `learning_rate = 0.002`
- `lambda_rule_usage = 0.01`
- `threshold min trades = 30`

---

## 7. Resultado do treino consolidado

No treino único consolidado com a configuração final, o threshold selecionado foi:

- `0.10`

Métricas desse treino único:

- validação direcional: `61.11%`
- teste direcional: `48.65%`
- validação carteira: `profit factor = 1.34`
- teste carteira: `profit factor = 0.76`

Esse resultado isolado é importante por dois motivos:

- ele mostra que ainda existe sensibilidade de regime;
- ele também reforça por que o walk-forward é mais confiável do que um único split temporal.

Em outras palavras, o treino único desta etapa não deve ser usado como evidência principal de robustez.

---

## 8. Resultado do backtest do modelo final

No backtest de `5000` barras com o modelo final salvo, os números ficaram assim:

- retorno total: `1.19%`
- retorno anualizado: `5.78%`
- drawdown máximo: `7.95%`
- win rate: `51.61%`
- profit factor: `1.04`
- payoff ratio: `0.97`
- expectancy: `19.18`
- Sharpe: `0.51`
- Sortino: `0.18`
- Calmar: `0.73`
- SQN: `0.19`
- total de trades: `62`

Esse backtest ficou melhor do que a versão causal anterior, mas ainda é um resultado moderado. Ele sugere viabilidade, mas não sustenta sozinho uma conclusão forte.

---

## 9. Resultado do walk-forward consolidado

O walk-forward final foi executado com:

- 4 folds crescentes;
- treino expandindo ao longo do tempo;
- validação por carteira;
- teste sempre fora da janela usada para calibrar o limiar.

### 9.1 Resultado agregado

Resumo agregado:

- média do `profit factor`: `1.0149`
- mediana do `profit factor`: `0.9662`
- média do `win rate`: `50.39%`
- média da acurácia direcional: `50.44%`
- média do `drawdown` máximo: `14.66%`
- média do retorno dos folds: `-2.56%`
- folds com `profit factor > 1`: `2/4`
- folds com retorno positivo: `2/4`

### 9.2 Resultado por fold

Fold 1:

- threshold: `0.07`
- profit factor: `1.006`
- retorno: `0.72%`
- win rate: `50.42%`
- trades: `238`

Fold 2:

- threshold: `0.12`
- profit factor: `0.926`
- retorno: `-1.15%`
- win rate: `48.39%`
- trades: `31`

Fold 3:

- threshold: `0.07`
- profit factor: `0.865`
- retorno: `-15.74%`
- win rate: `46.52%`
- trades: `230`

Fold 4:

- threshold: `0.10`
- profit factor: `1.262`
- retorno: `5.93%`
- win rate: `56.25%`
- trades: `48`

### 9.3 Interpretação honesta

Este foi o principal ganho metodológico da etapa:

- a versão causal inicial tinha ficado com média de `profit factor` em torno de `0.88` e nenhum fold positivo;
- a versão refinada subiu para média ligeiramente acima de `1.0`, com `2` folds positivos.

Isso não significa que o sistema está pronto ou validado de forma definitiva.

O que dá para afirmar com honestidade é:

- houve melhora real em relação à versão causal anterior;
- a estratégia saiu de um regime claramente desfavorável para um regime marginalmente positivo em média;
- a vantagem estatística observada ainda é modesta e não está distribuída de forma uniforme entre todos os folds.

---

## 10. Métricas acadêmicas adotadas

Para a dissertação, esta etapa consolidou um conjunto mais adequado de métricas.

### 10.1 Métricas de carteira

- retorno total
- retorno anualizado
- drawdown máximo
- win rate
- profit factor
- payoff ratio
- expectancy
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- SQN

### 10.2 Métricas do score fuzzy

- acurácia direcional
- cobertura de ativações
- correlação score-target
- desvio padrão do score
- dominância média de regra
- número de regras efetivamente ativas

Isso foi importante porque o trabalho passou a avaliar duas camadas:

1. a qualidade estatística do sinal fuzzy;
2. o comportamento operacional da carteira.

---

## 11. Artefatos gerados nesta etapa

### 11.1 Documentação

- `docs/ETAPA_01_OTIMIZACAO_ANFIS.md`
- `docs/ETAPA_02_WALKFORWARD_E_REFINO_SMC.md`

### 11.2 Artefatos do modelo

- `anfis_trained_mt5.pt`
- `anfis_trained_mt5_meta.json`

### 11.3 Backtest e treino

- `outputs/plots/mt5_training_loss.png`
- `outputs/plots/backtest_results.png`

### 11.4 Walk-forward

- `outputs/walkforward/walkforward_summary.csv`
- `outputs/walkforward/walkforward_summary.json`
- `outputs/walkforward/walkforward_equity_curves.png`
- `outputs/walkforward/walkforward_fold_metrics.png`
- `outputs/walkforward/walkforward_threshold_heatmap.png`
- `outputs/walkforward/fold_01_training_loss.png`
- `outputs/walkforward/fold_02_training_loss.png`
- `outputs/walkforward/fold_03_training_loss.png`
- `outputs/walkforward/fold_04_training_loss.png`

### 11.5 Comparações visuais da evolução fuzzy

- `outputs/plots/stage2_mf_adaptation_comparison.png`
- `outputs/plots/stage2_consequent_comparison.png`
- `outputs/plots/stage2_summary_panels.png`

---

## 12. O que esta etapa resolveu de fato

As melhorias desta etapa resolveram quatro pontos importantes:

- a leitura SMC ficou mais rica do ponto de vista causal;
- o threshold passou a ser calibrado com um critério mais consistente;
- o projeto ganhou um protocolo walk-forward reproduzível;
- os outputs passaram a ter utilidade direta para a dissertação.

---

## 13. O que ainda continua em aberto

Mesmo com a melhora, ainda existem limitações claras:

- a vantagem média ainda é pequena;
- a distribuição de desempenho entre folds ainda é irregular;
- o Fold 3 continuou fraco, indicando sensibilidade a regime;
- o score fuzzy ainda não mostra correlação forte e estável em todos os cenários;
- a validação por carteira ainda depende bastante da escolha do horizonte e da barreira ATR.

---

## 14. Próximos passos sugeridos

Os próximos passos mais naturais, depois desta etapa, são:

- testar múltiplos horizontes de barreira no walk-forward;
- investigar se vale separar melhor informação bullish e bearish de `FVG` e `sweep`;
- comparar `reward_to_risk` diferentes na mesma janela walk-forward;
- documentar em texto acadêmico a diferença entre resultado pontual de backtest e robustez entre folds;
- transformar as figuras geradas nesta etapa em material direto para a seção de resultados e discussão.

---

## 15. Conclusão resumida da etapa

Se eu resumisse esta etapa em poucas linhas:

1. a primeira versão causal do SMC ficou honesta, mas fraca;
2. o `trend_strength` foi refinado para absorver melhor o viés direcional de `FVG` e `sweep`;
3. o critério de seleção do threshold foi revisado para reduzir superajuste;
4. o `swing_window=5` apresentou melhor equilíbrio do que `4` nesta rodada;
5. o walk-forward saiu de uma média claramente negativa para uma média de `profit factor` ligeiramente acima de `1`;
6. a vantagem estatística encontrada ainda é modesta, mas agora já existe evidência mais defensável de melhora em relação à versão causal anterior.
