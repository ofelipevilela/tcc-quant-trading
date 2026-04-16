# Experimentos iterativos do ANFIS causal_v3

Este documento registra a bateria de testes executada em 16/04/2026 com dados reais obtidos pelo MetaTrader 5. O objetivo foi partir de um baseline do modo `causal_v3`, comparar ajustes de treinamento e de engenharia do alvo, e registrar os efeitos observados nas métricas acadêmicas e operacionais.

Os testes usaram o ativo `USTEC.r`, timeframe M15, 20.000 barras recentes do MT5, quatro folds walk-forward e pipeline causal `causal_v3`. Como os dados foram puxados novamente do MT5, a janela temporal não é exatamente a mesma dos resultados antigos registrados em `outputs/walkforward_compare/mode_comparison_summary.json`. Por isso, a comparação principal deste documento é entre as iterações executadas nesta mesma bateria.

Após a primeira bateria, foi executada uma segunda rodada com 50.000 barras do MT5 e uma reformulação do alvo supervisionado. Essa rodada buscou aproximar o aprendizado da forma como setups de SMC/ICT são avaliados na prática operacional: não apenas por direção após um número fixo de candles, mas pela relação entre risco assumido e alvo de preço atingido.

## Métricas acompanhadas

- `MAE`: erro absoluto médio entre o retorno alvo e o score previsto. Foi usado como métrica acadêmica principal por ser menos sensível a outliers.
- `RMSE`: erro quadrático médio com raiz. Penaliza erros grandes com mais força que o MAE.
- `R2`: coeficiente de determinação. Valores próximos ou abaixo de zero indicam que o modelo ainda explica pouco da variância do alvo contínuo.
- `DA bruta`: acurácia direcional em todas as amostras do teste, comparando o sinal de `y_pred` com o sinal de `y_true`.
- `DA filtrada`: acurácia direcional apenas quando o score passou do threshold operacional escolhido na validação.
- `Coverage`: fração do teste em que o modelo gerou sinal acima do threshold.
- `Profit Factor`: razão entre lucro bruto e perda bruta. Valores acima de 1 indicam resultado operacional positivo no recorte avaliado.
- `Win Rate`: percentual de trades vencedores.
- `Retorno total`: variação percentual do capital no fold.
- `Max Drawdown`: maior queda percentual da curva de capital.
- `Rule dominance`: maior uso médio entre as regras fuzzy. Quanto maior, mais concentrado o modelo fica em poucas regras.
- `Active rules 95`: quantidade de regras com uso médio acima de 5%.
- `MF shift`: deslocamento médio dos centros e sigmas das funções de pertinência após o treinamento.

## Resumo comparativo

| Iteração | Ajuste principal | Horizonte | LR consequentes | `lambda_rule_usage` | MAE teste | R2 teste | DA bruta | DA filtrada | Coverage | PF | Win Rate | Retorno | Max DD | Trades médios | PF > 1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | `AdamTrainer` com scheduler/early stopping, sem regularização de uso das regras | 15 | 5.0 | 0.00 | 0.9351 | -0.0090 | 50.10% | 52.60% | 38.14% | 0.912 | 47.87% | -9.62% | 20.51% | 270.75 | 0/4 |
| Iteração 1 | Inclusão de penalidade de diversidade de uso das regras | 15 | 5.0 | 0.01 | 0.9357 | -0.0117 | 49.89% | 50.04% | 51.24% | 0.933 | 48.25% | -11.38% | 23.81% | 350.00 | 1/4 |
| Iteração 2 | LR igual para MFs e consequentes; thresholds mais amplos | 15 | 1.0 | 0.01 | 0.9359 | -0.0091 | 49.75% | 52.12% | 27.20% | 1.010 | 50.36% | 1.15% | 12.98% | 202.25 | 2/4 |
| Iteração 3 | Horizonte de barreira reduzido de 15 para 10 candles | 10 | 1.0 | 0.01 | 0.9166 | -0.0060 | 50.19% | 56.63% | 7.98% | 1.131 | 52.52% | 6.87% | 7.03% | 76.00 | 2/4 |

## Interpretação por iteração

### Baseline

O baseline confirmou que o modo `causal_v3` ainda contém algum sinal direcional quando o score é filtrado por threshold, pois a DA filtrada ficou em 52,60%. Mesmo assim, o resultado operacional foi negativo: PF médio de 0,912, retorno médio de -9,62% e nenhum fold com PF acima de 1.

Essa combinação sugere que havia acerto direcional localizado, mas ele ainda não se convertia em desempenho de carteira. O R2 negativo também indica que o score contínuo do ANFIS ainda explicava pouco a variação do target.

Artefatos: `outputs/experiment_20260416_110110/`.

### Iteração 1

A primeira melhoria adicionou uma penalidade de diversidade de uso das regras (`lambda_rule_usage=0.01`). A intenção era reduzir o colapso em poucas regras fuzzy e preservar melhor a interpretabilidade da base neuro-fuzzy.

O efeito observado foi misto. A dominância média das regras caiu de 0,420 para 0,350 e o número médio de regras ativas subiu de 3,75 para 4,25, indicando maior distribuição de ativação. Porém, as métricas de resultado pioraram: DA filtrada caiu para 50,04%, PF ficou em 0,933 e o retorno médio caiu para -11,38%.

Interpretação: deixar a ativação fuzzy mais distribuída não melhorou o poder preditivo por si só. Isso reforça a hipótese de que regularização ajuda a forma do modelo, mas não substitui qualidade informacional das features e do alvo.

Artefatos: `outputs/experiment_20260416_110833/`.

### Iteração 2

A segunda melhoria manteve a regularização de uso das regras, mas reduziu o learning rate dos consequentes para o mesmo valor das funções de pertinência (`consequent_lr_mult=1.0`). Também foram ampliados os thresholds candidatos para `[0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20]`.

Essa alteração melhorou a parte operacional: PF médio 1,010, retorno médio 1,15% e 2/4 folds com PF acima de 1. O drawdown médio caiu para 12,98%. No entanto, a DA bruta permaneceu abaixo de 50%, e a DA filtrada ficou em 52,12%, próxima do baseline.

Interpretação: reduzir a velocidade dos consequentes parece ter estabilizado a carteira, mas não resolveu plenamente o aprendizado direcional. A melhora veio mais da combinação entre treino menos agressivo e calibração de threshold do que de uma regressão claramente melhor.

Artefatos: `outputs/experiment_iter02_20260416_111608/`.

### Iteração 3

A terceira melhoria alterou a engenharia do alvo: o horizonte da barreira ATR foi reduzido de 15 para 10 candles. A hipótese era que um alvo mais curto ficaria mais alinhado com setups SMC de reação local, como Sweep -> CISD -> FVG, reduzindo ruído de barras muito distantes do evento.

Essa foi a melhor configuração da bateria. O MAE caiu para 0,9166, a DA bruta subiu para 50,19%, a DA filtrada chegou a 56,63%, o PF médio ficou em 1,131, o retorno médio foi 6,87% e o drawdown médio caiu para 7,03%.

Por outro lado, a melhora veio com cobertura baixa: apenas 7,98% das amostras ficaram ativadas após threshold, com média de 76 trades por fold. Além disso, apenas 2/4 folds tiveram PF acima de 1. Portanto, o resultado é minimamente satisfatório para registro experimental, mas ainda não deve ser apresentado como robusto.

Artefatos: `outputs/experiment_iter03_h10_20260416_112808/`.

## Detalhamento da melhor iteração

| Fold | Epochs | Threshold | DA filtrada | Coverage | PF | Win Rate | Retorno | Max DD | Trades | MAE | R2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 0.20 | 61.80% | 11.65% | 1.506 | 60.34% | 26.39% | 7.77% | 116 | 0.9100 | -0.0020 |
| 2 | 49 | 0.12 | 59.57% | 7.05% | 1.319 | 57.53% | 10.42% | 4.69% | 73 | 0.9219 | -0.0079 |
| 3 | 38 | 0.15 | 50.96% | 7.85% | 0.842 | 45.90% | -5.17% | 7.09% | 61 | 0.9163 | -0.0120 |
| 4 | 35 | 0.15 | 54.21% | 5.35% | 0.855 | 46.30% | -4.18% | 8.57% | 54 | 0.9184 | -0.0020 |

## Evolução das funções de pertinência

A Iteração 3 também passou a exportar gráficos `fold_XX_mf_evolution.png`, comparando as funções de pertinência antes e depois do treinamento. Isso permite mostrar visualmente como o ANFIS ajustou centros e sigmas das MFs.

Arquivos gerados:

- `outputs/experiment_iter03_h10_20260416_112808/fold_01_mf_evolution.png`
- `outputs/experiment_iter03_h10_20260416_112808/fold_02_mf_evolution.png`
- `outputs/experiment_iter03_h10_20260416_112808/fold_03_mf_evolution.png`
- `outputs/experiment_iter03_h10_20260416_112808/fold_04_mf_evolution.png`

Na melhor iteração, o deslocamento médio dos centros foi 0,3181 e o deslocamento médio dos sigmas foi 0,3488. Isso indica que houve adaptação real das MFs, mas sem mudanças extremas de escala. O maior deslocamento observado foi de 1,2120 para centros e 1,1831 para sigmas.

## Conclusão parcial

A bateria indica que o caminho mais promissor não foi apenas mexer em métricas ou regularização, mas ajustar a formulação do problema de aprendizado. A redução do horizonte do alvo para 10 candles melhorou MAE, DA filtrada, PF, retorno e drawdown em relação ao baseline.

Mesmo assim, o resultado deve ser interpretado com cautela. O R2 permaneceu negativo, a cobertura ficou baixa e somente metade dos folds apresentou PF acima de 1. A conclusão tecnicamente honesta é que o modelo encontrou um recorte operacional mais seletivo e promissor, mas ainda não há evidência suficiente para afirmar robustez estatística do método.

## Rodada 50k com alvo por risco-retorno

Depois da Iteração 3, foi feita uma nova rodada de testes com 50.000 barras recentes do MT5. A motivação veio de uma limitação metodológica do alvo anterior: o alvo por horizonte de barras se aproxima de uma lógica temporal, em que o resultado depende de onde o preço estará após certo número de candles. Na prática operacional de SMC/ICT, porém, a avaliação de um setup costuma ser mais ligada ao preço: o stop é posicionado em uma região estrutural, como o topo ou fundo associado ao sweep, e o alvo é definido em função do risco ou de uma região de liquidez oposta.

Para tornar isso sistemático, foi implementado um alvo por risco-retorno (`target_mode="rr"`). Para cada candle, o algoritmo estima o resultado hipotético de uma operação comprada e de uma operação vendida, usando stop estrutural e alvo em múltiplos do risco. O valor supervisionado passa a representar qual direção teria melhor resultado em R dentro do horizonte máximo definido.

As principais alterações dessa rodada foram:

- aumento da base para 50.000 barras;
- manutenção do modo causal `causal_v3`;
- uso de horizonte máximo de 40 candles apenas como tempo limite da simulação;
- criação de stop estrutural com prioridade para sweep recente e fallback para `leg_low`/`leg_high` ou ATR;
- comparação entre RR 1:1 e RR 1:2;
- preservação da linha temporal completa no backtest (`min_activation=0.0`), para evitar distorção causada por remover candles antes da simulação.

Foram feitos testes preliminares com filtro de ativação antes do treino, mas eles foram tratados apenas como sondagem. A versão usada como referência nesta seção é a de linha temporal completa, pois ela preserva melhor a continuidade temporal do mercado e torna o backtest mais honesto.

### Comparação entre alvo antigo, RR 1:1 e RR 1:2

| Configuração | Barras | Alvo | DA bruta | DA filtrada | Coverage | PF | Win Rate | Payoff | Retorno médio | Max DD | PF > 1 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Iteração 3 | 20.000 | Barreira ATR, horizonte 10 | 50.19% | 56.63% | 7.98% | 1.131 | 52.52% | n/d | 6.87% | 7.03% | 2/4 |
| RR 1:1 | 50.000 | Stop estrutural, alvo 1R | 52.65% | 55.72% | 23.71% | 1.026 | 50.41% | 1.00 | 6.75% | 10.98% | 2/4 |
| RR 1:2 | 50.000 | Stop estrutural, alvo 2R | 53.69% | 55.25% | 36.39% | 1.392 | 42.04% | 1.92 | 58.08% | 13.77% | 4/4 |

O resultado mais importante da rodada foi a mudança de perfil entre RR 1:1 e RR 1:2. O RR 1:1 manteve uma taxa de acerto próxima de 50%, payoff próximo de 1 e PF levemente acima de 1. Isso indica uma configuração quase neutra: o modelo acerta um pouco mais do que erra, mas a vantagem operacional ainda é pequena.

O RR 1:2 teve comportamento diferente. O win rate caiu para 42,04%, o que seria esperado em uma estratégia que busca alvos maiores. Porém, o payoff médio ficou em 1,92 e o PF subiu para 1,392, com os quatro folds apresentando PF acima de 1. Essa combinação sugere que, nessa configuração, o modelo pareceu capturar melhor situações em que o preço desenvolve deslocamento suficiente para compensar uma menor taxa de acerto.

Essa leitura é coerente com a hipótese operacional do SMC/ICT adotada no projeto: após uma sequência como Sweep -> CISD -> FVG, o setup pode não gerar acertos frequentes em todos os contextos, mas tende a ser mais valioso quando há deslocamento direcional relevante. Portanto, nesta rodada, o modelo se mostrou mais interessante como filtro de setups com assimetria de retorno do que como classificador de direção de curto prazo.

### Detalhamento do melhor teste: RR 1:2

| Fold | Epochs | Threshold | DA bruta | DA filtrada | Coverage | PF | Win Rate | Retorno | Max DD | Trades | MAE | R2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 43 | 0.05 | 57.00% | 55.81% | 43.60% | 1.406 | 42.52% | 70.95% | 13.67% | 254 | 0.8163 | -0.0078 |
| 2 | 28 | 0.07 | 52.23% | 52.20% | 22.73% | 1.276 | 39.46% | 36.68% | 12.45% | 185 | 0.8654 | -0.0020 |
| 3 | 35 | 0.05 | 54.53% | 54.85% | 67.77% | 1.451 | 44.15% | 101.58% | 16.78% | 265 | 0.8340 | 0.0013 |
| 4 | 26 | 0.20 | 51.00% | 58.14% | 11.47% | 1.437 | 42.05% | 23.12% | 12.19% | 88 | 0.8335 | -0.0021 |

Artefatos: `outputs/experiment_rr50k_full_rr2_20260416_164258/`.

As métricas agregadas desse teste foram:

- MAE médio: 0,8373;
- RMSE médio: 0,8832;
- R2 médio: -0,0027;
- DA bruta média: 53,69%;
- DA filtrada média: 55,25%;
- coverage médio: 36,39%;
- profit factor médio: 1,392;
- win rate médio: 42,04%;
- payoff médio: 1,92;
- retorno médio por fold: 58,08%;
- max drawdown médio: 13,77%;
- folds com PF acima de 1: 4/4;
- folds com retorno positivo: 4/4.

O R2 continuou próximo de zero, o que indica que o modelo ainda não explica bem a variância contínua do alvo. No entanto, em séries financeiras de curto prazo, especialmente em mercados altamente eficientes, um R2 baixo não invalida automaticamente o resultado. Para este trabalho, a leitura mais adequada é comparar o R2 com as métricas direcionais e operacionais. Neste caso, mesmo com R2 negativo, o modelo apresentou DA acima de 50%, PF acima de 1 em todos os folds e payoff compatível com a estrutura de RR 1:2.

Também é importante interpretar o retorno anualizado com bastante cautela. O teste RR 1:2 superou a meta de 12% ao ano nos quatro folds dentro do cálculo anualizado do backtest, mas os valores anualizados ficaram muito altos por causa da combinação entre janelas curtas, risco fixo por trade e composição de capital. Portanto, para a monografia, esse número deve ser usado apenas como indicador comparativo interno, não como promessa de desempenho real.

## Conclusão da rodada RR

A rodada com 50.000 barras e alvo por risco-retorno representa a melhora mais relevante até este ponto do projeto. A principal evidência não está em uma grande elevação do R2, mas na conversão do sinal em desempenho operacional: o RR 1:2 obteve PF médio de 1,392, retorno positivo em todos os folds e payoff próximo do esperado para uma estratégia com alvo de 2R.

Isso sugere que a hipótese de engenharia do alvo estava correta: para esse tipo de setup, modelar apenas o retorno após uma quantidade fixa de barras pode esconder parte da lógica operacional. Ao representar o problema em termos de risco, stop estrutural e alvo em R, o treinamento fica mais próximo da decisão real de trading.

Ainda assim, a conclusão precisa permanecer cautelosa. O teste ainda não inclui custos de transação detalhados, slippage, variação de spread, validação em outros ativos ou uma janela final congelada de out-of-sample. Portanto, o resultado pode ser apresentado como um avanço experimental forte dentro do escopo avaliado, mas não como comprovação definitiva de lucratividade em operação real.

## Próximos ajustes recomendados

1. Congelar o RR 1:2 com 50.000 barras como novo modo de referência experimental.
2. Inserir custos operacionais mais realistas no backtest, incluindo spread, slippage e comissão.
3. Rodar sensibilidade de risco por trade, por exemplo 0,25%, 0,50% e 1,00%, para reduzir distorções do retorno anualizado.
4. Criar uma janela final de out-of-sample congelada, não usada na escolha de thresholds ou hiperparâmetros.
5. Testar robustez em outros recortes temporais e, se possível, em outros ativos correlatos.
6. Melhorar as features de confluência, separando Sweep isolado de Sweep + CISD + FVG alinhado.
7. Adicionar filtro de regime lateral, pois rompimentos falsos em consolidação ainda podem prejudicar a estabilidade do modelo.
