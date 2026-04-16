# Material visual para orientadora - metricas ANFIS

Gerado em: 16/04/2026 17:04:30

## Resultado principal

A melhor configuracao desta rodada foi o ANFIS `causal_v3` com 50.000 barras,
target por risco-retorno e RR 1:2. O resultado medio nos 4 folds foi:

- Profit Factor: 1.392
- Retorno medio por fold: 58.08%
- DA filtrada: 55.25%
- Coverage: 36.39%
- Win Rate: 42.04%
- Payoff: 1.92
- Max Drawdown: 13.77%
- R2: -0.0027
- Folds com PF > 1: 4/4

Interpretacao curta: o RR 1:2 nao venceu por ter o maior win rate, mas por
combinar payoff proximo de 2R com Profit Factor positivo nos quatro folds.
Isso sugere que o modelo esta mais adequado para filtrar deslocamentos com
assimetria do que para buscar alta taxa de acerto em alvo curto.

## Imagens geradas

1. `00_resumo_executivo_rr2.png`
   - Mostra os principais indicadores da melhor configuracao.
   - Serve como slide de abertura da discussao.

2. `01_parametros_experimentos.png`
   - Compara os parametros das configuracoes testadas: quantidade de barras,
     tipo de target, horizonte, RR, stop e numero de folds.
   - A mudanca mais importante foi sair do alvo temporal por barreira ATR para
     alvo em R com stop estrutural.

3. `02_metricas_academicas.png`
   - Compara MAE, RMSE, R2 e acuracia direcional.
   - MAE e RMSE medem erro de regressao; quanto menores, melhor.
   - R2 proximo de zero e comum em series financeiras de curto prazo e deve ser
     interpretado junto com metricas operacionais.
   - DA filtrada mede a acuracia direcional apenas quando o modelo gerou sinal.

4. `03_metricas_operacionais.png`
   - Compara Profit Factor, Win Rate, Payoff, retorno, drawdown e coverage.
   - O RR 1:2 teve Win Rate menor, mas payoff e Profit Factor melhores.

5. `04_interpretabilidade_regras_mfs.png`
   - Mostra dominancia de regras, quantidade de regras ativas e deslocamento das
     funcoes de pertinencia.
   - Ajuda a explicar que o modelo neuro-fuzzy nao e apenas caixa-preta: tambem
     e possivel observar adaptacao das regras e das MFs.

6. `05_rr2_metricas_por_fold.png`
   - Detalha a melhor configuracao fold a fold.
   - O ponto mais relevante e que o Profit Factor ficou acima de 1 nos quatro
     folds, indicando consistencia dentro da janela testada.

7. `06_meta_12_por_cento_anualizada.png`
   - Compara cada fold com a meta de 12% ao ano.
   - A leitura deve ser cautelosa, pois a anualizacao usa janelas curtas e
     composicao de capital.

8. `07_curvas_capital_rr2.png`
   - Mostra as curvas de capital por fold.
   - Ajuda a visualizar retorno e drawdown, nao apenas metricas finais.

9. `08_curvas_aprendizado_rr2.png`
   - Agrupa as curvas de aprendizado exportadas pelo treinamento.
   - Serve para discutir estabilidade, reducao de erro e possiveis sinais de
     overfitting.

10. `09_funcoes_pertinencia_rr2.png`
    - Agrupa os graficos de evolucao das funcoes de pertinencia por fold.
    - E util para mostrar visualmente que o ANFIS ajustou seus conjuntos fuzzy.

11. `10_esquema_target_rr.png`
    - Figura conceitual do novo target por risco-retorno.
    - Explica a diferenca entre alvo por tempo e alvo por preco/risco.

## Explicacao das metricas

- MAE: erro absoluto medio. Menor e melhor.
- RMSE: erro quadratico medio com raiz. Penaliza erros grandes.
- R2: proporcao da variancia explicada pelo modelo. Em mercado financeiro
  intraday, valores proximos de zero ainda podem coexistir com algum valor
  operacional quando ha vantagem direcional ou assimetria.
- DA bruta: percentual de vezes em que o sinal previsto acertou o sinal real.
- DA filtrada: DA considerando apenas sinais acima do threshold operacional.
- Coverage: percentual de amostras em que o modelo gerou sinal operacional.
- Profit Factor: lucro bruto dividido pela perda bruta. Acima de 1 indica
  resultado positivo no recorte avaliado.
- Win Rate: percentual de trades vencedores.
- Payoff: ganho medio dos trades vencedores dividido pela perda media dos
  trades perdedores.
- Retorno total: variacao percentual do capital no fold.
- Max Drawdown: maior queda percentual da curva de capital.
- Rule dominance: concentracao media de ativacao em poucas regras fuzzy.
- Active rules 95: quantidade media de regras com uso relevante.
- MF shift: deslocamento medio dos centros/sigmas das funcoes de pertinencia.

## Cautela metodologica

Os resultados sao promissores dentro da bateria executada, mas ainda nao devem
ser apresentados como comprovacao definitiva de lucratividade real. A proxima
etapa recomendada e incluir custos operacionais, slippage, spread variavel e uma
janela final congelada de out-of-sample.

Artefatos usados como melhor configuracao:
`D:/CODES/TCC/tcc-quant-trading/outputs/experiment_rr50k_full_rr2_20260416_164258`
