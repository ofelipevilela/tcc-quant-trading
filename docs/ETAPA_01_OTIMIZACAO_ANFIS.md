# Etapa 1: Otimizacao Inicial do ANFIS em Mercado Real

## 1. Objetivo desta etapa

Esta etapa teve como objetivo tirar o sistema neuro-fuzzy de um estado claramente degenerado e colocá-lo em um regime minimamente utilizável para testes com dados reais do MT5.

O foco não foi "provar" que o sistema já estava pronto, e sim resolver os primeiros gargalos que impediam qualquer avaliação séria:

- o modelo estava praticamente colapsado para uma saída quase constante;
- o backtest não estava abrindo trades de forma coerente;
- o treino e a simulação estavam operando com lógicas parcialmente desalinhadas;
- a escala herdada do Mamdani estava conflitando com o target real do treino.

Em outras palavras, antes de discutir desempenho de carteira, era necessário primeiro garantir que:

1. a rede realmente variasse sua saída;
2. essa saída pudesse ser convertida em sinais operáveis;
3. o backtest estivesse avaliando o mesmo tipo de objetivo usado no treino.

---

## 2. Estado inicial observado

Antes das correções, o pipeline rodava, mas os resultados mostravam um comportamento ruim e pouco confiável.

### 2.1 Sintomas principais

- a `MSE loss` caía muito rápido no começo e depois entrava em platô;
- o backtest antigo registrava zero ou praticamente zero trades;
- o output do modelo ficava concentrado perto de um valor quase fixo;
- o threshold de ativação do backtest estava incompatível com a escala real da rede.

### 2.2 Problema central identificado

O problema mais importante foi este:

- os consequentes iniciais do ANFIS vinham da lógica Mamdani na faixa de aproximadamente `[-80, 80]`;
- o target de treino real estava na faixa `[-1, 1]`;
- isso criava um desbalanceamento de escala entre o que a rede produzia no início e o que ela precisava aprender a prever.

Na prática, a otimização acabava encontrando um ponto cômodo em que quase toda a rede colapsava para uma saída próxima de zero. Quando isso acontece, o modelo "aprende" muito pouco sobre estrutura direcional e o backtest deixa de receber sinais fortes o suficiente para operar.

---

## 3. Diagnostico executado

Para não corrigir no escuro, foi feito um diagnóstico direto em cima do código e dos artefatos reais do projeto.

### 3.1 Verificações executadas

Foram rodados:

```bash
.\.venv\Scripts\python.exe -m pytest tests\test_anfis.py -q
.\.venv\Scripts\python.exe -u train_real_market.py
.\.venv\Scripts\python.exe -u run_backtest.py
```

Também foram executados scripts auxiliares de inspeção para medir:

- média, desvio padrão, mínimo e máximo do output do ANFIS;
- quantidade de ativações acima de vários thresholds;
- correlação entre score predito e target futuro;
- quantidade média de ativação por regra fuzzy;
- comportamento das funções de pertinência após o treino.

### 3.2 Resultado do diagnostico

O diagnóstico mostrou que o modelo anterior estava praticamente colapsado:

- `scores_mean ~= 0.057`
- `scores_std ~= 0.0019`
- `scores_min ~= 0.0024`
- `scores_max ~= 0.0576`
- correlação com target próxima de zero e levemente negativa

Isso explicava diretamente o backtest sem trades:

- com limiar de ativação em torno de `0.35`, nenhum score passava da barreira;
- o motor não recebia nenhum gatilho de entrada;
- o problema não era apenas o backtest, mas o fato de a rede já sair quase neutra em praticamente todas as barras.

### 3.3 Outra observacao importante

Foi confirmado que, na primeira etapa, a tradução SMC ainda estava relativamente simples do ponto de vista operacional:

- `trend_strength` vinha de BOS e CISD acumulados;
- `price_zone` vinha da posição do preço no range entre swings;
- `fvg_quality` e `sweep_quality` eram essencialmente magnitudes com propagação temporal.

Esse desenho ainda era útil como ponto de partida, mas já indicava que seria necessário um refinamento posterior para deixar a leitura SMC/ICT mais robusta e menos "crua".

---

## 4. Hipoteses de trabalho consideradas

Depois do diagnóstico, as hipóteses principais foram:

### Hipótese 1

O colapso da saída estava sendo causado por incompatibilidade de escala entre consequentes iniciais e target real.

### Hipótese 2

O target baseado apenas no `close` futuro, mesmo normalizado por ATR, não estava totalmente alinhado com a lógica de decisão que o backtest usava para abrir e fechar posições.

### Hipótese 3

Mesmo que o modelo conseguisse aprender alguma direção, o backtest ainda poderia continuar ruim se a regra de execução estivesse desacoplada da forma como o target foi construído.

---

## 5. Mudancas implementadas

As mudanças da Etapa 1 ficaram concentradas em três arquivos:

- `train_real_market.py`
- `run_backtest.py`
- `backtest/engine.py`

### 5.1 Reescrita do treino real

Arquivo principal:

- `train_real_market.py`

Mudanças principais:

#### 5.1.1 Target por barreiras ATR

Foi criado um target novo chamado `atr_barrier`, em vez de depender apenas do `close` no horizonte final.

Ideia:

- para cada barra, define-se uma barreira superior e uma inferior com base no ATR;
- se a superior for atingida primeiro, o alvo é positivo;
- se a inferior for atingida primeiro, o alvo é negativo;
- se nenhuma for atingida, usa-se um retorno suavizado como fallback.

Por que isso melhora:

- aproxima melhor a ideia de "sinal operacional" do que o uso puro do fechamento futuro;
- fica mais coerente com a lógica de entradas e saídas do backtest;
- aproxima treino e simulação.

#### 5.1.2 Reescala dos consequentes

Antes do treino, os consequentes do ANFIS passaram a ser divididos por `80.0`.

Motivo:

- a base de regras veio de uma semântica Mamdani/TSK centrada em `[-80, 80]`;
- o target real do treino ficou em `[-1, 1]`;
- reescalar os consequentes reduziu o choque de escala no início da otimização.

#### 5.1.3 Split temporal

O treino passou a usar divisão cronológica:

- treino
- validação
- teste

Isso é importante porque:

- evita mistura indevida de padrões temporais;
- é mais compatível com o problema de mercado;
- permite calibrar threshold na validação e avaliar no teste.

#### 5.1.4 Loss com penalizacao leve de dominancia de regra

Além do `MSE`, foi adicionada uma penalização leve sobre o uso médio das regras.

Objetivo:

- evitar que quase toda a inferência desabe sobre uma regra dominante;
- forçar uma ativação um pouco mais distribuída;
- reduzir o risco de colapso.

#### 5.1.5 Early stopping, gradient clipping e clamp de MFs

Foram adicionados:

- early stopping;
- `clip_grad_norm_`;
- clamp de `sigma` das MFs.

Motivo:

- estabilizar o treino;
- evitar degeneração de parâmetros;
- impedir que sigmas ficassem efetivamente inúteis.

#### 5.1.6 Calibracao automatica do threshold

Em vez de fixar o limiar manualmente, o pipeline passou a selecionar um threshold com base na validação.

Critério usado:

- directional accuracy nas ativações;
- cobertura de sinais;
- intensidade média do target nas barras ativadas.

O threshold escolhido nessa etapa foi:

- `0.10`

### 5.2 Ajustes do motor de backtest

Arquivo principal:

- `backtest/engine.py`

Mudanças principais:

#### 5.2.1 Threshold compatível com a escala real da rede

O backtest antigo ainda estava pensando em limiares grandes, compatíveis com um regime em `[-100, 100]`.

Depois da reescala e do treino com target real:

- a rede passou a operar efetivamente em torno de `[-1, 1]`;
- o limiar precisava acompanhar isso.

#### 5.2.2 Stop e alvo por ATR

O motor passou a operar com:

- `stop` por ATR
- `target` por ATR
- `time exit` pelo mesmo horizonte do target

Isso foi importante para alinhar:

- o que o modelo aprende;
- o que o backtest cobra da simulação.

#### 5.2.3 Registro mais rico das operacoes

As operações passaram a guardar:

- motivo de saída (`TARGET`, `STOP`, `TIME`);
- número de barras em posição;
- retorno percentual da operação.

Isso melhora a análise posterior e ajuda na escrita da dissertação.

### 5.3 Ajustes do runner do backtest

Arquivo principal:

- `run_backtest.py`

Mudanças principais:

- leitura automática de `anfis_trained_mt5_meta.json`;
- uso automático do `recommended_threshold`;
- uso automático de `horizon` e `atr_mult`;
- log de motivos de saída e tempo médio de permanência.

---

## 6. Resultados da Etapa 1

### 6.1 Resultado do treino

Depois das mudanças, o comportamento do modelo deixou de ser quase constante.

No treino salvo ao final da Etapa 1:

- `Scores validacao: mean=0.0085 std=0.1727 min=-0.7498 max=0.6884`
- `Scores teste: mean=0.0122 std=0.1851 min=-0.6083 max=0.7131`

Isso já mostra uma diferença relevante em relação ao estado anterior:

- antes, a saída praticamente não variava;
- depois, a rede voltou a espalhar sinais em uma faixa útil para decisão.

### 6.2 Resultado de validacao e teste offline

Com threshold calibrado em `0.10`:

- validação: `64.68%` de acerto direcional nas ativações
- teste: `61.07%` de acerto direcional nas ativações

Esses números não devem ser tratados como "prova de lucratividade", mas como um indicativo inicial de que a rede voltou a responder de forma direcionalmente útil.

### 6.3 Resultado do backtest

Backtest da primeira etapa:

- retorno total alto;
- win rate acima de 60%;
- profit factor acima de 1;
- grande quantidade de trades;
- predominância de saídas por `TARGET`.

Em janelas menores de verificação, os resultados também se mantiveram positivos:

- 2000 barras: `win rate ~= 57.78%`, `profit factor ~= 1.41`
- 2500 barras: `win rate ~= 60.04%`, `profit factor ~= 1.49`
- 3000 barras: `win rate ~= 59.61%`, `profit factor ~= 1.48`

Esses resultados foram suficientes para considerar a Etapa 1 bem-sucedida como correção estrutural do pipeline.

---

## 7. O que melhorou de fato

### 7.1 Melhorias concretas

- o modelo deixou de colapsar para saída quase constante;
- o backtest passou a abrir trades de forma consistente;
- treino e execução passaram a compartilhar a mesma lógica geral de barreiras;
- o threshold deixou de ser arbitrário e passou a ser calibrado em validação;
- o pipeline passou a salvar metadados úteis para reprodutibilidade.

### 7.2 O que ainda nao pode ser afirmado

Mesmo com melhora real, ainda não é correto afirmar que:

- o sistema está pronto para operação real;
- a vantagem estatística já está plenamente consolidada;
- o desempenho observado será estável em qualquer janela de mercado;
- o modelo já capturou bem toda a semântica interpretativa do SMC/ICT.

---

## 8. Limitacoes identificadas ao fim da Etapa 1

Esta parte é importante para a dissertação.

### 8.1 Possivel simplificacao excessiva da traduçao SMC

Embora o pipeline SMC já estivesse funcional, a tradução ainda era relativamente direta:

- BOS e CISD para tendência;
- posição no range para `price_zone`;
- magnitude de FVG e sweep como intensidades.

Isso é útil, mas ainda simplifica uma leitura que, na prática operacional, costuma depender de contexto.

### 8.2 Necessidade de walk-forward

Mesmo com validação temporal, ainda faltava uma verificação mais robusta com múltiplas janelas sequenciais de treino e teste.

Por isso, a próxima etapa prevista ficou sendo:

- validação walk-forward;
- comparação entre folds;
- agregação de métricas out-of-sample.

### 8.3 Ponto critico detectado depois da Etapa 1

Após a primeira rodada de melhoria, ficou evidente a necessidade de revisar um detalhe mais profundo da tradução SMC:

- a detecção de swings usava uma janela simétrica;
- isso é útil para confirmar pivôs graficamente;
- mas, se não houver tratamento causal, pode introduzir atraso de confirmação mal representado no uso operacional.

Esse ponto não invalida a Etapa 1 como correção estrutural, mas mostra que a Etapa 2 precisaria ser mais rigorosa do ponto de vista temporal e causal.

---

## 9. Arquivos e artefatos gerados

Artefatos preservados da primeira etapa:

- `outputs/stage1/anfis_trained_mt5_stage1.pt`
- `outputs/stage1/anfis_trained_mt5_stage1_meta.json`
- `outputs/stage1/mt5_training_loss_stage1.png`
- `outputs/stage1/backtest_results_stage1.png`

Artefatos principais da etapa:

- `anfis_trained_mt5.pt`
- `anfis_trained_mt5_meta.json`
- `outputs/plots/mt5_training_loss.png`
- `outputs/plots/backtest_results.png`

---

## 10. Leitura resumida da Etapa 1

Se eu precisasse resumir esta etapa em poucas linhas:

1. o sistema estava colapsando para quase nenhum sinal;
2. isso foi rastreado principalmente a um problema de escala e desalinhamento entre treino e backtest;
3. o target foi reformulado para barreiras ATR;
4. o treino foi estabilizado;
5. o threshold foi calibrado automaticamente;
6. o backtest passou a operar de forma coerente;
7. o sistema saiu do estado inviável para um estado utilizável, embora ainda com limitações importantes.

---

## 11. Proximos passos planejados

A partir desta etapa, os próximos passos naturais ficaram definidos como:

- implementar validação walk-forward;
- revisar a causalidade temporal da tradução SMC;
- tornar a leitura de FVG e sweep mais contextual;
- criar métricas mais acadêmicas para a dissertação;
- gerar visuais comparando o modelo inicial, o ajustado e os resultados por fold.

---

## 12. Comandos principais desta etapa

```bash
.\.venv\Scripts\python.exe -m pytest tests\test_anfis.py -q
.\.venv\Scripts\python.exe -u train_real_market.py
.\.venv\Scripts\python.exe -u run_backtest.py
```

---

## 13. Observacao final

Esta etapa foi uma etapa de saneamento técnico.

Ela não encerra o trabalho, mas foi essencial para permitir que as próximas melhorias sejam feitas sobre um sistema que realmente produz sinal, opera no backtest e pode ser analisado com alguma seriedade.
