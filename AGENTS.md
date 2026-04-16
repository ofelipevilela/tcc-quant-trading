# AGENTS.md

## Contexto do projeto
TCC em Engenharia Computacional: sistema de trading algorítmico que formaliza a metodologia Smart Money Concepts (SMC/ICT) — usada pelo aluno como daytrader manual — via Lógica Fuzzy e Neuro Fuzzy (ANFIS). O objetivo foi automatizar um operacional discricionário usando inferência fuzzy interpretável + aprendizado por backpropagation.

**Repositório dual**: contém tanto o código Python do sistema quanto a monografia em LaTeX (`MonografiaTCCLatex/`). O agente deve ser capaz de atuar no código e na escrita acadêmica.

---

## Estado atual do projeto (Abril 2026)

Pipeline completo e funcional: `MT5 → OHLCV → tradução SMC → 4 inputs ANFIS → treino/threshold → backtest`

### Etapas concluídas
- **Mamdani**: sistema Fuzzy Mamdani completo com 4 antecedentes, base de 26 regras, saída bidirecional `Trade_Signal [-100, 100]`. Ver `CHANGELOG_ARQUITETURA_FUZZY.md`.
- **Etapa 1 — ANFIS + dados reais**: resolução do colapso de saída, reescala de consequentes, target por barreira ATR, split temporal, early stopping. Ver `docs/ETAPA_01_OTIMIZACAO_ANFIS.md`.
- **Etapa 2 — Walk-forward + refino SMC**: causalização do pipeline SMC, revisão do critério de threshold, walk-forward com 4 folds. Ver `docs/ETAPA_02_WALKFORWARD_E_REFINO_SMC.md`.
- **Etapa 3 — causal_v3**: redesenho completo das features causais inspirado em análise do VFI Pine. Dois problemas corrigidos: (1) trend_strength separado em 3 componentes com cap independente; (2) sweep_quality substituído por setup_phase — máquina de estados Sweep→CISD→FVG. Ver `docs/ETAPA_03_CAUSAL_V3_E_RESULTADOS_FINAIS.md`.

### Métricas consolidadas — modo de referência

**legacy_like** (limite superior — usa info futura de pivot, fim de viés de look-ahead declarado):
- PF médio: **1.44** | Win Rate: **59.4%** | DirAcc: **~61%** | 4/4 folds positivos

**causal_v3** (modo causalmente correto, melhor resultado sem look-ahead):
- PF médio: **0.99** | Win Rate: **50.31%** | DirAcc: **53.25%** | 3/4 folds positivos
- Primeiro modo causal com DirAcc consistentemente acima de 50% em todos os folds

---

## Arquitetura do código

| Módulo | Papel |
|--------|-------|
| `smc/indicators.py` | Tradução de OHLCV → features SMC (BOS, CISD, FVG, Sweep, swing) |
| `smc/feature_factory.py` | Modos de feature: `causal_raw`, `legacy_like`, `causal_v2`, `causal_v3` (modo principal) |
| `anfis/` | Modelo ANFIS (TSK), trainer Adam, pipeline de dados, avaliação |
| `fuzzy/` | Sistema Mamdani (scikit-fuzzy), funções de pertinência, visualização |
| `backtest/engine.py` | Motor de simulação; `simulate_trading_from_scores()` separado dos scores |
| `backtest/performance.py` | Métricas: PF, win rate, drawdown, Sharpe, Sortino, Calmar, SQN |
| `real_market_utils.py` | Lógica compartilhada do pipeline real (target ATR, treino, threshold) |
| `data/mt5_client.py` | Conexão MT5, download OHLCV |
| `config/settings.py` | Parâmetros das variáveis fuzzy |
| `MonografiaTCCLatex/` | Monografia em LaTeX (template institucional) |

### 4 inputs do ANFIS
1. `trend_strength` — direção estrutural: BOS recency decay (±40) + EMA50/200 HTF bias (±40) + EMA9/21 momentum (±20), range `[-100, 100]`
2. `price_zone` — posição no range [0,1]: Discount / Equilibrium / Premium (com leg lock)
3. `fvg_quality` — razão FVG_size / ATR, zerada quando FVG contra a tendência
4. `sweep_quality` — no causal_v3, é preenchido por `setup_phase`: máquina de estados [0,3] rastreando Sweep→CISD→FVG

---

## Comandos principais

```powershell
# Ativar ambiente
.\.venv\Scripts\Activate.ps1

# Treinar ANFIS com dados reais do MT5
python train_real_market.py

# Rodar backtest com modelo salvo
python run_backtest.py

# Walk-forward completo (modo único, usa feature_mode de real_market_utils)
python run_walkforward.py

# Comparação de modos (principal script de pesquisa)
python run_walkforward_mode_comparison.py

# Testes unitários
python -m pytest tests/ -q
```

**Ambiente**: Python 3.10+, `.venv` local. Dependências em `requirements.txt`.  
**MT5**: necessário MetaTrader 5 instalado e conta ativa para `data/mt5_client.py`.

---

## Documentação interna

| Arquivo | Conteúdo |
|---------|----------|
| `docs/ETAPA_01_OTIMIZACAO_ANFIS.md` | Diagnóstico e correções do colapso inicial do ANFIS |
| `docs/ETAPA_02_WALKFORWARD_E_REFINO_SMC.md` | Walk-forward, refino SMC, critério de threshold |
| `CHANGELOG_ARQUITETURA_FUZZY.md` | Evolução do sistema Mamdani (Price_Zone, saída bidirecional) |
| `resumo_dissertacao.md` | Síntese técnica para a dissertação |
| `outputs/walkforward/walkforward_summary.json` | Resultados do walk-forward consolidado |
| `outputs/walkforward_compare/mode_comparison_summary.json` | Comparação legacy_like vs causal_raw |

---

## Contexto do projeto (monografia)
Este repositório contém materiais de um Trabalho de Conclusão de Curso (TCC) em Engenharia Computacional, escrito em português do Brasil, com foco em mercado financeiro, análise técnica, Smart Money Concepts (SMC), ICT e temas correlatos.

O objetivo do agente neste projeto é ajudar na escrita, revisão, organização e consistência técnica da monografia, preservando a naturalidade de um texto de graduação e respeitando os limites do que realmente foi estudado, implementado e validado pelo aluno.

## Diretriz principal
Escreva sempre em português do Brasil.
Mantenha um tom acadêmico natural, técnico e claro.
Evite soar como texto corporativo, publicitário, institucional ou excessivamente polido.
Evite vocabulário rebuscado demais quando houver opção mais simples e natural.
Palavras em inglês no texto da monografia devem estar sempre em itálico (\textit{termo pessoalmente}).

## Perfil de escrita desejado
Considere a voz de um aluno de 25 anos de Engenharia Computacional:
- tecnicamente interessado
- sério
- objetivo
- com boa capacidade de explicação
- sem tentar parecer um pesquisador sênior
- sem usar palavras difíceis só para parecer sofisticado

O texto deve parecer humano, direto e coerente.
Prefira clareza a elegância.
Prefira precisão a floreio.

## Regras de segurança acadêmica
Nunca invente:
- referências bibliográficas
- dados experimentais
- resultados
- métricas
- conclusões não sustentadas
- implementações que não existam no projeto

Quando faltar evidência, diga isso claramente e proponha:
- marcar como limitação
- registrar como hipótese
- inserir um TODO
- reformular de modo mais honesto

## Regras de LaTeX
Preserve a estrutura do template.
Não altere preâmbulo, packages, macros, labels ou convenções sem motivo claro.
Não quebre comandos LaTeX.
Não remova `\\cite`, `\\ref`, `\\label`, `\\input`, `\\includegraphics` ou ambientes sem verificar o impacto.
Sempre formate termos em inglês com \textit{termo}.

## Regras de citação
Use apenas chaves de citação já existentes no projeto, salvo quando o usuário explicitamente pedir para adicionar nova bibliografia.
Se houver nova bibliografia, deixe claro que a referência precisa ser conferida antes de entrar na versão final.
Nunca crie citação fictícia.

## Estratégia de trabalho
Antes de escrever uma seção nova:
1. identifique o objetivo da seção
2. levante os fatos disponíveis no projeto
3. separe o que é fato, interpretação e hipótese
4. proponha uma microestrutura curta
5. só então redija

Antes de revisar:
1. preserve o sentido original
2. corte repetições
3. reduza trechos com “cara de IA”
4. melhore a fluidez
5. mantenha o nível de graduação

## Skills do projeto
Ao notar que a tarefa se encaixa bem, use as skills apropriadas:
- `escrita-capitulos` para estruturar e redigir seções e subseções
- `latex-guardiao` para proteger o template e a sintaxe LaTeX
- `citacoes-e-referencias` para revisar citações e coerência bibliográfica
- `naturalizador-academico` para reduzir artificialidade e ajustar o tom
- `mercado-financeiro-smc-ict` para termos e explicações do domínio financeiro
- `resultados-e-discussao` para transformar achados reais em texto acadêmico

## O que evitar fortemente
Evite expressões como:
- "nos dias atuais"
- "de suma importância"
- "vale destacar"
- "é inegável que"
- "robusto" em excesso
- "revolucionário"
- "extremamente relevante"
- "solução inovadora" sem prova
- parágrafos longos que giram sem dizer algo concreto

## Forma preferida de resposta no projeto
Quando editar texto:
- entregue o trecho já em LaTeX
- preserve estrutura existente
- explique alterações só quando isso for útil

Quando a tarefa envolver análise:
- primeiro diga o problema
- depois mostre a correção sugerida
- por fim, entregue a versão revisada

## Critério final de qualidade
Antes de concluir qualquer tarefa de escrita, verifique:
- está claro?
- está natural?
- parece texto de graduação?
- está tecnicamente honesto?
- evita exagero?
- evita inventar coisa?
- evita parecer texto genérico de IA?