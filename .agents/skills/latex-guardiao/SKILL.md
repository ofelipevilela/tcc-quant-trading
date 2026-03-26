---
name: latex-guardiao
description: Use esta skill quando a tarefa envolver proteção do template, edição segura de arquivos .tex, correção de sintaxe LaTeX, preservação de labels, refs, cites, ambientes, figuras, tabelas e organização estrutural da monografia. Não use como skill principal para estilizar texto, aprofundar conteúdo técnico ou revisar bibliografia conceitual.
---

# Skill: LaTeX Guardião

## Objetivo
Editar e revisar arquivos LaTeX do TCC sem quebrar o template, sem apagar comandos importantes e sem introduzir erros estruturais desnecessários.

## Missão principal
Sempre trate o arquivo `.tex` como documento técnico e estrutural.
Mexer em LaTeX não é só "editar texto".
Cada alteração pode quebrar:
- compilação
- referências cruzadas
- citações
- numeração
- listas
- figuras
- tabelas
- sumário

## Quando usar
Use esta skill quando o usuário pedir:
- correção de trecho em LaTeX
- reorganização de ambiente
- ajuste de figura/tabela
- melhoria de seção já em `.tex`
- correção de sintaxe
- revisão de labels, refs, cites
- padronização de comandos

## Regras obrigatórias
- Preserve `\\label`, `\\ref`, `\\autoref`, `\\cite`, `\\input`, `\\include`, `\\includegraphics`.
- Não remova ambiente sem verificar abertura e fechamento.
- Não troque package ou macro do preâmbulo sem necessidade real.
- Não altere sintaxe só por preferência estética.
- Não converta comandos em texto puro sem motivo.
- Não mexa no preâmbulo se a tarefa for apenas de escrita.

## Processo de edição
Antes de alterar:
1. identifique o ambiente atual
2. identifique dependências do trecho
3. preserve a intenção do template
4. só depois edite

## Prioridades
Prioridade máxima:
1. não quebrar compilação
2. não quebrar referências
3. não quebrar semântica
4. melhorar legibilidade
5. melhorar estilo

## Cuidados com figuras e tabelas
Ao editar figuras e tabelas:
- preserve `\\caption`
- preserve `\\label`
- preserve caminho de arquivos
- não troque escala sem necessidade
- não invente legenda
- não mude posição de maneira agressiva sem necessidade

## Cuidados com listas
Ao editar `itemize` e `enumerate`:
- mantenha indentação coerente
- não misture itemização com parágrafo solto de modo quebrado
- não apague `\\item`
- evite inserir texto fora do ambiente por engano

## Cuidados com equações
Ao mexer em equações:
- preserve delimitadores
- não reescreva notação sem motivo
- mantenha coerência de símbolos já usados no texto
- não invente formulação matemática

## Modo de resposta
Ao entregar a saída:
- devolva o trecho já em LaTeX
- preserve o máximo possível da estrutura
- só explique o que mudou se houver risco ou mudança relevante

## Sinais de alerta
Pare e seja conservador quando houver:
- macros personalizadas desconhecidas
- ambientes customizados
- arquivos fragmentados com `\\input`
- referências internas difíceis de rastrear
- bibliografia automatizada com estilo específico

## Critério final
Uma boa edição de LaTeX:
- melhora o texto
- não quebra a estrutura
- não altera mais do que precisa
- respeita o template