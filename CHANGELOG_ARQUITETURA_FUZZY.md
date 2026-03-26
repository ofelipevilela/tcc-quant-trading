# Changelog de Arquitetura Fuzzy

## Objetivo desta revisão

Esta revisão consolidou duas mudanças estruturais no sistema fuzzy do projeto:

1. Redução de dimensionalidade da variável `Price_Zone`.
2. Substituição da saída escalar unidirecional por uma saída bidirecional encapsulada.

---

## 1. Por que mudamos `Price_Zone`

Antes, a zona de preço era dividida em cinco rótulos:

- `Deep_Discount`
- `Discount`
- `Equilibrium`
- `Premium`
- `Deep_Premium`

Essa modelagem aumentava a granularidade nominal, mas também ampliava o espaço combinatório da base de regras.

Considerando os antecedentes do sistema:

- `Trend_Strength`: 3 conjuntos
- `Price_Zone`: 5 conjuntos
- `FVG_Quality`: 3 conjuntos
- `Sweep_Quality`: 2 conjuntos

o espaço total de combinações possíveis era:

`3 x 5 x 3 x 2 = 90 combinações`

Após a refatoração de `Price_Zone` para apenas três conjuntos:

- `Discount`
- `Equilibrium`
- `Premium`

o novo espaço passa a ser:

`3 x 3 x 3 x 2 = 54 combinações`

Isso representa uma redução de 40% no espaço combinatório potencial da base de regras.

### Justificativa conceitual

O princípio adotado foi o da parcimônia. Em lógica fuzzy, não é necessário criar um rótulo extra como `Deep_Premium` para representar intensidade máxima da zona. Essa intensidade já pode ser capturada pelo próprio grau de pertinência.

Exemplo:

- um valor `price_zone = 0.85` já pode pertencer ao conjunto `Premium` com grau muito alto, próximo de 1.0;
- logo, o sistema já interpreta esse contexto como premium forte sem precisar de um novo rótulo linguístico.

Em outras palavras, a "profundidade" da zona passa a ser representada de forma contínua pelo grau de pertinência, e não de forma discreta por mais rótulos.

Isso reduz redundância sem perder capacidade de discriminação.

---

## 2. Por que mudamos a saída

Antes, a saída era modelada como um `Trade_Score` em `[0, 100]`.

Esse formato tinha uma limitação importante: o valor de saída representava apenas magnitude ou qualidade do setup, mas não continha a direção da decisão. Para determinar se o resultado final era compra ou venda, o sistema precisava voltar a consultar os inputs, especialmente `Trend_Strength` e `Price_Zone`.

Isso quebrava o encapsulamento da inferência.

### Problema do modelo anterior

Se a saída precisa de lógica externa para dizer se o setup é long ou short, então a saída não representa plenamente a decisão do sistema. Nesse cenário, parte da decisão está no consequente e parte continua espalhada nos antecedentes.

### Novo modelo

A saída passou a ser um `Trade_Signal` bidirecional no universo `[-100, 100]`, com cinco conjuntos:

- `Venda_Forte`
- `Venda`
- `Neutro`
- `Compra`
- `Compra_Forte`

Com isso:

- sinais negativos representam venda;
- sinais positivos representam compra;
- sinais próximos de zero representam neutralidade, conflito ou ausência de vantagem.

### Ganho de encapsulamento

Agora a própria saída já carrega:

- a direção do sinal;
- a intensidade do sinal;
- a possibilidade de neutralização.

Ou seja, a decisão final passa a ser derivada do próprio consequente, sem depender de heurísticas externas baseadas nos inputs.

---

## 3. Resolução de conflitos via centroide bidirecional

O novo universo bidirecional também melhora a interpretação de conflitos.

Quando regras de compra e venda são ativadas simultaneamente com intensidades semelhantes, a agregação fuzzy produz uma geometria distribuída em lados opostos do eixo. Na defuzzificação por centroide:

- forças compradoras puxam o centroide para o lado positivo;
- forças vendedoras puxam o centroide para o lado negativo;
- conflitos ou compensações tendem a empurrar o resultado para perto de zero.

Esse comportamento é desejável do ponto de vista quantitativo e conceitual, porque o valor neutro emerge naturalmente da competição entre regras, em vez de depender de tratamento externo.

Assim, o zero passa a representar uma resolução fuzzy de conflito, e não apenas ausência arbitrária de ação.

---

## 4. Resultado prático da revisão

As mudanças desta revisão produziram os seguintes efeitos:

- simplificação semântica de `Price_Zone`;
- redução do risco de explosão combinatória na base de regras;
- melhora do encapsulamento do sistema de decisão;
- maior coerência entre inferência, direção do trade e interpretação do centroide;
- atualização dos scripts de simulação e visualização para refletir sinais positivos para long e negativos para short.

---

## 5. Arquivos impactados

- `config/settings.py`
- `fuzzy/membership_functions.py`
- `fuzzy/fuzzy_system.py`
- `fuzzy/visualization.py`
- `main.py`
- `demo_inference.py`
- `demo_mamdani_visual.py`
- `demo_mamdani_completo.py`

