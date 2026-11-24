# Semana 29 — Propensity Score Matching e Instrumental Variables

**Fase 6: Causal Inference e Experimentation** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Estimar efeitos causais em observational data, quando as pessoas **escolhem** receber o tratamento. Duas ferramentas: Propensity Score Matching (PSM), para quando os confounders são observados, e Instrumental Variables (IV), para quando não são.

## Por que isso importa para um AI Architect

"Usuários que usam a feature de IA retêm 30% mais" é a frase mais comum e mais enganosa em apresentações de produto: quem adota a feature já era diferente. Saber separar o efeito da feature do perfil de quem a escolhe evita investimentos baseados em correlação, e é o que se espera de alguém em nível sênior em um time de dados.

## Conceitos

### 1. Confounding e selection bias

> **Em palavras simples.** **Confounding** ocorre quando uma terceira característica influencia tanto quem recebe o tratamento quanto o resultado. Se quem toma certo remédio é, em média, mais rico e mais saudável, comparar quem toma com quem não toma mistura o efeito do remédio com o da riqueza. **Selection bias** é o problema geral: os grupos já eram diferentes antes de qualquer tratamento.

Um **confounder** é uma variável que afeta tanto a chance de receber o tratamento quanto o outcome. Pessoas focadas em saúde baixam o personal trainer de IA **e** têm colesterol mais baixo, com ou sem o app.

`diferença observada = efeito causal + selection bias`

Um **DAG** (directed acyclic graph) torna as suposições explícitas: `health_consciousness → uses_app`, `health_consciousness → cholesterol`, `uses_app → cholesterol`. O caminho que passa pelo confounder é um **backdoor path**; ajustar pelo confounder o fecha.

Nem toda variável deve ser controlada:

- **Confounder** (causa comum): controle.
- **Mediator** (está no caminho causal, ex.: "horas de exercício depois de instalar o app"): não controle, ou você remove parte do efeito.
- **Collider** (efeito comum do tratamento e do outcome): não controle, ou você **cria** uma associação espúria.

Regra prática: só use covariáveis medidas **antes** do tratamento.

> **No mundo real.** Durante anos, estudos observacionais, como o *Nurses' Health Study*, indicaram que a terapia de reposição hormonal reduzia doenças cardíacas em mulheres. Em 2002, o experimento randomizado **Women's Health Initiative**, com mais de 16 mil participantes, foi interrompido antes do previsto: a terapia aumentava o risco. As mulheres que optavam pela terapia eram mais saudáveis e tinham mais acesso a cuidados médicos; o benefício aparente vinha dessa diferença.

### 2. As hipóteses de identificação

> **Em palavras simples.** Para estimar um efeito causal sem sorteio, é preciso assumir duas coisas. Primeiro, que todas as características que influenciam o tratamento e o resultado foram observadas (**unconfoundedness**). Segundo, que para cada perfil de pessoa existem tratados e não tratados (**overlap**). A primeira não pode ser testada com os dados; depende de conhecer bem o processo.

Para estimar o efeito com dados observacionais ajustando por covariáveis `X`:

- **Unconfoundedness / ignorability (selection on observables):** dado `X`, o tratamento é "como se fosse" sorteado. Todos os confounders estão em `X`. **Não é testável.**
- **Overlap / positivity (common support):** para todo perfil `X`, há chance de ser tratado e de não ser. É verificável.

> **No mundo real.** LaLonde (1986) fez o teste mais conhecido dessas hipóteses. Ele partiu de um experimento randomizado de treinamento profissional, com efeito conhecido, trocou o grupo de controle sorteado por pessoas de pesquisas nacionais e aplicou os métodos estatísticos da época. As estimativas divergiram muito do valor experimental, algumas com o sinal oposto. O paper abalou a confiança nos métodos observacionais e motivou duas décadas de pesquisa.

### 3. Propensity score

> **Em palavras simples.** O **propensity score** é a probabilidade de uma pessoa receber o tratamento, calculada a partir das suas características. Ele resume dezenas de variáveis em um número. Duas pessoas com o mesmo score, uma tratada e outra não, são comparáveis: é como se, entre elas, o tratamento tivesse sido decidido por sorteio.

`e(X) = P(T = 1 | X)`: a probabilidade de receber o tratamento dadas as características. Resultado central (Rosenbaum & Rubin): se unconfoundedness vale dado `X`, também vale dado `e(X)`. Comparar unidades com o **mesmo** propensity score equivale a comparar unidades sorteadas. Reduz-se um problema de muitas dimensões a uma.

O propensity score é um **balancing score**: seu propósito é equilibrar as covariáveis entre os grupos, não prever bem o tratamento. Um modelo de propensity com AUC altíssimo é má notícia: significa pouca sobreposição.

> **No mundo real.** Dehejia e Wahba (1999) voltaram aos dados de LaLonde usando propensity score, conceito proposto por Rosenbaum e Rubin em 1983. Restringindo a comparação a pessoas com scores semelhantes, obtiveram estimativas próximas do resultado experimental, de cerca de 1.800 dólares de ganho anual de renda. O resultado reabilitou os métodos observacionais, com a ressalva, muito discutida depois, de que ele depende das variáveis disponíveis.

### 4. PSM passo a passo

> **Em palavras simples.** No **propensity score matching**, para cada pessoa tratada procura-se uma não tratada com score parecido, formando pares. Descartam-se os que ficam sem par. Depois, confere-se se os dois grupos ficaram de fato parecidos em cada característica (**balance**). Só então se comparam os resultados. A etapa de conferência é a que distingue uma análise séria.

1. **Estimar** `e(X)` com logistic regression (ou outro classifier), usando características pré-tratamento.
2. **Verificar o overlap:** histogramas do score por grupo. Onde só há tratados (ou só controles), não existe comparação possível; apare essas regiões (trimming).
3. **Matching:** para cada tratado, o controle com score mais próximo (nearest neighbor), dentro de um **caliper** (tolerância; regra comum: 0,2 desvio-padrão do logit do score). Opções: 1:1 ou 1:k; com ou sem reposição.
4. **Checar o balance:** a **SMD (standardized mean difference)** de cada covariável deve cair para abaixo de 0,1 em módulo. Use SMD, não p-values (que dependem do tamanho da amostra). Um "love plot" mostra antes e depois.
5. **Estimar:** diferença média do outcome entre pares. Isso estima o **ATT**.

Se o balance não for atingido, volte ao passo 1 e melhore o modelo de propensity (interações, termos não lineares). O outcome só deve ser olhado depois que o balance estiver aceitável: é o equivalente observacional de pré-registrar a análise.

> **No mundo real.** O estudo de Connors et al. (JAMA, 1996) aplicou o método a 5.735 pacientes de UTI para avaliar o cateterismo cardíaco direito, procedimento muito usado e nunca testado em experimento. Comparando pacientes com probabilidade semelhante de receber o cateter, o procedimento apareceu associado a **maior** mortalidade em 30 dias. O resultado provocou a realização de experimentos randomizados, que não encontraram o benefício que se supunha.

### 5. Alternativas ao matching

> **Em palavras simples.** Há outras formas de usar o propensity score além de formar pares. A **inverse probability weighting** dá mais peso às pessoas que receberam um tratamento improvável para o seu perfil. Os métodos **doubly robust** combinam o score com um modelo do resultado e acertam se pelo menos um dos dois estiver correto. Em geral, aproveitam melhor os dados do que o matching.

- **IPW (Inverse Probability Weighting):** pesa cada tratado por `1/e(X)` e cada controle por `1/(1 − e(X))`, criando uma pseudo-população em que o tratamento independe de `X`. Estima o ATE. Sensível a scores próximos de 0 ou 1 (pesos enormes); usa-se truncamento ou pesos estabilizados.
- **Regression adjustment:** modelar o outcome em função de `T` e `X`.
- **Doubly robust (AIPW):** combina um modelo de outcome com um de propensity; é consistente se **pelo menos um** dos dois estiver correto. É a recomendação atual.
- **Double Machine Learning** e **meta-learners** (S-, T-, X-learner): estendem a ideia para modelos flexíveis e para efeitos heterogêneos (CATE / uplift modeling).

> **No mundo real.** O paper *Why Propensity Scores Should Not Be Used for Matching* (King e Nielsen, 2019) mostrou, com simulações e dados reais, que formar pares pelo score pode aumentar o desequilíbrio entre os grupos e a dependência das escolhas do analista, sobretudo quando se descartam muitas observações. Os autores recomendam outras formas de matching ou o uso do score como peso. O paper mudou a recomendação padrão em várias áreas.

### 6. O limite: unobserved confounding

> **Em palavras simples.** Todos esses métodos só corrigem diferenças **observadas**. Se existe um fator importante que não foi medido, como motivação ou gravidade da doença, nenhum ajuste resolve. O que se pode fazer é uma **sensitivity analysis**: calcular quão forte teria de ser um fator oculto para anular o resultado. Se precisasse ser implausivelmente forte, a conclusão se sustenta.

PSM só corrige o que foi **medido**. Se a motivação da pessoa não está nos dados e afeta tudo, o viés continua lá, e nenhum diagnóstico de balance o revela. O Passo 9 do notebook demonstra: balance perfeito nas covariáveis observadas, estimativa errada.

O que fazer: **sensitivity analysis** (quão forte precisaria ser um confounder oculto para anular o resultado? Rosenbaum bounds, E-value), buscar proxies do confounder, ou mudar de estratégia de identificação.

> **No mundo real.** Nos anos 1950, críticos como o estatístico Ronald Fisher diziam que a associação entre cigarro e câncer de pulmão poderia vir de um fator genético oculto. Cornfield e colegas (1959) responderam com uma conta: para explicar um risco nove vezes maior entre fumantes, esse fator teria de ser pelo menos nove vezes mais comum entre fumantes do que entre não fumantes. Nenhum candidato chegava perto. Foi a primeira sensitivity analysis formal.

### 7. Instrumental variables

> **Em palavras simples.** Uma **instrumental variable** é algo que empurra as pessoas para o tratamento por um motivo sem relação com o resultado, como um sorteio ou uma regra arbitrária. Comparando quem recebeu o empurrão com quem não recebeu, mede-se o efeito do tratamento sem a contaminação das escolhas pessoais. É um sorteio emprestado da natureza ou de uma regra.

Quando há confounder oculto, procura-se um **instrument** `Z`: uma variável que

1. **Relevance:** afeta a chance de receber o tratamento. (Testável: first stage forte.)
2. **Exclusion restriction:** só afeta o outcome **através** do tratamento. (Não testável: é argumento.)
3. **Independence (exogeneity):** não está relacionada aos confounders ocultos. (Idealmente, é sorteada ou "as good as random".)

Para a interpretação como LATE, acrescenta-se **monotonicity**: o instrumento não faz ninguém mudar no sentido contrário (no defiers).

Exemplo: a distância da mesa até a sala de treinamento. Quem senta perto faz mais o curso de IA (relevance); a distância não muda o salário por outra via (exclusion) e não tem relação com a motivação (independence).

> **No mundo real.** Angrist (1990) usou o sorteio de convocação para a Guerra do Vietnã, que definia a ordem de chamada pelas datas de nascimento. A data sorteada influenciava a chance de servir e não tinha nenhuma relação com a capacidade de ganhar dinheiro. Com esse instrumento, ele estimou que, no início dos anos 1980, os veteranos brancos ganhavam cerca de 15% menos do que ganhariam sem ter servido.

### 8. 2SLS e o Wald estimator

> **Em palavras simples.** O cálculo com instrumento é uma divisão. No numerador, o quanto o resultado muda entre quem recebeu o empurrão e quem não recebeu. No denominador, o quanto o empurrão aumentou a adesão ao tratamento. O **2SLS** faz isso em duas regressões encadeadas. Se o empurrão altera pouco a adesão (**weak instrument**), a divisão é por um número pequeno, e o resultado fica instável.

**Two-Stage Least Squares:**

1. **First stage:** `T ~ Z`. Guarde o tratamento previsto `T̂`. Ele contém só a variação do tratamento causada pelo instrumento, livre do confounder.
2. **Second stage:** `Y ~ T̂`. O coeficiente é o efeito causal.

Com instrumento binário, isso se reduz ao **Wald estimator**:

`efeito = (E[Y | Z=1] − E[Y | Z=0]) / (E[T | Z=1] − E[T | Z=0])`

O numerador é a **reduced form** (efeito do instrumento sobre o outcome, também chamado intention-to-treat); o denominador é o first stage.

Cuidados:

- **Weak instrument:** se o first stage é fraco (F-statistic abaixo de ~10, pela regra clássica), o denominador fica perto de zero, a estimativa é instável e enviesada na direção do OLS. O notebook mostra a explosão.
- **LATE (Local Average Treatment Effect):** o IV estima o efeito nos **compliers**, aqueles cuja decisão é mudada pelo instrumento. Não é o efeito em quem faria o curso de qualquer jeito (always-takers) nem em quem nunca faria (never-takers). Outro instrumento pode dar outro número, e ambos estarem certos.
- Os erros-padrão do segundo estágio feito "à mão" estão errados; use uma biblioteca (`linearmodels.IV2SLS`).
- IV troca viés por variância: os intervalos são bem mais largos que os do OLS.

> **No mundo real.** Angrist e Krueger (1991) usaram o trimestre de nascimento como instrumento para a escolaridade: pelas regras escolares americanas, quem nasce no início do ano pode abandonar a escola com menos tempo de estudo. Estimaram um retorno de cerca de 7,5% de renda por ano de escola. Bound, Jaeger e Baker (1995) mostraram depois que o instrumento era fraco, e que isso podia enviesar o resultado mesmo com centenas de milhares de observações.

### 9. Instrumentos em produtos digitais

> **Em palavras simples.** Em produtos digitais, os instrumentos são abundantes: todo A/B test é um sorteio. Se um teste sorteou quem recebeu um incentivo para usar uma funcionalidade, o sorteio serve de instrumento para medir o efeito de **usar** a funcionalidade, e não apenas o de receber o incentivo. Isso responde a perguntas que o teste original não fazia.

O melhor instrumento é um sorteio que você controla: um **encouragement design**. Sorteia-se quem recebe um banner, um e-mail ou um desconto convidando a usar a feature. O convite é o instrumento (relevante e aleatório por construção); a adoção é o tratamento. É a forma padrão de medir o efeito de features opt-in, e o caso de **non-compliance** em A/B tests: nem todos os atribuídos ao tratamento o usam.

Outras fontes de instrumentos: atribuição aleatória de atendentes ou revisores (judge / examiner designs), falhas técnicas aleatórias, regras de elegibilidade com corte arbitrário (o que leva a **Regression Discontinuity Design**, um parente próximo).

> **No mundo real.** Peysakhovich e Eckles (2018), então no **Facebook**, propuseram usar o acervo de milhares de A/B tests já realizados como um conjunto de instrumentos: cada teste desloca um pouco certos comportamentos dos usuários, e a combinação de todos permite estimar o efeito desses comportamentos sobre resultados de longo prazo. O trabalho mostra como o histórico de experimentos de uma empresa pode ser reutilizado para perguntas novas.

## Erros comuns

- Controlar por variáveis medidas depois do tratamento.
- Avaliar o propensity model pela accuracy em vez do balance.
- Ignorar regiões sem overlap.
- Checar balance com p-values.
- Apresentar PSM como prova de causalidade sem discutir confounders não observados.
- Usar um instrumento fraco.
- Defender um instrumento só pela relevance, sem argumentar a exclusion restriction.
- Interpretar o LATE como efeito para toda a população.

## Leitura e vídeos (fim de semana)

- Matheus Facure, *Causal Inference for the Brave and True*: Chapters *Graphical Causal Models*, *Propensity Score*, *Doubly Robust Estimation*, *Instrumental Variables* e *Non Compliance and LATE*.
- Scott Cunningham, *Causal Inference: The Mixtape*: Chapters *Directed Acyclic Graphs*, *Matching and Subclassification* e *Instrumental Variables*.
- Angrist & Pischke, *Mastering 'Metrics*, Chapter 3 (Instrumental Variables).
- Austin, *An Introduction to Propensity Score Methods for Reducing the Effects of Confounding in Observational Studies*.
- King & Nielsen, *Why Propensity Scores Should Not Be Used for Matching* (para conhecer a crítica).
- Cinelli, Forney & Pearl, *A Crash Course in Good and Bad Controls*.
- Bibliotecas: `DoWhy`, `EconML`, `causalml`, `linearmodels`.

## Perguntas de autoavaliação

1. Por que comparar usuários e não usuários de uma feature costuma superestimar o efeito dela?
2. O balance após o matching ficou perfeito. Isso garante uma estimativa sem viés?
3. Quais as condições de um instrumento e qual delas é testável?
4. Por que o IV estima um efeito "local"? Quem são os compliers?
5. Qual a diferença entre confounder, mediator e collider, e qual deles você controla?
6. Como um encouragement design mede o efeito de uma feature opt-in?

## Entregável da semana

Notebook resolvido e uma nota em `docs/`: uma afirmação correlacional que você já ouviu sobre uma feature, os confounders prováveis e um instrumento ou encouragement design que você proporia.

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/06-Phase-6-Causal-Inference/notebooks/semana_29_exercicio.ipynb) · [`semana_29_exercicio.ipynb`](../notebooks/semana_29_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/06-Phase-6-Causal-Inference/notebooks/semana_29_gabarito.ipynb) · [`semana_29_gabarito.ipynb`](../notebooks/semana_29_gabarito.ipynb) |

[← Semana 28](/06-Phase-6-Causal-Inference/docs/Semana-28-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 30 →](/06-Phase-6-Causal-Inference/docs/Semana-30-Conceitos.md)
