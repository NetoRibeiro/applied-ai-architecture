# Semana 28 — Difference-in-Differences (DiD)

**Fase 6: Causal Inference e Experimentation** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Estimar o efeito de uma intervenção quando não há sorteio, usando Difference-in-Differences (DiD): comparar a mudança no grupo tratado com a mudança no grupo de controle. Você vai aplicar o método, testar sua hipótese central (parallel trends) e ver o que o spillover faz com a estimativa.

## Por que isso importa para um AI Architect

Muitos lançamentos de IA não podem ser testados com A/B: a feature entra por região, por cliente ou para todos de uma vez. O DiD é o método quasi-experimental mais usado nesses casos e o primeiro que uma diretoria vai pedir para "provar o impacto". Saber quando ele vale, e quando não vale, evita atribuir à IA o que foi sazonalidade.

## Conceitos

### 1. A ideia

> **Em palavras simples.** Quando não é possível sortear, pode-se comparar mudanças. **Difference-in-Differences** observa dois grupos antes e depois de um evento que atingiu apenas um deles. Calcula-se o quanto cada grupo mudou e subtrai-se uma mudança da outra. O que sobra é o efeito do evento, já descontado o que teria acontecido de qualquer forma.

Dois grupos (treated e control), dois períodos (pre e post). O grupo de controle fornece o **counterfactual da tendência**: o que teria acontecido com o tratado sem a intervenção.

| | Pre | Post | Diferença |
|---|---|---|---|
| Treated (SP) | `Ȳ_T,pre` | `Ȳ_T,post` | `Δ_T` |
| Control (RJ) | `Ȳ_C,pre` | `Ȳ_C,post` | `Δ_C` |

`DiD = Δ_T − Δ_C`

A primeira diferença (no tempo) remove tudo o que é fixo em cada grupo: SP sempre vende mais que RJ. A segunda (entre grupos) remove o que mudou igualmente para todos: sazonalidade, inflação, uma campanha nacional.

Por que as comparações ingênuas falham:

- **Before/after em SP:** atribui à intervenção toda a tendência e a sazonalidade.
- **SP vs. RJ no post:** atribui à intervenção a diferença de nível que já existia.

O DiD estima o **ATT** (Average Treatment effect on the Treated): o efeito em SP, não o que seria em outras cidades.

> **No mundo real.** Em abril de 1992, Nova Jersey elevou o salário mínimo de 4,25 para 5,05 dólares; a vizinha Pensilvânia, não. Card e Krueger (1994) entrevistaram 410 restaurantes de fast-food nos dois lados da divisa, antes e depois. A teoria previa queda de emprego em Nova Jersey; a comparação das mudanças não a encontrou. O estudo mudou o debate econômico e contribuiu para o Nobel de David Card, em 2021.

### 2. A hipótese: parallel trends

> **Em palavras simples.** O método depende de uma hipótese: sem o evento, os dois grupos teriam seguido trajetórias paralelas (**parallel trends**). Não é preciso que estivessem no mesmo nível, só que evoluíssem do mesmo jeito. Isso não pode ser provado, mas pode ser examinado: se, nos períodos anteriores ao evento, as duas curvas já andavam juntas, a hipótese ganha credibilidade.

Sem a intervenção, tratado e controle teriam evoluído em paralelo. Os **níveis** podem ser diferentes; as **tendências**, não. É uma hipótese sobre o counterfactual e, portanto, não pode ser provada. Pode ser **tornada plausível**:

- **Gráfico das séries no pre-period:** andam juntas? É o primeiro e mais importante diagnóstico.
- **Placebo test (no tempo):** usando só o pre-period, finja que a intervenção ocorreu antes. O "efeito" estimado deve ser zero.
- **Placebo outcome:** aplique o DiD a uma métrica que a intervenção não poderia afetar.
- **Event study:** estime um efeito para cada período relativo à intervenção. Os coeficientes dos **leads** (antes) devem ficar em torno de zero; os dos **lags** (depois) mostram a dinâmica do efeito.

Ressalva importante: tendências paralelas no passado não garantem que continuariam paralelas. E um pre-trend test "não significativo" pode ser só falta de power. A hipótese também depende da escala: paralelas em nível não são paralelas em log.

> **No mundo real.** Os estudos sobre a expansão do Medicaid nos EUA em 2014, adotada por alguns estados e recusada por outros, mostram a prática atual. Miller, Johnson e Wherry (2021), ao estimar o efeito sobre a mortalidade de adultos de baixa renda, apresentam um gráfico com os coeficientes de cada ano: nos anos anteriores à expansão, a diferença entre os grupos de estados fica próxima de zero; depois, a mortalidade cai nos estados que expandiram.

### 3. DiD como regressão

> **Em palavras simples.** O DiD pode ser calculado com uma regressão. Incluem-se três termos: pertencer ao grupo tratado, estar no período posterior e a combinação dos dois. O coeficiente da combinação é o efeito. A vantagem da regressão é permitir acrescentar outros fatores, tratar muitos grupos e períodos e obter a margem de erro diretamente.

`Y = β₀ + β₁·treated + β₂·post + β₃·(treated × post) + ε`

- `β₀`: nível do controle no pre.
- `β₁`: diferença de nível entre os grupos.
- `β₂`: mudança comum no tempo.
- `β₃`: a estimativa DiD.

Em `statsmodels`: `smf.ols("y ~ treated * post", df)`.

Com vários grupos e períodos, usa-se **Two-Way Fixed Effects (TWFE)**: `Y_it = α_i + λ_t + β·D_it + ε_it`, com um efeito fixo por unidade (`α_i`, absorve tudo o que é constante na unidade) e um por período (`λ_t`, absorve choques comuns).

> **No mundo real.** O estudo de Autor (2003) sobre o crescimento do trabalho temporário nos EUA é a referência de como apresentar isso. Para medir o efeito de decisões judiciais estaduais que dificultaram demissões, ele incluiu na regressão termos para os anos anteriores e posteriores a cada decisão. O gráfico resultante, com efeito nulo antes e crescente depois, tornou-se o modelo do **event study** usado hoje em praticamente todo trabalho com DiD.

### 4. Standard errors

> **Em palavras simples.** A margem de erro de um DiD é fácil de subestimar. As observações do mesmo grupo ao longo do tempo se parecem muito entre si, e tratá-las como independentes faz o resultado parecer mais preciso do que é. A correção é calcular os **standard errors** agrupados por unidade (**clustered**), por exemplo por estado ou por loja.

Observações da mesma unidade ao longo do tempo são correlacionadas (**serial correlation**). OLS comum trata cada linha como independente e produz erros-padrão muito pequenos: o clássico resultado é que DiD com erros ingênuos "encontra" efeitos em placebos com frequência muito acima de 5%.

Use **cluster-robust standard errors** agrupados na unidade em que o tratamento varia (a loja, a cidade). Com poucos clusters tratados (uma ou duas cidades), mesmo isso é frágil: a inferência por placebo/permutação e o synthetic control (Semana 30) são mais honestos.

> **No mundo real.** O paper *How Much Should We Trust Differences-in-Differences Estimates?* (Bertrand, Duflo e Mullainathan, 2004) fez um teste simples: atribuiu leis fictícias, sorteadas, a estados americanos e estimou o efeito delas sobre salários reais. Com os erros-padrão usuais, as leis inexistentes apareciam como significativas em até **45%** das vezes, quando o esperado era 5%. Desde então, agrupar os erros por unidade é obrigatório.

### 5. Ameaças à identificação

> **Em palavras simples.** O DiD falha quando algo além do evento afeta os grupos de forma diferente. Exemplos: outro acontecimento na mesma época; o grupo tratado foi escolhido justamente por estar em um momento atípico; as pessoas mudaram de grupo por causa do evento; ou anteciparam o evento e mudaram o comportamento antes. Cada ameaça deve ser discutida abertamente.

| Ameaça | Exemplo | Efeito |
|---|---|---|
| Non-parallel trends | SP já crescia mais rápido | Viés de sinal imprevisível |
| Concurrent shock | Concorrente fecha em SP na mesma semana | Atribui ao tratamento o que é do choque |
| Anticipation | Usuários mudam de comportamento antes do lançamento | Contamina o pre-period |
| **Spillover** | A campanha da IA em SP viraliza e atrai compradores do RJ | O controle também é tratado: **subestima** |
| Composition change | Novos usuários diferentes entram no grupo tratado | Mistura efeito com seleção |
| Ashenfelter's dip | A unidade foi escolhida para o tratamento **porque** ia mal | Regression to the mean vira "efeito" |

A última é comum em empresas: pilotos são lançados na loja ou no time com pior desempenho, que melhoraria de qualquer forma.

> **No mundo real.** Ashenfelter (1978), ao avaliar programas de treinamento profissional, notou que a renda dos participantes **caía** pouco antes de entrarem no programa: as pessoas se inscreviam depois de perder o emprego. Como a renda tenderia a se recuperar de qualquer modo, comparar o antes e o depois exagerava o efeito do treinamento. O fenômeno ficou conhecido como **Ashenfelter's dip** e é o exemplo padrão de seleção por um momento atípico.

### 6. Lidando com spillover

> **Em palavras simples.** **Spillover** é o tratamento vazar para o grupo de controle. Se uma promoção em uma cidade atrai clientes da cidade vizinha, a vizinha deixa de ser uma boa comparação: ela também foi afetada. As saídas são escolher controles mais distantes, excluir uma zona de transição, ou medir o vazamento como parte do resultado.

- Escolher um controle **distante** da fonte de contaminação (outra região sem contato com a campanha).
- Usar um terceiro grupo: comparar o controle contaminado com um controle limpo **mede o spillover** (o notebook faz isso com RJ vs. BH).
- Excluir a zona de fronteira (donut design).
- Reportar a estimativa com o controle contaminado como **lower bound** do efeito.

Quando o spillover é positivo (o controle se beneficia), o impacto total no negócio é **maior** que o DiD: efeito direto mais o ganho indireto no controle. Vale reportar os dois.

> **No mundo real.** No estudo *Worms* (Miguel e Kremer, 2004), escolas do Quênia receberam tratamento contra verminoses em fases sorteadas. Os autores mostraram que o tratamento reduzia a transmissão também para crianças não tratadas da mesma escola e de escolas próximas. Estudos anteriores, que sorteavam crianças dentro da mesma escola, haviam subestimado o benefício, justamente porque o grupo de controle também melhorava.

### 7. Staggered adoption

> **Em palavras simples.** Muitas vezes o tratamento não chega a todos ao mesmo tempo: uma funcionalidade é liberada por regiões, uma lei é adotada por estados em anos diferentes (**staggered adoption**). A regressão tradicional, nesses casos, acaba usando unidades já tratadas como controle das tratadas depois, e o resultado pode sair distorcido. Métodos recentes fazem apenas as comparações válidas.

Quando unidades adotam o tratamento em datas diferentes e o efeito varia no tempo ou entre unidades, o TWFE tradicional usa unidades **já tratadas** como controle de unidades tratadas depois ("forbidden comparisons") e pode dar pesos negativos e até sinal trocado (a decomposição de Goodman-Bacon mostra isso).

Estimadores modernos tratam o caso: **Callaway & Sant'Anna**, **Sun & Abraham**, **de Chaisemartin & D'Haultfœuille**, **Borusyak et al.** A ideia comum é comparar cada coorte de adoção apenas com unidades ainda não tratadas (ou nunca tratadas). Saiba que o problema existe antes de aplicar a receita simples a um rollout gradual.

> **No mundo real.** Goodman-Bacon (2021) demonstrou que a estimativa tradicional é uma média ponderada de todas as comparações possíveis entre grupos, incluindo as problemáticas, e ilustrou com as leis de divórcio unilateral adotadas pelos estados americanos em anos distintos. No mesmo ano, Callaway e Sant'Anna publicaram um estimador que usa apenas controles ainda não tratados. Os dois papers mudaram a prática da área e têm implementações abertas em R e Python.

### 8. Extensões

> **Em palavras simples.** O DiD tem extensões. A **triple difference** acrescenta uma terceira comparação, com um subgrupo que não deveria ser afetado, para descontar choques que atingiram o grupo tratado por outras razões. O **synthetic DiD** combina o método com a construção de um grupo de comparação ponderado. O **event study** mostra o efeito período a período.

- **Triple differences (DDD):** acrescenta uma terceira dimensão (ex.: uma categoria de produto não afetada) para remover choques específicos do grupo tratado.
- **DiD com covariáveis** e **doubly robust DiD:** tornam parallel trends condicional a características observadas.
- **Synthetic DiD:** combina DiD com os pesos do synthetic control.

> **No mundo real.** A triple difference foi popularizada por Gruber (1994), ao estudar leis estaduais americanas que obrigaram os planos de saúde a cobrir despesas de maternidade. Ele comparou estados com e sem a lei, antes e depois, e ainda mulheres casadas em idade fértil com trabalhadores não afetados pela lei. A terceira comparação mostrou que o custo do benefício foi repassado aos salários do grupo beneficiado.

### 9. Como comunicar

> **Em palavras simples.** Um resultado de DiD convence quando é apresentado com transparência: o gráfico das duas trajetórias, a hipótese dita em palavras simples, a evidência de que as tendências eram paralelas, o efeito com a margem de erro e os testes de robustez. Quem lê deve conseguir enxergar o resultado no gráfico antes de ver a tabela.

Um resultado de DiD para a diretoria deve conter: a estimativa com intervalo; o gráfico das séries; o resultado do placebo; a hipótese de parallel trends dita com todas as letras; e as ameaças consideradas. "A feature gerou +8, IC [7,6; 8,3], supondo que SP teria seguido a tendência de BH" é uma frase honesta. "A feature gerou +8" não é.

> **No mundo real.** Quando a Academia Sueca explicou ao público o Nobel de Economia de 2021, usou o estudo do salário mínimo de Card e Krueger com um único gráfico: duas linhas, Nova Jersey e Pensilvânia, antes e depois do aumento. O material mostra que um resultado causal sólido pode ser comunicado a quem não é especialista com uma figura e uma frase sobre o que teria acontecido sem a mudança.

## Erros comuns

- Before/after sem grupo de controle.
- Não olhar o gráfico do pre-period.
- Erros-padrão sem cluster.
- Escolher o grupo de controle depois de ver qual dá o melhor resultado.
- Usar como controle uma unidade exposta ao tratamento.
- Aplicar TWFE simples a adoção escalonada com efeitos heterogêneos.
- Ignorar que a unidade tratada foi escolhida por desempenho ruim.

## Leitura e vídeos (fim de semana)

- Matheus Facure, *Causal Inference for the Brave and True*: Chapters *Difference-in-Differences*, *Panel Data and Fixed Effects* e *The Difference-in-Differences Saga* (staggered adoption).
- Scott Cunningham, *Causal Inference: The Mixtape*, Chapter *Difference-in-Differences* (disponível online).
- Card & Krueger (1994), *Minimum Wages and Employment*: leia um resumo; é o exemplo canônico.
- Bertrand, Duflo & Mullainathan, *How Much Should We Trust Differences-in-Differences Estimates?* (abstract e introdução), sobre serial correlation.
- Goodman-Bacon, *Difference-in-Differences with Variation in Treatment Timing* (abstract e figuras).
- Roth et al., *What's Trending in Difference-in-Differences? A Synthesis of the Recent Econometrics Literature* (um guia para a literatura recente).

## Perguntas de autoavaliação

1. O que cada uma das duas diferenças remove?
2. Parallel trends exige que os grupos tenham o mesmo nível? Explique.
3. Como um placebo test aumenta a confiança na estimativa? O que ele não garante?
4. O spillover para o grupo de controle superestima ou subestima o efeito? Por quê?
5. Por que os erros-padrão precisam ser clustered?
6. Qual o problema do TWFE em staggered adoption?

## Perguntas do entrevistador

1. A empresa lançou uma funcionalidade de IA em apenas três estados sem sorteio prévio, e um colega compara antes e depois apenas nesses estados para declarar sucesso. Como você usa Difference-in-Differences para corrigir tendências temporais, testa parallel trends e evita viés em adoção escalonada (staggered adoption)?

## Entregável da semana

Notebook resolvido e um parágrafo em `docs/`, escrito para um diretor: qual foi o efeito do lançamento, com que margem, e quais hipóteses sustentam o número.

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/06-Phase-6-Causal-Inference/notebooks/semana_28_exercicio.ipynb) · [`semana_28_exercicio.ipynb`](../notebooks/semana_28_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/06-Phase-6-Causal-Inference/notebooks/semana_28_gabarito.ipynb) · [`semana_28_gabarito.ipynb`](../notebooks/semana_28_gabarito.ipynb) |

[← Semana 27](/06-Phase-6-Causal-Inference/docs/Semana-27-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 29 →](/06-Phase-6-Causal-Inference/docs/Semana-29-Conceitos.md)
