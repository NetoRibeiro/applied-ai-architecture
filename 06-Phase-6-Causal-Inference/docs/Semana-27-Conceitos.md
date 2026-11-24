# Semana 27 — A/B Tests, Statistical Power e Spillover em Redes

**Fase 6: Causal Inference e Experimentation** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Dominar o randomized controlled trial (A/B test) como gold standard de causalidade: hipótese, sample size, análise e os erros clássicos. Depois, ver esse padrão falhar em produtos com interação entre usuários (interference / spillover) e corrigir com cluster randomization.

## Por que isso importa para um AI Architect

Um modelo com métricas offline melhores só vale dinheiro se mudar o comportamento dos usuários, e isso se prova com experimento. Em produtos com network effects (compartilhamento, colaboração, marketplaces), o A/B test ingênuo erra a medida, e quem assina a arquitetura de experimentação precisa saber por quê.

## Conceitos

### 1. Potential outcomes e o problema fundamental

> **Em palavras simples.** Para saber o efeito de uma ação sobre uma pessoa, seria preciso ver dois mundos: aquele em que ela recebeu a ação e aquele em que não recebeu. Só um deles acontece. Esse é o **problema fundamental** da inferência causal. A saída é comparar grupos: se as pessoas forem distribuídas por sorteio, os grupos são parecidos em tudo, e a diferença entre eles pode ser atribuída à ação.

Para cada usuário existem dois resultados potenciais: `Y(1)` com a feature e `Y(0)` sem. O efeito causal individual é `Y(1) − Y(0)`. O **fundamental problem of causal inference**: nunca observamos os dois para a mesma pessoa. O que falta é o **counterfactual**.

Comparar quem usa com quem não usa mistura efeito com **selection bias**: `diferença observada = efeito causal + diferença prévia entre os grupos`.

A **randomization** resolve isso em média: grupos sorteados são iguais, em expectativa, em tudo, observável ou não, exceto no tratamento. A diferença de médias estima o **ATE (Average Treatment Effect)**.

> **No mundo real.** O teste da vacina Salk contra a poliomielite, em 1954, é o exemplo clássico. Na parte randomizada do estudo, cerca de 400 mil crianças foram sorteadas entre vacina e placebo, sem que elas ou os médicos soubessem quem recebeu o quê. A taxa de pólio foi de 28 por 100 mil entre as vacinadas e de 71 por 100 mil no grupo placebo. Como o sorteio igualou os grupos, a diferença só podia vir da vacina.

### 2. Anatomia de um A/B test

> **Em palavras simples.** Um **A/B test** é um experimento com sorteio aplicado a um produto. Os usuários são divididos ao acaso em dois grupos: um vê a versão atual (**control**), o outro vê a versão nova (**treatment**). Define-se antes qual métrica será comparada e por quanto tempo. Ao final, a diferença na métrica entre os grupos é a estimativa do efeito da mudança.

1. **Hipótese e primary metric** definidas antes (uma só; ver OEC, Semana 32), mais as guardrail metrics.
2. **Randomization unit:** user, session, device, cluster. A **analysis unit** deve ser a mesma (ou mais agregada). Randomizar por usuário e analisar por page view viola a independência e gera intervalos estreitos demais.
3. **Sample size**, a partir de: baseline rate, MDE (minimum detectable effect), significance level `α` e power `1 − β`.
4. **Execução** por tempo fixo, cobrindo ao menos um ciclo semanal completo.
5. **Análise:** estimativa do efeito com confidence interval.

Pré-registrar esses itens (em um documento, antes de ligar o teste) é o que impede escolher a análise depois de ver os dados.

> **No mundo real.** Em 2012, um funcionário do **Bing** propôs alterar a forma como os títulos dos anúncios eram exibidos. A ideia ficou meses na fila, por parecer pouco importante. Quando finalmente foi testada em um A/B test, a receita subiu 12%, o equivalente a mais de 100 milhões de dólares por ano só nos Estados Unidos, sem prejuízo às métricas de experiência do usuário. Foi, segundo Ron Kohavi, a ideia mais lucrativa da história do Bing, e ninguém a havia reconhecido de antemão.

### 3. Erros, power e sample size

> **Em palavras simples.** Um teste pode errar de duas formas: apontar um efeito que não existe (**false positive**) ou deixar passar um efeito que existe (**false negative**). O **power** é a chance de detectar um efeito real de certo tamanho. Ele depende do número de usuários: efeitos pequenos exigem amostras enormes. Por isso o **sample size** é calculado antes de o teste começar.

| | Efeito não existe (H₀ verdadeira) | Efeito existe |
|---|---|---|
| Teste rejeita H₀ | False positive (Type I, taxa `α`) | Acerto (power `1 − β`) |
| Teste não rejeita | Acerto | False negative (Type II, taxa `β`) |

É a confusion matrix da Semana 1 aplicada a decisões.

Para duas proporções:

`n por grupo ≈ (z_{1−α/2} + z_{1−β})² · [p₁(1−p₁) + p₂(1−p₂)] / (p₂ − p₁)²`

O `n` cresce com o **inverso do quadrado** do efeito: detectar um efeito duas vezes menor exige quatro vezes mais amostra. Com baseline de 5%, detectar +10% relativo pede cerca de 31 mil usuários por grupo; +2% relativo, 750 mil.

Um teste **underpowered** que "não deu significativo" não prova que a feature não funciona. E, quando um teste underpowered dá significativo, o efeito estimado tende a estar inflado (winner's curse).

> **No mundo real.** O paper *A/B Testing Intuition Busters* (Kohavi, Deng e Vermeer, 2022) analisa exemplos publicados de testes com amostras pequenas demais. Em um deles, o power estimado era de apenas 3%. Nessa situação, um resultado estatisticamente significativo quase certamente exagera o efeito real, e pode até ter o sinal trocado: é a chamada maldição do vencedor. Um teste sem power não é apenas inconclusivo, ele engana.

### 4. Interpretando o resultado

> **Em palavras simples.** O resultado de um teste é uma estimativa com margem de erro, não uma certeza. O **confidence interval** mostra a faixa de efeitos compatíveis com os dados. Se a faixa inclui o zero, não dá para afirmar que houve efeito. E um resultado bom demais para ser verdade geralmente é um erro de medição: deve ser investigado antes de comemorado.

- **p-value:** a probabilidade de observar uma diferença tão grande quanto a observada **se não houvesse efeito**. Não é a probabilidade de a hipótese ser verdadeira.
- **Confidence interval:** a faixa de efeitos compatíveis com os dados. Reporte sempre: "+0,5 pp, IC 95% [+0,1; +0,9]" diz muito mais que "p = 0,02".
- **Statistical vs. practical significance:** com amostra enorme, um efeito irrelevante é significativo. Compare o intervalo com o MDE de negócio.

> **No mundo real.** Na **Microsoft**, segundo os dados publicados por Kohavi e colegas, apenas cerca de um terço das ideias testadas melhora a métrica que pretendia melhorar; um terço não tem efeito detectável e um terço piora. Em produtos já muito otimizados, como o Bing, a taxa de sucesso é ainda menor. Sem experimento, a empresa teria lançado as três categorias com a mesma confiança.

### 5. Armadilhas clássicas

> **Em palavras simples.** Há erros clássicos que invalidam um teste. **Peeking** é olhar o resultado todos os dias e parar assim que parecer bom: isso multiplica os falsos positivos. **Sample ratio mismatch** é os grupos não terem o tamanho esperado, sinal de defeito no sorteio ou na coleta. Outros são mudar a métrica depois de ver os dados e encerrar o teste antes de completar uma semana.

- **Peeking:** olhar o p-value todos os dias e parar no primeiro "significativo". Com 20 olhadas, o false positive rate passa de 5% para mais de 25% (o notebook simula). Soluções: duração fixa, ou **sequential testing** (group sequential, mSPRT / always-valid p-values), que ajusta os limites para permitir monitoramento contínuo.
- **SRM (Sample Ratio Mismatch):** o sorteio era 50/50 e chegaram 50.000 vs. 48.700 usuários. Um chi-square test acusa. SRM indica bug na atribuição, no logging ou em filtros pós-randomização; o resultado do teste **não é confiável**, em qualquer direção.
- **Multiple comparisons:** vinte métricas (ou vinte segmentos) a 5% rendem, em média, um "significativo" por acaso. Correções: Bonferroni (conservadora), Benjamini–Hochberg (controla o false discovery rate).
- **Novelty e primacy effects:** o ganho dos primeiros dias some, ou o efeito demora a aparecer.
- **Simpson's paradox:** o efeito agregado tem sinal oposto ao de cada segmento quando a composição dos grupos difere.
- **Twyman's law:** todo resultado que parece interessante ou surpreendente provavelmente está errado. Verifique antes de comemorar.

**A/A tests** (dois grupos com a mesma experiência) validam a plataforma: o p-value deve se distribuir uniformemente e o false positive rate deve ficar em `α`.

> **No mundo real.** Um estudo da Microsoft (Fabijan et al., KDD 2019) verificou que cerca de **6%** dos experimentos da empresa apresentavam sample ratio mismatch, e que, nesses casos, as conclusões frequentemente estavam erradas, às vezes com o sinal invertido. As causas iam de filtros de bots aplicados de forma desigual a falhas no registro de eventos. As grandes plataformas de experimentação passaram a checar isso automaticamente antes de exibir qualquer resultado.

### 6. Variance reduction

> **Em palavras simples.** Parte da variação de uma métrica não tem relação com o experimento: há usuários que sempre compram muito e outros que quase nunca compram. **Variance reduction** usa o comportamento de cada usuário **antes** do teste para descontar essa diferença previsível. Com menos ruído, o mesmo efeito é detectado com menos usuários ou em menos tempo.

**CUPED** usa a métrica do próprio usuário no período anterior ao teste como covariável para remover variância. Reduz o sample size necessário em 30 a 50% em métricas com forte correlação temporal, sem introduzir viés. É padrão em plataformas maduras.

> **No mundo real.** A técnica **CUPED** foi publicada por pesquisadores da Microsoft (Deng, Xu, Kohavi e Walker, 2013). Aplicada a métricas do Bing, reduziu a variância em cerca de **50%**, o que equivale a obter o mesmo poder estatístico com metade dos usuários ou em metade do tempo. Tornou-se recurso padrão das plataformas de experimentação de empresas como Netflix, Airbnb e Booking.

### 7. SUTVA e interference

> **Em palavras simples.** Um A/B test supõe que o que acontece com um usuário não afeta os demais (**SUTVA**). Em muitos produtos isso é falso: em um marketplace, se o grupo de teste compra mais, sobra menos estoque para o grupo de controle; em uma rede social, a novidade se espalha entre amigos. Essa **interference** contamina a comparação e costuma exagerar o efeito.

A análise padrão assume **SUTVA** (Stable Unit Treatment Value Assumption): o resultado de um usuário não depende do tratamento dos outros. Em produtos sociais isso é falso: o usuário tratado cria um álbum com a IA e convida um usuário do controle, cuja experiência muda. Isso é **interference** ou **spillover**, e ele **contamina o controle**.

Consequência típica: o controle melhora "de carona", a diferença entre grupos encolhe e o teste **subestima** o efeito de lançar para todos. Em marketplaces o sinal pode ser o oposto: tratamento e controle competem pelo mesmo estoque (motoristas, anúncios), e o teste **superestima**.

Há dois efeitos distintos: o **direct effect** (ter a feature) e o **total effect** de lançar para todos (direto + spillover). O A/B por usuário estima algo próximo do primeiro; a decisão de lançamento precisa do segundo.

> **No mundo real.** Blake e Coey, do **eBay**, descreveram o problema em *Why Marketplace Experimentation Is Harder than It Seems* (2014). Em um teste que afetava lances em leilões, os compradores dos dois grupos disputavam os mesmos itens: quando os do grupo de teste davam lances mais altos, venciam leilões que os do controle teriam vencido. A comparação simples entre os grupos superestimava de forma substancial o efeito real.

### 8. Cluster-randomized trials

> **Em palavras simples.** Quando os usuários se influenciam, a solução é sortear **grupos** em vez de indivíduos: cidades, lojas, turmas, comunidades. Todos de um grupo recebem o mesmo tratamento, e a interferência fica contida dentro dele. O custo é estatístico: o que conta são os grupos, não as pessoas, e é preciso ter muitos grupos para ter precisão.

Sorteiam-se grupos inteiros (famílias, equipes, cidades, comunidades do grafo social): todos os membros de um cluster recebem o mesmo tratamento. A interference fica contida dentro dos clusters.

O preço:

- A unidade de análise passa a ser o **cluster**. Usuários do mesmo cluster são correlacionados; tratá-los como independentes produz intervalos estreitos demais. Use **cluster-robust standard errors** ou analise médias por cluster.
- **Design effect:** `DEFF = 1 + (m − 1) · ICC`, onde `m` é o tamanho médio do cluster e `ICC` a intraclass correlation. O **effective sample size** é `N / DEFF`. Com `m = 5` e `ICC = 0,2`, `DEFF = 1,8`: quase metade da amostra é "perdida".
- Menos unidades independentes, logo menos power. É o tradeoff entre **bias** (individual) e **variance** (cluster).

Variações:

- **Graph cluster randomization:** particionar o grafo social em comunidades e sortear comunidades.
- **Switchback experiments:** alternar tratamento e controle no tempo para uma região inteira (marketplaces, logística, pricing).
- **Geo experiments:** sortear cidades ou regiões.
- **Ego-cluster / exposure-based designs:** medir o efeito em função da fração de vizinhos tratados.

> **No mundo real.** O **Google** publicou a metodologia dos seus **geo experiments** (Vaver e Koehler, 2011) para medir o efeito real da publicidade: regiões geográficas inteiras são sorteadas para ter os anúncios ligados ou desligados, e comparam-se as vendas. Sortear usuários não funcionaria, porque a mesma pessoa vê anúncios em vários dispositivos e compra em loja física. É o desenho padrão para medir marketing.

### 9. Experimentação em produtos de IA

> **Em palavras simples.** Testar produtos de IA traz dificuldades adicionais. As respostas variam a cada execução, o que aumenta o ruído. A qualidade é difícil de resumir em um número. Os efeitos podem demorar: a novidade encanta no início, ou o usuário leva tempo para confiar. E o custo e a latência mudam junto com a qualidade, de modo que se observam várias métricas ao mesmo tempo.

Particularidades: a unidade costuma ser o usuário (não a requisição), porque a experiência com um assistente é cumulativa; respostas não determinísticas aumentam a variância; métricas de qualidade (satisfação, resolução) são lentas e ruidosas, o que pressiona por proxies (Semana 32); e custo e latência entram como guardrail metrics.

> **No mundo real.** Um experimento controlado sobre o **GitHub Copilot** (Peng et al., 2023) recrutou 95 desenvolvedores e sorteou metade para usar o assistente na tarefa de implementar um servidor HTTP em JavaScript. O grupo com Copilot concluiu a tarefa **55,8%** mais rápido. É um dos poucos resultados causais, e não apenas relatos, sobre o efeito de assistentes de código, e também um lembrete dos limites: uma tarefa só, de curta duração.

## Erros comuns

- Parar o teste quando "deu significativo".
- Ignorar um SRM porque o resultado agradou.
- Randomizar por sessão em um produto com estado por usuário.
- Analisar por evento quando a randomização foi por usuário.
- Procurar o segmento em que "funcionou" depois do teste.
- Concluir "não funciona" de um teste underpowered.
- Usar A/B por usuário em produto com interação social e reportar o resultado como o efeito do lançamento.

## Leitura e vídeos (fim de semana)

- Kohavi, Tang & Xu, *Trustworthy Online Controlled Experiments*: Chapters 1 a 3, *Sample Ratio Mismatch*, *Variance Estimation and Improved Sensitivity* e *Leakage and Interference between Variants*.
- Matheus Facure, *Causal Inference for the Brave and True*: Chapters 1 (*Introduction to Causality*), 2 (*Randomised Experiments*) e 3 (*Stats Review*).
- Evan Miller, *How Not To Run an A/B Test*.
- Deng et al., *Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data* (CUPED; abstract e seção 3).
- Blogs de engenharia: LinkedIn, *Detecting Network Effects: Randomizing Over Randomized Experiments*; Lyft, *Experimentation in a Ridesharing Marketplace*; DoorDash, *Switchback Tests*.
- Ugander et al., *Graph Cluster Randomization: Network Exposure to Multiple Universes* (abstract).

## Perguntas de autoavaliação

1. Por que a randomization elimina confounders que você nem conhece?
2. O que acontece com o false positive rate quando se espia o teste todo dia?
3. Em um produto com compartilhamento, o A/B por usuário superestima ou subestima o efeito do lançamento total? E em um marketplace com estoque limitado?
4. Com clusters de 5 pessoas e ICC de 0,2, qual o design effect?
5. O que é SRM e por que invalida o teste mesmo quando o resultado é positivo?
6. O que um p-value de 0,03 significa, e o que ele não significa?

## Entregável da semana

Notebook resolvido e uma nota em `docs/`: para um produto com interação social que você conheça, qual seria a randomization unit correta e por quê.

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/06-Phase-6-Causal-Inference/notebooks/semana_27_exercicio.ipynb) · [`semana_27_exercicio.ipynb`](../notebooks/semana_27_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/06-Phase-6-Causal-Inference/notebooks/semana_27_gabarito.ipynb) · [`semana_27_gabarito.ipynb`](../notebooks/semana_27_gabarito.ipynb) |

[← Semana 26](/05-Phase-5-MLOps-and-Agents/docs/Semana-26-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 28 →](/06-Phase-6-Causal-Inference/docs/Semana-28-Conceitos.md)
