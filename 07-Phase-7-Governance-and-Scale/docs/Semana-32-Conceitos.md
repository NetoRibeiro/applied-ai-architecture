# Semana 32 — OEC: a Métrica Certa e o Capstone Final

**Fase 7: Governance e Engineering at Scale** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Aprender a definir o **OEC** (Overall Evaluation Criterion): a métrica única, mensurável no curto prazo, que prevê o valor de longo prazo e contra a qual os experimentos são decididos. Você vai demonstrar com dados que uma métrica fácil pode ser um misleading proxy e construir uma métrica melhor. A semana fecha a trilha com o projeto final.

## Por que isso importa para um AI Architect

Sistemas de IA otimizam o que se manda otimizar. Um recommender treinado para cliques entrega cliques, inclusive os que destroem a confiança do usuário. A decisão mais alavancada de um líder técnico de dados não é o modelo: é a métrica. E convencer a diretoria a trocar a métrica fácil pela correta exige evidência, não opinião.

## Conceitos

### 1. O que é um OEC

> **Em palavras simples.** O **OEC** (*Overall Evaluation Criterion*) é a medida única, combinada de antemão, pela qual os experimentos são julgados. Sem ele, cada equipe escolhe a métrica que favorece a sua ideia, e todo teste vira sucesso. Um bom OEC pode ser medido em poucas semanas e, ao mesmo tempo, indica o que a empresa quer alcançar em anos.

Uma medida quantitativa do objetivo do experimento que seja:

- **Measurable in the short term** (dentro da duração de um teste, 1 a 4 semanas);
- **Predictive of long-term value** (retention, LTV, receita de 12 meses);
- **Sensitive** (move-se quando o produto muda de verdade; variância baixa o bastante para detectar efeitos);
- **Hard to game** (melhorá-la sem melhorar o produto deve ser difícil);
- **Single** (uma decisão não pode depender de escolher, depois, qual de vinte métricas olhar);
- **Understandable** (as pessoas precisam entender o que estão otimizando).

Conceitos vizinhos: a **North Star metric** orienta a estratégia da empresa; o OEC é a sua tradução operacional para decisões de experimento.

> **No mundo real.** Kohavi e colegas relatam um caso do **Bing**: um defeito no algoritmo de ranking piorou os resultados da busca. Os usuários, sem achar o que queriam, fizeram mais consultas e clicaram em mais anúncios: as consultas por usuário subiram 10% e a receita por usuário, 30%. Se o critério fosse consultas ou receita de curto prazo, o defeito teria sido aprovado. O OEC do Bing passou a se basear em sessões por usuário.

### 2. A taxonomia de métricas

> **Em palavras simples.** As métricas têm papéis diferentes. As **goal metrics** representam o objetivo final e mudam devagar. As **driver metrics** são indicadores de curto prazo que antecipam o objetivo. As **guardrail metrics** são limites que não podem ser violados. As **debug metrics** ajudam a entender o que aconteceu. Confundir os papéis leva a otimizar a coisa errada.

| Tipo | Papel | Exemplo |
|---|---|---|
| **Goal / success metrics** | O que a empresa realmente quer; lentas e ruidosas | Retention em 90 dias, LTV |
| **Driver / surrogate metrics** | Movem-se rápido e antecedem o goal | Sessões com sucesso, retorno em 7 dias |
| **Guardrail metrics** | Não devem piorar | Latência, error rate, reclamações, custo |
| **Diagnostic / debug metrics** | Explicam por que o OEC se moveu | Funil por etapa, métricas por segmento |

O OEC é construído a partir de driver metrics, validado contra goal metrics e protegido por guardrails.

> **No mundo real.** O framework **HEART**, publicado por pesquisadores do Google (Rodden, Hutchinson e Fu, 2010), organiza as métricas de experiência do usuário em cinco categorias: *Happiness*, *Engagement*, *Adoption*, *Retention* e *Task success*, cada uma desdobrada em objetivos, sinais e métricas. Criado para as equipes de produto do próprio Google, tornou-se uma referência para sair do hábito de medir apenas page views.

### 3. O misleading proxy

> **Em palavras simples.** Um **proxy** é uma métrica fácil de medir, usada no lugar do objetivo verdadeiro, que é difícil de medir. O risco é o proxy poder ser aumentado sem que o objetivo melhore, ou até com o objetivo piorando. Quando uma medida vira meta, as pessoas e os algoritmos encontram formas de aumentá-la que ninguém pretendia.

Cliques, tempo de tela, mensagens enviadas ao chatbot: fáceis de medir, fáceis de mover e apenas **correlacionados** com o valor. Sob otimização, a correlação se rompe.

**Goodhart's law:** quando uma medida vira alvo, deixa de ser uma boa medida. **Campbell's law** diz o mesmo sobre indicadores sociais. O mecanismo: as pessoas (e os modelos) encontram as formas mais baratas de mover o número, e as mais baratas raramente são as que criam valor.

Exemplos conhecidos:

- **Bing, queries per user:** um bug que piorou os resultados aumentou as consultas e a receita por usuário no curto prazo, porque as pessoas precisavam buscar de novo. A métrica correta passou a ser **sessions per user** (o usuário volta) e sucesso por sessão.
- **Clickbait:** títulos sensacionalistas aumentam o CTR e reduzem a retention.
- **Engagement / time spent:** otimizar tempo de tela favorece conteúdo que prende, não o que satisfaz.
- **Chatbot, messages per session:** um assistente que responde de forma incompleta gera mais mensagens.
- **Support, ticket deflection:** um bot que dificulta falar com humano "reduz tickets" e aumenta churn.

Em GenAI o risco é maior, porque o modelo é um otimizador eficiente: o que estiver na reward function, ele encontra.

> **No mundo real.** O banco americano **Wells Fargo** adotou como meta o número de produtos por cliente, tomado como indicador de relacionamento. Pressionados por metas agressivas, funcionários abriram milhões de contas e cartões sem autorização dos clientes. Em 2016, o banco foi multado em 185 milhões de dólares; as penalidades posteriores chegaram a bilhões. A métrica subiu por anos, enquanto o que ela deveria representar, a confiança do cliente, era destruído.

### 4. Como construir um OEC

> **Em palavras simples.** Construir um OEC é traduzir o objetivo de longo prazo em algo mensurável em semanas. Listam-se os sinais de curto prazo ligados ao objetivo, verifica-se com dados históricos quais realmente o antecipam e combinam-se esses sinais em uma medida, com pesos explícitos. Inclui-se também o custo do que se quer evitar. O resultado é revisto periodicamente.

1. **Defina o long-term outcome** que importa.
2. **Reúna experimentos passados** em que tanto as métricas de curto prazo quanto o outcome de longo prazo foram medidos. Esse acervo (experiment repository, parte da "institutional memory") é um ativo; sem ele, comece a guardá-lo agora, mantendo **holdout groups** de longa duração.
3. **Ajuste um surrogate index:** um modelo que preveja o efeito de longo prazo a partir dos efeitos de curto prazo.
4. **Valide out-of-sample:** em experimentos que não entraram no ajuste, o OEC acerta o **sinal** do efeito de longo prazo? É o que importa para decidir.
5. **Simplifique** para uma fórmula que as pessoas entendam e não possam manipular trivialmente: uma combinação ponderada de poucas métricas.
6. **Reavalie periodicamente:** o produto muda e a relação entre proxies e valor muda junto.

A hipótese por trás do surrogate index (surrogacy): o tratamento afeta o outcome de longo prazo **apenas através** das métricas de curto prazo incluídas. Se uma mudança afeta a retention por um canal que nenhum proxy capta, o índice erra. É por isso que a validação out-of-sample e os guardrails são obrigatórios.

> **No mundo real.** Na **Amazon**, o OEC inicial do sistema de e-mails promocionais era a receita gerada pelos cliques. Por esse critério, mandar mais e-mails sempre parecia bom. A equipe então incluiu no cálculo o valor perdido com cada cliente que cancelava a inscrição e deixava de receber qualquer oferta futura. Com o novo critério, mais da metade das campanhas, antes consideradas positivas, passou a ter valor negativo, segundo o relato de Kohavi.

### 5. Guardrail metrics

> **Em palavras simples.** **Guardrail metrics** são as métricas que um experimento não pode piorar, mesmo que melhore o objetivo principal: tempo de carregamento, taxa de erros, reclamações, custo. Elas protegem contra ganhos obtidos às custas de algo importante. Define-se de antemão o quanto de piora é tolerável; acima disso, a mudança não é lançada.

Métricas que não se quer melhorar, só **não piorar**. Um experimento que ganha no OEC e viola um guardrail não é lançado.

- **Organizational guardrails:** latência, error rate, crashes, custo por requisição, receita, cancelamentos, reclamações.
- **Trust-related guardrails:** SRM e outras verificações de validade do experimento (Semana 27).
- **Em IA generativa:** hallucination rate, false refusal rate, incidentes de segurança, custo em tokens, latência p95.

Como o objetivo é mostrar que **não piorou**, o teste adequado é de **non-inferiority**, com uma margem tolerável definida antes: ausência de diferença significativa em um teste underpowered não prova segurança.

> **No mundo real.** A latência é o guardrail mais conhecido. Experimentos do **Bing**, nos quais o tempo de resposta foi aumentado de propósito, mostraram que cada 100 milissegundos a mais custavam cerca de 0,6% de receita. Com esse número, qualquer nova funcionalidade passou a ser avaliada também pelo atraso que causava: o ganho precisava superar o custo. Um engenheiro que tornasse o servidor 10 ms mais rápido pagava o próprio salário anual.

### 6. A decisão como classificação

> **Em palavras simples.** Decidir lançar ou não lançar é uma classificação, com dois erros possíveis: lançar algo que não funciona e deixar de lançar algo que funciona. Como a maioria das ideias não funciona, uma parte relevante dos resultados positivos é falsa, mesmo com os critérios estatísticos usuais. É a mesma lógica de precision da Semana 1, aplicada a decisões de produto.

Lançar ou não lançar é uma classificação (Semana 1). Uma métrica de decisão ruim produz false positives (lança o que prejudica) e false negatives (descarta o que ajudaria). Compare métricas pela precision e pelo recall das **decisões** que elas geram, e pelo valor de longo prazo acumulado. É o que o notebook faz, e é o argumento mais convincente para uma diretoria: não "a métrica é teoricamente melhor", e sim "com ela teríamos deixado de lançar 78 mudanças prejudiciais".

> **No mundo real.** Kohavi, Deng e Vermeer (2022) fizeram a conta com taxas de sucesso reais. Quando apenas 8% das ideias testadas são de fato boas, como reportado para a busca do Airbnb, cerca de **26%** dos resultados estatisticamente significativos são falsos positivos, e não os 5% que muitos supõem. A recomendação dos autores é repetir os experimentos de resultado surpreendente ou exigir um critério mais rigoroso.

### 7. O efeito de segunda ordem

> **Em palavras simples.** Uma mudança pode ter um efeito imediato positivo e um efeito tardio negativo. Mais anúncios aumentam a receita hoje e ensinam o usuário a ignorar anúncios amanhã. Notificações insistentes aumentam os acessos nesta semana e os cancelamentos no mês seguinte. Experimentos curtos captam o primeiro efeito; para ver o segundo, é preciso manter um grupo de controle por mais tempo.

A métrica não só decide quais experimentos são lançados; ela decide **quais experimentos o time propõe**. Um time cobrado por cliques aprende a produzir mudanças que geram cliques. Em um ano, o portfólio de ideias muda. O Passo 9 do notebook simula isso: é o custo real de um mau OEC e a razão de ele ser uma decisão de liderança, não de análise.

> **No mundo real.** O paper *Focus on the Long-Term: It's Good for Users and Business* (Hohnhold, O'Brien e Tang, Google, 2015) mediu esse efeito com experimentos de longa duração na busca. Usuários expostos a mais anúncios, ou a anúncios piores, passavam a clicar menos em anúncios ao longo do tempo. Com base nisso, o Google reduziu a carga de anúncios na busca em celulares em cerca de 50%, concluindo que o ganho de longo prazo compensava a perda imediata.

### 8. Sensibilidade e variância

> **Em palavras simples.** Uma boa métrica precisa ser **sensível**: mover-se quando o produto de fato melhora ou piora. Métricas muito ruidosas ou muito lentas não distinguem uma boa ideia de uma ruim em tempo útil. Avalia-se a sensibilidade com experimentos passados de resultado conhecido: a métrica teria apontado a direção certa? Com que frequência?

Um OEC perfeito que nunca atinge significância é inútil. Técnicas: variance reduction (CUPED, Semana 27), winsorization ou capping de outliers, métricas de taxa em vez de totais, e a escolha de componentes com melhor signal-to-noise. Há um tradeoff entre fidelidade ao longo prazo (métricas lentas e ruidosas) e sensibilidade (métricas rápidas e, às vezes, enganosas); o OEC é o ponto de equilíbrio escolhido conscientemente.

> **No mundo real.** Deng e Shi, da Microsoft, descreveram esse processo no paper *Data-Driven Metric Development for Online Controlled Experiments* (2016). Eles usaram o acervo de experimentos anteriores do Bing como gabarito e avaliaram cada métrica candidata por duas propriedades: a direção (ela se move para o lado certo quando a mudança é reconhecidamente boa?) e a sensibilidade (com que frequência ela detecta o efeito?).

### 9. Convencendo a organização

> **Em palavras simples.** Adotar decisões baseadas em experimentos é uma mudança de cultura, não só de ferramenta. A opinião da pessoa mais graduada deixa de ser o critério, e isso encontra resistência. O que convence são casos concretos: uma ideia em que ninguém apostava e que deu grande resultado, ou uma ideia querida que os dados mostraram ser prejudicial.

- Mostre **casos concretos** em que a métrica atual e o outcome de longo prazo divergiram.
- **Quantifique:** "se tivéssemos decidido os últimos 40 testes pelo OEC, a retention acumulada seria X pontos maior".
- Proponha **transição**, não ruptura: reportar as duas métricas por um trimestre.
- Alinhe **incentivos**: se o bônus do time é por cliques, nenhum documento muda o comportamento.
- Formalize a fórmula e sua validação em um RFC (Semana 31), com revisão periódica.

> **No mundo real.** Na campanha presidencial de Barack Obama em 2008, Dan Siroker testou combinações de imagem e de botão na página de cadastro. A equipe preferia um vídeo; a combinação vencedora foi uma foto de família com o botão *Learn More*, com taxa de cadastro cerca de 40% maior. A campanha estimou em 60 milhões de dólares as doações adicionais. O caso é usado até hoje para mostrar que a intuição de especialistas erra.

## Erros comuns

- Escolher a métrica que é fácil de mover e chamá-la de sucesso.
- Vinte "primary metrics" e a escolha feita depois do resultado.
- OEC sem guardrails.
- Validar o surrogate index nos mesmos experimentos usados para ajustá-lo.
- Concluir que um guardrail "não piorou" a partir de um teste sem power.
- Uma fórmula tão complexa que ninguém entende o que está otimizando.
- Nunca reavaliar os pesos.
- Ignorar que a métrica molda as ideias que o time gera.

## Leitura e vídeos (fim de semana)

- Kohavi, Tang & Xu, *Trustworthy Online Controlled Experiments*: Chapters *Organizational Metrics*, *Metrics for Experimentation and the Overall Evaluation Criterion* e *Speed Matters* (um estudo de caso de guardrail).
- Athey et al., *The Surrogate Index: Combining Short-Term Proxies to Estimate Long-Term Treatment Effects More Rapidly and Precisely* (abstract e introdução).
- Kohavi et al., *Online Controlled Experiments at Large Scale* e *Seven Rules of Thumb for Web Site Experimenters* (o caso do Bing).
- Netflix Technology Blog, série sobre experimentação e *proxy metrics*; Spotify Engineering, textos sobre guardrail metrics e non-inferiority.
- Hohnhold, O'Brien & Tang, *Focusing on the Long-term: It's Good for Users and Business* (Google; ads blindness).
- Opcional: Thomas & Uminsky, *Reliance on Metrics is a Fundamental Challenge for AI*.

## Perguntas de autoavaliação

1. Liste as propriedades de um bom OEC.
2. Dê um exemplo, do seu contexto, de métrica que subiria se o produto piorasse.
3. Qual a diferença entre OEC e guardrail metric? Por que guardrails pedem um teste de non-inferiority?
4. Como você validaria que um OEC prevê o longo prazo?
5. Qual a hipótese por trás de um surrogate index e como ela pode falhar?
6. Por que um mau OEC causa dano mesmo que todos os experimentos sejam analisados corretamente?

## Entregável da semana

Notebook resolvido e um memorando de uma página para a diretoria: "Por que deixaremos de decidir experimentos por cliques", com os números da simulação adaptados ao seu contexto.

---

## Projeto final da trilha

Um sistema pequeno e completo, documentado como você documentaria no trabalho. Sugestão de escopo:

1. **Knowledge base** de um domínio à sua escolha (50 a 200 documentos reais e públicos).
2. **RAG pipeline** com chunking justificado, hybrid search e reranking (Fase 3).
3. **Golden set** de 50 perguntas, incluindo unanswerable questions, e relatório de avaliação com Recall@K, NDCG, faithfulness e refusal rates (Semanas 15 e 16).
4. **Um componente fine-tuned** com LoRA: query router, structured extractor ou o próprio generator (Fase 4), comparado a zero-shot e few-shot.
5. **Interface agent-first:** a busca exposta como tool com input e output schemas (Semana 26).
6. **Observability:** traces por etapa e um input guardrail com threshold justificado (Semana 25).
7. **Experiment plan:** como você mediria o impacto em produção, com randomization unit, OEC e guardrails (Fase 6 e esta semana).
8. **Documentos:** um ADR, um model card e um manifest de reprodução (Semanas 21 e 31).

### Checklist final: o que um AI Architect deve conseguir explicar

- [ ] Por que accuracy engana e como escolher entre precision e recall pelo custo do erro.
- [ ] O que o dot product mede e por que vector databases normalizam embeddings.
- [ ] O que `loss.backward()` calcula e o que a learning rate controla.
- [ ] Como funciona um Transformer block e por que o custo da attention é quadrático.
- [ ] O multi-stage ranking funnel e a métrica certa para cada stage.
- [ ] A RAG triad e como separar retrieval failure de generation failure.
- [ ] A fórmula do LoRA, o que o QLoRA quantiza e quando fine-tuning não é a resposta.
- [ ] O que determina latency e throughput de um LLM e como fazer capacity planning.
- [ ] Como projetar um output analítico para ser consumido por um agent.
- [ ] Quando usar A/B, cluster randomization, DiD, PSM, IV e synthetic control, e a hipótese frágil de cada um.
- [ ] Como escrever um ADR e tornar um resultado reproduzível com um comando.
- [ ] Como definir e defender um OEC.

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/07-Phase-7-Governance-and-Scale/notebooks/semana_32_exercicio.ipynb) · [`semana_32_exercicio.ipynb`](../notebooks/semana_32_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/07-Phase-7-Governance-and-Scale/notebooks/semana_32_gabarito.ipynb) · [`semana_32_gabarito.ipynb`](../notebooks/semana_32_gabarito.ipynb) |

[← Semana 31](/07-Phase-7-Governance-and-Scale/docs/Semana-31-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md)
