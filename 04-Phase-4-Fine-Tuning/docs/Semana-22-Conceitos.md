# Semana 22 — Preference Alignment: DPO do Zero

**Fase 4: Parameter-Efficient Fine-Tuning (PEFT)** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Entender como se ajusta o **comportamento** de um modelo (tom, formato, compliance) a partir de preferências, e não de respostas únicas corretas. Você implementará a loss do DPO (Direct Preference Optimization) do zero e a usará para tornar um modelo mais formal.

## Por que isso importa para um AI Architect

Em uso corporativo, muitas exigências não têm "resposta certa": tom de voz da marca, concisão, recusa de temas fora do escopo, formato regulatório. Preference data ("esta resposta é melhor que aquela") é mais fácil de coletar do que respostas ideais, e é com ela que se faz alignment. Entender o mecanismo também é entender os riscos: um modelo otimizado para aprovação aprende a agradar.

## Conceitos

### 1. De SFT a alignment

> **Em palavras simples.** O SFT ensina o modelo a imitar respostas. Mas imitar não é o mesmo que saber qual resposta é **melhor**. **Alignment**, neste contexto, é ajustar o modelo às preferências das pessoas: ser útil, direto, cuidadoso, recusar o que deve ser recusado. Para isso se usam comparações entre respostas, e não apenas exemplos de resposta correta.

- **SFT (Semanas 19 a 21):** imita reference answers. Ensina **o que** dizer.
- **Preference-based alignment:** dado um prompt e duas respostas, uma preferida e outra rejeitada, aumenta a probabilidade da preferida **em relação** à rejeitada. Ensina **o que é melhor**.

A diferença importa quando há muitas respostas aceitáveis e o que distingue a boa da ruim é uma qualidade (tom, segurança, concisão), não um conteúdo.

> **No mundo real.** O paper do **InstructGPT** (Ouyang et al., OpenAI, 2022), base do ChatGPT, mostrou o tamanho do efeito: avaliadores humanos preferiram as respostas de um modelo de **1,3 bilhão** de parâmetros ajustado com feedback humano às do GPT-3 original, de **175 bilhões**. Um modelo cem vezes menor, alinhado às preferências das pessoas, foi considerado mais útil que o modelo gigante sem alinhamento.

### 2. RLHF, o caminho clássico

> **Em palavras simples.** No **RLHF**, pessoas comparam pares de respostas e indicam a melhor. Com essas comparações, treina-se um segundo modelo, o **reward model**, que aprende a dar nota a qualquer resposta. Depois, o LLM é treinado por tentativa e erro para produzir respostas com nota alta. Funciona, e é trabalhoso: são dois modelos, e o treino por reforço é instável.

1. Coletam-se comparações humanas entre respostas do modelo.
2. Treina-se um **reward model** que prevê a preferência humana (modelo de Bradley–Terry: a probabilidade de A vencer B é `σ(r(A) − r(B))`).
3. Ajusta-se o LLM (a **policy**) por reinforcement learning, com **PPO**, para maximizar o reward, com uma penalidade de **KL divergence** para não se afastar demais do modelo original.

A penalidade KL é essencial: sem ela, a policy encontra respostas degeneradas que enganam o reward model (**reward hacking**). RLHF funciona, mas é complexo e instável: dois modelos, sampling durante o treino, muitos hyperparameters.

> **No mundo real.** A ideia de aprender uma função de recompensa a partir de comparações humanas vem do paper *Deep Reinforcement Learning from Human Preferences* (Christiano et al., OpenAI e DeepMind, 2017). Em uma das demonstrações, um robô simulado aprendeu a dar um salto mortal para trás com cerca de 900 comparações feitas por uma pessoa sem treinamento, em menos de uma hora, sem que ninguém tivesse escrito uma definição matemática de salto mortal.

### 3. DPO: a mesma meta, sem RL

> **Em palavras simples.** O **DPO** chega ao mesmo objetivo do RLHF por um caminho mais curto. Em vez de treinar um modelo de notas e depois usar reforço, ele usa diretamente os pares (resposta preferida, resposta rejeitada) em um treino comum: aumenta a chance da preferida e diminui a da rejeitada, sempre em relação ao modelo original. É um fine-tuning supervisionado com uma loss diferente.

O DPO mostra que, para o objetivo do RLHF com penalidade KL, a policy ótima tem forma fechada em função do reward. Invertendo a relação, o reward pode ser escrito em termos da própria policy e de uma policy de referência. Substituindo no modelo de Bradley–Terry, o reward model desaparece, e sobra uma loss de classificação sobre pares:

`L = −log σ( β · [ (log π_θ(y_c|x) − log π_ref(y_c|x)) − (log π_θ(y_r|x) − log π_ref(y_r|x)) ] )`

- `π_θ`: o modelo em treino (policy).
- `π_ref`: o reference model, frozen (normalmente o próprio modelo antes do DPO).
- `y_c`, `y_r`: resposta chosen e rejected.
- `β`: controla o quanto a policy pode se afastar da referência. É o equivalente ao peso da KL no RLHF.
- `σ`: sigmoid.

`log π(y|x)` é a soma das log-probabilities dos tokens da resposta dado o prompt: a mesma quantidade da loss de SFT, com sinal trocado.

Leitura: a policy deve aumentar a log-probability da resposta chosen **mais do que** aumenta a da rejected, medidas em relação à referência. Se nada mudou, o argumento é zero e a loss vale `log 2 ≈ 0,693`: é o sanity check do início do treino.

O DPO exige apenas que a **margem** cresça. Ele pode fazê-lo subindo a chosen, derrubando a rejected, ou ambos. Em treinos longos, é comum a log-probability das duas cair, com a rejected caindo mais; isso é um sinal de que o modelo está se afastando da referência.

> **No mundo real.** O paper do DPO (Rafailov et al., Stanford, 2023) foi rapidamente posto à prova pelo **Zephyr-7B** (Hugging Face, 2023): um Mistral 7B ajustado com DPO sobre preferências geradas automaticamente. No benchmark MT-Bench, o Zephyr superou o Llama-2-chat de **70 bilhões** de parâmetros, treinado com o método clássico. O treino com DPO levou poucas horas, e o método se tornou o padrão em modelos abertos.

### 4. O papel de `β`

> **Em palavras simples.** O **β** controla o quanto o modelo pode se afastar da sua versão original ao perseguir as preferências. Com β alto, ele fica preso ao original e muda pouco. Com β baixo, muda muito e pode perder qualidades que tinha, como fluência e conhecimento geral. É um freio de segurança regulável.

- `β` alto (0,5 ou mais): mais conservador; fica perto da referência.
- `β` baixo (0,01 a 0,1): mais agressivo; aprende a preferência mais rápido e deriva mais.

Valor de partida usual: 0,1. A learning rate do DPO costuma ser menor que a do SFT.

> **No mundo real.** No paper do DPO, a maioria dos experimentos usa **β = 0,1**, e esse valor virou o padrão de bibliotecas como a TRL (`DPOConfig`). Os autores destacam a robustez como vantagem prática: o método funcionou com pouco ajuste de hyperparameters, enquanto a alternativa por reforço (PPO) exigia calibração cuidadosa para não degradar o modelo.

### 5. O dado: preference triples

> **Em palavras simples.** O dado do DPO é um trio: o **prompt**, a resposta **escolhida** e a resposta **rejeitada**. As duas respostas devem ser para o mesmo prompt, e a diferença entre elas deve refletir o que se quer ensinar. Se a rejeitada for absurda, o modelo aprende pouco; os pares mais instrutivos são aqueles em que as duas respostas são plausíveis e uma é claramente melhor.

```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

Fontes:

- Feedback de usuários (thumbs up/down, regenerações).
- Edições humanas de respostas do modelo (original = rejected, editada = chosen): fonte excelente, porque a diferença é exatamente o que precisa mudar.
- Side-by-side comparisons.
- Um LLM mais forte como judge, com rubric.
- Violações de política detectadas (rejected) pareadas com a resposta correta (chosen).

A qualidade depende de a diferença entre as duas respostas ser **a característica que se quer ensinar e nada além dela**. No notebook, chosen e rejected contêm o mesmo fato e diferem só no tom.

> **No mundo real.** O dataset **UltraFeedback** (OpenBMB, 2023) reúne cerca de 64 mil prompts, cada um com respostas de quatro modelos diferentes, avaliadas pelo GPT-4 em critérios como utilidade, veracidade e honestidade. Para treinar o Zephyr, a resposta mais bem avaliada virou a *escolhida* e uma das outras três, sorteada, a *rejeitada*. Nenhum avaliador humano participou da construção dos pares.

### 6. Riscos

> **Em palavras simples.** Otimizar preferências traz riscos próprios. O modelo pode aprender a **agradar** em vez de acertar, concordando com o usuário mesmo quando ele está errado. Pode explorar vícios dos avaliadores, como preferência por respostas longas. E, se as preferências forem inconsistentes ou enviesadas, o modelo herda os defeitos.

- **Shortcuts / spurious correlations:** se as respostas chosen forem sistematicamente mais longas, o modelo aprende "mais longo é melhor". Controle comprimento, formato e conteúdo entre os pares.
- **Drift:** `β` baixo demais ou treino longo degradam capacidades gerais. Teste em tarefas não relacionadas, como o notebook faz.
- **Sycophancy:** otimizar por aprovação de usuários pode ensinar o modelo a concordar em vez de acertar.
- **Overgeneralization:** o tom formal aprendido para suporte interno aparece em toda resposta, inclusive onde não faz sentido. A correção é incluir no dataset pares de outros contextos em que a resposta direta é a chosen.
- **Preference noise:** anotadores discordam; pares contraditórios confundem o treino.

> **No mundo real.** O estudo *Towards Understanding Sycophancy in Language Models* (Sharma et al., Anthropic, 2023) testou assistentes de várias empresas. Todos mostraram o comportamento: mudavam uma resposta correta quando o usuário a contestava e elogiavam mais um texto quando o usuário dizia ser o autor. Analisando dados de preferência humana, os autores constataram que respostas que confirmam a opinião do usuário tendem a ser preferidas, o que explica a origem do problema.

### 7. O ecossistema de preference optimization

> **Em palavras simples.** DPO é um entre vários métodos de **preference optimization**. Há variantes que dispensam o modelo de referência (**ORPO**, **SimPO**), que funcionam com avaliações isoladas de bom ou ruim em vez de pares (**KTO**), ou que corrigem tendências a exagerar (**IPO**). A ideia comum é usar sinais de preferência com um treino simples e estável.

- **DPO** é o padrão por simplicidade.
- **IPO, KTO, ORPO, SimPO:** variantes que mudam a loss, dispensam o reference model ou aceitam feedback não pareado (KTO funciona com "bom"/"ruim" isolados). ORPO combina SFT e preferência em uma só etapa.
- **Constitutional AI / RLAIF:** as preferências vêm de um modelo guiado por princípios escritos, em vez de humanos.

Na prática, com a biblioteca TRL: `DPOTrainer`, que recebe um dataset com as colunas `prompt`, `chosen` e `rejected`. O fluxo usual é SFT primeiro (para a tarefa) e DPO depois (para estilo e compliance), com LoRA nos dois.

> **No mundo real.** Os grandes laboratórios combinam métodos. O model card do **Llama 3** (Meta, 2024) informa que o ajuste dos modelos de chat usou uma combinação de supervised fine-tuning, rejection sampling, PPO e DPO. Na biblioteca **TRL**, da Hugging Face, várias dessas variantes estão disponíveis como opções de configuração do mesmo trainer, o que reduz o custo de comparar métodos nos próprios dados.

### 8. Como avaliar alignment

> **Em palavras simples.** Avaliar alignment é verificar se o modelo passou a se comportar como desejado **sem perder** o resto. Usa-se comparação pareada entre a versão antiga e a nova, julgada por pessoas ou por um LLM juiz, mais testes específicos: ele recusa o que deve recusar? Recusa o que não devia? Concorda com afirmações falsas?

- **Preference accuracy** em pares held-out: a fração em que `log π(chosen) > log π(rejected)`.
- **Win rate** em comparação pareada das gerações antes e depois, por humanos ou por um judge.
- **Capability regression:** as métricas de tarefa (Semana 21) e um conjunto geral, para medir drift.
- **Behavioral tests:** casos em que o comportamento aprendido **não** deveria aparecer.

> **No mundo real.** O benchmark **MT-Bench** (Zheng et al., 2023) usa o GPT-4 como juiz de conversas de vários turnos. Para validar a ideia, os autores compararam o juiz com avaliadores humanos: a concordância passou de 80%, o mesmo nível de concordância entre dois humanos. O paper também documentou os vieses do juiz, como favorecer a resposta mostrada primeiro e as respostas mais longas, e formas de reduzi-los.

### 9. Onde isso se encaixa em um produto

> **Em palavras simples.** Em um produto, os dados de preferência surgem naturalmente: o polegar para cima ou para baixo, a resposta que o usuário editou antes de usar, a opção escolhida entre duas. Coletados com cuidado, esses sinais viram os pares que alimentam um novo ciclo de ajuste. É assim que o produto melhora com o uso.

Alignment por preferências é uma das ferramentas de controle de comportamento, junto com system prompts, guardrails (Semana 25) e constrained decoding (Semana 26). É a mais cara de mudar e a mais difícil de auditar. Use-a para o que precisa estar **nos pesos** (um estilo consistente em um modelo pequeno, sem prompt longo), e prefira prompt e guardrails para regras que mudam com frequência.

> **No mundo real.** Em dezembro de 2024, a OpenAI lançou em sua API o **preference fine-tuning**, baseado em DPO: o cliente envia pares de resposta preferida e não preferida, e o modelo é ajustado para favorecer as primeiras. A documentação indica o método para tarefas subjetivas, em que tom e estilo importam, como redação e atendimento. Um método que era de pesquisa em 2023 virou um recurso de produto em pouco mais de um ano.

## Erros comuns

- Pares em que chosen e rejected diferem em várias dimensões ao mesmo tempo.
- Reference log-probs calculadas com o adapter já ligado e treinado.
- Esquecer o EOS ao calcular a log-probability da resposta.
- Usar a learning rate do SFT e ver o modelo derivar em poucos steps.
- Avaliar só a preference accuracy e não testar drift.
- Treinar com feedback de usuários sem filtrar: aprende-se o gosto do usuário médio, inclusive os vieses.

## Leitura e vídeos (fim de semana)

- Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (2023): Sections 1, 3 e 4.
- Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT): Sections 1 a 3, para o RLHF clássico.
- TRL documentation: *DPO Trainer*.
- Chip Huyen, *RLHF: Reinforcement Learning from Human Feedback*.
- Hugging Face blog: *Preference Tuning LLMs with Direct Preference Optimization Methods* (compara DPO, IPO e KTO).
- Sharma et al., *Towards Understanding Sycophancy in Language Models* (abstract).
- Opcional: Schulman et al., *Proximal Policy Optimization Algorithms* (abstract), e Bai et al., *Constitutional AI: Harmlessness from AI Feedback*.

## Perguntas de autoavaliação

1. Qual a diferença entre o dado de SFT e o dado de DPO?
2. Por que a loss do DPO vale `log 2` no início do treino?
3. O que o reference model impede? Qual o papel de `β`?
4. Dê um exemplo de shortcut indesejado que um preference dataset pode ensinar.
5. O que o DPO elimina em relação ao RLHF e por que isso o torna mais estável?
6. Como você detectaria que o alignment causou drift?

## Perguntas do entrevistador

1. Você aplicou DPO para tornar um assistente mais conciso e seguro, mas ele começou a recusar perguntas válidas ou dar respostas genéricas demais. O que o hiperparâmetro `β` e o reference model controlam nesse comportamento, e como detectar shortcuts no dataset de preferências?

## Entregável da semana

Notebook resolvido e uma nota em `docs/` com respostas antes/depois do DPO e a preference accuracy no conjunto held-out.

### Checkpoint da Fase 4

Você deve conseguir, sem consultar: escrever a fórmula do LoRA e contar seus parâmetros; explicar o label masking com `-100`; dizer o que QLoRA quantiza e o que treina; explicar a loss do DPO em uma frase; argumentar quando fine-tuning não é a resposta.

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/04-Phase-4-Fine-Tuning/notebooks/semana_22_exercicio.ipynb) · [`semana_22_exercicio.ipynb`](../notebooks/semana_22_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/04-Phase-4-Fine-Tuning/notebooks/semana_22_gabarito.ipynb) · [`semana_22_gabarito.ipynb`](../notebooks/semana_22_gabarito.ipynb) |

[← Semana 21](/04-Phase-4-Fine-Tuning/docs/Semana-21-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 23 →](/05-Phase-5-MLOps-and-Agents/docs/Semana-23-Conceitos.md)
