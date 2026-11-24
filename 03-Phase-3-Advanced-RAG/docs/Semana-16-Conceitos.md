# Semana 16 — RAG Evaluation: Faithfulness, Context e Capstone

**Fase 3: Advanced RAG, Search e Reranking** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Fechar a Fase 3 avaliando um sistema RAG completo com rigor: separar a qualidade do retrieval da qualidade da generation, medir faithfulness ao contexto, medir a capacidade de recusar (abstention) e comparar configurações com números.

## Por que isso importa para um AI Architect

"Parece que está respondendo bem" não é critério de aceite. Um arquiteto precisa de um evaluation set, métricas definidas e um processo que rode a cada mudança de prompt, modelo, chunking ou index. Sem isso, toda alteração é uma aposta, e regressões só aparecem pela reclamação do usuário.

## Conceitos

### 1. A RAG triad

> **Em palavras simples.** Um RAG pode falhar de três formas, e a **RAG triad** mede cada uma. **Context relevance**: os trechos buscados têm a ver com a pergunta? **Faithfulness** (ou *groundedness*): a resposta diz apenas o que está nos trechos? **Answer relevance**: a resposta atende ao que foi perguntado? Uma resposta pode ser fiel aos trechos e inútil, ou útil e inventada; por isso são três medidas.

| Dimensão | Pergunta | Compara | Se falha, o culpado é |
|---|---|---|---|
| **Context relevance** | Os chunks recuperados servem para responder? | Query × context | Retrieval |
| **Faithfulness** (groundedness) | Tudo o que a resposta afirma está no contexto? | Answer × context | Generation (hallucination) |
| **Answer relevance** | A resposta atende à pergunta? | Answer × query | Generation / prompt |

Com ground truth disponível, somam-se:

- **Context recall:** o chunk que contém a resposta foi recuperado?
- **Context precision:** dos chunks recuperados, quantos são relevantes, e estão no topo?
- **Answer correctness:** a resposta bate com a reference answer?

As três primeiras são **reference-free** (não precisam de gabarito e servem para monitorar produção); as três últimas são **reference-based** (precisam do golden set).

> **No mundo real.** A formulação vem de duas ferramentas abertas de 2023: o **TruLens**, que cunhou o nome *RAG triad*, e o **RAGAS** (Es et al., 2023), que propôs métricas calculadas por um LLM sem exigir gabarito humano para cada pergunta. O paper do RAGAS validou as métricas comparando-as com julgamentos humanos em perguntas sobre páginas da Wikipedia, e a concordância mais alta foi justamente em faithfulness.

### 2. Volta à Semana 1

Faithfulness é precision: das afirmações feitas, quantas têm suporte? Um false positive aqui é uma hallucination entregue ao usuário. Context recall é recall: da informação necessária, quanto foi recuperado? E a recusa correta ("não encontrei") é um true negative que quase ninguém mede.

Faithful não é o mesmo que correto:

- **Faithful e errada:** o contexto recuperado está desatualizado e a resposta o reproduz fielmente.
- **Correta e unfaithful:** o modelo ignorou o contexto e respondeu com conhecimento parametric que, por acaso, estava certo. Em um sistema corporativo isso é um defeito: a resposta não é rastreável.

### 3. Como medir faithfulness

> **Em palavras simples.** Medir **faithfulness** é conferir a resposta frase por frase. Quebra-se a resposta em afirmações simples e, para cada uma, verifica-se se os trechos fornecidos a sustentam. A nota é a fração de afirmações sustentadas. Essa conferência pode ser feita por pessoas, por um modelo de inferência textual (**NLI**) ou por um LLM atuando como juiz.

**NLI (Natural Language Inference).** Um modelo classifica o par (premise = contexto, hypothesis = afirmação) em *entailment*, *neutral* ou *contradiction*. Rápido, local, determinístico e barato. Limitações: degrada com contextos longos (truncation em 512 tokens) e com respostas que combinam vários chunks. Boa prática: dividir a resposta em frases e verificar cada uma contra cada chunk, agregando pelo máximo.

**LLM-as-a-judge.** O procedimento do RAGAS: (1) um LLM decompõe a resposta em **atomic claims**; (2) para cada claim, um LLM julga se ela é suportada pelo contexto; (3) `faithfulness = claims suportadas / total de claims`. Mais flexível e mais caro.

Cuidados com LLM judges:

- **Valide contra humanos:** meça a concordância (Cohen's kappa, ou simplesmente precision e recall do judge) em uma amostra de 50 a 100 casos anotados.
- Use uma **rubric** explícita e, de preferência, saída binária ou em poucas categorias; escalas de 1 a 10 são ruidosas.
- Peça o raciocínio antes do veredito.
- Use um judge mais forte que o modelo avaliado, e de preferência de outra família (self-preference bias).
- Fixe `temperature=0` e versione o prompt do judge como código.

**Frameworks:** RAGAS, DeepEval, TruLens e as evals do Phoenix e do LangSmith empacotam essas métricas. Nesta semana você as implementa para entender o que os frameworks calculam; em um projeto real, use-os.

> **No mundo real.** O **FActScore** (Min et al., 2023) aplicou esse método a biografias geradas por LLMs, conferindo cada afirmação contra a Wikipedia. O ChatGPT da época atingiu apenas **58%** de afirmações sustentadas, e a taxa caía ainda mais para pessoas menos conhecidas. O estudo também mostrou que um avaliador automático reproduzia o julgamento humano com erro inferior a 2%, o que viabilizou a medição em escala.

### 4. Answer correctness

> **Em palavras simples.** **Answer correctness** compara a resposta do sistema com uma resposta de referência. Comparar palavra por palavra é injusto: *1945* e *no ano de 1945* dizem o mesmo. Comparar por significado, com embeddings ou com um LLM juiz, é mais justo e exige cuidado, porque o juiz também erra. Na prática, combina-se mais de uma medida e valida-se o juiz com uma amostra revisada por pessoas.

Em ordem crescente de qualidade e custo:

- **Exact match / F1 de tokens:** só para respostas curtas e factuais.
- **Semantic similarity** entre embeddings da resposta e da referência: barato; não distingue "10 dias" de "20 dias" (frases quase idênticas, fatos opostos).
- **LLM judge com a referência:** o mais confiável para respostas abertas.

No notebook usamos similarity como triagem; saiba o limite dela.

> **No mundo real.** O estudo *Evaluating Open-Domain Question Answering in the Era of Large Language Models* (Kamalloo et al., 2023) reavaliou manualmente respostas a perguntas do Natural Questions. A métrica tradicional de coincidência exata de texto subestimava muito os LLMs: uma grande parte das respostas marcadas como erradas estava correta, só que escrita de outra forma ou mais completa que o gabarito. Com avaliação humana, a ordem dos sistemas mudou.

### 5. Abstention

> **Em palavras simples.** **Abstention** é a capacidade de dizer *não sei* quando a resposta não está nos documentos. Um sistema que nunca se abstém inventa; um que se abstém demais é inútil. Mede-se como um classificador: das perguntas sem resposta, em quantas ele se absteve? Das que tinham resposta, em quantas ele se recusou sem necessidade?

Duas taxas, que puxam em direções opostas:

- **Correct refusal rate:** das perguntas sem resposta na base, em quantas o sistema recusou? Baixa → hallucination.
- **False refusal rate:** das perguntas com resposta, em quantas recusou indevidamente? Alta → sistema inútil.

O prompt de grounding e o `k` deslocam o equilíbrio. Medir só uma delas leva a otimizar na direção errada.

> **No mundo real.** O dataset **SQuAD 2.0** (Rajpurkar et al., Stanford, 2018) acrescentou ao original mais de 50 mil perguntas **sem resposta** no texto, escritas para parecer respondíveis. Um sistema que obtinha 86% de F1 na versão anterior caiu para 66% na nova: os modelos sabiam achar respostas, mas não sabiam reconhecer a ausência delas. Saber se abster passou a ser tratado como uma capacidade a medir separadamente.

### 6. O golden set

> **Em palavras simples.** O **golden set** é a prova oficial do seu sistema: um conjunto de perguntas com as respostas corretas e os documentos que as sustentam, revisado por quem entende do assunto. Deve refletir as perguntas reais dos usuários, incluir casos difíceis e perguntas sem resposta. É pequeno, de dezenas a poucas centenas de itens, e é o ativo mais valioso do projeto.

- 50 a 200 perguntas **reais** de usuários (logs, tickets, entrevistas), com reference answer e documento-fonte.
- Inclua **unanswerable questions**.
- Inclua casos difíceis: multi-hop (várias fontes), negações, números, siglas, perguntas ambíguas, perguntas sobre conteúdo desatualizado.
- Versione junto com o código. Cada bug de produção vira um caso novo (é um regression test).
- **Synthetic test sets:** um LLM pode gerar perguntas a partir dos chunks. Útil para começar e para volume; as perguntas tendem a ser mais fáceis e mais parecidas com o texto do que as reais. Revise uma amostra.

> **No mundo real.** O Google mantém, segundo suas próprias publicações, cerca de 16 mil avaliadores externos que julgam resultados de busca seguindo um manual público de mais de 150 páginas, as *Search Quality Rater Guidelines*. Esses julgamentos não alteram resultados diretamente: servem de gabarito para avaliar mudanças no algoritmo antes do lançamento. É o mesmo papel do golden set, em escala industrial.

### 7. Component-wise vs. end-to-end

> **Em palavras simples.** Avaliar **end-to-end** é olhar só a resposta final. Avaliar **component-wise** é medir cada etapa: a busca trouxe o trecho certo? O modelo o usou bem? A avaliação final diz se há problema; a por componente diz onde. Se a busca não traz o documento, melhorar o prompt não adianta.

Avalie o retriever isoladamente (Recall@K, NDCG; Semana 15), o generator isoladamente (dado o contexto correto, a resposta é fiel?) e o sistema inteiro. A avaliação por componente diz **onde** investir.

| | Generation faithful | Generation unfaithful |
|---|---|---|
| **Contexto certo recuperado** | Sucesso | Generation failure: ajustar prompt ou modelo |
| **Contexto certo ausente** | Deveria ter recusado; verifique | Hallucination por falta de contexto: corrigir retrieval **e** o prompt de recusa |

Classificar cada erro em uma dessas células transforma "o RAG está ruim" em uma lista de trabalho priorizada.

> **No mundo real.** O benchmark **RGB** (Chen et al., 2023) separou as capacidades que um LLM precisa ter dentro de um RAG e testou cada uma isoladamente: resistir a trechos irrelevantes, recusar-se quando nenhum trecho contém a resposta, combinar informações de vários trechos e lidar com trechos que contêm erros. Os modelos avaliados mostraram fraquezas distintas em cada item, algo invisível em uma nota única.

### 8. Comparando configurações

> **Em palavras simples.** Para decidir entre duas configurações de RAG, comparam-se as duas **nas mesmas perguntas** e verifica-se se a diferença é maior que o ruído. Como as avaliações com LLM juiz variam entre execuções e os golden sets são pequenos, uma diferença de dois pontos pode não ser nada. O **bootstrap** pareado dá um intervalo de confiança para a diferença.

Mude **uma variável por vez** (chunk size, `k`, reranker, prompt, modelo) e compare no mesmo golden set. Com 50 a 200 casos, diferenças pequenas são ruído: use o bootstrap da Semana 15. Registre cada execução (configuração, métricas, versão do golden set) como um experimento rastreável (Semana 24).

> **No mundo real.** O uso de bootstrap para comparar sistemas de linguagem foi popularizado por Koehn (2004), em tradução automática: ele mostrou que, com poucas centenas de frases de teste, diferenças pequenas na métrica BLEU frequentemente não eram significativas, e propôs reamostrar o conjunto de teste para estimar a confiança. O mesmo procedimento se aplica diretamente a um golden set de RAG.

### 9. Continuous evaluation

> **Em palavras simples.** A avaliação não termina no lançamento. Os documentos mudam, as perguntas dos usuários mudam e o próprio LLM pode ser atualizado pelo fornecedor. **Continuous evaluation** é rodar o golden set automaticamente a cada mudança e amostrar respostas reais de produção para revisão, de modo que uma piora seja detectada pela equipe antes dos usuários.

- **Offline (CI):** o golden set roda a cada pull request que toca prompt, modelo, chunking ou index. Um threshold por métrica bloqueia regressões.
- **Online:** uma amostra do tráfego de produção é avaliada com as métricas reference-free, além dos sinais do usuário (thumbs up/down, reformulação da pergunta, escalonamento para humano).
- O loop: traces de produção → revisão de erros → novos casos no golden set → correção → reavaliação. A Semana 25 trata da observabilidade que alimenta esse ciclo.

> **No mundo real.** O estudo *How Is ChatGPT's Behavior Changing over Time?* (Chen, Zaharia e Zou, 2023) comparou as versões de março e de junho de 2023 do GPT-4 nas mesmas tarefas. Em uma delas, identificar números primos, a accuracy caiu de 84% para 51%, sem aviso aos usuários da API. O estudo é o argumento mais citado para fixar a versão do modelo e manter testes de regressão próprios.

## Erros comuns

- Avaliar só end-to-end e não saber se o problema é retrieval ou generation.
- Golden set sem unanswerable questions.
- Confiar em um LLM judge que nunca foi comparado a julgamento humano.
- Usar semantic similarity como medida de correção factual.
- Mudar três coisas de uma vez e atribuir o ganho a uma delas.
- Golden set construído só com perguntas sintéticas fáceis.

## Leitura e vídeos (fim de semana)

- Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation* (paper curto) e a seção *Metrics* da documentação do RAGAS.
- TruLens documentation: *The RAG Triad*.
- Hamel Husain, *Your AI Product Needs Evals* e *Creating a LLM-as-a-Judge That Drives Business Results*.
- Eugene Yan, *Patterns for Building LLM-based Systems & Products* (seções *Evals* e *RAG*) e *Evaluating the Effectiveness of LLM-Evaluators*.
- Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (os vieses de LLM judges).
- Opcional: Honovich et al., *TRUE: Re-evaluating Factual Consistency Evaluation* (NLI para faithfulness).

## Perguntas de autoavaliação

1. Uma resposta pode ser faithful e errada? E correta e unfaithful? Dê um exemplo de cada.
2. Por que incluir unanswerable questions no golden set?
3. Seu RAG tem context recall 0,95 e faithfulness 0,70. Onde está o problema?
4. Quais os riscos de usar um LLM judge sem validação, e como validá-lo?
5. Por que semantic similarity é uma medida fraca de answer correctness?
6. Qual a diferença entre métricas reference-free e reference-based e onde cada uma é usada?

## Entregável da semana

Notebook resolvido, a tabela comparativa de configurações e um relatório de uma página em `docs/` com: métricas, os três piores erros classificados na matriz de diagnóstico e a próxima melhoria que você faria.

### Checkpoint da Fase 3

Você deve conseguir, sem consultar: desenhar o multi-stage ranking funnel; definir Recall@K, MRR e NDCG; explicar HNSW e seus três parâmetros; nomear a RAG triad e dizer quem é o culpado quando cada dimensão falha.

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/03-Phase-3-Advanced-RAG/notebooks/semana_16_exercicio.ipynb) · [`semana_16_exercicio.ipynb`](../notebooks/semana_16_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/03-Phase-3-Advanced-RAG/notebooks/semana_16_gabarito.ipynb) · [`semana_16_gabarito.ipynb`](../notebooks/semana_16_gabarito.ipynb) |

[← Semana 15](/03-Phase-3-Advanced-RAG/docs/Semana-15-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 17 →](/04-Phase-4-Fine-Tuning/docs/Semana-17-Conceitos.md)
