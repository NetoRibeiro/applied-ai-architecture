# Semana 13 — Chunking e o Primeiro RAG Pipeline

**Fase 3: Advanced RAG, Search e Reranking** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Montar o primeiro RAG pipeline de ponta a ponta com um open-weight LLM rodando localmente: dividir documentos em chunks, indexar, fazer retrieval, montar o prompt e gerar uma resposta grounded no contexto. Você vai comparar três chunking strategies e ver a hallucination aparecer e desaparecer.

## Por que isso importa para um AI Architect

Chunking é a decisão mais subestimada do RAG. Um chunk cortado no meio de uma frase não pode ser recuperado direito por nenhum embedding model nem respondido por nenhum LLM. E o grounding prompt é a primeira linha de defesa contra hallucination. Quando um RAG vai mal, a causa está muito mais frequentemente na ingestão do que no modelo.

## Conceitos

### 1. Por que RAG

> **Em palavras simples.** Um LLM só sabe o que estava nos dados de treino: não conhece os documentos internos da sua empresa nem o que aconteceu ontem, e pode inventar com segurança. **RAG** (*Retrieval-Augmented Generation*) resolve isso em dois tempos: primeiro busca os trechos relevantes em uma base de documentos, depois entrega esses trechos ao modelo junto com a pergunta. O modelo responde consultando, não de cabeça.

Um LLM tem conhecimento **parametric** (nos pesos): congelado na data do treino, sem os dados privados da empresa, e sem como citar a fonte. RAG acrescenta conhecimento **non-parametric**: documentos recuperados em query time e colocados no contexto.

| | RAG | Fine-tuning | Long context (colar tudo no prompt) |
|---|---|---|---|
| Fatos atualizáveis | Sim, basta reindexar | Não | Sim |
| Citação da fonte | Sim | Não | Parcial |
| Access control por usuário | Sim (no retriever) | Não | Difícil |
| Custo por requisição | Médio | Baixo | Alto |
| Ensina formato, tom, tarefa | Não | Sim | Não |

RAG ensina **fatos**; fine-tuning ensina **comportamento**. São complementares.

> **No mundo real.** O termo vem do paper de Lewis et al. (Facebook AI, 2020), que ligou um gerador a um index com 21 milhões de trechos da Wikipedia. Além de responder melhor a perguntas factuais, o sistema tinha uma propriedade nova: para atualizar o conhecimento, bastava **trocar o index**, sem retreinar o modelo. Os autores demonstraram isso trocando a Wikipedia de 2016 pela de 2018 e observando as respostas sobre líderes mundiais mudarem.

### 2. O pipeline

> **Em palavras simples.** Um sistema de RAG tem duas linhas de produção. A primeira, feita com antecedência, prepara os documentos: lê, divide em trechos, calcula os embeddings e guarda. A segunda acontece a cada pergunta: transforma a pergunta em embedding, busca os trechos mais próximos, monta o prompt e pede a resposta ao LLM. Cada etapa pode falhar de um jeito próprio, e por isso cada uma é medida separadamente.

**Ingestion (offline):** load → parse/clean → chunk → embed → index com metadata.

**Query (online):** pergunta → embed → retrieve top-k → montar o prompt com o contexto → generate → devolver resposta com fontes.

O **parsing** costuma ser a etapa mais trabalhosa na vida real: PDFs com colunas, tabelas, cabeçalhos e rodapés repetidos, documentos escaneados. Lixo no parsing vira lixo nos chunks.

> **No mundo real.** Em 2023, a **Morgan Stanley** colocou em uso, para os seus consultores financeiros, um assistente construído com GPT-4 sobre uma base de cerca de **100 mil documentos** internos de pesquisa e procedimentos. Segundo o estudo de caso publicado pela OpenAI, o trabalho central não foi o modelo, e sim a preparação dos documentos e um conjunto de avaliação montado com especialistas para verificar se as respostas vinham das fontes corretas.

### 3. Por que dividir em chunks

> **Em palavras simples.** Documentos inteiros são grandes demais para funcionar bem como unidade de busca. Um único embedding para um manual de cem páginas é uma média borrada de todos os assuntos. E o modelo de embeddings só lê um número limitado de tokens. Dividir o documento em **chunks** dá a cada trecho um vetor focado e permite entregar ao LLM só a parte que interessa.

- O embedding model tem `max_seq_length` e um único vetor dilui documentos longos: um vetor que representa 30 páginas não representa bem nenhuma delas.
- O LLM paga, em tokens, latência e atenção, por tudo o que entra no contexto.
- A unidade recuperada deve ser a menor que ainda responde sozinha a uma pergunta.

> **No mundo real.** O `all-MiniLM-L6-v2` foi configurado para ler no máximo **256 tokens**; o que passa disso é cortado em silêncio. Um contrato de 30 páginas indexado inteiro seria representado apenas pelo seu primeiro parágrafo. Esse limite, documentado no model card, é o tipo de detalhe que explica por que um RAG não encontra uma informação que está, comprovadamente, no documento.

### 4. Chunking strategies

> **Em palavras simples.** Há várias formas de cortar um documento. O corte por tamanho fixo é simples e pode partir uma frase ao meio. O corte recursivo tenta respeitar parágrafos e frases. O corte por estrutura segue títulos e seções. A **overlap** repete um pouco do fim de um trecho no início do seguinte, para não perder o que ficou na fronteira. A melhor estratégia depende do tipo de documento e deve ser medida.

| Strategy | Como | Prós | Contras |
|---|---|---|---|
| Fixed-size | A cada `n` caracteres/tokens, com overlap | Trivial, previsível | Corta frases e tabelas ao meio |
| Recursive / structure-aware | Tenta separar por heading, depois parágrafo, depois frase | Respeita a estrutura do documento | Depende de o documento ter estrutura |
| Semantic chunking | Quebra onde a similaridade entre frases vizinhas cai | Chunks tematicamente coesos | Mais caro; sensível ao threshold |
| Document / section-based | Uma página, uma cláusula, um item de FAQ | Ideal quando a fonte já é atômica | Tamanhos muito desiguais |

**Chunk size:** meça em tokens do embedding model, não em caracteres. Chunks pequenos (100–300 tokens) dão precisão no retrieval; chunks grandes (500–1.000) dão contexto para a geração. Não existe valor universal: é um hiperparâmetro a ser avaliado (Semana 16).

**Overlap** (10–20%) reduz o risco de a resposta ficar dividida entre dois chunks, ao custo de redundância no index.

> **No mundo real.** O `RecursiveCharacterTextSplitter`, do LangChain, tornou-se o padrão de fato da indústria: tenta dividir primeiro por parágrafos, depois por linhas, depois por frases, até caber no tamanho pedido. Estudos comparativos publicados em 2024, como o relatório da Chroma sobre estratégias de chunking, mostraram que o tamanho do chunk e a overlap alteram de forma mensurável o recall da busca, e que os valores padrão das bibliotecas raramente são os melhores.

### 5. Técnicas que desacoplam "o que se busca" de "o que se entrega"

> **Em palavras simples.** Um trecho pequeno é bom para a busca, pois o vetor fica focado; um trecho grande é bom para a resposta, pois traz contexto. Algumas técnicas usam os dois: busca-se pelo trecho pequeno e entrega-se ao LLM o parágrafo ou a seção em volta. Outra ideia é acrescentar a cada trecho uma frase que o situe no documento antes de calcular o embedding.

- **Parent-document / small-to-big retrieval:** indexar chunks pequenos (precisos) e entregar ao LLM o chunk pai ou a seção inteira (contexto).
- **Sentence-window retrieval:** indexar frases; entregar a frase mais uma janela de vizinhas.
- **Contextual chunk headers:** prefixar cada chunk com o título do documento e da seção antes de gerar o embedding. Um chunk solto como "O prazo é de 10 dias úteis" não diz de que prazo se trata; com o header, diz. Custo quase zero, ganho consistente.
- **Contextual retrieval:** usar um LLM para escrever uma frase de contexto para cada chunk na ingestão.

Guarde sempre no payload: `doc_id`, título, seção, posição, data, URL da fonte. É o que permite citar, filtrar e depurar.

> **No mundo real.** A Anthropic publicou em 2024 a técnica **Contextual Retrieval**: antes de indexar, um LLM escreve para cada chunk uma ou duas frases explicando de que documento e de que seção ele vem. Nos testes divulgados, isso reduziu em 35% a taxa de falhas da busca; combinado com BM25, em 49%; e, com reranking, em 67%. O ganho vem de resolver chunks que, isolados, não dizem a que empresa ou período se referem.

### 6. O grounding prompt

> **Em palavras simples.** O **grounding prompt** é a instrução que amarra o modelo aos documentos: responda apenas com base nos trechos abaixo, cite de qual trecho veio cada afirmação e, se a resposta não estiver ali, diga que não sabe. Sem essa amarra, o modelo completa as lacunas com o que parece plausível, e a resposta sai fluente e errada.

Elementos que funcionam:

1. Instrução para responder **somente** com base no contexto fornecido.
2. Uma saída explícita para quando a resposta não está no contexto ("Não encontrei essa informação"). Sem ela, o modelo completa com conhecimento parametric ou inventa.
3. Chunks numerados e pedido de **citation** por número (`[1]`, `[2]`).
4. Contexto claramente delimitado e separado da pergunta.
5. Instruções de formato (tamanho, idioma).

Instruções ficam no `system` message; contexto e pergunta, no `user` message. Conteúdo recuperado é **dado**, não instrução: um documento que contém "ignore as instruções anteriores" é uma indirect prompt injection (Semana 25).

> **No mundo real.** Em 2024, um tribunal canadense condenou a **Air Canada** a indenizar um passageiro (caso *Moffatt v. Air Canada*): o chatbot do site havia descrito uma política de reembolso por luto que não existia. A empresa argumentou que o chatbot era responsável pelas próprias palavras, e o tribunal rejeitou a tese. O caso é citado como o exemplo de que a empresa responde pelo que o assistente afirma, e de por que as respostas precisam estar presas às fontes.

### 7. Quantos chunks (`k`) e em que ordem

> **Em palavras simples.** Quantos trechos entregar ao modelo? Poucos, e a resposta pode ficar de fora. Muitos, e o modelo se perde, o custo sobe e a resposta demora. A ordem também importa: os modelos prestam mais atenção ao começo e ao fim do prompt do que ao meio. Por isso o `k` é ajustado com medição e os melhores trechos vão para as pontas.

`input tokens ≈ system prompt + k × chunk size + pergunta`. Dobrar `k` dobra o custo e o tempo de prefill, e nem sempre melhora a resposta: chunks irrelevantes são distração.

**Lost in the middle:** LLMs usam melhor a informação que está no início e no fim do contexto; o que fica no meio tende a ser ignorado. Coloque os chunks mais relevantes nas pontas e mantenha `k` tão pequeno quanto o recall permitir. É um dos motivos para reranking (Semana 14).

> **No mundo real.** O paper *Lost in the Middle* (Liu et al., Stanford, 2023) mediu esse efeito. Com 20 documentos no prompt e a resposta em apenas um deles, a accuracy dos modelos formava um **U**: alta quando o documento certo era o primeiro ou o último, e bem mais baixa quando estava no meio. Em um dos testes, o GPT-3.5 com a resposta no meio saiu-se pior do que sem documento nenhum.

### 8. Failure modes de um RAG

> **Em palavras simples.** Quando um RAG responde mal, a causa pode estar em pontos diferentes: o documento não estava na base; estava, mas a busca não o trouxe; a busca trouxe, mas ele ficou fora dos trechos enviados; foi enviado, e o modelo o ignorou ou o interpretou mal. Saber em qual ponto falhou é o que indica o conserto certo.

| Falha | Onde nasce | Sintoma |
|---|---|---|
| Chunk certo não foi recuperado | Parsing, chunking, embedding, index | "Não encontrei" ou resposta inventada |
| Chunk certo veio, mas enterrado | Ranking, `k` grande demais | Resposta ignora o chunk |
| Chunk certo veio, resposta errada | LLM, prompt | Resposta contradiz o contexto |
| Chunks conflitantes ou desatualizados | Data lifecycle | Resposta com a versão antiga |
| Pergunta sem resposta na base | Cobertura | Modelo inventa em vez de recusar |
| Pergunta precisa de vários documentos | Retrieval de passo único | Resposta parcial |

Separar **retrieval failure** de **generation failure** é o primeiro passo de qualquer diagnóstico (Semana 16).

> **No mundo real.** O paper *Seven Failure Points When Engineering a Retrieval Augmented Generation System* (Barnett et al., 2024) analisou três sistemas reais, em pesquisa, educação e biomedicina, e catalogou sete pontos de falha, de *conteúdo ausente* a *resposta incompleta*. Duas conclusões dos autores: a validação de um RAG só é possível com o sistema em operação, e a robustez não é projetada de uma vez, ela evolui com o uso.

### 9. Além do RAG ingênuo

> **Em palavras simples.** O RAG básico faz uma única busca e responde. Variações mais elaboradas reescrevem a pergunta antes de buscar, fazem várias buscas encadeadas, deixam o modelo decidir se precisa buscar, ou organizam os documentos em um grafo de entidades e relações para responder a perguntas sobre o conjunto. Cada camada acrescenta qualidade em certos casos e sempre acrescenta custo e latência.

Para situar o que vem depois: **query rewriting** e **HyDE** (gerar uma resposta hipotética e buscar por ela), **multi-query** (várias reformulações), **hybrid search e reranking** (Semana 14), **agentic RAG** (o modelo decide quando e o que buscar, em vários passos), **GraphRAG** (retrieval sobre um knowledge graph). Todos partem do pipeline desta semana.

> **No mundo real.** O **GraphRAG**, publicado pela Microsoft Research em 2024, usa um LLM para extrair entidades e relações dos documentos, agrupa-as em comunidades e gera resumos de cada uma. O alvo são perguntas globais, como *quais são os temas principais deste acervo?*, nas quais o RAG comum falha porque nenhum trecho isolado contém a resposta. O custo de indexação, porém, é bem maior, pois o LLM lê todo o acervo antes da primeira pergunta.

## Erros comuns

- Medir chunk size em caracteres e estourar o `max_seq_length` do embedding model.
- Zero overlap com fixed-size chunking.
- Prompt sem instrução de recusa.
- Aumentar `k` para "garantir" e piorar a resposta.
- Não guardar metadata de origem e não conseguir citar nem depurar.
- Avaliar o RAG só lendo meia dúzia de respostas.

## Leitura e vídeos (fim de semana)

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020): abstract e Introduction.
- Greg Kamradt, *5 Levels of Text Splitting* (vídeo e notebook).
- Anthropic, *Introducing Contextual Retrieval* (engineering blog).
- Liu et al., *Lost in the Middle: How Language Models Use Long Contexts* (abstract e figuras).
- Barnett et al., *Seven Failure Points When Engineering a Retrieval Augmented Generation System*.
- Gao et al., *Retrieval-Augmented Generation for Large Language Models: A Survey* (seções sobre Naive, Advanced e Modular RAG).
- LangChain ou LlamaIndex documentation: *Text splitters* / *Node parsers*, para ver as implementações de referência.

## Perguntas de autoavaliação

1. Por que não indexar o documento inteiro como um único vetor?
2. Qual chunking strategy você usaria para contratos com cláusulas numeradas?
3. Cite os elementos de um bom grounding prompt.
4. O RAG respondeu errado. Como saber se a culpa é do retriever ou do LLM?
5. O que é "lost in the middle" e que decisão de design ele motiva?
6. Em que situação fine-tuning resolve o que RAG não resolve, e vice-versa?

## Perguntas do entrevistador

1. Seu pipeline RAG está recuperando trechos relevantes de PDFs corporativos, mas o LLM ainda omite informações importantes ou mistura fatos de seções diferentes. Como você diagnostica se o gargalo está na estratégia de chunking (tamanho, overlap, fronteiras semânticas), no prompt de grounding ou no limite de contexto?

## Entregável da semana

Notebook resolvido e uma tabela em `docs/` comparando as três chunking strategies no seu corpus.

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/03-Phase-3-Advanced-RAG/notebooks/semana_13_exercicio.ipynb) · [`semana_13_exercicio.ipynb`](../notebooks/semana_13_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/03-Phase-3-Advanced-RAG/notebooks/semana_13_gabarito.ipynb) · [`semana_13_gabarito.ipynb`](../notebooks/semana_13_gabarito.ipynb) |

[← Semana 12](/03-Phase-3-Advanced-RAG/docs/Semana-12-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 14 →](/03-Phase-3-Advanced-RAG/docs/Semana-14-Conceitos.md)
