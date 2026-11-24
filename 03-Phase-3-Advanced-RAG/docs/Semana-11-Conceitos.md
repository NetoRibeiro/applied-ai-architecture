# Semana 11 — Embeddings e Semantic Search (Exact vs. HNSW)

**Fase 3: Advanced RAG, Search e Reranking** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Usar um embedding model real para semantic search e entender o que muda quando a base cresce: a exact search fica cara e entram os índices de Approximate Nearest Neighbor (ANN), em especial o HNSW. Você vai medir o tradeoff entre recall, latência e memória.

## Por que isso importa para um AI Architect

A qualidade de um RAG é limitada pelo que o retriever traz: o LLM não responde com o que não recebeu. E o custo de infraestrutura de um RAG é dominado pelo vector index. Escolher embedding model, dimensão, similarity metric e parâmetros do index é a primeira decisão de arquitetura de qualquer sistema de busca, e a mais cara de reverter.

## Conceitos

### 1. Bi-encoder

> **Em palavras simples.** Um **bi-encoder** transforma cada texto, sozinho, em uma lista de números (o **embedding**), de modo que textos com significado parecido fiquem próximos em um mapa. A pergunta e os documentos são convertidos separadamente; por isso os documentos podem ser processados uma única vez, com antecedência. Buscar passa a ser achar os pontos do mapa mais próximos da pergunta, mesmo que ela não use as mesmas palavras do documento.

Um embedding model codifica query e documento **separadamente** no mesmo vector space. Os documentos são codificados uma vez, na ingestão (offline); em query time só a pergunta passa pelo modelo. É isso que torna a busca escalável: comparar com um milhão de documentos é uma matrix multiplication, não um milhão de forwards. O contraste é o cross-encoder (Semana 14).

> **No mundo real.** O paper **Dense Passage Retrieval** (Karpukhin et al., Facebook AI, 2020) aplicou um bi-encoder a 21 milhões de trechos da Wikipedia. No dataset Natural Questions, formado por perguntas reais feitas ao Google, o trecho com a resposta estava entre os 20 primeiros resultados em 78,4% dos casos, contra 59,1% do BM25, a busca por palavras-chave. Foi o resultado que colocou a busca por embeddings no centro dos sistemas de perguntas e respostas.

### 2. Como embedding models são treinados

> **Em palavras simples.** Um embedding model aprende por comparação (**contrastive learning**). Ele recebe pares que deveriam ficar juntos, como uma pergunta e a sua resposta, e muitos textos que não têm relação. O treino aproxima os pares verdadeiros no mapa e afasta os demais. Depois de ver milhões de pares, o modelo passa a organizar o mapa por significado.

Por **contrastive learning**: o modelo recebe pares (query, documento relevante) e é treinado para aproximar os positivos e afastar os negativos. Com **in-batch negatives**, os outros documentos do mesmo batch servem de negativos. **Hard negatives** (documentos parecidos, mas não relevantes) são o que mais melhora a qualidade.

Consequências práticas:

- O modelo aprende a noção de similaridade **dos dados de treino**. Um modelo treinado em perguntas e respostas da web pode ir mal em contratos jurídicos ou códigos de produto.
- A similarity metric correta (cosine ou dot product) é a usada no treino.
- Alguns modelos são **assimétricos** e exigem prefixos ou instruções diferentes para query e documento (`query:` / `passage:`). Omiti-los degrada o retrieval sem erro.

> **No mundo real.** O `all-MiniLM-L6-v2`, um dos embedding models mais baixados do Hugging Face Hub, foi treinado com mais de **1 bilhão de pares** de frases reunidos de fontes públicas: perguntas e respostas do Reddit e do Stack Exchange, títulos e resumos de artigos científicos, perguntas duplicadas do Quora, entre outras. Ninguém rotulou esses pares à mão: eles já existiam na estrutura dos sites. A técnica central é usar os outros exemplos do mesmo batch como negativos (**in-batch negatives**).

### 3. Como escolher um embedding model

> **Em palavras simples.** Não existe um embedding model melhor para tudo. Um modelo excelente em inglês pode ser fraco em português; um bom em frases curtas pode falhar em documentos longos; um maior custa mais memória e tempo. A escolha parte de rankings públicos para montar uma lista curta, e termina com um teste nos seus próprios dados, que é o único que decide.

| Critério | O que olhar |
|---|---|
| Idioma | Modelos multilingual para português; desempenho por idioma no MTEB |
| Dimension | 384 a 1.024 é comum; mais dimensão = mais memória e latência |
| Max sequence length | 256 a 8.192 tokens; texto além disso é truncado em silêncio |
| Simetria | Prefixos de query/passage exigidos? |
| Domínio | Avalie no **seu** dado; leaderboard é triagem, não decisão |
| Licença e hosting | Open e local vs. API paga; confidencialidade dos dados |
| Matryoshka | Alguns modelos permitem truncar o vetor (ex.: 768 → 256) com pouca perda |

O **MTEB** (Massive Text Embedding Benchmark) compara modelos em retrieval, clustering, classification e outras tarefas. Olhe a coluna de Retrieval e os datasets no seu idioma.

Trocar de embedding model obriga a **re-embed toda a base**: vetores de modelos diferentes não são comparáveis.

> **No mundo real.** O benchmark **MTEB** (Muennighoff et al., 2022) avaliou dezenas de modelos em 8 tipos de tarefa e 58 datasets, cobrindo mais de cem idiomas. A conclusão principal do paper é que nenhum modelo vence em todas as tarefas: o melhor em busca não é o melhor em clustering ou em classificação. O leaderboard público do MTEB virou a referência inicial para escolher modelos, com a ressalva de que posição no ranking não garante resultado no seu domínio.

### 4. Exact search (flat index)

> **Em palavras simples.** **Exact search** compara a pergunta com todos os vetores guardados, um por um, e devolve os mais próximos. O resultado é sempre o correto, e o custo cresce na mesma proporção do acervo. Para alguns milhares ou centenas de milhares de documentos, é rápido o bastante e dispensa qualquer estrutura sofisticada. Também serve de gabarito para medir o quanto os métodos aproximados erram.

Comparar a query com todos os `N` vetores: `O(N · d)`. Recall perfeito por definição. Viável até algumas centenas de milhares de vetores, e mais além com GPU. Sempre comece por ela: é o **ground truth** contra o qual se mede qualquer ANN index. No FAISS: `IndexFlatIP` (inner product) e `IndexFlatL2`.

> **No mundo real.** A biblioteca **FAISS**, da Meta, descrita no paper *Billion-scale similarity search with GPUs* (Johnson, Douze e Jégou, 2017), chama esse index de `IndexFlat`. A documentação o recomenda como ponto de partida e como referência de qualidade. Uma conta com dados reais: 1 milhão de vetores de 384 dimensões (o tamanho do `all-MiniLM-L6-v2`) ocupam cerca de 1,5 GB, e uma busca exata nesse volume leva dezenas de milissegundos em uma CPU moderna.

### 5. HNSW (Hierarchical Navigable Small World)

> **Em palavras simples.** **HNSW** é um index que organiza os vetores em uma rede de vizinhos com vários níveis, como um sistema de estradas: rodovias com poucas saídas no nível de cima, ruas locais embaixo. A busca começa pelas rodovias, chega rápido à região certa e só então percorre as ruas. Assim, visita uma fração mínima dos vetores e quase sempre acha os mesmos resultados da busca exata.

Um graph em camadas. A camada inferior contém todos os vetores, cada um ligado aos vizinhos próximos. Cada camada superior contém um subconjunto aleatório cada vez menor, com conexões "longas". A busca começa no topo, caminha gulosamente para o vizinho mais próximo da query, desce de camada e repete, refinando. É a mesma ideia de uma skip list aplicada a um graph de vizinhança.

| Parâmetro | O que controla | Efeito de aumentar |
|---|---|---|
| `M` | Conexões por nó | Mais recall e mais memória |
| `efConstruction` | Largura da busca ao construir | Graph melhor, indexação mais lenta |
| `efSearch` | Candidatos mantidos na busca | Mais recall, mais latência; **ajustável sem reindexar** |

Características operacionais: busca em tempo aproximadamente logarítmico; o graph fica em RAM; inserções são baratas; deleções são custosas (em geral marcadas e limpas depois). É o index padrão de pgvector, Qdrant, Weaviate, Milvus e OpenSearch.

> **No mundo real.** O algoritmo foi publicado por Malkov e Yashunin (2016) e se tornou o index padrão de quase todo o mercado: é usado no **Lucene** (e, portanto, no Elasticsearch e no OpenSearch), no **Qdrant**, no **Weaviate** e, desde a versão 0.5.0 (2023), no **pgvector**. No comparativo público **ANN-Benchmarks**, implementações de HNSW aparecem de forma consistente entre as melhores na relação entre recall e consultas por segundo.

### 6. Outras famílias de index

> **Em palavras simples.** Há outras formas de acelerar a busca. **IVF** divide os vetores em bairros e só procura nos bairros mais próximos da pergunta. **Product Quantization** guarda uma versão comprimida de cada vetor, que ocupa muito menos memória ao custo de alguma precisão. Índices em disco mantêm a maior parte dos dados no SSD. Cada família troca memória, velocidade e exatidão de um jeito diferente.

- **IVF (Inverted File):** agrupa os vetores em `nlist` células por k-means; a busca examina só as `nprobe` células mais próximas. Menos memória que HNSW; exige treino do index.
- **Product Quantization (PQ):** divide cada vetor em subvetores e substitui cada um pelo ID do centroide mais próximo. Compressão de 10× a 100×, com perda.
- **Scalar quantization:** `float32` → `int8` (4× menor) com perda pequena. **Binary quantization:** 1 bit por dimensão (32× menor), para uma primeira triagem.
- **DiskANN:** graph em SSD, para bases que não cabem em RAM.

Combinação comum: buscar em vetores comprimidos e fazer **rescoring** dos top candidatos com os vetores originais.

> **No mundo real.** O **DiskANN** (Microsoft Research, NeurIPS 2019) mostrou que é possível indexar **1 bilhão de vetores** em uma única máquina com 64 GB de RAM, mantendo o grafo em SSD e só vetores comprimidos na memória, com recall acima de 95% e latência de poucos milissegundos. Antes disso, um acervo desse tamanho exigia um cluster com centenas de gigabytes de RAM. A técnica é usada em produtos de busca da Microsoft.

### 7. O triângulo recall × latency × memory

> **Em palavras simples.** Em busca vetorial, três coisas disputam entre si: a qualidade do resultado (**recall**), a rapidez da resposta (**latency**) e a memória ocupada. Melhorar uma costuma piorar outra. Os parâmetros do index, como `ef_search` no HNSW, são os botões que deslocam o sistema dentro desse triângulo. Não existe configuração ótima universal: existe a adequada ao seu requisito.

Não existe index melhor em tudo. A pergunta de arquitetura é: qual o recall mínimo aceitável, e qual o menor custo que o entrega?

Atenção ao vocabulário: aqui, **recall@k do index** é a fração dos verdadeiros `k` vizinhos mais próximos (segundo a exact search) que o ANN devolveu. Não é o recall de relevância da Semana 15. São duas fontes de erro distintas:

1. O embedding model não coloca o documento certo entre os vizinhos (erro de modelo).
2. O index não devolve os verdadeiros vizinhos (erro de index).

Antes de trocar de modelo, rode a mesma query com exact search. Se o resultado melhora, o problema é o index.

> **No mundo real.** O projeto **ANN-Benchmarks** (Aumüller, Bernhardsson e Faithfull) publica, para dezenas de bibliotecas e datasets reais, gráficos de recall contra consultas por segundo. Em todas as curvas se vê o mesmo formato: passar de 90% para 99% de recall custa uma queda grande de velocidade, muitas vezes de várias vezes. É a evidência empírica de que os últimos pontos de recall são os mais caros e de que o alvo precisa ser decidido com o negócio.

### 8. Dimensionando memória

> **Em palavras simples.** A memória de um index vetorial se calcula com uma multiplicação: número de vetores × número de dimensões × 4 bytes, mais um acréscimo pela estrutura do index. Dobrar as dimensões dobra a conta. Fazer esse cálculo antes de escolher o modelo e o banco evita surpresas de custo depois.

Vetores `float32`: `N × d × 4` bytes. O graph HNSW acrescenta cerca de `N × M × 2 × 4` bytes.

| N | d | Vetores | Graph (M=32) |
|---|---|---|---|
| 1M | 768 | 3,1 GB | 0,26 GB |
| 10M | 768 | 30,7 GB | 2,6 GB |
| 100M | 1.024 | 410 GB | 25,6 GB |

A partir de dezenas de milhões, quantization e sharding deixam de ser opcionais.

> **No mundo real.** A OpenAI oferece o `text-embedding-3-small`, com 1.536 dimensões, e o `text-embedding-3-large`, com 3.072. Para 1 milhão de documentos, são cerca de 6 GB contra 12 GB só de vetores, antes do index. Os dois modelos aceitam o parâmetro `dimensions`, que encurta o vetor com pouca perda de qualidade, técnica conhecida como **Matryoshka Representation Learning**. A escolha de dimensão é, na prática, uma decisão de custo de infraestrutura.

### 9. Semantic vs. lexical

> **Em palavras simples.** **Semantic search** encontra textos pelo significado, mesmo com palavras diferentes: *carro* acha *automóvel*. **Lexical search** encontra textos pelas palavras exatas e é imbatível para códigos de produto, siglas e nomes próprios, que a busca semântica costuma confundir. As duas falham em situações opostas; por isso os melhores sistemas combinam ambas.

Dense retrieval captura paráfrase e sinônimos ("descanso anual" → "férias") e funciona entre idiomas. Em compensação, é fraco em **exact match**: códigos, siglas, números de processo, nomes próprios raros. O notebook mostra os dois lados; a solução é hybrid search (Semana 14).

> **No mundo real.** O benchmark **BEIR** (Thakur et al., 2021) testou modelos de busca em 18 datasets de domínios variados, sem treino específico em cada um. O resultado surpreendeu: o BM25, técnica lexical dos anos 1990, superou vários modelos de embeddings em boa parte dos datasets. Modelos densos treinados em um domínio perdiam desempenho ao mudar de assunto. O estudo é o principal argumento empírico a favor da busca híbrida (Semana 14).

## Erros comuns

- Escolher o modelo pelo topo do leaderboard sem avaliar no próprio dado.
- Indexar documentos maiores que o `max_seq_length` e não perceber a truncation.
- Esquecer os prefixos de query/passage em modelos assimétricos.
- Usar L2 em um index cujos vetores não estão normalizados quando o modelo espera cosine.
- Ajustar `efSearch` sem medir recall contra a exact search.
- Misturar na mesma collection vetores de versões diferentes do embedding model.

## Leitura e vídeos (fim de semana)

- Sentence-Transformers documentation: *Semantic Search*; *Pretrained Models*; *Training Overview* (para a loss contrastiva).
- Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs* (Introduction e Section 3).
- Pinecone Learn, *Faiss: The Missing Manual* (capítulos sobre HNSW, IVF e Product Quantization).
- Muennighoff et al., *MTEB: Massive Text Embedding Benchmark* (abstract e a descrição das tasks) e o MTEB Leaderboard.
- Karpukhin et al., *Dense Passage Retrieval for Open-Domain Question Answering* (abstract e seção 3), sobre bi-encoders e in-batch negatives.
- FAISS wiki: *Guidelines to choose an index*.

## Perguntas de autoavaliação

1. Por que um bi-encoder escala e um cross-encoder não?
2. Qual parâmetro do HNSW você ajusta em produção sem reconstruir o index?
3. Estime a memória para 5 milhões de vetores de 1.024 dimensões em `float32`. E em `int8`?
4. Seu RAG trouxe um documento errado. Como distinguir falha do embedding model de falha do index?
5. O que são hard negatives e por que importam no treino de embedding models?
6. Por que trocar de embedding model exige reindexar tudo?

## Entregável da semana

Notebook resolvido e o gráfico recall × latency do HNSW com a sua escolha de `efSearch` justificada em uma frase.

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/03-Phase-3-Advanced-RAG/notebooks/semana_11_exercicio.ipynb) · [`semana_11_exercicio.ipynb`](../notebooks/semana_11_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/03-Phase-3-Advanced-RAG/notebooks/semana_11_gabarito.ipynb) · [`semana_11_gabarito.ipynb`](../notebooks/semana_11_gabarito.ipynb) |

[← Semana 10](/02-Phase-2-Deep-Learning/docs/Semana-10-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 12 →](/03-Phase-3-Advanced-RAG/docs/Semana-12-Conceitos.md)
