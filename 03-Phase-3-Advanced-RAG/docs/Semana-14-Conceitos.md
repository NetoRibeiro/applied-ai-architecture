# Semana 14 — Hybrid Search e Multi-Stage Reranking

**Fase 3: Advanced RAG, Search e Reranking** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Construir o funil de busca moderno em três estágios: query understanding, hybrid retrieval (lexical BM25 + dense) e reranking com cross-encoder. É a arquitetura de referência de sistemas de Search & Discovery e de recommender systems.

## Por que isso importa para um AI Architect

Dense retrieval sozinho erra em códigos, siglas e nomes próprios; lexical search sozinho erra em paráfrases. Em produção, quase todo sistema sério é híbrido e tem reranking. Saber quanto de latência e custo cada estágio adiciona, e quanto de qualidade devolve, é o trabalho central de quem projeta busca e recomendação.

## Conceitos

### 1. O multi-stage ranking funnel

> **Em palavras simples.** Um bom sistema de busca trabalha em etapas, como um processo seletivo. A primeira etapa é rápida e pouco exigente: reduz milhões de documentos a algumas centenas de candidatos. A segunda é lenta e criteriosa: examina só esses candidatos e escolhe os melhores. Usar o avaliador caro em tudo seria inviável; usar só o barato daria resultados medianos.

| Stage | Entrada → saída | Técnica | Otimiza |
|---|---|---|---|
| 1. Query understanding | Texto → intent, filtros, query reescrita | Classifier, regras, LLM | Direcionar a busca |
| 2. Retrieval (candidate generation) | Milhões → centenas | BM25 + bi-encoder + fusion | **Recall** e velocidade |
| 3. Reranking | Centenas → dezenas ordenadas | Cross-encoder, learning-to-rank | **Precision** no topo |
| 4. Business rules / re-ranking final | Lista → lista final | Diversity, freshness, permissões | Produto |

Cada stage é mais caro por item e vê menos itens que o anterior. Um erro de recall no stage 2 é irrecuperável: o reranker só reordena o que recebeu. A mesma estrutura vale para recommender systems (candidate generation → ranking → re-ranking).

> **No mundo real.** O paper *Deep Neural Networks for YouTube Recommendations* (Covington et al., 2016) descreve exatamente essa arquitetura: uma rede de **candidate generation** reduz um acervo de milhões de vídeos a algumas centenas, e uma rede de **ranking**, com muito mais informações sobre cada vídeo e sobre o usuário, ordena esses poucos. O desenho em funil virou o padrão de sistemas de recomendação e de busca em grande escala.

### 2. BM25

> **Em palavras simples.** **BM25** é a fórmula clássica de busca por palavras. Ela dá mais pontos a um documento quando as palavras da pergunta aparecem nele, com três refinamentos: palavras raras valem mais que palavras comuns, repetir a mesma palavra muitas vezes rende cada vez menos, e documentos longos não levam vantagem só por serem longos. Não entende sinônimos, mas é rápida, barata e difícil de bater em termos exatos.

Função de scoring lexical, evolução do TF-IDF. Para cada termo da query presente no documento:

`score = IDF(t) · [ tf · (k₁ + 1) ] / [ tf + k₁ · (1 − b + b · |d| / avgdl) ]`

- **IDF (inverse document frequency):** raridade do termo no corpus. "VPN-4031" vale muito mais que "de".
- **TF com saturation (`k₁`, tipicamente 1,2 a 2,0):** a décima ocorrência de um termo acrescenta pouco. Com `k₁ = 0`, a frequência é ignorada.
- **Length normalization (`b`, tipicamente 0,75):** documentos longos não ganham só por serem longos. `b = 0` desliga a normalização.

A estrutura de dados por trás é o **inverted index** (termo → lista de documentos), o que torna a busca muito rápida.

Pontos fortes: exact match, códigos, nomes, números, termos raros; zero treino; interpretável. Ponto fraco: **vocabulary mismatch** (sinônimos e paráfrases). A qualidade depende da **analysis chain**: tokenization, lowercasing, stemming, stopwords, e ela é específica por idioma. Para português, um analyzer com stemming e remoção de acentos faz diferença.

**Learned sparse retrieval** (SPLADE e similares) é o meio-termo: um modelo neural produz vetores esparsos sobre o vocabulário, com expansão de termos aprendida, e ainda usa inverted index.

> **No mundo real.** A fórmula nasceu no sistema Okapi, nas conferências TREC dos anos 1990, e continua em uso: é a função de relevância padrão do **Lucene** desde a versão 6 e do **Elasticsearch** desde a 5.0 (2016), onde substituiu o TF-IDF. Ou seja, a maior parte das caixas de busca de sites e sistemas corporativos que você usa no dia a dia ordena os resultados com BM25.

### 3. Hybrid search e fusion

> **Em palavras simples.** **Hybrid search** roda duas buscas, uma por palavras e outra por significado, e junta as listas. O jeito mais usado de juntar, **Reciprocal Rank Fusion**, ignora as notas, que não são comparáveis, e olha só as posições: um documento que aparece bem colocado nas duas listas sobe; um que aparece em uma só fica no meio.

Scores de BM25 (sem limite superior) e de cosine (entre −1 e 1) estão em escalas incomparáveis. Duas formas de combinar:

**Reciprocal Rank Fusion (RRF)** ignora os scores e usa só as posições:

`RRF(d) = Σ_rankings 1 / (k + rank_r(d))`, com `k = 60` por convenção.

Um documento bem colocado em qualquer das listas sobe; um bem colocado nas duas sobe mais. Não tem parâmetros a ajustar, é robusto a outliers e difícil de bater. O `k` suaviza a diferença entre as primeiras posições.

**Weighted score fusion:** normalizar os scores (min-max ou z-score) e combinar com um peso `α`: `α · dense + (1 − α) · sparse`. Permite calibrar, mas exige dados rotulados para ajustar `α` e é sensível à normalização.

Hybrid costuma ser a opção mais **robusta**, não necessariamente a melhor em cada tipo de query. Robustez é o que se quer em produção.

> **No mundo real.** O **RRF** foi proposto por Cormack, Clarke e Büttcher (SIGIR 2009), em um paper de duas páginas que mostrou que essa conta simples, com a constante `k = 60`, superava métodos de fusão mais complexos nos dados do TREC. Quinze anos depois, é a forma de busca híbrida oferecida de fábrica por Elasticsearch, OpenSearch, Azure AI Search e Qdrant, quase sempre com o mesmo valor padrão de 60.

### 4. Cross-encoder (reranker)

> **Em palavras simples.** Um **cross-encoder** lê a pergunta e o documento juntos, palavra por palavra, e dá uma nota de relevância. É muito mais preciso que comparar dois embeddings calculados separadamente, e muito mais lento, porque nada pode ser preparado com antecedência. Por isso é usado como **reranker**: reordena só as dezenas de candidatos que a primeira etapa trouxe.

| | Bi-encoder | Cross-encoder |
|---|---|---|
| Entrada | Query e documento separados | Par `(query, documento)` concatenado |
| Interação | Só no dot product final | Full attention entre todos os tokens dos dois |
| Qualidade | Boa | Muito melhor |
| Custo | Documentos pré-computados | Um forward por par, em query time |
| Uso | Recuperar entre milhões | Reordenar 20 a 200 candidatos |

O cross-encoder é um `AutoModelForSequenceClassification` com um único logit de relevância, treinado em datasets como MS MARCO (mMARCO para versões multilingual). Como não há vetor pré-computável, o custo é linear no número de candidatos: é a principal alavanca de latência do funil.

**Late interaction (ColBERT)** é o meio-termo: guarda um vetor por token e calcula a relevância com MaxSim. Mais preciso que bi-encoder, mais barato que cross-encoder, com index muito maior.

LLMs também são usados como rerankers (pointwise ou listwise), com custo e latência maiores.

> **No mundo real.** O paper *Passage Re-ranking with BERT* (Nogueira e Cho, 2019) aplicou um cross-encoder para reordenar os 1.000 primeiros resultados do BM25 no dataset **MS MARCO**, com perguntas reais do Bing. O MRR@10 passou de cerca de 0,17 para cerca de 0,36, mais que o dobro, e o método assumiu o topo do leaderboard. Desde então, acrescentar um reranker é a melhoria com retorno mais previsível em um pipeline de busca.

### 5. Query understanding

> **Em palavras simples.** A pergunta do usuário nem sempre é uma boa consulta: pode ser curta, vaga ou usar termos que os documentos não usam. **Query understanding** é tratar a pergunta antes de buscar: reescrevê-la, expandi-la com sinônimos, dividi-la em subperguntas ou extrair dela os filtros (*notas fiscais de 2023* vira um filtro de ano).

- **Intent / domain classification:** escolher o index ou aplicar um metadata filter (Semana 12). Pode ser o classifier da Semana 10.
- **Query rewriting e expansion:** corrigir, expandir siglas, resolver correferência em conversas ("e para o Brasil?"), gerar variações (multi-query). **HyDE:** gerar uma resposta hipotética e buscar por ela.
- **Filter extraction (self-querying):** "contratos de 2024" → `year = 2024`.
- **Routing:** decidir se a pergunta precisa de vector search, de SQL, de uma API, ou de nada.

Todo componente aqui é um classifier ou um extractor, e falha como tal. Quando o router erra o domínio com um hard filter, a query perde todo o recall. Mitigações: confidence threshold, buscar nos top-2 domínios, fallback sem filtro, ou usar o domínio como **boost** em vez de filtro.

> **No mundo real.** A técnica **HyDE** (*Hypothetical Document Embeddings*, Gao et al., 2022) pede a um LLM que escreva uma resposta hipotética para a pergunta, mesmo que imprecisa, e usa o embedding dessa resposta para buscar. A lógica é que uma resposta se parece mais com os documentos do que uma pergunta. Nos experimentos do paper, o método melhorou a busca sem nenhum dado rotulado em vários datasets e idiomas.

### 6. Latency budget

> **Em palavras simples.** Cada etapa da busca consome tempo, e a soma precisa caber no que o usuário tolera. O **latency budget** é a divisão desse tempo total entre as etapas: tantos milissegundos para a busca, tantos para o reranker, tantos para o LLM. É ele que define quantos candidatos o reranker pode examinar e de que tamanho pode ser o modelo.

Um orçamento típico de 300 ms pode ser dividido em: query understanding 20 ms, retrieval 30 ms, reranking 150 ms, margem 100 ms. O reranker domina. Alavancas: número de candidatos, `max_length` dos pares, batching na GPU, um reranker menor ou distilled, e cache.

A curva "candidatos × qualidade" achata rápido: depois de certo ponto, mais candidatos custam latência e não trazem ganho. O notebook mede esse ponto.

> **No mundo real.** Dois relatos de engenharia muito citados deram origem a essa disciplina: na Amazon, experimentos internos indicaram que cada 100 ms de atraso custava cerca de 1% das vendas; no Google, um teste que aumentou o tempo de resposta em meio segundo reduziu o tráfego em cerca de 20%. Os números são dos anos 2000, mas o princípio se mantém: a velocidade é parte da qualidade percebida.

### 7. Depois do ranking

> **Em palavras simples.** Depois de ordenar, ainda há trabalho: remover resultados quase idênticos, garantir variedade para que os primeiros lugares não digam todos a mesma coisa, aplicar regras de negócio e cortar a lista no tamanho certo. Para um RAG, isso evita gastar o espaço do prompt com cinco trechos que repetem a mesma informação.

- **Diversity:** evitar dez resultados quase idênticos. **MMR (Maximal Marginal Relevance)** equilibra relevância e novidade.
- **Freshness** e **authority:** boosts por data e por fonte.
- **Deduplication** de chunks sobrepostos.
- Em recommender systems: exploration, fairness, regras de negócio.

> **No mundo real.** A técnica clássica é a **Maximal Marginal Relevance** (Carbonell e Goldstein, 1998), que escolhe cada próximo resultado equilibrando relevância para a pergunta e novidade em relação ao que já foi escolhido. Criada para resumos e busca de notícias, ela está hoje embutida em bibliotecas como LangChain e em vector databases como o Qdrant, justamente para diversificar os chunks enviados ao LLM.

### 8. Learning to rank, em uma frase

> **Em palavras simples.** **Learning to rank** é treinar um modelo para ordenar resultados usando exemplos de boas e más ordenações. Em vez de uma fórmula fixa, o modelo aprende a combinar dezenas de sinais: a nota do BM25, a similaridade semântica, a data, a popularidade, os cliques anteriores.

Quando há dados de interação (cliques, compras), treina-se um ranker com features da query, do item e do usuário (LambdaMART, gradient-boosted trees, redes neurais), otimizando diretamente NDCG. É o padrão em e-commerce e feeds. As métricas para avaliá-lo são o tema da próxima semana.

> **No mundo real.** Em 2010, o **Yahoo! Learning to Rank Challenge** distribuiu dados reais do seu buscador, com centenas de sinais por par de consulta e documento. A equipe vencedora, da Microsoft Research, usou o **LambdaMART**, baseado em gradient boosted trees. Variações desse método (disponíveis em XGBoost e LightGBM) continuam sendo o padrão para ranking com muitos sinais tabulares em buscadores e lojas online.

## Erros comuns

- Somar scores de BM25 e cosine sem normalizar.
- BM25 com tokenization ingênua em português (sem tratar acentos, sem stemming).
- Passar candidatos demais ao reranker e estourar a latência.
- Truncar o documento no par `(query, doc)` e cortar justamente o trecho relevante.
- Hard filter baseado em um router sem fallback.
- Avaliar o reranker em queries nas quais o retrieval já não trouxe o documento certo.

## Leitura e vídeos (fim de semana)

- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (seções introdutórias) ou a série *Practical BM25* do blog da Elastic.
- Cormack, Clarke & Buettcher, *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods* (paper curto).
- Sentence-Transformers documentation: *Cross-Encoders* e *Retrieve & Re-Rank*.
- Nogueira & Cho, *Passage Re-ranking with BERT* (abstract e seção 3).
- Khattab & Zaharia, *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT* (abstract e figura 1).
- Thakur et al., *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models* (a conclusão de que BM25 é um baseline forte fora do domínio).
- Opcional: Covington et al., *Deep Neural Networks for YouTube Recommendations*, para o funil em recommender systems.

## Perguntas de autoavaliação

1. Em que tipo de query o BM25 vence o dense retrieval?
2. Por que o RRF usa posições e não scores?
3. Por que não usar o cross-encoder direto sobre toda a base?
4. Seu budget é de 200 ms e o reranker leva 4 ms por par. Quantos candidatos cabem?
5. O que os parâmetros `k₁` e `b` do BM25 controlam?
6. O router de domínio erra 10% das vezes. Que mecanismos você colocaria para limitar o dano?

## Perguntas do entrevistador

1. Usuários estão buscando por códigos exatos de erro e SKUs no RAG e a busca puramente densa retorna documentos conceitualmente parecidos, mas errados. Como você arquiteta um funil híbrido (BM25 + dense retrieval + RRF + cross-encoder reranker) mantendo a latência dentro do SLA?

## Entregável da semana

Notebook resolvido e um diagrama (Mermaid ou imagem) do funil com a latência medida em cada stage, em `docs/`.

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/03-Phase-3-Advanced-RAG/notebooks/semana_14_exercicio.ipynb) · [`semana_14_exercicio.ipynb`](../notebooks/semana_14_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/03-Phase-3-Advanced-RAG/notebooks/semana_14_gabarito.ipynb) · [`semana_14_gabarito.ipynb`](../notebooks/semana_14_gabarito.ipynb) |

[← Semana 13](/03-Phase-3-Advanced-RAG/docs/Semana-13-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 15 →](/03-Phase-3-Advanced-RAG/docs/Semana-15-Conceitos.md)
