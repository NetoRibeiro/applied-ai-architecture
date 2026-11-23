# Semana 12 — Vector Databases: Qdrant, pgvector e Metadata Filtering

**Fase 3: Advanced RAG, Search e Reranking** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Passar de um index em memória para um vector database: collections, payloads (metadata), filtering, upsert e delete. A prática usa o Qdrant em local mode; a leitura cobre o `pgvector` em detalhe e os critérios para escolher entre as opções.

## Por que isso importa para um AI Architect

Em uma empresa, a busca nunca é só por similaridade: é "os documentos mais parecidos **que este usuário pode ver**, **vigentes**, **da unidade dele**". Metadata filtering, access control, multi-tenancy e o ciclo de vida dos documentos são o que separa uma demo de um sistema de produção, e são onde acontecem os incidentes de vazamento de dados em RAG.

## Conceitos

### 1. O que um vector database acrescenta a um index

> **Em palavras simples.** Um index vetorial é só a estrutura de busca. Um **vector database** acrescenta o que um sistema de verdade precisa: guardar os dados em disco, inserir e apagar sem reconstruir tudo, filtrar por atributos (como data ou cliente), controlar acesso, fazer backup e crescer em várias máquinas. É a diferença entre um motor e um carro.

| Capacidade | Index (FAISS) | Vector database |
|---|---|---|
| ANN search | Sim | Sim |
| Metadata e filtering | Não | Sim |
| Insert, update, delete | Limitado | Sim |
| Persistence, replication, backup | Não | Sim |
| Access control | Não | Sim (ou via banco hospedeiro) |
| Hybrid search (dense + sparse) | Não | Cada vez mais comum |

Vocabulário (Qdrant): uma **collection** contém **points**; cada point tem `id`, `vector` e `payload` (um JSON com metadata). Outros sistemas usam index/namespace, record/document e metadata.

> **No mundo real.** A própria documentação do **FAISS** o define como uma biblioteca, não um banco: ele não persiste dados sozinho, não filtra por metadados e não atende várias aplicações ao mesmo tempo. Produtos como **Qdrant**, **Weaviate**, **Milvus** e **Pinecone** surgiram para preencher essa lacuna, e bancos existentes ganharam busca vetorial (PostgreSQL com pgvector, Elasticsearch, MongoDB). Em 2023, o plugin de retrieval do ChatGPT já listava mais de uma dezena de vector databases compatíveis.

### 2. Post-filtering, pre-filtering e filtered search

> **Em palavras simples.** Quase toda busca real tem um filtro: só documentos deste cliente, só deste ano. **Post-filtering** busca primeiro e filtra depois, e pode devolver menos resultados do que o pedido, ou nenhum. **Pre-filtering** filtra primeiro e busca no que sobrou, o que fica lento se sobrar muito. **Filtered search** aplica o filtro durante a navegação do index, combinando as duas vantagens.

- **Post-filtering:** busca os `k` vizinhos mais próximos e depois descarta os que não passam no filtro. Se o filtro for seletivo, sobram poucos resultados, ou nenhum.
- **Pre-filtering:** filtra primeiro e faz exact search no que sobrou. Correto, mas lento quando o subconjunto é grande.
- **Filtered (in-graph) search:** o filtro é aplicado durante a navegação do graph HNSW. É o que o Qdrant faz (filterable HNSW) e o que o `pgvector` aproxima com iterative scans.

O problema de fundo: o graph HNSW foi construído sobre todos os pontos. Quando o filtro elimina a maioria, o graph "se desconecta" e a busca não chega aos pontos válidos. Bons sistemas escolhem a estratégia pela **selectivity** do filtro: muito seletivo → exact search no subconjunto; pouco seletivo → HNSW com filtro.

**Regra de segurança:** permissões e tenant isolation têm de ser aplicados **antes ou durante** a busca. Post-filtering para access control é um vazamento esperando para acontecer, porque qualquer bug na etapa posterior expõe dados de outro tenant ao LLM.

> **No mundo real.** No **pgvector**, até a versão 0.7, uma consulta com filtro seletivo sobre um index HNSW podia devolver bem menos linhas que o `LIMIT` pedido: o index trazia os vizinhos mais próximos e o filtro descartava quase todos. A versão 0.8.0 (2024) introduziu os **iterative index scans**, que continuam percorrendo o index até completar a quantidade solicitada. É um exemplo concreto de um problema de post-filtering resolvido no próprio banco.

### 3. Payload indexes

> **Em palavras simples.** Um **payload index** é um índice sobre os atributos que acompanham cada vetor, como `tenant_id` ou `data`. Funciona como o índice remissivo de um livro: sem ele, para aplicar um filtro o banco precisa olhar registro por registro. Com ele, o filtro vira uma consulta direta. É o que mantém a busca filtrada rápida quando o acervo cresce.

Campos usados em filtros precisam de index próprio (`create_payload_index` no Qdrant; um B-tree ou GIN index no Postgres). Sem isso, cada filtro vira um scan. É o mesmo raciocínio de um `WHERE` em SQL.

> **No mundo real.** No **Qdrant**, os campos de payload não são indexados automaticamente: é preciso criar o index para cada campo usado em filtros (`create_payload_index`). A documentação recomenda fazê-lo antes da carga dos dados, porque o banco usa esses índices para construir um grafo HNSW que continua navegável sob filtro. Esquecer esse passo é uma causa frequente de buscas filtradas lentas em coleções grandes.

### 4. pgvector

> **Em palavras simples.** **pgvector** é uma extensão que ensina o PostgreSQL a guardar vetores e a buscar por proximidade. A vantagem é não precisar de outro sistema: os vetores ficam na mesma tabela que os demais dados, com as mesmas transações, permissões e backups, e os filtros são SQL comum. Para muitas empresas que já usam PostgreSQL, é o caminho mais curto.

Extension do PostgreSQL que adiciona o tipo `vector` e ANN indexes.

```sql
CREATE EXTENSION vector;
CREATE TABLE documents (
  id bigserial PRIMARY KEY,
  content text, area text, year int, tenant_id int,
  embedding vector(384)
);
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
CREATE INDEX ON documents (tenant_id, area);

SET hnsw.ef_search = 100;
SELECT id, content, 1 - (embedding <=> $1) AS similarity
FROM documents
WHERE tenant_id = 42 AND area = 'ti' AND year >= 2024
ORDER BY embedding <=> $1
LIMIT 5;
```

Operators: `<=>` cosine distance, `<#>` negative inner product, `<->` L2 distance. O operator class do index (`vector_cosine_ops`, `vector_ip_ops`, `vector_l2_ops`) precisa corresponder ao operator da query; se não corresponder, o planner ignora o index e faz sequential scan sem avisar. Use `EXPLAIN ANALYZE` para conferir.

Outros pontos:

- `hnsw` (melhor recall/latência, mais memória e build mais lento) vs. `ivfflat` (build rápido, exige dados para treinar as lists; `ivfflat.probes` controla o recall).
- `halfvec` (float16) e binary quantization para reduzir memória.
- Iterative index scans (`hnsw.iterative_scan`) para filtros seletivos.
- **Row-Level Security (RLS)** do Postgres aplica multi-tenancy no próprio banco: a policy filtra as linhas antes de qualquer query. É uma das maiores vantagens de manter os vetores no Postgres.
- JOINs e transações ACID com o resto dos dados da aplicação.

> **No mundo real.** O pgvector ganhou index HNSW na versão 0.5.0 (agosto de 2023) e é oferecido pelos principais serviços gerenciados de PostgreSQL, como Amazon RDS, Google Cloud SQL, Azure e Supabase. Um limite real que orienta a escolha do embedding model: o tipo `vector` só pode ser indexado até **2.000 dimensões**, de modo que um embedding de 3.072 dimensões precisa ser reduzido ou guardado com o tipo `halfvec`.

### 5. Como escolher

> **Em palavras simples.** A escolha de onde guardar os vetores depende mais da operação do que do algoritmo. Perguntas que decidem: quantos vetores, quantas buscas por segundo, quão seletivos são os filtros, quem vai operar o sistema e o que a empresa já usa. Um banco que a equipe já domina costuma valer mais que um especializado que ninguém sabe manter.

| Critério | pgvector | Dedicated (Qdrant, Milvus, Weaviate) | Managed (Pinecone e similares) |
|---|---|---|---|
| Já usa Postgres | Grande vantagem: JOINs, transações, RLS, um só backup | Mais um sistema para operar | Mais um fornecedor |
| Escala | Confortável até dezenas de milhões | Centenas de milhões ou mais; sharding nativo | Elástica |
| Filtros complexos | SQL completo | Bons, API própria | Bons |
| Hybrid search | Via full-text search do Postgres + fusão manual | Nativo em vários | Nativo |
| Operação | Seu time de banco já sabe | Exige conhecimento novo | Terceirizada |
| Dados sensíveis | Ficam no seu ambiente | Ficam no seu ambiente (self-hosted) | Saem do seu ambiente |

Regra prática: comece no Postgres que você já tem; migre quando uma **medição** (não uma suposição) mostrar que ele não atende. Esta decisão será o tema do seu primeiro ADR, na Semana 31.

> **No mundo real.** A **Instacart** relatou em seu blog de engenharia a migração da busca semântica, antes apoiada em um serviço separado baseado em FAISS, para o **pgvector** dentro do PostgreSQL que já servia a busca textual. A motivação declarada foi operacional: um único sistema, com os atributos dos produtos e os vetores no mesmo lugar, simplificou os filtros e a atualização dos dados. É a decisão típica de quem prioriza simplicidade de operação.

### 6. Multi-tenancy

> **Em palavras simples.** **Multi-tenancy** é atender vários clientes no mesmo sistema garantindo que um nunca veja os dados do outro. Em busca vetorial, há dois caminhos: uma coleção separada para cada cliente, ou uma coleção única em que todo registro leva a etiqueta do cliente e toda consulta filtra por ela. O primeiro isola mais; o segundo escala melhor para milhares de clientes pequenos.

Três padrões: uma collection por tenant (isolamento forte; caro com milhares de tenants), partition/namespace por tenant, ou um campo `tenant_id` com filtro obrigatório e index. No terceiro, a obrigatoriedade do filtro precisa ser garantida pela camada de acesso (ou por RLS), nunca pela boa vontade de quem escreve a query.

> **No mundo real.** A documentação do **Qdrant** recomenda, na maioria dos casos, uma única coleção com particionamento por payload, marcando o campo do cliente como `is_tenant`, em vez de milhares de coleções, que consomem recursos fixos cada uma. O **Pinecone** oferece o conceito de **namespaces** com a mesma finalidade. Em ambos, a regra de segurança é a mesma: o filtro de tenant é aplicado pelo servidor da aplicação, nunca deixado a cargo do LLM ou do usuário.

### 7. Data lifecycle

> **Em palavras simples.** Os dados de um vector database envelhecem: documentos são criados, alterados e apagados, e os vetores precisam acompanhar. Há um ponto mais sério: vetores de modelos diferentes não são comparáveis. Trocar de embedding model obriga a recalcular o acervo inteiro. Por isso se registra, junto a cada vetor, qual modelo o gerou.

Documentos mudam. O ingestion pipeline precisa de:

- **Stable IDs** derivados da fonte (ex.: hash de `doc_id + chunk_index`), para que o upsert seja **idempotent**: reprocessar não duplica.
- **Delete** propagado quando o documento some da fonte ou perde vigência.
- Campos de **version** e **valid_from / valid_to**.
- Registro de **embedding model version** em cada point. Uma migração de modelo se faz em uma collection nova, com troca atômica de alias no final (blue/green), nunca misturando vetores.

Um RAG que responde com a política do ano passado é pior do que não ter RAG.

> **No mundo real.** Em janeiro de 2024, a OpenAI desligou seus embedding models de primeira geração (como os da família `text-search` e `text-similarity`), conforme anunciado em sua página de deprecations. Quem tinha índices construídos com eles precisou gerar todos os embeddings de novo com um modelo mais recente, pois os vetores antigos deixaram de poder ser comparados com consultas novas. Ter um processo de reindexação pronto é parte do projeto, não um extra.

### 8. Consistência e operação

> **Em palavras simples.** Operar um vector database envolve o que se cuida em qualquer banco: os dados gravados já aparecem na busca seguinte? O que acontece se uma máquina cai? Quanto tempo leva para restaurar um backup? E há particularidades: apagar um vetor de um grafo HNSW normalmente só o marca como removido, e o espaço só é recuperado em uma reorganização posterior.

Muitos vector databases indexam de forma assíncrona: um point recém-inserido pode demorar a aparecer nas buscas. Verifique o modelo de consistência antes de prometer "busca em tempo real". Monitore: tamanho do index em RAM, latência p95, recall em um conjunto de queries de controle, e lag de indexação.

> **No mundo real.** No **Lucene**, base do Elasticsearch e do OpenSearch, cada segmento do index tem o seu próprio grafo HNSW, e documentos apagados ficam apenas marcados até que os segmentos sejam fundidos (*merge*), quando o grafo é reconstruído. Por isso índices com muitas atualizações podem ficar mais lentos e ocupar mais memória do que o esperado, e a própria documentação do Elasticsearch discute o custo dos merges em índices vetoriais.

## Erros comuns

- Post-filtering para controle de acesso.
- Filtrar por campo sem payload index.
- Operator class do index diferente do operator da query no pgvector.
- IDs aleatórios a cada ingestão, gerando duplicatas.
- Não propagar deletes da fonte.
- Reindexar com um embedding model novo na mesma collection.

## Leitura e vídeos (fim de semana)

- README do repositório `pgvector` (inteiro: types, operators, HNSW, IVFFlat, filtering, iterative scans).
- Qdrant documentation: *Concepts* → *Collections*, *Points*, *Payload*, *Filtering*, *Indexing*; e o artigo *Filtrable HNSW*.
- PostgreSQL documentation: *Row Security Policies*.
- Opcional: *ANN-Benchmarks* (site) para ver como os tradeoffs recall × queries per second são reportados.
- Opcional: documentação de multi-tenancy do Qdrant ou do Weaviate.

## Perguntas de autoavaliação

1. Por que post-filtering é inaceitável para access control?
2. O que faz o operator `<=>` no pgvector e o que acontece se o operator class do index for outro?
3. Sua empresa já roda PostgreSQL e tem 2 milhões de documentos. Que opção você propõe e por quê?
4. Como você garante que um documento excluído na fonte desapareça das respostas do RAG?
5. O que torna um upsert idempotent?
6. Como migrar de um embedding model para outro sem downtime?

## Entregável da semana

Notebook resolvido e uma decision table preenchida para um cenário real seu (ou hipotético) em `docs/`.

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/ai-architect-foundations/blob/main/03-Phase-3-Advanced-RAG/notebooks/semana_12_exercicio.ipynb) · [`semana_12_exercicio.ipynb`](../notebooks/semana_12_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/ai-architect-foundations/blob/main/03-Phase-3-Advanced-RAG/notebooks/semana_12_gabarito.ipynb) · [`semana_12_gabarito.ipynb`](../notebooks/semana_12_gabarito.ipynb) |

[← Semana 11](/03-Phase-3-Advanced-RAG/docs/Semana-11-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 13 →](/03-Phase-3-Advanced-RAG/docs/Semana-13-Conceitos.md)
