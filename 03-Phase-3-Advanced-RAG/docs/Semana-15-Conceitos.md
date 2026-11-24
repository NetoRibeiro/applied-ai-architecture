# Semana 15 — Ranking Metrics: Recall@K, MRR e NDCG

**Fase 3: Advanced RAG, Search e Reranking** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Dominar as ranking metrics usadas em search e recommendation: Precision@K, Recall@K, MRR, MAP e NDCG. Você vai implementá-las do zero, compará-las em sistemas diferentes e aprender a dizer se uma diferença é real ou ruído.

## Por que isso importa para um AI Architect

Precision e recall (Semana 1) tratam o resultado como um conjunto. Em busca, a **ordem** importa: o usuário, ou o LLM, olha o topo da lista. Ranking metrics são a linguagem de times de Search & Discovery e a base de qualquer decisão do tipo "vale a pena colocar um reranker?" ou "o embedding model novo é melhor?".

## Conceitos

### 1. Do conjunto à lista ordenada

> **Em palavras simples.** Avaliar uma busca não é só perguntar se os documentos certos foram encontrados, mas **em que posição** apareceram. As pessoas olham os primeiros resultados e raramente descem a página; um LLM recebe só os primeiros trechos. Um documento certo na posição 50 vale quase o mesmo que não ter sido encontrado. Por isso as métricas de ranking dão peso à posição.

| Métrica | Pergunta que responde | Graded relevance? | Rank-aware? |
|---|---|---|---|
| Precision@K | Dos K mostrados, quantos são relevantes? | Não | Não (dentro do top-K) |
| Recall@K | Dos relevantes que existem, quantos estão no top-K? | Não | Não (dentro do top-K) |
| Hit Rate@K | Há pelo menos um relevante no top-K? | Não | Não |
| MRR | Em que posição aparece o **primeiro** relevante? | Não | Sim |
| MAP | Precision média ao longo de todos os relevantes | Não | Sim |
| NDCG@K | Os **mais** relevantes estão no topo? | **Sim** | Sim |

O ground truth são os **relevance judgments** (qrels): para cada query, quais documentos são relevantes e, opcionalmente, em que grau.

> **No mundo real.** Um estudo de rastreamento ocular de Joachims et al. (Cornell, 2005) com resultados do Google mostrou que os usuários olham e clicam de forma desproporcional nos primeiros resultados. Quando os pesquisadores inverteram a ordem em segredo, pondo os piores resultados no topo, as pessoas continuaram clicando nos primeiros. A posição influencia o clique quase tanto quanto a relevância, fenômeno conhecido como **position bias**.

### 2. Precision@K e Recall@K

> **Em palavras simples.** **Precision@K** responde: dos `K` primeiros resultados, quantos prestam? **Recall@K** responde: de tudo o que prestava, quanto apareceu entre os `K` primeiros? A primeira mede a limpeza da lista; a segunda, a cobertura. Em RAG, o recall da busca é um teto: o que não foi recuperado não pode ser usado na resposta.

`Precision@K = (relevantes no top-K) / K`

`Recall@K = (relevantes no top-K) / (total de relevantes)`

Recall@K é a métrica do **retrieval stage**: se o documento certo não está entre os candidatos, nenhum reranker ou LLM o recupera depois. Em RAG, `K` é o número de chunks entregues ao próximo stage.

Cuidado com o teto: se uma query tem 20 relevantes, o Recall@5 máximo é 0,25; se tem 1 relevante, a Precision@10 máxima é 0,1. Compare sempre com o máximo atingível.

> **No mundo real.** O paper do **DPR** reporta justamente essa métrica: no Natural Questions, o trecho com a resposta estava entre os 20 primeiros em 78,4% das perguntas e entre os 100 primeiros em 85,4%. A diferença mostra o trade-off de engenharia: buscar 100 trechos em vez de 20 recupera mais respostas, mas multiplica por cinco o que as etapas seguintes precisam processar.

### 3. MRR (Mean Reciprocal Rank)

> **Em palavras simples.** **MRR** (*Mean Reciprocal Rank*) mede em que posição aparece o **primeiro** resultado correto. Se está em 1º lugar, vale 1; em 2º, vale 1/2; em 10º, vale 1/10. Faz-se a média sobre muitas perguntas. É a métrica certa quando basta um bom resultado, como em perguntas com uma única resposta.

Para cada query, `RR = 1 / rank do primeiro relevante` (0 se não aparece). O MRR é a média sobre as queries.

Exemplo: três queries com o primeiro relevante nas posições 1, 4 e "não encontrado" → `(1 + 0,25 + 0) / 3 = 0,417`.

Indicado quando existe **uma** resposta certa: factoid QA, navigational queries, autocomplete. Ignora tudo o que vem depois do primeiro acerto. A queda é brusca no topo: da posição 1 para a 2, perde-se metade.

> **No mundo real.** O MRR foi introduzido na trilha de *question answering* do **TREC-8** (1999) e é a métrica oficial do leaderboard de passage ranking do **MS MARCO** (MRR@10), construído com perguntas reais do Bing, cada uma com, tipicamente, um único trecho marcado como relevante. Como só há uma resposta marcada por pergunta, medir a posição do primeiro acerto é a escolha natural.

### 4. MAP (Mean Average Precision)

> **Em palavras simples.** **MAP** (*Mean Average Precision*) avalia a lista inteira quando há vários documentos corretos: premia sistemas que colocam **todos** eles o mais alto possível. Para cada pergunta, calcula a precision a cada acerto encontrado e tira a média; depois, a média entre as perguntas. É uma nota única para a qualidade geral da ordenação.

Para cada query, a **Average Precision** é a média das Precision@k calculadas em cada posição `k` onde há um documento relevante. O MAP é a média entre queries. Considera todos os relevantes e a ordem, mas só com relevância binária. Era o padrão em IR acadêmico antes do NDCG.

> **No mundo real.** O MAP foi por décadas a métrica principal das avaliações do **TREC**, a série de conferências do NIST que desde 1992 padroniza a avaliação de sistemas de busca. A ferramenta `trec_eval`, usada até hoje em pesquisa, o calcula por padrão. É adequado a cenários como busca jurídica ou de patentes, em que o usuário quer encontrar todos os documentos pertinentes, não apenas um.

### 5. NDCG (Normalized Discounted Cumulative Gain)

> **Em palavras simples.** **NDCG** lida com relevância em graus: um resultado pode ser perfeito, bom, razoável ou inútil. A métrica soma os pontos de cada resultado com um desconto que cresce com a posição, e divide pelo máximo que a ordenação ideal alcançaria. O resultado vai de 0 a 1 e permite comparar perguntas diferentes.

Três ideias em sequência:

1. **Gain:** cada resultado tem um grau `rel` (0 = irrelevante, 1 = relacionado, 2 = resposta exata).
2. **Discount:** o ganho vale menos quanto mais abaixo na lista. `DCG@K = Σᵢ relᵢ / log₂(i + 1)`, com `i` começando em 1.
3. **Normalization:** divide-se pelo DCG da ordenação ideal (`IDCG@K`), para que o resultado fique entre 0 e 1 e seja comparável entre queries com números diferentes de relevantes.

`NDCG@K = DCG@K / IDCG@K`

Exemplo: ranking com graus `[0, 2, 0, 1, 0]`. `DCG = 0/1 + 2/1,585 + 0/2 + 1/2,322 + 0 = 1,693`. Ideal `[2, 1, 0, 0, 0]`: `IDCG = 2/1 + 1/1,585 = 2,631`. `NDCG@5 = 0,643`.

Duas convenções de gain, e é obrigatório dizer qual se usa:

- **Linear:** `rel` (é a do `sklearn.metrics.ndcg_score`).
- **Exponential:** `2^rel − 1`, que premia muito mais os graus altos (comum em web search e learning-to-rank).

> **No mundo real.** Proposto por Järvelin e Kekäläinen (2002), o NDCG é a métrica principal do benchmark **BEIR** (nDCG@10) e do leaderboard de retrieval do MTEB. Também foi a métrica dos grandes desafios de ranking com dados de buscadores comerciais, como o do Yahoo! em 2010, cujos julgamentos tinham cinco graus de relevância, de *ruim* a *perfeito*, atribuídos por avaliadores profissionais.

### 6. Qual métrica para qual sistema

> **Em palavras simples.** Cada métrica corresponde a uma necessidade. Se basta uma resposta certa, MRR. Se é preciso achar tudo, recall e MAP. Se a relevância tem graus e a ordem do topo importa, NDCG. Em RAG, o que mais pesa é o Recall@K da busca, pois o LLM consegue ignorar trechos ruins, mas não consegue usar um trecho que não recebeu.

| Sistema | Métrica principal | Motivo |
|---|---|---|
| Retriever de RAG | Recall@K | O LLM só usa o que chega |
| Reranker | NDCG@K, MRR | Ordem no topo |
| QA com uma resposta | MRR, Hit Rate@K | Só o primeiro acerto conta |
| E-commerce, recommendation | NDCG@K | Graded relevance, vários itens bons |
| Recommender (complementares) | Coverage, diversity, novelty | Relevância não é tudo |

O `K` não é arbitrário: deve refletir o que o consumidor vê. Dez resultados em uma página web; 3 a 5 chunks no contexto de um LLM.

> **No mundo real.** O **Netflix Prize** (2006–2009) premiava a redução do erro na previsão de notas (RMSE). Anos depois, a Netflix escreveu em seu blog técnico que deixou de tratar a recomendação como previsão de nota e passou a tratá-la como problema de ranking: o que importa é quais títulos aparecem nas primeiras posições de cada fileira, e isso se mede com métricas de ranking, não com RMSE. A métrica escolhida define o produto que se constrói.

### 7. De onde vêm os relevance labels

> **Em palavras simples.** Para calcular qualquer métrica é preciso saber quais documentos são relevantes para cada pergunta: os **relevance labels**. Eles podem vir de avaliadores humanos (caro e confiável), de cliques dos usuários (abundante e enviesado) ou de um LLM atuando como juiz (barato, exige validação). Como ninguém julga todos os documentos, os labels são sempre incompletos.

- **Human annotation:** cara e lenta; é o gold standard. Comece com 50 a 200 queries **reais**. Meça inter-annotator agreement: se dois anotadores discordam muito, a definição de relevância está vaga.
- **Implicit feedback** (cliques, dwell time, compras): abundante e enviesado. **Position bias:** o primeiro resultado recebe cliques por ser o primeiro. **Presentation bias** e **selection bias:** só há feedback sobre o que foi mostrado. Corrigir exige modelos de clique ou randomização parcial (interleaving).
- **LLM-as-a-judge:** barato e escalável; exige validação contra uma amostra anotada por humanos e uma rubrica clara.
- **Pooling:** como é impossível julgar o corpus inteiro, julgam-se os top resultados de vários sistemas. Documentos nunca recuperados por nenhum sistema ficam sem julgamento e são tratados como irrelevantes, o que prejudica sistemas novos que encontram documentos diferentes.

> **No mundo real.** O **TREC** resolve a incompletude com **pooling**: só os primeiros resultados de cada sistema participante são julgados por humanos. Já o **MS MARCO** tem em média cerca de um trecho marcado como relevante por pergunta, embora existam outros corretos. Um estudo de 2021 (Arabzadeh et al.) mostrou que, com frequência, avaliadores preferiam o primeiro resultado dos melhores sistemas ao trecho marcado como correto: o sistema era punido por acertar.

### 8. Statistical significance

> **Em palavras simples.** Se o sistema A teve 0,71 e o B teve 0,73, B é melhor? Depende de quantas perguntas foram testadas e de quanto os resultados variam entre elas. Um teste de **statistical significance** estima a chance de a diferença ser acaso. Com poucas perguntas, diferenças pequenas não significam nada, e trocar de sistema por causa delas é jogar moeda.

Com 50 queries, uma diferença de 0,02 em NDCG pode ser sorte. As métricas variam muito de query para query.

O **paired bootstrap** reamostra as queries com reposição milhares de vezes e calcula a diferença entre os dois sistemas em cada reamostra; se o intervalo de 95% não inclui zero, a diferença é confiável. "Paired" porque os dois sistemas são avaliados nas mesmas queries, o que remove a variância entre queries. Alternativas: paired t-test e randomization test.

Reporte sempre o intervalo, não só a média. E olhe a distribuição: uma média igual pode esconder um sistema que melhorou 30 queries e destruiu 5.

> **No mundo real.** As avaliações do **TREC** usam tradicionalmente conjuntos de 50 consultas, número apoiado em estudos de Voorhees e Buckley sobre a estabilidade das comparações. Smucker, Allan e Carterette (2007) compararam testes de significância em dados do TREC e recomendaram o teste de randomização, o bootstrap e o teste t pareado, desaconselhando testes mais fracos. A prática consolidada é sempre comparar os sistemas **pergunta a pergunta**.

### 9. Offline vs. online

> **Em palavras simples.** Avaliação **offline** usa um conjunto fixo de perguntas com gabarito: é rápida, repetível e barata, mas mede só o que o gabarito cobre. Avaliação **online** observa usuários reais: mede o que importa, mas é lenta e arriscada. As duas se completam: a offline filtra as ideias ruins; a online confirma as boas.

Uma melhora de NDCG offline é uma **hipótese**. A confirmação vem de um experimento com usuários (Fase 6): click-through rate, time to success, query reformulation rate, abandonment, conversão. Offline e online divergem quando os labels estão desatualizados, quando o position bias contaminou os dados, ou quando a métrica offline não capta o que o usuário valoriza. **Interleaving** (misturar os resultados de dois rankers na mesma página) é uma forma sensível de comparar online com pouco tráfego.

> **No mundo real.** A Netflix descreveu em seu blog técnico (2017) o uso de **interleaving**: em vez de mostrar o ranking A a um grupo e o B a outro, misturam-se os dois na mesma tela e observa-se de qual vêm os títulos assistidos. Segundo a empresa, o método identifica o melhor algoritmo com mais de **cem vezes menos usuários** do que um A/B test tradicional, o que permite testar muito mais ideias antes do A/B definitivo.

### 10. Error analysis

> **Em palavras simples.** Um número médio diz que o sistema erra, não por quê. **Error analysis** é abrir os casos em que a busca falhou, ler as perguntas e os resultados, e agrupar os erros por causa: termos técnicos não reconhecidos, perguntas ambíguas, documento ausente da base, gabarito errado. É o que transforma uma métrica em uma lista de consertos.

Métricas agregadas dizem **quanto**; a análise por query diz **por quê**. Segmente por tipo de query (head vs. tail, com código vs. em linguagem natural, curta vs. longa) e leia as piores. É dessa leitura que saem as melhorias.

> **No mundo real.** No benchmark **BEIR**, a análise dos erros revelou um viés dos próprios dados: como os documentos relevantes de vários datasets haviam sido encontrados originalmente com buscas por palavras, sistemas lexicais levavam vantagem, e modelos densos eram penalizados por achar documentos corretos que ninguém havia rotulado. Só a leitura dos casos, não a métrica agregada, mostrou que parte da diferença era defeito do gabarito.

## Erros comuns

- Reportar NDCG sem dizer o `K` nem a convenção de gain.
- Comparar sistemas com conjuntos de queries diferentes.
- Tratar documentos não julgados como irrelevantes sem perceber.
- Concluir a partir de uma diferença de média sem intervalo de confiança.
- Usar cliques crus como relevance labels.
- Otimizar MRR em um produto em que o usuário quer vários resultados.

## Leitura e vídeos (fim de semana)

- Manning, Raghavan & Schütze, *Introduction to Information Retrieval*, Chapter 8 (*Evaluation in information retrieval*). Disponível online.
- Järvelin & Kekäläinen, *Cumulated Gain-Based Evaluation of IR Techniques* (o paper do NDCG; Introduction e Section 2).
- Sakai, *Statistical Reform in Information Retrieval?* ou Smucker et al., *A Comparison of Statistical Significance Tests for Information Retrieval Evaluation*.
- Joachims et al., *Accurately Interpreting Clickthrough Data as Implicit Feedback* (position bias).
- BEIR e MTEB: quais métricas reportam (NDCG@10, Recall@100) e por quê.
- Ferramentas: `pytrec_eval` e `ranx`, bibliotecas de referência para estas métricas.

## Perguntas de autoavaliação

1. Dois sistemas têm o mesmo Recall@10. Como um pode ser muito melhor que o outro?
2. Calcule o MRR de três queries cujos primeiros relevantes estão nas posições 1, 4 e "não encontrado".
3. Por que o NDCG precisa da normalização pelo ideal?
4. Por que cliques são labels enviesados?
5. Qual a diferença entre linear e exponential gain e quando a segunda faz sentido?
6. Por que o bootstrap deve ser pareado?

## Entregável da semana

Notebook resolvido e, em `docs/`, uma tabela comparando três sistemas com intervalo de confiança da diferença entre os dois melhores.

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/03-Phase-3-Advanced-RAG/notebooks/semana_15_exercicio.ipynb) · [`semana_15_exercicio.ipynb`](../notebooks/semana_15_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/03-Phase-3-Advanced-RAG/notebooks/semana_15_gabarito.ipynb) · [`semana_15_gabarito.ipynb`](../notebooks/semana_15_gabarito.ipynb) |

[← Semana 14](/03-Phase-3-Advanced-RAG/docs/Semana-14-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 16 →](/03-Phase-3-Advanced-RAG/docs/Semana-16-Conceitos.md)
