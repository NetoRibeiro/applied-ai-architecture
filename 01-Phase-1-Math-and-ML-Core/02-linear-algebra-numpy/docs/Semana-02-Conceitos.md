# Semana 2 — A Geometria dos Dados: Linear Algebra com NumPy

**Fase 1: Math Foundations e ML Core** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Entender geometricamente vectors, dot product e matrix multiplication, e reconhecer neles a mecânica exata de embeddings e vector search. Ao final você terá escrito um mini vector database em NumPy e saberá ler e depurar shapes, a habilidade mais usada em Deep Learning.

## Por que isso importa para um AI Architect

Todo sistema RAG se resume a uma pergunta: "quais vetores estão mais próximos deste vetor?". As escolhas de arquitetura (distance metric, normalização, dimensão do embedding, tipo de index) são decisões de álgebra linear. E toda camada de uma rede neural, inclusive a attention de um Transformer, é uma sequência de matrix multiplications. Sem a intuição geométrica, essas decisões viram tentativa e erro.

## Conceitos

### 1. Vector: ponto e seta ao mesmo tempo

> **Em palavras simples.** Um **vector** é uma lista ordenada de números que descreve alguma coisa. Um endereço no mapa é um vetor de dois números (latitude e longitude). Um **embedding** é o mesmo princípio aplicado a significado: um modelo converte uma palavra ou frase em uma lista de centenas de números, e essa lista funciona como o "endereço" do texto em um mapa de ideias. Textos que querem dizer coisas parecidas recebem endereços vizinhos.

Um vector de `d` números é um ponto em um espaço de `d` dimensões e também uma seta da origem até ele. Um text embedding é isso: uma frase virou um ponto em 384, 768 ou 1.536 dimensões, e frases de sentido parecido caem em pontos próximos. "Próximo" é o que esta semana define com precisão.

> **No mundo real.** Em 2013, o Google publicou os vetores **word2vec** treinados em cerca de 100 bilhões de palavras do Google News: 3 milhões de palavras e expressões, cada uma com 300 números. Ficou famoso o teste de aritmética: `vector("king") − vector("man") + vector("woman")` resulta em um ponto cujo vizinho mais próximo é `vector("queen")`. A relação homem/mulher virou uma direção no espaço, sem que ninguém a tivesse programado. Esses vetores ainda podem ser baixados e explorados com a biblioteca `gensim`.

### 2. Norm: o comprimento

> **Em palavras simples.** A **norm** é o comprimento de um vetor, a distância do ponto até a origem, calculada pelo teorema de Pitágoras estendido a muitas dimensões. **Normalizar** é esticar ou encolher o vetor até ele ter comprimento 1, sem mudar para onde aponta. Serve para comparar só a direção: dois textos sobre o mesmo assunto, um curto e um longo, devem ser considerados parecidos apesar do tamanho diferente.

A **L2 norm** (norma euclidiana): `‖a‖ = sqrt(a₁² + a₂² + … + a_d²)`. Dividir um vetor pela própria norma gera um **unit vector** (comprimento 1), que preserva a direção e descarta a magnitude. Essa operação é a **L2 normalization**, e é o que `normalize_embeddings=True` faz no `sentence-transformers`.

Existe também a L1 norm (soma dos valores absolutos), que aparece em regularização (Lasso) e em sparse retrieval.

> **No mundo real.** A documentação da API de embeddings da **OpenAI** informa que os vetores já são entregues normalizados, com comprimento 1. A consequência prática, descrita na própria documentação: cosine similarity pode ser calculada só com o dot product, que é mais rápido, e o ranking por cosine e por Euclidean distance é idêntico. A maioria dos modelos do `sentence-transformers` oferece o mesmo com `normalize_embeddings=True`.

### 3. Dot product: a operação central

> **Em palavras simples.** O **dot product** é uma conta simples: multiplicam-se os números das duas listas, posição por posição, e soma-se tudo. O resultado é um único número que mede o quanto os dois vetores "concordam". É alto quando os dois têm valores grandes nas mesmas posições, perto de zero quando não têm relação e negativo quando apontam para lados opostos. Quase tudo o que uma rede neural faz por dentro é uma sequência de dot products.

`a · b = Σ aᵢ bᵢ = ‖a‖ ‖b‖ cos(θ)`

As duas formas são a mesma coisa. A primeira é como se calcula; a segunda é o que significa: o quanto um vetor aponta na direção do outro, ponderado pelos comprimentos.

- `a · b > 0`: apontam para o mesmo lado.
- `a · b = 0`: são **orthogonal** (independentes). Vetores one-hot diferentes sempre dão zero.
- `a · b < 0`: apontam para lados opostos.

Outra leitura útil: `a · b` é o comprimento da **projection** de `a` sobre `b`, vezes o comprimento de `b`. "Quanto de `b` existe em `a`."

O dot product está em toda parte: um neurônio calcula `w · x + b`; a attention calcula `q · k` para cada par de tokens; um retriever calcula `query · doc` para cada documento.

> **No mundo real.** No **Netflix Prize** (2006–2009; cerca de 100 milhões de avaliações de 480 mil usuários para 17.770 filmes), a técnica central das equipes vencedoras foi a **matrix factorization**: cada usuário e cada filme recebe um vetor de algumas dezenas de números, e a nota prevista é o **dot product** entre o vetor do usuário e o do filme. Se uma posição do vetor representa algo como "gosto por ação", o usuário com valor alto ali e o filme com valor alto ali produzem uma nota alta. O método está descrito em *Matrix Factorization Techniques for Recommender Systems* (Koren, Bell e Volinsky, 2009).

### 4. Cosine similarity

> **Em palavras simples.** **Cosine similarity** mede o ângulo entre dois vetores, ignorando o comprimento. O resultado vai de 1 (mesma direção: mesmo assunto) a 0 (sem relação) e a −1 (direções opostas). É como comparar a direção para onde duas pessoas estão andando, sem ligar para a velocidade de cada uma. É a medida padrão de semelhança entre textos.

`cos(θ) = (a · b) / (‖a‖ ‖b‖)`

É o dot product com o efeito do comprimento removido. Varia de −1 a 1. **Se os vetores já estão normalizados, cosine similarity e dot product são idênticos**, e é por isso que quase todo vector database normaliza na ingestão e usa inner product, que é mais barato (uma multiplicação a menos, e sem divisão).

Exemplo numérico: `a = [3, 4]`, `b = [4, 3]`. `a · b = 12 + 12 = 24`. `‖a‖ = ‖b‖ = 5`. `cos = 24 / 25 = 0,96`.

> **No mundo real.** O sistema de recomendação "clientes que compraram este item também compraram" da **Amazon**, descrito por Linden, Smith e York em 2003 (*Amazon.com Recommendations: Item-to-Item Collaborative Filtering*), representa cada produto como um vetor com uma posição por cliente (comprou ou não comprou) e mede a semelhança entre dois produtos pela **cosine similarity** desses vetores. Como a tabela de produtos parecidos é calculada offline, a recomendação na página é uma simples consulta, o que permitiu escalar para dezenas de milhões de clientes.

### 5. As três métricas e quando coincidem

> **Em palavras simples.** Há três formas comuns de medir se dois vetores estão "perto": **dot product** (concordância, sensível ao comprimento), **cosine similarity** (só o ângulo) e **Euclidean distance** (a distância em linha reta entre os dois pontos). Quando todos os vetores têm comprimento 1, as três ordenam os resultados exatamente da mesma forma, e a escolha passa a ser de velocidade. Quando não têm, cada uma responde a uma pergunta diferente, e usar a errada piora a busca sem avisar.

| Métrica | Fórmula | Sensível à magnitude? | Uso típico |
|---|---|---|---|
| Dot product (inner product) | `a · b` | Sim | Embeddings normalizados; recommender systems (magnitude ≈ popularidade) |
| Cosine similarity | `a · b / (‖a‖‖b‖)` | Não | Semantic similarity de texto |
| Euclidean distance (L2) | `‖a − b‖` | Sim | Clustering, k-means |

Para unit vectors vale a identidade `‖a − b‖² = 2 − 2(a · b)`. Consequência prática: com embeddings normalizados, ordenar por menor L2, maior cosine ou maior dot product dá **o mesmo ranking**. A escolha da métrica no vector database passa a ser questão de custo, não de qualidade. Sem normalização, as três divergem, e usar a métrica diferente daquela com que o embedding model foi treinado degrada o retrieval sem dar erro.

> **No mundo real.** O **Spotify** criou e abriu a biblioteca **Annoy** (*Approximate Nearest Neighbors Oh Yeah*) para recomendar músicas: cada faixa e cada usuário é um vetor, e recomendar é buscar os vetores vizinhos. A biblioteca obriga a declarar a métrica ao criar o índice (`angular`, `euclidean` ou `dot`), justamente porque o resultado muda. A métrica `angular` do Annoy é a Euclidean distance entre vetores normalizados, uma aplicação direta da identidade que você verifica no Passo 8 do notebook.

### 6. Matrix: transformação e pilha de vetores

> **Em palavras simples.** Uma **matrix** é uma tabela de números, e tem dois usos. No primeiro, é uma máquina de transformar: multiplicar um vetor por uma matriz gira, estica ou achata esse vetor (é o que cada camada de uma rede neural faz). No segundo, é uma pilha de vetores: cada linha é um documento, e multiplicar a matriz por uma pergunta calcula, de uma só vez, a semelhança da pergunta com todos os documentos.

**Como transformação:** multiplicar um vetor por uma matriz o gira, estica, reflete ou projeta. As colunas da matriz dizem para onde vão os vetores da base. Uma camada de rede neural é isso: `y = Wx + b`, uma linear transformation seguida de um deslocamento.

**Como pilha:** uma matriz `N × d` são `N` embeddings empilhados. `M @ q` calcula, de uma vez, o dot product da query `q` com todos os `N` documentos. Uma exact vector search é uma única matrix multiplication seguida de um `argsort`.

**Matrix multiplication** `A @ B`: o elemento `(i, j)` do resultado é o dot product da linha `i` de `A` com a coluna `j` de `B`. Por isso o número de colunas de `A` precisa ser igual ao número de linhas de `B`. Compor duas transformações é multiplicar as matrizes, e a ordem importa (`A @ B ≠ B @ A` em geral).

> **No mundo real.** O **PageRank**, algoritmo que deu origem ao Google (Brin e Page, 1998), trata a web inteira como uma matriz gigante em que cada posição diz se a página A aponta para a página B. A importância de todas as páginas é obtida multiplicando repetidamente essa matriz por um vetor de pontuações, até os números pararem de mudar (*power iteration*). Uma única operação, **matrix × vector**, repetida algumas dezenas de vezes, ordenava centenas de milhões de páginas.

### 7. Shapes, axes e broadcasting

> **Em palavras simples.** O **shape** é o formato de uma tabela de números: quantas linhas, quantas colunas e, às vezes, mais dimensões. Toda operação entre tabelas exige formatos compatíveis, como peças de encaixe. **Broadcasting** é o atalho pelo qual o NumPy "repete" automaticamente a tabela menor para casar com a maior, por exemplo subtrair a mesma lista de médias de todas as linhas. **Vectorization** é fazer a conta na tabela inteira de uma vez, em vez de linha por linha, o que é dezenas a centenas de vezes mais rápido.

Regra de shapes: `(N, d) @ (d,) → (N,)`. `(N, d) @ (d, M) → (N, M)`. `(B, T, d) @ (d, k) → (B, T, k)`.

`axis` indica a dimensão que **desaparece** na operação: em uma matriz `(N, d)`, `sum(axis=0)` devolve `(d,)` e `sum(axis=1)` devolve `(N,)`. `keepdims=True` mantém a dimensão com tamanho 1, o que é necessário para o broadcasting funcionar na normalização: `M / np.linalg.norm(M, axis=1, keepdims=True)`.

**Broadcasting:** NumPy alinha os shapes pela direita; dimensões de tamanho 1 (ou ausentes) são esticadas. `(N, d) − (d,)` subtrai o vetor de cada linha. `(N, 1) * (1, M)` produz `(N, M)`. É poderoso e é a origem de bugs silenciosos: `(N,) − (N, 1)` produz uma matriz `(N, N)` sem erro nenhum.

**Vectorization:** substituir loops Python por operações de array. O ganho vem de o loop passar a rodar em C/BLAS com instruções SIMD e sem overhead do interpretador. Diferenças de 50 a 1.000 vezes são normais.

> **No mundo real.** O dataset **MNIST** (dígitos manuscritos coletados pelo NIST e organizados por LeCun, Cortes e Burges) tem 60.000 imagens de treino de 28 × 28 pixels. Em NumPy, isso é um único array de shape `(60000, 28, 28)`. A preparação clássica cabe em duas linhas sem nenhum loop: `X.reshape(60000, 784)` transforma cada imagem em um vetor, e `(X - X.mean(axis=0)) / 255` subtrai de cada uma das 60.000 imagens a "imagem média", de shape `(784,)`, por broadcasting.

### 8. Rank, em uma frase

> **Em palavras simples.** O **rank** de uma tabela diz quanta informação realmente independente ela contém. Uma planilha com 100 colunas em que 97 são combinações das outras 3 tem rank 3: no fundo, bastam três colunas e uma receita para reconstruir o resto. Tabelas de **low rank** podem ser guardadas de forma muito mais compacta. Essa ideia é a base da compressão e, mais adiante na trilha, do LoRA.

O **rank** de uma matriz é o número de direções independentes que ela contém. Uma matriz `d × k` de rank `r` pode ser escrita como o produto de uma `d × r` por uma `r × k`. Guarde isso: é a ideia inteira por trás de LoRA (Semana 18).

> **No mundo real.** A técnica **Latent Semantic Analysis** (Deerwester et al., 1990) aplica isso a texto. Monta-se uma matriz enorme de palavras × documentos e usa-se a SVD para aproximá-la por uma de **low rank**, tipicamente com 100 a 300 dimensões. O efeito colateral útil: palavras diferentes que aparecem nos mesmos contextos ("carro" e "automóvel") acabam próximas no espaço reduzido, e uma busca por uma encontra documentos que só usam a outra. Foi um dos primeiros sistemas de busca semântica e um antepassado direto dos embeddings.

### 9. The curse of dimensionality

> **Em palavras simples.** Em espaços com centenas de dimensões, a intuição do dia a dia falha: pontos sorteados ao acaso ficam todos mais ou menos à mesma distância uns dos outros, e "o mais próximo" quase não se distingue do "mais distante". Essa é a **curse of dimensionality**. Embeddings funcionam apesar dela porque não são pontos ao acaso: o treino organiza o espaço. A consequência prática é que um número de semelhança não tem significado universal; cada modelo tem a sua escala, e a nota de corte precisa ser medida para cada um.

Em alta dimensão, pontos aleatórios ficam todos quase à mesma distância uns dos outros e quase ortogonais: o cosine entre dois vetores aleatórios em `d` dimensões tem média 0 e desvio aproximadamente `1/√d`. Em 768 dimensões, isso dá cerca de 0,036.

Duas consequências práticas. Primeiro, um cosine de 0,35 entre duas frases, que parece baixo, está a quase dez desvios-padrão do acaso: é um sinal forte. Scores de similaridade não têm uma escala universal; cada embedding model tem a sua, e thresholds precisam ser calibrados por modelo. Segundo, embeddings funcionam porque **não** são aleatórios: o treino concentra o significado em um subespaço de dimensão efetiva muito menor que `d`.

> **No mundo real.** O **FaceNet** (Schroff, Kalenichenko e Philbin, Google, 2015) converte a foto de um rosto em um vetor de 128 números. Duas fotos são consideradas da mesma pessoa quando a distância entre os vetores fica abaixo de um **threshold** que os autores escolheram medindo em dados reais, não por intuição. Com isso, o modelo atingiu 99,63% de accuracy no benchmark *Labeled Faces in the Wild*. O valor desse threshold só serve para aquele modelo; outro modelo de reconhecimento facial precisa do seu.

## Erros comuns

- Comparar embeddings não normalizados com dot product quando o modelo foi treinado para cosine.
- Esquecer `keepdims=True` ao normalizar linhas e obter um erro de broadcasting (ou, pior, um resultado errado sem erro).
- Confundir `*` (elemento a elemento) com `@` (matrix multiplication).
- Misturar `float64` (padrão do NumPy) com `float32` (padrão de embeddings e de PyTorch): dobra a memória e reduz a velocidade.
- Usar um threshold de similaridade copiado de outro embedding model.

## Leitura e vídeos (fim de semana)

- 3Blue1Brown, *Essence of Linear Algebra*: Chapter 1 (*Vectors*), 2 (*Linear combinations, span, and basis vectors*), 3 (*Linear transformations and matrices*), 4 (*Matrix multiplication as composition*), 7 (*Inverse matrices, column space and null space*, para rank) e 9 (*Dot products and duality*).
- NumPy documentation: *NumPy: the absolute basics for beginners* e *Broadcasting*.
- Jay Alammar, *The Illustrated Word2vec*, para ver word vectors na prática.
- Opcional: Gilbert Strang, *Introduction to Linear Algebra* (MIT OCW 18.06), Lectures 1 e 2.

## Perguntas de autoavaliação

1. Dois vetores têm dot product 50. Dá para dizer se são semelhantes? O que falta saber?
2. Por que `M @ q` é muito mais rápido que um loop Python sobre as linhas de `M`?
3. Se todos os embeddings estão normalizados, ordenar por Euclidean distance e por cosine similarity dá o mesmo ranking? Demonstre com a identidade.
4. Qual o shape de `A @ B` se `A` é `(32, 128)` e `B` é `(128, 10)`? E de `A.sum(axis=1, keepdims=True)`?
5. Em um embedding de 1.024 dimensões, qual o desvio aproximado do cosine entre vetores aleatórios? O que isso diz sobre um score de 0,2?

## Entregável da semana

Notebook resolvido e uma nota curta em `docs/` explicando, com suas palavras, por que vector databases normalizam os embeddings e o que acontece se a métrica do index não for a do embedding model.

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/ai-architect-foundations/blob/main/01-Phase-1-Math-and-ML-Core/02-linear-algebra-numpy/notebooks/semana_02_exercicio.ipynb) · [`semana_02_exercicio.ipynb`](../notebooks/semana_02_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/ai-architect-foundations/blob/main/01-Phase-1-Math-and-ML-Core/02-linear-algebra-numpy/notebooks/semana_02_gabarito.ipynb) · [`semana_02_gabarito.ipynb`](../notebooks/semana_02_gabarito.ipynb) |

[← Semana 1](/01-Phase-1-Math-and-ML-Core/01-sklearn-pipelines/docs/Semana-01-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 3 →](/01-Phase-1-Math-and-ML-Core/03-calculus-gradient/docs/Semana-03-Conceitos.md)
