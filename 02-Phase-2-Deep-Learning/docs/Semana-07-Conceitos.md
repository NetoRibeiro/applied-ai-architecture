# Semana 7 — Self-Attention do Zero

**Fase 2: Deep Learning e o Ecossistema Hugging Face** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Implementar o mecanismo de attention do paper *Attention Is All You Need* linha por linha: scaled dot-product attention, causal mask, multi-head attention e positional encoding. Ao final você saberá dizer o shape de cada tensor dentro de uma attention layer.

## Por que isso importa para um AI Architect

Attention explica os limites práticos dos LLMs: por que o custo cresce com o quadrado do contexto, por que existe um KV cache, por que LoRA é aplicado em `q_proj` e `v_proj`, por que um modelo "perde" informação no meio de um contexto longo e por que encoders e decoders servem a tarefas diferentes.

## Conceitos

### 1. O problema que a attention resolve

> **Em palavras simples.** Para entender uma palavra, muitas vezes é preciso olhar outra que está longe na frase. Os modelos antigos liam o texto em fila, passando um resumo de palavra em palavra, e o começo se perdia até chegar ao fim. A **attention** permite que cada palavra consulte diretamente todas as outras e decida em quais prestar atenção, sem intermediários. Além de lembrar melhor, isso pode ser calculado de uma vez só, o que aproveita a GPU.

No bigram model, cada token só enxerga o anterior. RNNs e LSTMs estenderam o contexto passando um hidden state de token em token, mas com dois problemas: a informação de tokens distantes se degrada no caminho, e o processamento é sequencial, sem paralelismo. Attention resolve os dois: cada posição consulta **diretamente** todas as outras, em uma única operação matricial paralelizável.

> **No mundo real.** A attention nasceu na tradução automática (Bahdanau, Cho e Bengio, 2014): os sistemas da época comprimiam a frase inteira em um único vetor e pioravam muito em frases longas. Em 2016, o **Google Translate** adotou um sistema neural com attention (GNMT), e o Google reportou redução de 55% a 85% nos erros de tradução em vários pares de idiomas em relação ao sistema anterior, baseado em frases.

### 2. Query, Key e Value

> **Em palavras simples.** A attention funciona como uma busca. Cada palavra emite uma pergunta (**query**: "o que estou procurando?"), uma etiqueta (**key**: "o que eu tenho a oferecer?") e um conteúdo (**value**: "o que entrego se for escolhida?"). Compara-se a pergunta de uma palavra com as etiquetas de todas as outras; as que combinam mais contribuem mais com o seu conteúdo. É uma biblioteca em que todos os livros respondem ao mesmo tempo, cada um na proporção em que é relevante.

Cada token gera três vetores, por três linear projections aprendidas (`W_Q`, `W_K`, `W_V`):

- **Query (q):** "o que estou procurando?"
- **Key (k):** "o que eu ofereço?"
- **Value (v):** "o que eu entrego se for escolhido?"

A afinidade entre a posição `i` e a posição `j` é o dot product `qᵢ · kⱼ` (Semana 2). É um soft lookup em um dicionário: em vez de recuperar um único value pela key exata, recupera-se uma média ponderada de todos os values, com pesos dados pela similaridade entre query e keys. É, literalmente, uma vector search feita dentro do modelo, em cada layer.

> **No mundo real.** O texto do Google que apresentou o **Transformer** ao público (2017) usa a frase *"The animal didn't cross the street because it was too tired"*. Ao processar a palavra *it*, o modelo concentra a attention em *animal*. Trocando o final para *"because it was too wide"*, a attention de *it* passa para *street*. Resolver a quem o pronome se refere é justamente o tipo de problema que os tradutores automáticos anteriores erravam.

### 3. Scaled dot-product attention

> **Em palavras simples.** A conta da attention tem três passos: medir a afinidade entre cada par de palavras (com um dot product), transformar as afinidades em pesos que somam 100% e fazer uma média ponderada dos conteúdos. Como cada palavra é comparada com todas as outras, o trabalho cresce com o **quadrado** do tamanho do texto: um texto duas vezes maior exige quatro vezes mais contas.

`Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) · V`

Passo a passo, com os shapes para um batch `B`, sequência `T` e head dimension `d_k`:

1. `Q Kᵀ` → `(B, T, T)`: a matriz de attention scores. A linha `i` contém a afinidade do token `i` com cada token `j`.
2. Dividir por `√d_k`: sem isso, a variância do dot product cresce com a dimensão, os scores ficam grandes, o softmax satura em um único token e os gradientes ficam minúsculos.
3. `softmax` em cada linha → attention weights que somam 1.
4. Multiplicar por `V` → `(B, T, d_k)`: cada token recebe a média ponderada dos values.

**Complexidade:** `O(T²)` em tempo e memória. Dobrar o contexto quadruplica a attention matrix. É a razão de fundo para limites de context window e para o custo de prompts longos.

> **No mundo real.** O **GPT-2** aceita 1.024 tokens: cada cabeça de attention monta uma tabela de 1.024 × 1.024, cerca de um milhão de afinidades. Um modelo com contexto de 128 mil tokens, calculado da mesma forma, precisaria de uma tabela com mais de 16 bilhões de entradas por cabeça, por camada. É esse crescimento quadrático que explica por que contextos longos são caros e por que existem técnicas como a **Flash Attention**, que evita guardar a tabela inteira na memória.

### 4. Self-attention, cross-attention e masks

> **Em palavras simples.** Uma **mask** é uma venda nos olhos. Um modelo que gera texto não pode espiar as palavras futuras durante o treino, senão copiaria a resposta; a **causal mask** garante que cada palavra só enxergue as anteriores. Modelos feitos para compreender um texto pronto dispensam essa venda e olham para os dois lados. Essa é a diferença de fundo entre um **decoder** (gera) e um **encoder** (compreende).

- **Self-attention:** Q, K e V vêm da mesma sequência.
- **Cross-attention:** Q vem de uma sequência e K, V de outra (o decoder consultando o encoder em modelos de tradução; texto consultando imagem em modelos multimodais).

**Causal mask.** Um modelo gerador não pode olhar o futuro, senão copiaria a resposta durante o treino. Antes do softmax, as posições `j > i` recebem `−∞`, o que zera seus pesos. A attention matrix fica lower-triangular. Em PyTorch: `scores.masked_fill(mask == 0, float("-inf"))`.

**Padding mask.** Em batches com sequências de tamanhos diferentes, os tokens de padding também são mascarados para não receberem attention. É a `attention_mask` que o tokenizer devolve.

| | Encoder (BERT) | Decoder (GPT, Llama) |
|---|---|---|
| Mask | Só padding: attention bidirecional | Causal + padding |
| Cada token vê | Passado e futuro | Só o passado |
| Bom para | Representar texto (embeddings, classificação) | Gerar texto |

> **No mundo real.** O **BERT** (2018) e o **GPT** (2018) são quase a mesma arquitetura com masks diferentes. O BERT é treinado escondendo 15% das palavras de uma frase e pedindo ao modelo que as adivinhe olhando o contexto dos dois lados; por isso é bom em classificar e em buscar. O GPT só olha para trás e é treinado para continuar o texto; por isso é bom em escrever. Uma única decisão de mascaramento separou as duas famílias de modelos.

### 5. Multi-head attention

> **Em palavras simples.** Em vez de uma única attention, o modelo usa várias em paralelo, as **heads**. Cada uma pode se especializar em um tipo de relação: uma liga o verbo ao seu objeto, outra liga o pronome a quem ele se refere, outra presta atenção à palavra vizinha. É como ter vários leitores analisando a mesma frase, cada um com uma pergunta diferente, e depois juntar as anotações.

Em vez de uma attention com dimensão `d`, usam-se `h` attentions paralelas (heads), cada uma com dimensão `d/h`, com suas próprias projeções. As saídas são concatenadas e passam por uma projeção final (`o_proj`).

Cada head pode se especializar em um tipo de relação: sintaxe, coreference, posição relativa, cópia de tokens. O custo total é praticamente o mesmo de uma head grande.

Na implementação eficiente, uma única layer linear produz Q, K e V para todas as heads, e um `view` + `transpose` reorganiza `(B, T, d)` em `(B, h, T, d/h)`. `F.scaled_dot_product_attention` usa por baixo implementações otimizadas (Flash Attention quando disponível).

Nomes que você verá em `named_modules()` de modelos reais: `q_proj`, `k_proj`, `v_proj`, `o_proj`. São esses os `target_modules` do LoRA.

> **No mundo real.** O estudo *What Does BERT Look At?* (Clark et al., Stanford, 2019) examinou as 144 attention heads do BERT e encontrou especialistas: uma head que liga verbos aos seus objetos diretos e outra que liga pronomes possessivos ao substantivo correspondente, com acerto muito acima do acaso, sem que o modelo jamais tivesse recebido aulas de gramática. As funções surgiram sozinhas da tarefa de prever palavras escondidas.

### 6. Positional information

> **Em palavras simples.** A attention, sozinha, trata a frase como um saco de palavras: não sabe qual veio antes. "O cão mordeu o homem" e "o homem mordeu o cão" pareceriam iguais. A **positional information** é um carimbo de posição acrescentado a cada palavra. Há várias formas de carimbar; a mais usada hoje, **RoPE**, registra a distância relativa entre as palavras, e não a posição absoluta.

Self-attention é **permutation-equivariant**: embaralhar os tokens de entrada só embaralha a saída. Sem informação de posição, "o cão mordeu o homem" e "o homem mordeu o cão" seriam indistinguíveis. Soluções:

- **Sinusoidal positional encoding** (paper original): senos e cossenos de frequências diferentes somados ao token embedding. Sem parâmetros; posições próximas têm vetores parecidos.
- **Learned absolute positions** (GPT-2, BERT): um `nn.Embedding(max_len, d)`. Simples, mas não extrapola além de `max_len`.
- **RoPE, Rotary Position Embedding** (Llama, Qwen, Gemma, Mistral): rotaciona os vetores de query e key por um ângulo proporcional à posição, de modo que o dot product dependa da distância **relativa** entre tokens. É o padrão atual e generaliza melhor para contextos longos.
- **ALiBi** e relative position bias: somam um viés aos attention scores em função da distância.

> **No mundo real.** O **GPT-2** aprende um vetor para cada posição, de 1 a 1.024. A consequência é um limite rígido: não existe vetor para a posição 1.025, e o modelo simplesmente não consegue ler além disso. Modelos como **Llama**, **Mistral** e **Qwen** adotaram RoPE (Su et al., 2021), que codifica distâncias relativas e se comporta melhor em textos longos, um dos fatores que permitiram os contextos de dezenas ou centenas de milhares de tokens.

### 7. O que a attention não faz sozinha

> **Em palavras simples.** A attention movimenta informação entre as palavras, mas não a processa a fundo: ela junta e redistribui. O trabalho de transformar e de consultar conhecimento guardado fica com outra peça, que vem logo depois, uma pequena rede (**MLP**, ou **feed-forward**) aplicada a cada palavra separadamente. Uma reúne os dados; a outra pensa sobre eles.

A attention move informação entre posições, mas é uma operação essencialmente linear sobre os values. O processamento não linear de cada posição fica com o feed-forward (MLP) que vem depois. O bloco completo, com residual connections e normalization, é o tema da Semana 8.

> **No mundo real.** O estudo *Locating and Editing Factual Associations in GPT* (Meng et al., 2022), conhecido como **ROME**, investigou onde o modelo guarda fatos como "a Torre Eiffel fica em Paris". A resposta: principalmente nas camadas MLP do meio da rede, não na attention. Os autores conseguiram alterar um único fato modificando os pesos de uma dessas camadas, e o modelo passou a responder de forma coerente como se a torre ficasse em outra cidade.

### 8. Ligações com as próximas fases

- **KV cache (Semana 23):** na geração, as keys e values dos tokens anteriores não mudam; guardá-los evita recalcular todo o prefixo a cada novo token.
- **LoRA (Semana 18):** os adapters são inseridos nas projection matrices da attention.
- **"Lost in the middle" (Semana 13):** modelos tendem a usar melhor a informação do início e do fim do contexto.
- **Bi-encoder vs. cross-encoder (Semana 14):** no cross-encoder, query e documento entram juntos e a attention opera entre os tokens dos dois; por isso é mais preciso e mais caro.
- **Interpretability:** attention weights são úteis para inspeção, mas não são uma explicação fiel do raciocínio do modelo. Use como pista, não como prova.

## Erros comuns

- Esquecer a causal mask em um decoder: a loss de treino despenca e o modelo não gera nada útil.
- Aplicar a mask depois do softmax em vez de antes.
- Trocar `transpose(-2, -1)` e errar os shapes de `Q Kᵀ`.
- Esquecer `contiguous()` antes de `view` ao juntar as heads.
- Tratar `attention_mask` como opcional em batches com padding.

## Leitura e vídeos (fim de semana)

- Vaswani et al., *Attention Is All You Need* (2017): Sections 3.1 a 3.5.
- Jay Alammar, *The Illustrated Transformer*.
- 3Blue1Brown, *Neural Networks*: Chapter 5 (*Transformers, the tech behind LLMs*) e Chapter 6 (*Attention in transformers, step-by-step*).
- Andrej Karpathy, *Let's build GPT: from scratch, in code, spelled out* (primeira metade, até self-attention).
- Harvard NLP, *The Annotated Transformer* (o paper implementado linha a linha em PyTorch).
- Opcional: Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (abstract e seção 3).

## Perguntas de autoavaliação

1. Qual o shape de `Q Kᵀ` para um batch `B`, sequência `T` e dimensão `d`?
2. Por que dividir por `√d_k`?
3. O que acontece se você treinar um gerador sem causal mask?
4. Por que o custo da attention é quadrático no tamanho do contexto?
5. Qual a diferença entre self-attention e cross-attention? Dê um exemplo de uso de cada.
6. Por que RoPE substituiu as learned absolute positions nos LLMs atuais?

## Entregável da semana

Notebook resolvido e um attention heatmap comentado em `docs/`.

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/02-Phase-2-Deep-Learning/notebooks/semana_07_exercicio.ipynb) · [`semana_07_exercicio.ipynb`](../notebooks/semana_07_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/02-Phase-2-Deep-Learning/notebooks/semana_07_gabarito.ipynb) · [`semana_07_gabarito.ipynb`](../notebooks/semana_07_gabarito.ipynb) |

[← Semana 6](/02-Phase-2-Deep-Learning/docs/Semana-06-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 8 →](/02-Phase-2-Deep-Learning/docs/Semana-08-Conceitos.md)
