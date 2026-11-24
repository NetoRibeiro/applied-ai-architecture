# Semana 6 — Tokenization, Embeddings e um Bigram Language Model

**Fase 2: Deep Learning e o Ecossistema Hugging Face** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Entender como texto vira números (tokenization), como números viram vetores com significado (embeddings) e o que significa "language model": uma função que prevê o next token. Você treinará o language model mais simples possível, um bigram model, e medirá sua qualidade com cross-entropy e perplexity.

## Por que isso importa para um AI Architect

Custo, latência e context window de um LLM são medidos em tokens, não em palavras. Embeddings são a matéria-prima do RAG. E a loss que você vai minimizar aqui, next-token cross-entropy, é literalmente a mesma do pre-training e do fine-tuning de qualquer LLM. Entender este objetivo explica tanto a fluência quanto a hallucination: o modelo é treinado para produzir a continuação **plausível**, não a verdadeira.

## Conceitos

### 1. Tokenization

> **Em palavras simples.** **Tokenization** é cortar o texto em pedaços que o modelo conhece e trocar cada pedaço por um número. Os pedaços (**tokens**) não são letras nem palavras inteiras: são fragmentos frequentes. Palavras comuns viram um token só; palavras raras são montadas com vários, como peças de montar. O modelo nunca vê letras, só a sequência de números. É por tokens que se mede o tamanho de um texto, o limite de memória do modelo e o preço cobrado.

| Granularidade | Vocabulary size | Sequências | Problema |
|---|---|---|---|
| Character-level | ~100 | Muito longas | Cada token carrega pouco sentido |
| Word-level | Enorme e aberto | Curtas | Out-of-vocabulary (OOV) |
| **Subword** (BPE, WordPiece, SentencePiece/Unigram) | 30 mil a 250 mil | Médias | É o padrão dos LLMs |

**Byte-Pair Encoding (BPE)** começa com caracteres (ou bytes) e funde repetidamente o par adjacente mais frequente no corpus de treino, até atingir o vocabulary size desejado. Palavras comuns viram um token; palavras raras viram pedaços. Byte-level BPE (GPT-2 em diante) garante que qualquer string é representável, sem token `<unk>`.

**WordPiece** (BERT) é parecido, com um critério de fusão baseado em likelihood e o marcador `##` para continuação. **SentencePiece** trata o espaço como um caractere (`▁`) e não depende de pré-tokenização por palavras, o que ajuda em idiomas sem espaços.

Consequências práticas:

- Regra de bolso em inglês: 1 token ≈ 4 caracteres ≈ 0,75 palavra. Em português a razão é pior (mais tokens por palavra), porque a maioria dos tokenizers foi treinada com predominância de inglês. Isso entra na conta de custo e de context window.
- O tokenizer é parte do modelo. Usar um tokenizer diferente do usado no treino produz lixo.
- Aritmética, contagem de letras e manipulação de strings são difíceis para LLMs em parte porque o modelo não vê caracteres, vê tokens.

**Special tokens:** `<bos>`/`<s>` (begin of sequence), `<eos>`/`</s>` (end of sequence), `<pad>`, e os marcadores de papel dos chat templates. O modelo só aprende a parar de gerar se viu `<eos>` no treino (Semana 19).

Nesta semana usamos character-level para manter tudo visível.

> **No mundo real.** O **GPT-2** usa um vocabulário de 50.257 tokens, criado com **Byte-Pair Encoding** a partir do seu corpus de treino. A regra prática divulgada pela OpenAI para o inglês é que 1 token equivale a cerca de 4 caracteres, ou 100 tokens a cerca de 75 palavras. Em português a conta piora, porque o vocabulário foi montado com predominância de inglês: a mesma frase traduzida costuma consumir mais tokens. Dá para verificar com a biblioteca `tiktoken`.

### 2. Embedding: uma lookup table treinável

> **Em palavras simples.** Um **embedding** é um dicionário em que a definição de cada token não é um texto, e sim uma lista de números. No início essas listas são sorteadas. Durante o treino, o modelo vai ajustando cada lista de modo que tokens usados em situações parecidas terminem com listas parecidas. O significado não é programado: ele aparece porque ajuda a prever a próxima palavra.

`nn.Embedding(V, d)` é uma matriz `V × d`. Dar o token de índice `i` devolve a linha `i`. Matematicamente equivale a multiplicar um vetor one-hot pela matriz (ligação direta com a Semana 1), sem desperdiçar memória nem tempo.

As linhas começam aleatórias. O treino as move de modo que tokens usados em contextos parecidos fiquem próximos: é a **distributional hypothesis** ("you shall know a word by the company it keeps"). Ninguém rotula significado; ele emerge da tarefa de prever o próximo token.

Atenção à distinção que será importante na Fase 3: a embedding table de um LLM produz **static token embeddings** (um vetor por token, independente do contexto). Os vetores usados em RAG são **contextual sentence embeddings**, a saída de um encoder inteiro depois de várias layers de attention.

> **No mundo real.** No **GPT-2 small**, a tabela de embeddings tem 50.257 linhas (uma por token) e 768 colunas: cerca de **38,6 milhões** de números, quase um terço dos 124 milhões de parâmetros do modelo. Cada vez que você digita um prompt, o primeiro passo é uma simples consulta a essa tabela, token por token. A mesma tabela é reutilizada no fim do modelo para converter o resultado de volta em tokens.

### 3. Language model

> **Em palavras simples.** Um **language model** é um sistema que, dado o início de um texto, calcula a chance de cada palavra possível vir em seguida. É o autocompletar do celular levado ao extremo. Para escrever um texto longo, ele repete a operação: prevê uma palavra, acrescenta ao texto, prevê a próxima. Todo chatbot atual é, por baixo, essa máquina de prever a continuação.

Dado um contexto, produzir uma distribuição de probabilidade sobre o next token. Pela chain rule da probabilidade, a probabilidade de uma sequência se fatora em:

`P(x₁ … x_T) = Π P(xₜ | x₁ … xₜ₋₁)`

Um modelo que faz isso é **autoregressive**. O **bigram model** aproxima usando só o token anterior: `P(xₜ | xₜ₋₁)`. Um GPT usa todos os anteriores, via attention. A arquitetura muda; o objetivo é o mesmo.

> **No mundo real.** O **Smart Compose** do Gmail, lançado em 2018, é um language model que sugere o fim da frase enquanto você digita. O Google informou que o recurso poupa os usuários de digitar mais de um bilhão de caracteres por semana. O desafio de engenharia descrito pela equipe não foi só a qualidade: a sugestão precisa aparecer em menos de 100 milissegundos, ou chega depois de a pessoa já ter digitado a palavra.

### 4. Loss e perplexity

> **Em palavras simples.** A **loss** de um language model mede a surpresa: o quanto, em média, o modelo se espanta com a palavra que realmente vem a seguir. A **perplexity** traduz isso em algo mais intuitivo: entre quantas opções o modelo fica em dúvida a cada palavra. Perplexity 100 significa hesitar entre cem alternativas; perplexity 10, entre dez. Quanto menor, melhor o modelo conhece a língua.

- **Cross-entropy:** `−log P(token correto)`, em média sobre todas as posições. Medida em nats (log natural).
- **Sanity check:** um modelo que chuta uniformemente entre `V` tokens tem loss `ln(V)`. Para 65 caracteres, 4,17.
- **Perplexity** = `exp(loss)`. Lê-se como "entre quantas opções igualmente prováveis o modelo está em dúvida, em média". Loss 1,5 → perplexity 4,5.

Perplexity só é comparável entre modelos com o **mesmo tokenizer** e o mesmo dataset de avaliação. Um modelo com tokens maiores terá perplexity por token mais alta sem ser pior.

> **No mundo real.** O paper do **GPT-2** (2019) usou perplexity para mostrar que o modelo generalizava: sem nenhum treino específico, ele bateu o melhor resultado então publicado em vários benchmarks clássicos de modelagem de linguagem. No Penn Treebank, por exemplo, a perplexity caiu da faixa de 47 para a de 35. Foi um dos primeiros sinais de que um único modelo grande, treinado em texto genérico, podia superar modelos feitos sob medida.

### 5. O formato do dado de treino

> **Em palavras simples.** Para ensinar um language model, não é preciso ninguém rotular nada: o próprio texto é o gabarito. Pega-se um trecho, esconde-se a continuação e pede-se ao modelo para adivinhá-la, uma palavra por vez. Cada trecho de mil palavras rende mil exercícios de "qual é a próxima?". Por isso é possível treinar com quantidades de texto que nenhuma equipe conseguiria rotular à mão.

De um texto longo, sorteiam-se janelas de `T` tokens (`block_size` ou context length). A entrada é `x = tokens[i : i+T]` e o target é a mesma janela deslocada uma posição, `y = tokens[i+1 : i+T+1]`. Cada janela gera `T` exemplos de treino de uma vez: prever o token 2 a partir do 1, o 3 a partir de 1–2, e assim por diante. Isso se chama **teacher forcing**: no treino o modelo sempre recebe o contexto correto, não as próprias previsões.

Shapes: `x` e `y` são `(B, T)`; os logits são `(B, T, V)`. Para a `F.cross_entropy`, achata-se para `(B·T, V)` e `(B·T,)`.

> **No mundo real.** O **GPT-2** foi treinado no WebText, cerca de 8 milhões de páginas (40 GB de texto) coletadas a partir de links bem avaliados no Reddit, em janelas de até 1.024 tokens. Nenhum humano anotou esse material: a tarefa "preveja o próximo token" extrai o exercício do próprio texto. Essa forma de aprender, chamada **self-supervised learning**, é o que tornou viável treinar com a internet inteira.

### 6. Generation: decoding autoregressivo

> **Em palavras simples.** Na hora de escrever, o modelo tem uma lista de candidatos com as respectivas chances e precisa escolher um. Escolher sempre o mais provável dá um texto correto e repetitivo. Sortear de acordo com as chances dá variedade. A **temperature** é o botão que regula esse sorteio: baixa, o modelo fica conservador e previsível; alta, fica criativo e erra mais.

Prever → escolher um token → anexar ao contexto → repetir.

- **Greedy:** sempre o `argmax`. Determinístico, tende a repetição.
- **Sampling com temperature:** divide os logits por `T` antes do softmax. `T < 1` concentra a distribuição (texto conservador); `T > 1` achata (texto criativo e mais sujeito a erro); `T → 0` equivale a greedy.
- **Top-k** e **top-p (nucleus):** restringem o sampling aos tokens mais prováveis (Semana 9).

`torch.multinomial(probs, 1)` sorteia um índice segundo as probabilidades.

> **No mundo real.** Nas APIs comerciais de LLM, a `temperature` é um parâmetro exposto ao desenvolvedor, normalmente entre 0 e 2. A prática consolidada é usar valores próximos de zero quando a resposta precisa ser reproduzível (extração de dados, classificação, geração de código, avaliação automática) e valores mais altos para brainstorming e escrita criativa. Muitos problemas de "o modelo responde diferente a cada vez" em produção se resolvem ajustando esse único número.

### 7. O que a tabela do bigram aprende

> **Em palavras simples.** O **bigram model** é o language model mais simples que existe: para prever a próxima letra, olha só a letra atual. Depois de treinado, ele é uma tabela de "o que costuma vir depois de quê": depois de `q` vem `u`. Ele acerta combinações curtas e não consegue formar uma frase, porque não se lembra de nada além da última letra. É o ponto de partida para entender por que os modelos precisam de mais contexto.

A matriz `V × V` treinada converge para as frequências condicionais do corpus: a linha do token `q` concentra probabilidade em `u`. É o mesmo resultado que se obteria contando pares e normalizando; o treino por gradiente é só outra forma de chegar lá. A vantagem é que a abordagem por gradiente escala para modelos em que contar é impossível.

O limite do bigram é o contexto de um token: ele não tem como manter concordância, fechar aspas ou lembrar o assunto. Estender o contexto por contagem (n-grams) explode combinatorialmente. A solução é a attention, na próxima semana.

> **No mundo real.** A série *makemore*, de Andrej Karpathy, treina exatamente esse modelo em um arquivo com cerca de 32 mil nomes próprios em inglês e gera nomes novos que soam plausíveis. O exercício mostra que a tabela aprendida por gradient descent é praticamente idêntica à obtida simplesmente **contando** os pares de letras no arquivo. A partir daí, a série vai acrescentando contexto passo a passo até chegar ao Transformer, o mesmo caminho desta fase.

## Erros comuns

- Tokenizar com um tokenizer e rodar com o modelo de outro.
- Esquecer o deslocamento de uma posição entre `x` e `y` (o modelo aprende a copiar a entrada e a loss vai a zero).
- Comparar perplexity entre modelos com tokenizers diferentes.
- Estimar custo de API contando palavras em vez de tokens.
- Dividir train/validation embaralhando caracteres ou janelas sobrepostas: vaza informação. Divida por posição no texto.

## Leitura e vídeos (fim de semana)

- Andrej Karpathy, *The spelled-out intro to language modeling: building makemore* (Part 1, bigrams). O notebook desta semana segue a mesma linha.
- Hugging Face *LLM Course*, capítulo *The 🤗 Tokenizers library*: seções *Byte-Pair Encoding tokenization*, *WordPiece tokenization* e *Unigram tokenization*.
- Andrej Karpathy, *Let's build the GPT Tokenizer*.
- Jay Alammar, *The Illustrated Word2vec*.
- Opcional: Bengio et al., *A Neural Probabilistic Language Model* (2003), o artigo que introduziu embeddings aprendidos para language modeling.
- Ferramenta: o *Tokenizer Playground* da Hugging Face ou o `tiktoken`, para ver como frases em português são quebradas.

## Perguntas de autoavaliação

1. Por que `nn.Embedding` é equivalente a one-hot seguido de uma layer linear sem bias?
2. Vocabulary de 65 caracteres. Qual a loss esperada antes do treino? E a perplexity?
3. O que a temperature faz com a distribuição de probabilidade? O que acontece em `T → 0`?
4. Por que um bigram model nunca vai escrever uma frase coerente, por mais que treine?
5. Qual a diferença entre o token embedding de um LLM e o sentence embedding usado em RAG?
6. Por que um texto em português custa mais tokens que o equivalente em inglês?

## Entregável da semana

Notebook resolvido e amostras geradas em três temperatures, com uma frase comparando-as.

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/02-Phase-2-Deep-Learning/notebooks/semana_06_exercicio.ipynb) · [`semana_06_exercicio.ipynb`](../notebooks/semana_06_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/02-Phase-2-Deep-Learning/notebooks/semana_06_gabarito.ipynb) · [`semana_06_gabarito.ipynb`](../notebooks/semana_06_gabarito.ipynb) |

[← Semana 5](/02-Phase-2-Deep-Learning/docs/Semana-05-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 7 →](/02-Phase-2-Deep-Learning/docs/Semana-07-Conceitos.md)
