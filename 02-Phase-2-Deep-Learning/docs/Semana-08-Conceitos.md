# Semana 8 — Montando um Mini-GPT

**Fase 2: Deep Learning e o Ecossistema Hugging Face** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Juntar as peças das Semanas 6 e 7 em um Transformer decoder completo (um GPT em miniatura), treiná-lo e gerar texto. Depois desta semana, a arquitetura de um Llama ou Qwen deixa de ser black box: muda a escala, não a estrutura.

## Por que isso importa para um AI Architect

Ler um `config.json` (`hidden_size`, `num_hidden_layers`, `num_attention_heads`, `intermediate_size`, `max_position_embeddings`, `vocab_size`) e estimar parâmetros, memória e custo é rotina de quem escolhe modelos. E saber onde ficam as matrizes de cada block é pré-requisito para LoRA, quantization e para entender o que um inference server otimiza.

## Conceitos

### 1. O Transformer block

> **Em palavras simples.** O **Transformer block** é a unidade que se repete para formar o modelo. Cada bloco faz duas coisas em sequência: primeiro as palavras trocam informação entre si (attention), depois cada palavra processa o que recebeu (MLP). Um modelo de linguagem é uma pilha de dezenas desses blocos idênticos; a cada andar, a representação do texto fica um pouco mais refinada.

Cada block tem duas sublayers, cada uma envolvida por uma residual connection:

```
x = x + Attention(Norm(x))      # comunicação entre tokens
x = x + MLP(Norm(x))            # processamento de cada token
```

- **Attention** move informação entre posições.
- **MLP (feed-forward network, FFN)** processa cada posição isoladamente: expande para `4·d` (`intermediate_size`), aplica `GELU` e volta para `d`. É onde está a maior parte dos parâmetros e, segundo a pesquisa de interpretability, onde boa parte do conhecimento factual fica armazenada.

> **No mundo real.** O **GPT-2 small** empilha 12 blocos; o **GPT-3** (2020), 96 blocos, com 175 bilhões de parâmetros. A estrutura de cada bloco é essencialmente a mesma nos dois. A diferença de capacidade entre um modelo que completa frases e um que escreve ensaios veio de repetir mais vezes o mesmo bloco, torná-lo mais largo e treiná-lo com mais dados, não de uma arquitetura nova.

### 2. Residual connections e o residual stream

> **Em palavras simples.** Uma **residual connection** é um atalho: a entrada de cada etapa é somada à sua saída, de modo que a informação original nunca se perde, mesmo que a etapa faça pouco. Dá para imaginar uma esteira que atravessa a fábrica inteira; cada estação pega o que precisa da esteira e coloca de volta a sua contribuição. Essa esteira é o **residual stream**, e é o que permite treinar redes muito profundas.

`x + f(x)` cria um caminho direto para o gradiente da saída até a entrada. Sem isso, redes com dezenas de layers não treinam (vanishing gradient, Semana 3).

Uma forma útil de pensar: existe um **residual stream** de dimensão `d` que atravessa o modelo inteiro. Cada sublayer **lê** dele, calcula algo e **escreve** de volta por soma. O token embedding é a primeira escrita; o LM head lê o resultado final. Isso explica por que técnicas como LoRA funcionam somando pequenas atualizações e por que é possível inspecionar hidden states de layers intermediárias.

> **No mundo real.** A técnica **logit lens** (2020) explora essa esteira: aplicando a última etapa do GPT-2 ao conteúdo de camadas intermediárias, é possível ler o "palpite parcial" do modelo em cada andar. O que se observa é a previsão se formando aos poucos: nas primeiras camadas, um palpite vago; nas últimas, a palavra final. Só é possível porque todas as camadas escrevem no mesmo espaço compartilhado. A técnica é base de boa parte da pesquisa em interpretabilidade.

### 3. Normalization

> **Em palavras simples.** **Normalization** é um regulador de volume colocado entre as etapas. Sem ele, os números poderiam crescer ou encolher sem controle de camada em camada, e o treino ficaria instável. A `LayerNorm` reajusta os valores de cada palavra para uma faixa padrão antes de entregá-los à etapa seguinte.

`LayerNorm` normaliza cada vetor para média 0 e variância 1, com ganho e bias aprendidos. Diferente de `BatchNorm`, não depende do batch, o que é essencial para sequências de tamanhos variáveis.

- **Post-norm** (paper original): `x = Norm(x + f(x))`. Instável em redes profundas; exige warmup cuidadoso.
- **Pre-norm** (GPT-2 em diante): `x = x + f(Norm(x))`. O residual stream nunca é normalizado diretamente; treino mais estável. É o padrão.
- **RMSNorm** (Llama, Qwen, Gemma): só divide pela root mean square, sem centrar. Mais barato e igualmente eficaz.

> **No mundo real.** O **GPT-2** mudou a posição da normalização em relação ao Transformer original: passou a aplicá-la **antes** de cada subcamada (**pre-norm**), o que tornou o treino de modelos profundos mais estável e virou padrão. A família **Llama** foi além e trocou a `LayerNorm` pela **RMSNorm**, uma versão simplificada que faz menos contas com o mesmo efeito prático. São detalhes pequenos que aparecem em todo `config.json` atual.

### 4. O modelo completo

> **Em palavras simples.** O caminho completo de um texto dentro do modelo: os tokens viram vetores (embedding), recebem o carimbo de posição, atravessam a pilha de blocos e, no fim, uma última etapa (o **LM head**) converte o resultado em uma nota para cada token do vocabulário. A nota mais alta indica a continuação mais provável.

```
token IDs → token embedding (+ position) → [Block] × N → final Norm → LM head (Linear d → V) → logits
```

O **LM head** frequentemente compartilha pesos com a embedding table (**weight tying**): a mesma matriz `V × d` que transforma token em vetor é usada, transposta, para transformar vetor em logits. Economiza `V·d` parâmetros, o que em modelos pequenos é uma fração grande.

> **No mundo real.** No **GPT-2**, a tabela usada para transformar tokens em vetores na entrada é a mesma usada, ao contrário, para transformar vetores em tokens na saída. Esse reaproveitamento, chamado **weight tying** (Press e Wolf, 2017), economiza 38,6 milhões de parâmetros no modelo de 124 milhões e melhora a qualidade. É uma das razões de modelos pequenos conseguirem ter vocabulários grandes.

### 5. Contando parâmetros

> **Em palavras simples.** O tamanho de um modelo é o número de **parâmetros**, os valores ajustados no treino. Dá para estimá-lo com três informações: a largura dos vetores, o número de blocos e o tamanho do vocabulário. A conta mostra onde está o peso: dobrar a largura quadruplica os parâmetros; dobrar o número de blocos apenas os dobra. Com o total, estima-se quanta memória o modelo exige.

Para dimensão `d`, `N` layers e vocabulário `V` (ignorando bias e normas):

- Embeddings: `V · d` (uma vez, com weight tying)
- Por block: attention `≈ 4d²` (Q, K, V, O) + MLP `≈ 8d²` (`d → 4d → d`) = `≈ 12d²`
- Total `≈ V·d + 12·N·d²`

GPT-2 small: `d=768, N=12, V=50257` → 38,6M + 85M ≈ 124M. Confere.

A fórmula assume `intermediate_size = 4d`. Modelos recentes variam: usam activation com gate (SwiGLU, três matrizes no MLP) e `intermediate_size` diferente. Confira no `config.json`. A regra geral continua valendo: os parâmetros crescem **linearmente com `N`** e **quadraticamente com `d`**.

Memória dos pesos: `parâmetros × bytes por parâmetro` (4 em fp32, 2 em fp16, cerca de 0,5 em 4-bit).

> **No mundo real.** Quando a OpenAI lançou o **GPT-2**, o menor modelo foi anunciado com 117 milhões de parâmetros. A contagem estava errada: o número real é **124 milhões**, e a própria OpenAI corrigiu a informação depois. A fórmula desta semana (`V·d + 12·N·d²`, com 50.257 tokens, largura 768 e 12 blocos) dá 124 milhões. É um bom exemplo de por que vale saber fazer a conta em vez de confiar no número do anúncio.

### 6. Hyperparameters e o que cada um custa

> **Em palavras simples.** Os **hyperparameters** de arquitetura são as medidas do modelo, definidas antes do treino: a largura dos vetores, o número de blocos, o número de heads e o tamanho máximo do texto. Cada uma compra um tipo de capacidade e cobra um preço diferente em memória e em tempo. Escolher um modelo é, em boa parte, ler essas medidas.

| Hyperparameter | Efeito | Custo |
|---|---|---|
| `n_embd` / `hidden_size` (d) | Capacidade por token | Parâmetros crescem com `d²` |
| `n_layer` | Profundidade do processamento | Linear em parâmetros e em tempo |
| `n_head` | Diversidade de relações | Quase grátis |
| `block_size` / context length | Quanto o modelo enxerga | Memória de attention cresce com `T²` |
| `vocab_size` | Granularidade dos tokens | `V·d` parâmetros; softmax mais caro |

> **No mundo real.** A família **GPT-2** foi publicada em quatro tamanhos, variando só essas medidas: *small* (12 blocos, largura 768, 124 milhões de parâmetros), *medium* (24 blocos, 1.024, 355 milhões), *large* (36 blocos, 1.280, 774 milhões) e *XL* (48 blocos, 1.600, 1,5 bilhão). Mesma arquitetura, mesmo código, mesmos dados. A qualidade do texto gerado melhora de forma visível a cada degrau.

### 7. Variantes que você encontrará em modelos atuais

> **Em palavras simples.** A receita do Transformer de 2017 foi recebendo ajustes. Os modelos atuais trocaram algumas peças por versões mais eficientes: outra forma de marcar posições (**RoPE**), outra função de ativação (**SwiGLU**), outra normalização (**RMSNorm**) e uma attention que reaproveita parte das contas (**Grouped-Query Attention**). O desenho geral, uma pilha de blocos em volta de uma esteira, continua o mesmo.

- **RoPE** em vez de learned positions (Semana 7).
- **SwiGLU** em vez de `GELU` no MLP.
- **Grouped-Query Attention (GQA):** várias query heads compartilham as mesmas key/value heads, reduzindo o tamanho do KV cache.
- **Flash Attention:** implementação que evita materializar a attention matrix inteira; mesma matemática, muito menos memória.

A estrutura de blocks com residual stream é a mesma em todos.

> **No mundo real.** O **Llama 2** (Meta, 2023) e o **Mistral 7B** (2023) adotam esse conjunto: pre-norm com RMSNorm, SwiGLU e RoPE; o Mistral 7B e as versões maiores do Llama usam também Grouped-Query Attention, que reduz a memória gasta durante a geração. Ao abrir o `config.json` de quase qualquer modelo aberto lançado desde então, você encontra as mesmas escolhas. Conhecer esse "pacote padrão" permite ler um modelo novo em minutos.

### 8. Detalhes de treino que fazem diferença

> **Em palavras simples.** Treinar um Transformer exige alguns cuidados além do loop básico: começar com passos pequenos e ir aumentando (**warmup**), reduzir o passo aos poucos até o fim (**decay**), impedir passos exageradamente grandes (**gradient clipping**) e sortear os valores iniciais em uma escala adequada. Sem eles, é comum o treino desandar logo nos primeiros minutos.

- **Initialization:** pesos com desvio 0,02. Com weight tying e inicialização padrão `N(0, 1)`, os logits iniciais ficam enormes e a loss começa em dezenas em vez de `ln(V)`. É o sanity check da Semana 5 em ação.
- **Gradient clipping** (`clip_grad_norm_`, norma máxima 1,0): evita que um batch ruim destrua o modelo.
- **AdamW** com weight decay e, em treinos longos, **warmup** seguido de cosine decay.
- **Train vs. validation loss:** em um corpus pequeno, o mini-GPT decora rapidamente; a validation loss mostra quando.

> **No mundo real.** O paper do **GPT-3** (Brown et al., 2020) lista a receita usada em todos os tamanhos do modelo: otimizador Adam, gradient clipping com norma máxima 1,0, warmup no início e decaimento da learning rate em forma de cosseno, além de weight decay. São praticamente os mesmos itens que você configura no notebook desta semana e, mais adiante, no fine-tuning. A receita mudou muito pouco desde então.

### 9. Generation

> **Em palavras simples.** O modelo só enxerga uma janela limitada de texto, a **context window**. Tudo o que fica fora dela simplesmente não existe para ele. Em uma conversa longa, as primeiras mensagens saem da janela e são esquecidas, a não ser que o sistema as resuma ou as busque de volta. O tamanho dessa janela é uma das especificações mais importantes de um modelo.

O modelo só enxerga os últimos `block_size` tokens: `idx[:, -block_size:]`. O que fica para trás é esquecido. Em LLMs reais, estourar a context window dá erro ou truncamento silencioso, dependendo do servidor.

Sem KV cache (como no notebook), cada novo token reprocessa a sequência inteira: gerar `n` tokens custa `O(n²)` forwards de token. O cache resolve isso (Semana 23).

> **No mundo real.** O **GPT-2** enxerga 1.024 tokens. O modelo do ChatGPT no lançamento, o GPT-3.5, trabalhava com cerca de 4 mil tokens, algo como seis páginas de texto. Modelos atuais oferecem de 128 mil a mais de um milhão. Esse crescimento mudou o que se pode construir: de "responder a uma pergunta curta" para "analisar um contrato inteiro". Mas o custo cresce junto, e é por isso que RAG (Fase 3) continua necessário.

### 10. Encoder, decoder, encoder-decoder

> **Em palavras simples.** Há três famílias de Transformer. O **encoder** lê o texto inteiro, olhando para os dois lados, e produz uma representação: serve para classificar e para buscar. O **decoder** lê só para trás e escreve a continuação: serve para gerar. O **encoder-decoder** combina os dois: um lado lê, o outro escreve, como em tradução. A escolha depende de você precisar entender ou produzir texto.

| Família | Mask | Objetivo de pre-training | Exemplos | Uso |
|---|---|---|---|---|
| Encoder-only | Bidirecional | Masked language modeling (MLM) | BERT, RoBERTa, modelos de embedding | Classificação, retrieval, reranking, NER |
| Decoder-only | Causal | Next-token prediction | GPT, Llama, Qwen, Gemma, Mistral | Geração; hoje, quase tudo |
| Encoder-decoder | Ambas + cross-attention | Denoising / seq2seq | T5, BART | Tradução, sumarização |

Decoder-only dominou porque uma única arquitetura e um único objetivo escalam bem e cobrem qualquer tarefa formulada como texto. Encoders continuam sendo a escolha certa quando se quer um vetor ou uma classe, com custo muito menor.

> **No mundo real.** Em outubro de 2019, o **Google** anunciou que passara a usar o **BERT**, um encoder, para interpretar as buscas, e que a mudança afetava cerca de 1 em cada 10 consultas em inglês nos EUA. O exemplo divulgado: em *"2019 brazil traveler to usa need a visa"*, o sistema antigo ignorava a palavra *to* e devolvia resultados sobre americanos viajando ao Brasil; com o BERT, passou a entender a direção da viagem.

### 11. Scaling

> **Em palavras simples.** **Scaling** é a observação de que modelos de linguagem melhoram de forma previsível quando se aumentam, juntos, três recursos: o tamanho do modelo, a quantidade de dados e o poder computacional. Não basta aumentar um só. Um modelo enorme com poucos dados é desperdício; há uma proporção certa entre os três.

Em nível prático: mais parâmetros, mais dados e mais compute melhoram a loss de forma previsível (**scaling laws**). Para um arquiteto, a consequência é saber que modelos menores, bem ajustados a uma tarefa estreita (Fase 4), muitas vezes igualam modelos grandes genéricos naquela tarefa, por uma fração do custo.

> **No mundo real.** O paper **Chinchilla** (Hoffmann et al., DeepMind, 2022) testou essa proporção. Com o mesmo orçamento de computação do modelo Gopher, de 280 bilhões de parâmetros, treinou um modelo quatro vezes menor (70 bilhões) com quatro vezes mais dados (1,4 trilhão de tokens). O modelo menor venceu em praticamente todas as avaliações. A conclusão mudou a prática da indústria: muitos modelos grandes da época estavam subtreinados.

## Erros comuns

- Loss inicial muito acima de `ln(V)`: initialization ou weight tying.
- Esquecer de recortar o contexto a `block_size` na geração (erro de índice no position embedding).
- Estimar memória só pelos pesos, ignorando activations e KV cache.
- Assumir `intermediate_size = 4d` sem conferir o `config.json`.
- Comparar "tamanho" de modelos só pelo número de parâmetros, ignorando context length e vocab.

## Leitura e vídeos (fim de semana)

- Andrej Karpathy, *Let's build GPT: from scratch, in code, spelled out* (segunda metade) e o repositório `nanoGPT`.
- Jay Alammar, *The Illustrated GPT-2*.
- 3Blue1Brown, *Neural Networks*, Chapter 7 (*How might LLMs store facts*), sobre o papel do MLP.
- Elhage et al., *A Mathematical Framework for Transformer Circuits* (seção sobre residual stream), para a visão de leitura e escrita.
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2 paper), Section 2.
- Opcional: Touvron et al., *LLaMA: Open and Efficient Foundation Language Models*, Section 2 (pre-norm, SwiGLU, RoPE).

## Perguntas de autoavaliação

1. Qual sublayer troca informação entre tokens e qual não?
2. O que acontece com o treino se você remover as residual connections de um modelo de 12 layers?
3. Estime os parâmetros de um modelo com `d=2048`, `N=24`, `V=32000`. Quanto ocupa em fp16?
4. Por que um embedding model costuma ser um encoder e um chatbot um decoder?
5. O que é weight tying e por que afeta a loss inicial?
6. Qual a diferença entre pre-norm e post-norm?

## Perguntas do entrevistador

1. Ao analisar um Transformer decoder em produção, quais mecanismos (residual connections, LayerNorm pré-atenção e weight tying) evitam o colapso dos gradientes e reduzem o tamanho do modelo? Como o tamanho da janela de contexto e o número de heads afetam o orçamento de VRAM?

## Entregável da semana

Notebook resolvido, a loss curve e uma amostra de texto do seu mini-GPT comparada à do bigram da Semana 6.

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/02-Phase-2-Deep-Learning/notebooks/semana_08_exercicio.ipynb) · [`semana_08_exercicio.ipynb`](../notebooks/semana_08_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/02-Phase-2-Deep-Learning/notebooks/semana_08_gabarito.ipynb) · [`semana_08_gabarito.ipynb`](../notebooks/semana_08_gabarito.ipynb) |

[← Semana 7](/02-Phase-2-Deep-Learning/docs/Semana-07-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 9 →](/02-Phase-2-Deep-Learning/docs/Semana-09-Conceitos.md)
