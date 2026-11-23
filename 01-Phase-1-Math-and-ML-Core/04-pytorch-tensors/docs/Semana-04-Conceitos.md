# Semana 4 — A Ponte para Tensors: PyTorch e Autograd

**Fase 1: Math Foundations e ML Core** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Trocar NumPy por PyTorch entendendo o que muda de fato: tensors vivem em um device (CPU ou GPU), têm dtype explícito e carregam o histórico das operações, o que permite ao autograd calcular todos os gradients da Semana 3 automaticamente. Ao final você terá o training loop canônico de cinco passos na ponta dos dedos.

## Por que isso importa para um AI Architect

Todo o ecossistema de open-weight LLMs (Hugging Face `transformers`, `peft`, vLLM) é construído sobre PyTorch tensors. Erros de `device`, `dtype`, out-of-memory na GPU e graphs que não são liberados são os problemas operacionais mais comuns em fine-tuning e inference. Saber ler `shape`, `dtype` e `device` de um tensor resolve metade dos stack traces que você vai encontrar.

## Conceitos

### 1. Tensor vs. NumPy array

> **Em palavras simples.** Um **tensor** é uma tabela de números com qualquer quantidade de dimensões: uma lista (1 dimensão), uma planilha (2), uma pilha de planilhas (3) e assim por diante. É a mesma ideia do array do NumPy, com dois poderes a mais: pode morar na placa de vídeo (**GPU**), onde as contas são muito mais rápidas, e lembra de todas as operações feitas com ele, o que permite calcular gradients automaticamente.

| | `np.ndarray` | `torch.Tensor` |
|---|---|---|
| Onde vive | RAM (CPU) | CPU ou GPU (`.device`) |
| Differentiation | Não | Sim (`requires_grad=True`) |
| Default float dtype | `float64` | `float32` |
| Conversão | `torch.from_numpy(a)` compartilha memória na CPU | `t.numpy()` (só CPU, sem grad); `t.cpu().numpy()` a partir da GPU |

Três atributos para olhar sempre que algo falhar: `t.shape`, `t.dtype`, `t.device`.

`torch.from_numpy(a)` não copia: alterar o array altera o tensor. `torch.tensor(a)` copia.

> **No mundo real.** Para treinar um classificador de imagens no ImageNet, cada batch é um único tensor de shape `(256, 3, 224, 224)`: 256 imagens, 3 canais de cor, 224 × 224 pixels. São cerca de 38,5 milhões de números em um só objeto, processados em uma única chamada. Um batch de texto para um LLM é um tensor de shape `(batch, sequence_length)` com os IDs dos tokens. Todo dado, seja imagem, áudio ou texto, vira tensor antes de entrar em uma rede.

### 2. dtype

> **Em palavras simples.** O **dtype** é o tipo de número guardado no tensor e define quanta memória cada número ocupa e com quanta precisão. `float32` usa 4 bytes por número; `float16` usa 2 e guarda menos casas decimais. É como escolher entre anotar um preço com centavos ou arredondado: ocupa menos espaço e, para a maioria das contas, o resultado é o mesmo. Para modelos com bilhões de números, essa escolha decide se o modelo cabe ou não na máquina.

| dtype | Bytes | Uso |
|---|---|---|
| `float32` | 4 | Padrão de treino |
| `float16` / `bfloat16` | 2 | Mixed precision e inference de LLMs |
| `int64` (`long`) | 8 | Token IDs e class labels (exigido por `CrossEntropyLoss` e `nn.Embedding`) |
| `bool` | 1 | Masks (attention mask, causal mask) |

`bfloat16` tem a mesma faixa de expoente do `float32` com menos precisão na mantissa; é mais estável que `float16` para treino, mas depende de suporte do hardware (GPUs mais novas). Na T4 do Colab, use `float16`.

Erros clássicos: passar labels como `float` para `CrossEntropyLoss`, ou índices como `int32` para `nn.Embedding`.

> **No mundo real.** O **Llama 2 7B**, da Meta, tem cerca de 6,74 bilhões de parâmetros. Em `float32` isso são aproximadamente 27 GB só de pesos; em `float16`, cerca de 13,5 GB. É por isso que os pesos são distribuídos em 16 bits e que o modelo cabe, por pouco, em uma GPU de 16 GB como a T4 do Colab. Em 4 bits (Semana 20), cai para algo em torno de 4 GB e roda em um laptop.

### 3. Device

> **Em palavras simples.** O **device** é onde o tensor mora: na memória comum do computador (**CPU**) ou na memória da placa de vídeo (**GPU**). A GPU é como uma fábrica com milhares de operários fazendo contas simples ao mesmo tempo, ideal para multiplicar tabelas enormes. A regra prática: tudo o que participa de uma conta precisa estar no mesmo lugar. Misturar os dois é o erro mais comum de quem começa.

`x.to("cuda")` copia o tensor para a VRAM. Toda operação exige que os operandos estejam no mesmo device; caso contrário: `RuntimeError: Expected all tensors to be on the same device`. O padrão idiomático:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device); x = x.to(device)
```

Note a assimetria: `model.to(device)` move o módulo in-place; `x.to(device)` devolve um tensor novo e precisa ser reatribuído.

GPUs ganham em operações grandes e paralelas (matrix multiplication). Para tensors pequenos, o overhead de transferência CPU→GPU supera o ganho; por isso o notebook mede o ponto de virada.

> **No mundo real.** A **AlexNet** (2012) foi treinada em **duas placas GTX 580 de 3 GB**, placas de videogame da época, durante cinco a seis dias. O modelo não cabia em uma só, e os autores dividiram a rede entre as duas. O paper registra que o tamanho da rede foi limitado pela memória de GPU disponível. Foi o resultado que convenceu a área de que GPUs eram o caminho, e a razão de `.to("cuda")` estar em todo código de Deep Learning desde então.

### 4. Autograd: o computational graph dinâmico

> **Em palavras simples.** **Autograd** é o contador automático do PyTorch. Enquanto você faz as contas com os tensores, ele anota cada operação em um diagrama (o **computational graph**). Quando você pede `loss.backward()`, ele percorre esse diagrama de trás para frente e calcula sozinho todos os gradients que, na Semana 3, você calculou à mão. Você descreve a conta; ele cuida da derivada.

Quando um tensor tem `requires_grad=True`, cada operação sobre ele cria um nó no graph, registrado em `.grad_fn`. Os tensors que você cria são **leaf tensors**; os resultados de operações são intermediários. `loss.backward()` percorre o graph de trás para frente aplicando a chain rule (reverse-mode automatic differentiation) e deposita o resultado no atributo `.grad` de cada leaf.

O graph é **dinâmico** (define-by-run): é reconstruído a cada forward pass, o que permite `if`, loops e shapes variáveis no modelo. Depois do `backward()` ele é descartado.

Detalhes que causam bugs:

- **Gradients accumulate.** `.backward()` **soma** em `.grad`. É preciso zerar a cada step (`optimizer.zero_grad()`). Esse comportamento não é acidente: é o que permite **gradient accumulation**, simular um batch grande somando gradientes de vários batches pequenos antes de dar o step (útil quando a VRAM é curta).
- **`torch.no_grad()`** desliga o registro do graph: use em evaluation e inference. Economiza memória e tempo. `torch.inference_mode()` é a versão ainda mais enxuta.
- **`.item()`** extrai um número Python de um tensor escalar. Guardar o tensor `loss` em uma lista (`history.append(loss)`) mantém o graph inteiro vivo e produz um memory leak lento; guarde `loss.item()`.
- **`.detach()`** devolve um tensor que compartilha os dados mas está fora do graph.
- **In-place operations** em tensors que participam do graph (`x += 1`, `x.relu_()`) podem invalidar o cálculo do gradiente. Atualizações de parâmetros feitas à mão precisam estar dentro de `torch.no_grad()`.

> **No mundo real.** Em janeiro de 2020, a **OpenAI** anunciou que estava padronizando todo o seu desenvolvimento em PyTorch, citando a facilidade de testar ideias novas. Uma parte central dessa facilidade é o autograd dinâmico: o pesquisador escreve o modelo como código Python comum, com `if` e `for`, e os gradients saem automaticamente. O design está descrito em *PyTorch: An Imperative Style, High-Performance Deep Learning Library* (Paszke et al., 2019).

### 5. `nn.Module` e `nn.Parameter`

> **Em palavras simples.** Um `nn.Module` é uma peça de modelo: uma caixa que guarda seus próprios parâmetros e sabe o que fazer com uma entrada. Peças se encaixam dentro de peças, como blocos de montar, até formar o modelo inteiro. Um `nn.Parameter` é um número que o treino tem permissão para ajustar. Organizar o modelo assim permite pedir, em uma linha, "me dê todos os parâmetros" ou "mova tudo para a GPU".

`nn.Parameter` é um tensor com `requires_grad=True` que, quando atribuído a um `nn.Module`, é registrado automaticamente. É isso que faz `model.parameters()`, `model.to(device)` e `model.state_dict()` funcionarem.

`nn.Linear(in_features, out_features)` guarda `weight` com shape `(out, in)` e `bias` com shape `(out,)`, e calcula `x @ weight.T + bias`. A entrada precisa ter a última dimensão igual a `in_features`: por isso o `unsqueeze(1)` no notebook, que transforma `(N,)` em `(N, 1)`.

> **No mundo real.** O `bert-base-uncased`, um dos modelos mais baixados do Hugging Face Hub, é um `nn.Module` composto de 12 blocos Transformer, cada um feito de módulos menores (`nn.Linear`, `nn.LayerNorm`, `nn.Embedding`). A soma `sum(p.numel() for p in model.parameters())` devolve cerca de **110 milhões**, o número que aparece no paper. Qualquer modelo da biblioteca `transformers`, do DistilBERT ao Llama, obedece à mesma estrutura, e é por isso que o mesmo código de treino serve para todos.

### 6. O training loop canônico

> **Em palavras simples.** O **training loop** é a rotina que se repete milhares de vezes para ensinar o modelo, sempre em cinco passos: o modelo dá um palpite (**forward**), mede-se o erro (**loss**), apagam-se as anotações da rodada anterior (`zero_grad`), calcula-se quanto cada parâmetro contribuiu para o erro (**backward**) e ajusta-se cada parâmetro um pouco (`step`). É como treinar um arremesso: lançar, ver onde caiu, corrigir o movimento, repetir.

```python
pred = model(x)                  # 1. forward pass
loss = criterion(pred, y)        # 2. loss
optimizer.zero_grad()            # 3. zera os gradients do step anterior
loss.backward()                  # 4. backward pass: calcula os gradients
optimizer.step()                 # 5. atualiza os pesos
```

Esses cinco passos são os mesmos para uma linear regression e para o fine-tuning de um LLM de 8 bilhões de parâmetros. O `Trainer` da Hugging Face é este loop com logging, checkpoints e schedules em volta.

`torch.optim.SGD` implementa `θ ← θ − η∇L`; `Adam` e `AdamW` adaptam o step por parâmetro (Semana 3).

> **No mundo real.** O repositório **nanoGPT**, de Andrej Karpathy, reproduz o GPT-2 (124 milhões de parâmetros) treinando no dataset OpenWebText, e o coração do `train.py` são esses mesmos cinco passos, em poucas dezenas de linhas. O README informa que o treino completo leva cerca de quatro dias em oito GPUs A100. O loop que você escreve nesta semana para uma reta com dois parâmetros é, na estrutura, o mesmo que treina um modelo de linguagem.

### 7. Cronometrando na GPU

> **Em palavras simples.** Quando o Python manda uma conta para a GPU, ele não espera o resultado: entrega o pedido e segue em frente. Se você olhar o relógio logo depois, terá medido o tempo de **fazer o pedido**, não o de **fazer a conta**. `torch.cuda.synchronize()` é a instrução "espere a GPU terminar", necessária antes de qualquer medição de tempo honesta.

Chamadas CUDA são **assíncronas**: a função Python retorna antes de a GPU terminar. Para medir tempo corretamente, chame `torch.cuda.synchronize()` antes de ler o relógio. Faça também um **warm-up**: a primeira chamada inclui inicialização de contexto e alocação.

> **No mundo real.** O **MLPerf**, benchmark da indústria mantido pelo consórcio MLCommons e usado por NVIDIA, Google, Intel e outros para comparar hardware de IA, define regras rígidas de medição exatamente por isso. No MLPerf Training, o que se mede é o tempo total para treinar um modelo até uma qualidade-alvo, de ponta a ponta, e não o tempo de uma operação isolada. Medições ingênuas de GPU costumam errar por ordens de grandeza.

### 8. Memória de GPU

> **Em palavras simples.** A GPU tem a sua própria memória, a **VRAM**, bem menor que a do computador, e tudo precisa caber nela: os pesos do modelo, os dados do batch e todas as contas intermediárias. Quando não cabe, o programa para com o erro `CUDA out of memory`. Quase sempre a causa é uma de quatro: batch grande demais, textos longos demais, resultados antigos que ficaram guardados sem necessidade, ou avaliação feita sem desligar o cálculo de gradients.

- `torch.cuda.memory_allocated()`: memória ocupada por tensors vivos.
- `torch.cuda.memory_reserved()`: memória que o allocator do PyTorch reservou do driver (inclui cache). É o que o `nvidia-smi` mostra.
- `torch.cuda.max_memory_allocated()`: o pico, o número que interessa para dimensionar.
- `del x; torch.cuda.empty_cache()`: libera a referência e devolve o cache ao driver.

Um `CUDA out of memory` em treino vem, em ordem de frequência, de: batch size, sequence length, guardar tensors com graph, e não usar `no_grad` na evaluation.

> **No mundo real.** A GPU **T4** do Google Colab tem 16 GB de VRAM. Um modelo de 7 bilhões de parâmetros em `float16` ocupa cerca de 13,5 GB só com os pesos, sobrando pouco mais de 2 GB para todo o resto. Dá para gerar texto com batch pequeno; não dá para treinar. É essa conta que motiva as técnicas da Fase 4: carregar o modelo em 4 bits (cerca de 4 GB) e treinar só um adapter pequeno, o que torna o fine-tuning possível na mesma placa.

### 9. Reproducibility

> **Em palavras simples.** **Reproducibility** é conseguir repetir um experimento e obter o mesmo resultado. Treinar um modelo envolve vários sorteios: os pesos iniciais, a ordem dos dados, o que o dropout desliga. A **seed** é o número que fixa esses sorteios, como combinar de antemão o resultado de todos os dados que serão lançados. Sem seed fixa, não dá para saber se uma melhora veio da sua ideia ou da sorte.

`torch.manual_seed(seed)` fixa a geração de números aleatórios. Resultados ainda podem diferir entre CPU e GPU e entre versões, porque algumas operações CUDA são não determinísticas por padrão. Para comparação rigorosa existe `torch.use_deterministic_algorithms(True)`, com custo de velocidade.

> **No mundo real.** O paper *Deep Reinforcement Learning that Matters* (Henderson et al., 2018) rodou os mesmos algoritmos, com o mesmo código e os mesmos hiperparâmetros, mudando apenas a seed, e mostrou que dois grupos de cinco execuções podiam produzir curvas de resultado tão diferentes que pareciam métodos distintos. O trabalho ajudou a motivar o **reproducibility checklist** que a conferência NeurIPS passou a exigir dos autores a partir de 2019.

## Erros comuns

- Esquecer `optimizer.zero_grad()`: os gradients acumulam e o treino diverge.
- Avaliar o modelo sem `torch.no_grad()`: funciona, mas gasta memória à toa e pode causar OOM.
- `x.to(device)` sem reatribuir.
- Acumular `loss` em vez de `loss.item()`.
- Chamar `.numpy()` em um tensor na GPU ou com grad: use `.detach().cpu().numpy()`.
- Medir tempo de GPU sem `synchronize()` e concluir que a GPU é milhares de vezes mais rápida.

## Leitura e vídeos (fim de semana)

- PyTorch Tutorials, *Learn the Basics*: *Tensors*; *Automatic Differentiation with torch.autograd*; *Optimizing Model Parameters*.
- PyTorch documentation: *Autograd mechanics* (Notes); *CUDA semantics* (Notes); a página de `torch.Tensor`.
- PyTorch Tutorials: *Deep Learning with PyTorch: A 60 Minute Blitz*.
- Opcional: Andrej Karpathy, *micrograd* (repositório): um autograd engine completo em 150 linhas, a melhor forma de entender o que o PyTorch faz por baixo.

## Perguntas de autoavaliação

1. O que acontece se você esquecer `optimizer.zero_grad()`? E em que situação acumular gradients é desejado?
2. Por que avaliar o modelo dentro de `with torch.no_grad():`?
3. Você recebe `RuntimeError: Expected all tensors to be on the same device`. Qual a causa e a correção?
4. Por que `float32` e não `float64` é o padrão em Deep Learning? E por que LLMs usam `float16` ou `bfloat16` em inference?
5. Qual a diferença entre `memory_allocated()` e o que o `nvidia-smi` mostra?
6. O que é um leaf tensor e onde ficam armazenados os gradients depois de `backward()`?

## Entregável da semana

Notebook resolvido, a comparação de tempo CPU vs. GPU registrada em `docs/`, e o **commit de fechamento da Fase 1**.

### Checkpoint da Fase 1

Você deve conseguir, sem consultar: escrever a fórmula de cosine similarity; explicar o que `loss.backward()` faz; escrever os cinco passos do training loop; desenhar a confusion matrix e dizer qual célula custa mais em um caso de uso que você escolher.

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/ai-architect-foundations/blob/main/01-Phase-1-Math-and-ML-Core/04-pytorch-tensors/notebooks/semana_04_exercicio.ipynb) · [`semana_04_exercicio.ipynb`](../notebooks/semana_04_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/ai-architect-foundations/blob/main/01-Phase-1-Math-and-ML-Core/04-pytorch-tensors/notebooks/semana_04_gabarito.ipynb) · [`semana_04_gabarito.ipynb`](../notebooks/semana_04_gabarito.ipynb) |

[← Semana 3](/01-Phase-1-Math-and-ML-Core/03-calculus-gradient/docs/Semana-03-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 5 →](/02-Phase-2-Deep-Learning/docs/Semana-05-Conceitos.md)
