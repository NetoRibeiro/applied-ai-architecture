# Semana 9 — Hugging Face por Dentro: Tokenizers, Hidden States e Generation

**Fase 2: Deep Learning e o Ecossistema Hugging Face** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Abrir a caixa da biblioteca `transformers`: carregar um model e um tokenizer locais, inspecionar tokens, hidden states e logits, e reproduzir à mão o que `generate()` faz. Você também vai extrair sentence embeddings de um encoder com mean pooling, a peça que alimenta toda a Fase 3.

## Por que isso importa para um AI Architect

Quem só conhece LLMs por API não consegue diagnosticar truncation silenciosa, padding do lado errado, chat template ausente ou out-of-memory na GPU. Esses são os defeitos reais em produção com open-weight models, e nenhum deles levanta exceção: o modelo simplesmente responde pior.

## Conceitos

### 1. O trio: config, tokenizer, model

> **Em palavras simples.** Um modelo publicado vem em três partes. O **config** é a planta: descreve as medidas da arquitetura. O **tokenizer** é o tradutor entre texto e números. Os **weights** são os bilhões de valores aprendidos no treino. As três precisam ser do mesmo modelo, como chave, fechadura e porta. A biblioteca `transformers` baixa e monta tudo a partir do nome do modelo.

Um repositório no Hugging Face Hub contém:

- `config.json`: a arquitetura (as dimensões que você estimou na Semana 8).
- `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`: texto ↔ IDs, special tokens, chat template.
- `model.safetensors`: os pesos. `safetensors` é um formato seguro (não executa código, ao contrário de pickle) e mapeável em memória.
- `generation_config.json`: defaults de `generate()` (temperature, top_p, EOS).

As classes `Auto*` leem o config e instanciam a arquitetura correta:

| Classe | Saída | Uso |
|---|---|---|
| `AutoModel` | Hidden states | Embeddings |
| `AutoModelForCausalLM` | Logits do next token | Geração |
| `AutoModelForSequenceClassification` | Logits por classe | Classificação, reranking, NLI |
| `AutoModelForTokenClassification` | Logits por token | NER |

`from_pretrained` baixa para um cache local (`~/.cache/huggingface`). Argumentos que você usará sempre: `torch_dtype` (ou `dtype`), `device_map="auto"` (distribui as layers entre GPU e CPU via `accelerate`), `quantization_config` (Semana 20). Alguns modelos são **gated** e exigem aceitar a licença e um access token; os desta trilha não.

> **No mundo real.** No **Hugging Face Hub**, a página do `Qwen/Qwen2.5-0.5B-Instruct`, usado nesta trilha, mostra exatamente esses arquivos. O `config.json` informa `hidden_size: 896`, `num_hidden_layers: 24` e `vocab_size: 151936`; o `model.safetensors` tem cerca de 1 GB. Uma linha, `AutoModelForCausalLM.from_pretrained(...)`, lê a planta, constrói a arquitetura e preenche com os pesos. O mesmo comando funciona para qualquer um dos modelos do Hub.

### 2. O que o tokenizer devolve

> **Em palavras simples.** O tokenizer devolve duas listas. A primeira são os números dos tokens (`input_ids`). A segunda (`attention_mask`) marca quais posições são texto de verdade e quais são enchimento (**padding**), colocado para que todos os textos de um lote tenham o mesmo comprimento. Textos longos demais são cortados (**truncation**) sem aviso: o modelo nem fica sabendo que havia mais.

`tokenizer(texts, padding=True, truncation=True, max_length=..., return_tensors="pt")` devolve um `BatchEncoding` com:

- `input_ids`: os token IDs, shape `(B, T)`.
- `attention_mask`: 1 = token real, 0 = padding.

**Padding side:**

- **Encoders:** padding à direita.
- **Decoders em geração:** padding à **esquerda** (`tokenizer.padding_side = "left"`). O modelo gera a partir da última posição; se ela for um `<pad>`, a continuação sai degradada, sem erro nenhum.

Muitos decoders não têm `pad_token`; a convenção é `tokenizer.pad_token = tokenizer.eos_token`.

**Truncation:** texto além de `max_length` (ou do `model_max_length`) é cortado. Em embedding models isso significa que o final de um documento longo simplesmente não entra no vetor.

**Special tokens:** `tokenizer(text)` adiciona BOS/CLS/SEP conforme o modelo (`add_special_tokens=True`). Ao tokenizar um prompt já formatado por um chat template, use `add_special_tokens=False` para não duplicar o BOS.

> **No mundo real.** O **BERT** aceita no máximo 512 tokens. Em aplicações reais com documentos longos (contratos, prontuários, artigos), tudo o que passa desse limite é descartado pelo tokenizer por padrão, e o modelo classifica ou indexa o documento com base apenas no início. É uma fonte clássica de resultados ruins que ninguém percebe, porque nenhum erro aparece. Por isso sistemas de busca dividem os documentos em trechos antes (Semana 13).

### 3. Model outputs

> **Em palavras simples.** Além da resposta final, o modelo pode entregar o que "pensou" em cada andar: os **hidden states**, a representação do texto ao sair de cada bloco. Os primeiros andares capturam aspectos mais superficiais, como classe gramatical; os últimos, o significado e a tarefa. Olhar esses estados intermediários é como ver os rascunhos de quem está resolvendo um problema.

Os modelos devolvem um objeto com campos nomeados:

- `logits`: `(B, T, V)` em um causal LM.
- `hidden_states` (com `output_hidden_states=True`): tupla de `N + 1` tensors `(B, T, d)`: a saída dos embeddings e a de cada block.
- `attentions` (com `output_attentions=True`): os attention weights por layer.
- `past_key_values`: o KV cache.
- `loss`: quando se passam `labels`.

Layers inferiores codificam mais sintaxe e forma; as superiores, mais semântica e a tarefa. O último hidden state é o que alimenta o LM head.

> **No mundo real.** O estudo *BERT Rediscovers the Classical NLP Pipeline* (Tenney, Das e Pavlick, Google, 2019) mediu em que camada do BERT cada tipo de informação linguística aparece. O resultado segue a ordem dos antigos sistemas feitos à mão: primeiro classes de palavras, depois estrutura sintática, depois papéis semânticos e correferência. O modelo reconstruiu por conta própria as etapas que os linguistas computacionais programavam uma a uma.

### 4. De hidden states a sentence embedding

> **Em palavras simples.** O modelo produz um vetor para cada token, mas para comparar frases é preciso um vetor por frase. O **pooling** é a forma de resumir: a mais comum, **mean pooling**, tira a média dos vetores de todos os tokens, ignorando o enchimento. O resultado é o "endereço" da frase inteira no mapa de significados.

Um encoder devolve um vetor por token. Para ter um vetor por frase:

- **Mean pooling:** média dos vetores dos tokens reais. A `attention_mask` é indispensável: sem ela, o padding entra na média e o embedding de uma frase muda conforme o tamanho do batch.

  ```python
  mask = attention_mask.unsqueeze(-1).float()
  emb = (last_hidden_state * mask).sum(1) / mask.sum(1)
  ```
- **CLS pooling:** o vetor do primeiro token (`[CLS]`).
- **Last-token pooling:** o vetor do último token, usado em embedding models baseados em decoders.

O método correto é o que foi usado no treino do modelo; está no model card. Depois, L2 normalization (Semana 2) e comparação por dot product. A biblioteca `sentence-transformers` embrulha tudo isso em `model.encode()`.

> **No mundo real.** O paper **Sentence-BERT** (Reimers e Gurevych, 2019) deu o exemplo que tornou a técnica padrão. Para achar o par mais parecido entre 10.000 frases, o BERT original precisa analisar cada par junto: cerca de 50 milhões de comparações, ou **65 horas** de processamento. Com um vetor por frase, calculado uma única vez, a mesma busca leva cerca de **5 segundos**. É essa diferença que torna a busca semântica viável.

### 5. Decoding strategies

> **Em palavras simples.** Uma **decoding strategy** é a regra para escolher a próxima palavra a partir das chances calculadas pelo modelo. **Greedy** pega sempre a mais provável. **Top-k** sorteia só entre as `k` melhores. **Top-p** sorteia entre as melhores que, juntas, somam uma certa probabilidade. Cada regra muda o caráter do texto: mais seguro e repetitivo, ou mais variado e arriscado.

| Estratégia | Como funciona | Quando usar |
|---|---|---|
| Greedy (`do_sample=False`) | Sempre o token mais provável | Extração, JSON, tarefas determinísticas |
| Temperature | Divide os logits por `T` | Controle geral de variabilidade |
| Top-k | Amostra só entre os `k` mais prováveis | Corta a cauda |
| Top-p (nucleus) | Amostra do menor conjunto com probabilidade acumulada ≥ `p` | Padrão em chat |
| Beam search (`num_beams`) | Mantém as `b` melhores sequências | Tradução, sumarização |
| `repetition_penalty` | Penaliza tokens já gerados | Reduz loops |

Top-p se adapta ao formato da distribuição (poucos candidatos quando o modelo está confiante, muitos quando não está); top-k usa um número fixo. `temperature` só tem efeito com `do_sample=True`.

**Stopping:** `max_new_tokens` (limite de tokens gerados), `eos_token_id` e stop strings. `max_length` inclui o prompt, o que confunde; prefira `max_new_tokens`.

`generate()` devolve prompt + continuação. Para isolar a resposta: `out[0, inputs["input_ids"].shape[1]:]`.

> **No mundo real.** O paper *The Curious Case of Neural Text Degeneration* (Holtzman et al., 2019) mostrou que escolher sempre a continuação mais provável faz o GPT-2 entrar em laços, repetindo a mesma frase indefinidamente, e que sortear livremente produz texto incoerente. Os autores propuseram o **nucleus sampling** (top-p), que sorteia apenas no núcleo de palavras prováveis. É o método padrão das interfaces de chat desde então.

### 6. Chat templates

> **Em palavras simples.** Modelos de conversa foram treinados com um formato exato de roteiro, com marcações que indicam quem está falando: sistema, usuário ou assistente. O **chat template** é a fôrma que coloca a sua conversa nesse formato. Cada família de modelo tem a sua. Se a fôrma estiver errada, o modelo recebe um texto em um formato que nunca viu e responde mal, sem acusar erro.

Instruct models foram treinados com marcadores de papel específicos (`system`, `user`, `assistant`). Cada família tem o seu formato (ChatML com `<|im_start|>`, o formato do Llama, o do Gemma). O template é um Jinja template guardado no tokenizer:

```python
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

`add_generation_prompt=True` acrescenta o cabeçalho do turno do assistant, sinalizando que é a vez de o modelo responder. Usar o template errado, ou concatenar strings à mão, é uma das maiores causas de "o modelo aberto é ruim".

> **No mundo real.** O **Llama 2** espera as mensagens entre marcadores `[INST]` e `[/INST]`; modelos no formato **ChatML**, como o Qwen, usam `<|im_start|>` e `<|im_end|>`. Em 2023, a Hugging Face passou a guardar o template dentro do próprio tokenizer (`apply_chat_template`), justificando a mudança com o que chamou de erros silenciosos: muitos relatos de "este modelo aberto é ruim" eram, na verdade, prompts montados no formato de outro modelo.

### 7. Precisão e memória de GPU

> **Em palavras simples.** A memória que um modelo ocupa é o número de parâmetros multiplicado pelo tamanho de cada um. Guardar cada número com metade da precisão (16 bits em vez de 32) corta a memória pela metade, com perda de qualidade quase imperceptível. É uma conta simples que decide em qual equipamento o modelo roda e quanto custa mantê-lo no ar.

Memória dos pesos ≈ `parâmetros × bytes por parâmetro`:

| Precisão | Bytes | Modelo de 7B |
|---|---|---|
| `float32` | 4 | ~28 GB |
| `float16` / `bfloat16` | 2 | ~14 GB |
| 8-bit | 1 | ~7 GB |
| 4-bit | ~0,5 | ~4 a 5 GB |

A isso se somam activations e KV cache, que crescem com batch size e contexto. `model.get_memory_footprint()` informa o tamanho carregado.

> **No mundo real.** O **Mistral 7B** tem cerca de 7,2 bilhões de parâmetros. Em 32 bits, seriam uns 29 GB; em 16 bits, a forma em que é distribuído, cerca de 14,5 GB, o que cabe em uma única GPU de 16 GB. Em 4 bits, cai para algo em torno de 4 a 5 GB e roda em um notebook comum. A mesma conta, feita antes da compra, evita descobrir depois que o modelo escolhido não cabe no hardware disponível.

### 8. `pipeline`

> **Em palavras simples.** O `pipeline` é o modo automático da biblioteca: você informa a tarefa e o texto, e ele cuida de transformar o texto em números, rodar o modelo e traduzir o resultado. É ótimo para experimentar em minutos. Para um sistema de verdade, costuma-se abrir o capô e controlar cada etapa, porque é ali que se ajustam velocidade, memória e qualidade.

`pipeline("text-generation", model=...)` é o atalho de alto nível: tokeniza, roda e decodifica. É ótimo para protótipo. Em produção e em avaliação, trabalhar com tokenizer e model diretamente dá controle sobre batching, padding, dtype e memória. A lista de tasks suportadas muda entre versões da biblioteca.

> **No mundo real.** `pipeline("sentiment-analysis")`, sem mais nada, baixa por padrão o `distilbert-base-uncased-finetuned-sst-2-english`, um modelo treinado em críticas de cinema (dataset SST-2) com cerca de 91% de accuracy. Em três linhas de código tem-se um classificador de sentimento funcionando. O mesmo exemplo mostra o limite: é um modelo em inglês, de um domínio específico; aplicá-lo a avaliações de clientes em português exige outro modelo ou fine-tuning.

## Erros comuns

- Padding à direita em geração em batch com decoder.
- Mean pooling sem `attention_mask`.
- Montar o prompt de um instruct model sem o chat template.
- Duplicar o BOS ao tokenizar um prompt já formatado.
- Usar `max_length` achando que é o limite de tokens gerados.
- Ignorar o aviso de truncation e indexar documentos maiores que o `max_seq_length` do embedding model.
- Esquecer `model.eval()` e `torch.no_grad()` em inference.

## Leitura e vídeos (fim de semana)

- Hugging Face *LLM Course*: capítulos *Transformer models* e *Using 🤗 Transformers* (*Behind the pipeline*, *Models*, *Tokenizers*, *Handling multiple sequences*).
- `transformers` documentation: *Generation strategies*; *Chat templates*; *Padding and truncation*.
- Jay Alammar, *The Illustrated BERT, ELMo, and co.*
- Reimers & Gurevych, *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks* (abstract e seção 3), para a origem do mean pooling em sentence embeddings.
- Holtzman et al., *The Curious Case of Neural Text Degeneration* (nucleus sampling): abstract e figuras.

## Perguntas de autoavaliação

1. Para que serve a `attention_mask` no mean pooling?
2. Por que decoders usam left padding na geração em batch?
3. Quanta memória de GPU ocupam os pesos de um modelo de 3B de parâmetros em fp16?
4. Qual a diferença entre top-k e top-p?
5. O que `add_generation_prompt=True` acrescenta e o que acontece sem ele?
6. Por que `safetensors` substituiu arquivos `.bin` baseados em pickle?

## Entregável da semana

Notebook resolvido e uma tabela em `docs/` com o mesmo prompt gerado em quatro configurações de decoding.

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/02-Phase-2-Deep-Learning/notebooks/semana_09_exercicio.ipynb) · [`semana_09_exercicio.ipynb`](../notebooks/semana_09_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/02-Phase-2-Deep-Learning/notebooks/semana_09_gabarito.ipynb) · [`semana_09_gabarito.ipynb`](../notebooks/semana_09_gabarito.ipynb) |

[← Semana 8](/02-Phase-2-Deep-Learning/docs/Semana-08-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 10 →](/02-Phase-2-Deep-Learning/docs/Semana-10-Conceitos.md)
