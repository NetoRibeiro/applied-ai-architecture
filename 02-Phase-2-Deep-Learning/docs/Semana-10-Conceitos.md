# Semana 10 — Projeto: Fine-Tuning de um Encoder com Training Loop Manual

**Fase 2: Deep Learning e o Ecossistema Hugging Face** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Fechar a Fase 2 com um projeto completo: fine-tuning de um encoder pré-treinado para text classification, escrevendo o loop à mão (sem `Trainer`) e aplicando as técnicas que todo treino de Transformer usa: AdamW, warmup, mixed precision, gradient clipping, dynamic padding e avaliação com as métricas da Semana 1.

## Por que isso importa para um AI Architect

Nem todo problema pede um LLM gerador. Intent classifiers, query routers, prompt injection detectors, PII detectors e rerankers são encoders ajustados: custam uma fração, respondem em milissegundos e têm comportamento previsível. Saber quando um encoder de 100M de parâmetros resolve é uma decisão de arquitetura que economiza muito dinheiro e latência.

## Conceitos

### 1. Transfer learning

> **Em palavras simples.** **Transfer learning** é aproveitar o que um modelo já aprendeu em uma tarefa genérica para resolver uma tarefa específica. Um modelo que leu bilhões de frases já sabe gramática e vocabulário; para ensiná-lo a classificar chamados, não é preciso começar do zero, basta um ajuste fino com poucos exemplos (**fine-tuning**). É a diferença entre formar um profissional do berço e treinar alguém que já tem diploma.

O modelo pré-treinado já sabe representar linguagem. O fine-tuning adiciona um **classification head** (uma layer linear, às vezes com dropout, sobre o vetor da frase) e ajusta os pesos com uma learning rate baixa, por poucas epochs. Com poucos milhares de exemplos já se obtém bom resultado, porque só é preciso aprender a tarefa, não a língua.

O aviso `Some weights of ... were not initialized from the model checkpoint` ao carregar `AutoModelForSequenceClassification` é esperado: é o head novo, que começa aleatório.

Três regimes:

- **Feature extraction:** o encoder fica frozen; treina-se só o head. Rápido, menos preciso.
- **Full fine-tuning:** todos os pesos são atualizados. É o que fazemos aqui.
- **PEFT** (LoRA e afins): o encoder fica frozen e treinam-se adapters. Fase 4.

> **No mundo real.** O paper **ULMFiT** (Howard e Ruder, 2018) mediu o ganho: em classificação de textos, um modelo pré-treinado e ajustado com apenas **100 exemplos rotulados** igualou o desempenho de um modelo treinado do zero com cem vezes mais dados. Meses depois, o BERT consolidou a prática. Desde então, treinar um modelo de linguagem do zero para uma tarefa de negócio praticamente deixou de fazer sentido.

### 2. Hyperparameters típicos de fine-tuning

> **Em palavras simples.** No fine-tuning, os ajustes são delicados porque o modelo já sabe muito, e a meta é acrescentar sem estragar. Por isso os passos são bem pequenos (learning rate baixa), o treino é curto (poucas epochs) e os lotes são modestos. Usar os ajustes de um treino do zero em um modelo pré-treinado é como reformar uma casa com dinamite.

| Item | Ponto de partida | Observação |
|---|---|---|
| Optimizer | AdamW | `weight_decay=0.01` |
| Learning rate | 2e-5 a 5e-5 | ~100× menor que no treino do zero |
| Epochs | 2 a 4 | Mais que isso tende a overfitting |
| Batch size | 16 a 32 | Limitado pela VRAM |
| Warmup | 5 a 10% dos steps | Protege os pesos pré-treinados |
| Max sequence length | Pelo p95–p99 do dataset | Custo cresce com `T²` |
| Gradient clipping | Norma máxima 1,0 | Estabilidade |

A LR é o hiperparâmetro mais sensível. Alta demais destrói o pre-training; baixa demais não aprende a tarefa em poucas epochs.

> **No mundo real.** O paper do **BERT** recomenda uma grade pequena de valores para praticamente qualquer tarefa: batch size de 16 ou 32, learning rate de 5e-5, 3e-5 ou 2e-5, e de 2 a 4 epochs. Os autores relatam que testar essas poucas combinações bastou para obter os resultados publicados em onze tarefas diferentes. Essa grade continua sendo o ponto de partida recomendado para fine-tuning de encoders.

### 3. Learning rate schedule

> **Em palavras simples.** Um **learning rate schedule** é um plano para o tamanho do passo ao longo do treino. O mais comum começa quase parado, acelera aos poucos (**warmup**) e depois desacelera até parar. O warmup existe porque, no começo, a parte nova do modelo ainda está desajustada e produz correções bruscas; com passos pequenos, elas não danificam o que já estava aprendido.

**Linear warmup** sobe a LR de 0 ao valor alvo nos primeiros steps; depois ela decai (linear ou cosine) até zero. No início, o head recém-criado produz gradientes grandes e ruidosos, e as estatísticas do Adam ainda não estabilizaram; com LR cheia, esses gradientes danificariam os pesos pré-treinados. `get_linear_schedule_with_warmup` devolve um scheduler cujo `.step()` é chamado **a cada training step**, depois do `optimizer.step()`.

> **No mundo real.** O paper original do Transformer, *Attention Is All You Need* (2017), já usava warmup: a learning rate crescia linearmente durante os primeiros **4.000 steps** e depois diminuía. Os autores e trabalhos posteriores observaram que, sem isso, o treino frequentemente divergia logo no início. A prática foi mantida em praticamente todos os modelos seguintes, do BERT aos LLMs atuais.

### 4. Mixed precision (AMP)

> **Em palavras simples.** **Mixed precision** é fazer a maior parte das contas com números de precisão menor (16 bits em vez de 32). As contas ficam mais rápidas e ocupam metade da memória, e o resultado final quase não muda. Um mecanismo auxiliar, o `GradScaler`, evita que valores muito pequenos sejam arredondados para zero no caminho.

`torch.autocast` executa a maior parte das operações em `float16`, reduzindo memória e tempo quase pela metade em GPUs com tensor cores. Os pesos continuam em `float32`.

O risco é **underflow**: gradientes pequenos viram zero em fp16. O `GradScaler` multiplica a loss por um fator grande antes do `backward()`, e desfaz a escala antes do `optimizer.step()`. A sequência correta:

```python
with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss = model(**batch).loss
optimizer.zero_grad()
scaler.scale(loss).backward()
scaler.unscale_(optimizer)                         # necessário antes do clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer); scaler.update(); scheduler.step()
```

Com `bfloat16` (GPUs mais novas) o scaler é dispensável.

> **No mundo real.** A **NVIDIA** reporta ganhos de até cerca de três vezes na velocidade de treino com mixed precision em GPUs com *tensor cores*, como a V100 e a T4, sem perda de accuracy em uma série de modelos de visão e de linguagem. Na prática, é uma mudança de poucas linhas de código (`torch.autocast`) que permite treinar com lotes maiores ou modelos maiores no mesmo equipamento. Hoje é padrão em praticamente todo treino em GPU.

### 5. Dynamic padding e a `collate_fn`

> **Em palavras simples.** Os textos de um lote têm comprimentos diferentes e precisam ser igualados com enchimento. **Dynamic padding** é encher cada lote só até o tamanho do seu texto mais longo, em vez de encher todos até o máximo permitido. Como o modelo gasta contas também com o enchimento, a economia é grande quando a maioria dos textos é curta.

Em vez de preencher tudo até `max_length`, a `collate_fn` faz padding de cada batch só até a maior sequência daquele batch (`padding=True`, que equivale a `"longest"`). Como o custo da attention é quadrático em `T`, isso reduz muito o tempo quando os textos têm tamanhos variados. Um refinamento é **length bucketing**: agrupar exemplos de tamanho parecido no mesmo batch.

> **No mundo real.** No dataset **SST-2** (frases de críticas de cinema, parte do benchmark GLUE), as frases têm em média cerca de 20 palavras, enquanto o BERT aceita até 512 tokens. Preencher tudo até 512 significaria gastar mais de 90% das contas com enchimento. O `DataCollatorWithPadding` da biblioteca `transformers` implementa o dynamic padding e é o que os scripts oficiais de exemplo usam por padrão.

### 6. Gradient accumulation

> **Em palavras simples.** Às vezes a GPU não comporta o tamanho de lote desejado. **Gradient accumulation** resolve somando as correções de vários lotes pequenos antes de aplicar um único ajuste, o que equivale a um lote grande processado em prestações. Troca-se memória por tempo: o resultado é praticamente o mesmo, só demora mais.

Se a VRAM só comporta batch 8 e você quer o efeito de batch 32, acumule os gradientes de 4 batches antes de chamar `optimizer.step()` (dividindo a loss por 4). É o `gradient_accumulation_steps` do `Trainer`. O **effective batch size** é o produto dos dois.

> **No mundo real.** O repositório oficial do **BERT**, do Google, tem uma seção dedicada a erros de falta de memória: o fine-tuning do BERT-large com os ajustes do paper não cabia em uma GPU de 12 GB. Entre as saídas discutidas estão reduzir o comprimento máximo dos textos, reduzir o lote e acumular gradients. A opção `gradient_accumulation_steps` do `Trainer` existe para isso e é rotina em fine-tuning de LLMs em uma única GPU.

### 7. Catastrophic forgetting

> **Em palavras simples.** **Catastrophic forgetting** é o que acontece quando um modelo, ao aprender algo novo, apaga o que sabia antes. Diferente de uma pessoa, uma rede neural não acumula conhecimento naturalmente: os mesmos números internos são reaproveitados e sobrescritos. Um fine-tuning agressivo pode ensinar a tarefa nova e destruir a competência geral do modelo.

LR alta ou treino longo fazem o modelo perder o conhecimento geral do pre-training. Sintoma: a train loss cai rápido e a validation piora, ou o modelo passa a prever uma única classe. É a motivação para LoRA (Fase 4): congelar os pesos originais e treinar só um adapter. Técnicas intermediárias: layer-wise learning rate decay e congelar as primeiras layers.

> **No mundo real.** Em 2017, pesquisadores da **DeepMind** (Kirkpatrick et al.) mostraram o efeito com jogos de Atari: uma rede treinada em um jogo e depois em outro esquecia quase tudo do primeiro. Eles propuseram a técnica *Elastic Weight Consolidation*, que protege os parâmetros mais importantes para as tarefas antigas. Em LLMs, a resposta mais usada é outra: congelar o modelo e treinar só um pequeno módulo adicional (LoRA, Fase 4).

### 8. Evaluation

> **Em palavras simples.** Avaliar um modelo ajustado é medir, em dados que ele nunca viu, o quanto ele acerta e, principalmente, **como** ele erra. Um número só (accuracy) esconde demais: é preciso olhar classe por classe e ver quais são confundidas entre si. Também é prudente repetir o treino algumas vezes, porque em datasets pequenos o resultado varia com o sorteio inicial.

Accuracy sozinha não basta (Semana 1). Use `classification_report` por classe e a confusion matrix para ver quais classes o modelo confunde. Em datasets pequenos, o resultado varia com a seed: reporte média e desvio de três a cinco seeds antes de afirmar que uma configuração é melhor.

Separe validation de test: epochs, LR e early stopping são escolhidos em validation.

> **No mundo real.** O benchmark **GLUE** (2018) reúne nove tarefas de compreensão de texto e reporta uma métrica adequada a cada uma: accuracy em algumas, F1 nas de classes desbalanceadas, correlação nas de nota contínua. O BERT elevou a pontuação média do GLUE para 80,5, um salto de 7,7 pontos sobre o melhor resultado anterior. Ter um conjunto fixo de tarefas e métricas foi o que permitiu comparar modelos de forma justa naquele período.

### 9. Inference e custo

> **Em palavras simples.** Depois de treinado, o que importa é quanto o modelo custa para usar: quanto tempo leva para responder e quantas respostas entrega por segundo. Modelos menores respondem mais rápido e rodam em máquinas mais baratas. Muitas vezes vale trocar um pouco de qualidade por um modelo várias vezes mais leve.

Meça latência com batch (throughput) e sem batch (latência por requisição). Um DistilBERT classifica centenas de textos por segundo em uma T4 e roda bem em CPU. Para produção, as rotas de otimização são: export para ONNX, quantization int8, e distillation para um modelo ainda menor.

> **No mundo real.** O **DistilBERT** (Sanh et al., Hugging Face, 2019) foi criado por **knowledge distillation**: um modelo pequeno treinado para imitar o BERT. O resultado é 40% menor, 60% mais rápido e mantém 97% do desempenho do original nas tarefas do GLUE. É um dos modelos mais usados em produção para classificação de texto justamente por rodar bem até em CPU, e é o que você ajusta no notebook desta semana.

### 10. Encoder ajustado ou LLM com prompt?

> **Em palavras simples.** Para classificar textos, há dois caminhos. Um modelo pequeno, treinado para aquela tarefa, é rápido, barato e previsível, mas exige exemplos rotulados. Um LLM com um bom prompt não exige treino e se adapta a mudanças, mas custa mais e responde mais devagar. A escolha depende do volume, da pressa e de quantos exemplos existem.

| Critério | Encoder fine-tuned | LLM zero/few-shot |
|---|---|---|
| Dados rotulados | Precisa (centenas a milhares) | Não precisa |
| Latência | Milissegundos | Centenas de ms a segundos |
| Custo por requisição | Muito baixo | Alto |
| Consistência | Alta, determinístico | Variável |
| Mudança de classes | Exige retreinar | Muda o prompt |
| Explicação em texto | Não | Sim |

Caminho comum: começar com LLM few-shot para validar o problema e gerar rótulos, e destilar para um encoder quando o volume justificar.

> **No mundo real.** O filtro de spam do **Gmail** bloqueia, segundo o Google, mais de 99,9% do spam e do phishing, o que representa bilhões de mensagens indesejadas por dia. Nesse volume, com resposta exigida em milissegundos, usar um LLM gerador para cada e-mail seria inviável em custo e em tempo: o trabalho é feito por classificadores especializados. É o caso típico em que um modelo pequeno e ajustado é a arquitetura correta.

### 11. O `Trainer`

> **Em palavras simples.** O `Trainer` é a versão pronta do loop de treino. Em vez de escrever cada passo, você preenche uma ficha de configuração (taxa de aprendizado, número de epochs, tamanho do lote) e ele executa, registra as métricas e salva os resultados. Depois de ter escrito o loop à mão, cada campo da ficha deixa de ser mistério.

Tudo desta semana cabe em `TrainingArguments(learning_rate=..., num_train_epochs=..., warmup_ratio=..., fp16=True, per_device_train_batch_size=..., weight_decay=..., eval_strategy="epoch")` mais um `DataCollatorWithPadding`. Depois de escrever o loop à mão, cada argumento tem significado concreto.

> **No mundo real.** O repositório da biblioteca `transformers` traz o script de exemplo `run_glue.py`, construído sobre o `Trainer`. Com um comando, ele faz o fine-tuning do BERT em qualquer tarefa do GLUE; na tarefa MRPC (detectar se duas frases são paráfrases, cerca de 3.700 exemplos de treino), o processo leva poucos minutos em uma GPU. É o caminho que a maioria dos times usa no dia a dia, e os argumentos correspondem ao que você implementou nesta semana.

## Erros comuns

- LR de treino do zero (`1e-3`) em fine-tuning: o modelo colapsa para uma classe.
- Chamar `scheduler.step()` uma vez por epoch em vez de por step.
- Clipping sem `scaler.unscale_()` em AMP: o clip atua sobre gradientes escalados.
- `padding="max_length"` sem necessidade.
- Esquecer `model.eval()` na avaliação (o dropout fica ligado).
- Concluir a partir de uma única seed.

## Leitura e vídeos (fim de semana)

- Hugging Face *LLM Course*, capítulo *Fine-tuning a pretrained model*: *Processing the data* (dynamic padding) e *A full training loop*.
- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (Sections 1, 3 e 4.1).
- Sanh et al., *DistilBERT, a distilled version of BERT* (abstract e seção 2), sobre knowledge distillation.
- PyTorch documentation: *Automatic Mixed Precision package* e o recipe *Automatic Mixed Precision*.
- Sun et al., *How to Fine-Tune BERT for Text Classification?* (seções sobre learning rate e catastrophic forgetting).
- `transformers` documentation: *Trainer* e *Data collator*.

## Perguntas de autoavaliação

1. Por que a learning rate de fine-tuning é tão menor que a de treino do zero?
2. O que o warmup protege?
3. O que é dynamic padding e por que acelera o treino?
4. Para que serve o `GradScaler` e por que `unscale_` vem antes do gradient clipping?
5. Em que situações você escolheria um encoder ajustado em vez de um LLM com prompt?
6. Como simular batch size 64 em uma GPU que só comporta 16?

## Entregável da semana

Notebook resolvido, o classification report e um parágrafo: "em que parte de um sistema RAG corporativo eu usaria este classificador?".

### Checkpoint da Fase 2

Você deve conseguir, sem consultar: desenhar um Transformer block; explicar Q, K e V; escrever os cinco passos do training loop; estimar a memória de um modelo a partir do número de parâmetros; dizer por que decoders usam left padding.

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/02-Phase-2-Deep-Learning/notebooks/semana_10_exercicio.ipynb) · [`semana_10_exercicio.ipynb`](../notebooks/semana_10_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/02-Phase-2-Deep-Learning/notebooks/semana_10_gabarito.ipynb) · [`semana_10_gabarito.ipynb`](../notebooks/semana_10_gabarito.ipynb) |

[← Semana 9](/02-Phase-2-Deep-Learning/docs/Semana-09-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 11 →](/03-Phase-3-Advanced-RAG/docs/Semana-11-Conceitos.md)
