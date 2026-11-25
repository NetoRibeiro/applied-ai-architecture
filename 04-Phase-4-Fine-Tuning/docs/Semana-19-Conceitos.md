# Semana 19 — Fine-Tuning com PEFT/LoRA

**Fase 4: Parameter-Efficient Fine-Tuning (PEFT)** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Fazer o primeiro fine-tuning de um open-weight LLM com a biblioteca PEFT: preparar os exemplos com o chat template, aplicar label masking para que só a resposta conte na loss, configurar o LoRA, treinar, avaliar antes e depois e salvar o adapter.

## Por que isso importa para um AI Architect

Este é o procedimento que adapta um modelo aberto ao domínio da empresa sem enviar dados a terceiros. Os detalhes desta semana (label masking, EOS token, `target_modules`, padding) são exatamente os que fazem um fine-tuning "rodar sem erro e não funcionar".

## Conceitos

### 1. Supervised Fine-Tuning (SFT)

> **Em palavras simples.** **Supervised Fine-Tuning** (SFT) é ensinar pelo exemplo: mostra-se ao modelo uma instrução e a resposta desejada, e ele é ajustado para produzir aquela resposta. Tecnicamente é o mesmo treino de prever o próximo token, agora sobre conversas escolhidas. É a etapa que transforma um modelo que apenas continua textos em um assistente que responde.

O objetivo é o mesmo do pre-training: next-token prediction com cross-entropy. O que muda são os dados (pares instrução → resposta) e a escala. Cada exemplo vira uma única sequência:

`[prompt formatado pelo chat template] + [resposta] + [EOS]`

O modelo é treinado a prever o próximo token em todas as posições, recebendo sempre o prefixo correto (**teacher forcing**).

> **No mundo real.** Em 2023, a Databricks pediu a mais de 5.000 funcionários que escrevessem pares de instrução e resposta em categorias como resumo, extração e brainstorming. O resultado, o dataset **databricks-dolly-15k**, com cerca de 15 mil exemplos, foi usado para fazer o SFT de um modelo aberto de 2019. O modelo resultante, o Dolly, passou a seguir instruções de forma reconhecível, mostrando que o comportamento de assistente vem mais dos dados de SFT que do tamanho do modelo.

### 2. Label masking

> **Em palavras simples.** No treino, o modelo vê a conversa inteira, mas só deve ser cobrado pela parte que lhe cabe escrever: a resposta. **Label masking** é marcar os tokens da instrução para que não entrem no cálculo do erro. Sem isso, o modelo gasta esforço aprendendo a reproduzir perguntas, o que não serve para nada.

Não queremos ensinar o modelo a escrever o system prompt nem a pergunta do usuário: só a resposta. O tensor `labels` recebe `-100` nas posições do prompt; `-100` é o `ignore_index` padrão da cross-entropy do PyTorch, e essas posições não entram na loss. Padding também recebe `-100`.

```
input_ids: [system ... user ... <assistant>] [resposta ... <eos>]
labels:    [-100  ...  -100  ...  -100     ] [resposta ... <eos>]
```

Os modelos da Hugging Face fazem o shift de uma posição internamente: passa-se `labels` com o mesmo alinhamento de `input_ids`.

Treinar sem masking (loss sobre a sequência toda) também funciona, mas desperdiça capacidade em aprender a reproduzir prompts e dilui o sinal quando o prompt é longo e a resposta é curta. Na biblioteca TRL isso é feito por um data collator "completion only".

> **No mundo real.** Na biblioteca `transformers`, a convenção é o valor **-100**: toda posição de `labels` com esse número é ignorada pela cross-entropy. O repositório Alpaca-LoRA expunha a decisão como a opção `train_on_inputs`, e a biblioteca TRL oferece um collator dedicado a treinar apenas na resposta. Em datasets com instruções longas e respostas curtas, a diferença de qualidade entre mascarar e não mascarar é perceptível.

### 3. O EOS token

> **Em palavras simples.** O **EOS token** é o ponto final que o modelo aprende a escrever quando a resposta termina. Se ele não aparece no fim dos exemplos de treino, o modelo nunca aprende a parar: responde e continua escrevendo, inventando novas perguntas e respostas até atingir o limite de tamanho.

Se a resposta de treino não termina com o token de fim de turno, o modelo nunca aprende a parar e gera até `max_new_tokens`. É o defeito mais comum em fine-tunings caseiros. Em modelos com ChatML, o token de fim de turno é `<|im_end|>`; confira `tokenizer.eos_token` e o chat template do seu modelo.

Teste simples, que está no notebook: gere com `max_new_tokens` alto e veja se a resposta termina sozinha.

> **No mundo real.** No lançamento do **Llama 3** (abril de 2024), muitos usuários viram o modelo responder e seguir gerando texto indefinidamente. A causa: o modelo encerra cada turno com o token `<|eot_id|>`, diferente do EOS configurado por padrão, e várias ferramentas não o reconheciam como sinal de parada. A correção foi incluir esse token na lista de terminadores. Um único token mal configurado tornava inutilizável um modelo excelente.

### 4. Padding

> **Em palavras simples.** Para formar um batch, os textos mais curtos recebem enchimento (**padding**). É preciso decidir de que lado. No treino, o enchimento vai à direita. Na geração, vai à esquerda, porque o modelo continua o texto a partir do último token, e esse token não pode ser enchimento. Alguns modelos nem têm um token de padding definido, e é preciso escolher um.

- **No treino:** right padding, com `-100` nos labels do padding e `0` na `attention_mask`.
- **Na geração em batch:** left padding (Semana 9).
- Se o modelo não tem `pad_token`, usa-se o EOS como pad. Nesse caso é essencial mascarar o padding nos labels; senão o modelo aprende a emitir EOS em sequência.

> **No mundo real.** Os tokenizers da família **Llama** são distribuídos sem `pad_token`, e a primeira tentativa de montar um batch produz um erro. A solução mais comum, definir o padding como o próprio EOS, funciona, mas exige cuidado com o mascaramento dos labels. A documentação da `transformers` traz um aviso específico: em modelos decoder-only, gerar com padding à direita produz resultados errados sem nenhuma mensagem de erro.

### 5. Configurando o PEFT

> **Em palavras simples.** Configurar o **PEFT** é dizer à biblioteca onde colocar os adaptadores e de que tamanho. Informam-se o rank, o alpha, o dropout e os nomes das camadas que receberão o ajuste. A biblioteca congela o modelo original, insere os adaptadores e informa quantos parâmetros serão treinados, normalmente menos de 1% do total.

```python
config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    task_type="CAUSAL_LM")
model = get_peft_model(base_model, config)
model.print_trainable_parameters()
```

`get_peft_model` congela o base model e embrulha as layers escolhidas com LoRA. `print_trainable_parameters()` é o primeiro sanity check: espere algo entre 0,1% e 2%.

Outros campos úteis: `bias="none"`; `modules_to_save` (layers treinadas por inteiro, como embeddings quando se adicionam tokens novos); `use_dora=True`; `target_modules="all-linear"`.

> **No mundo real.** O paper do **QLoRA** (Dettmers et al., 2023) investigou onde colocar os adaptadores e chegou a uma conclusão que virou padrão: aplicar LoRA apenas às matrizes de query e value da attention, como no paper original, não bastava para igualar o full fine-tuning; era preciso aplicá-lo a **todas as camadas lineares** dos blocos. O número de camadas adaptadas pesou mais que o valor do rank.

### 6. Hyperparameters de partida

> **Em palavras simples.** Os valores iniciais de um fine-tuning com LoRA são conhecidos: learning rate bem mais alta que a do full fine-tuning (porque só os adaptadores, que começam do zero, são treinados), poucas epochs, warmup curto. São pontos de partida testados por muita gente, a ajustar conforme as curvas do seu treino.

| Item | Valor |
|---|---|
| Learning rate | 2e-4 (com warmup curto e cosine ou linear decay em treinos longos) |
| Epochs | 1 a 3 |
| Effective batch size | 8 a 32 (`per_device_batch_size × gradient_accumulation_steps`) |
| Max length | Pelo p99 do dataset (Semana 17) |
| Precision | Mixed precision (fp16 na T4; bf16 em GPUs mais novas) |
| Gradient clipping | 1,0 |
| Optimizer | AdamW |

No notebook o modelo é carregado em fp32 e o forward roda sob `torch.autocast`: é a combinação estável na T4. Carregar em fp16 puro e treinar sem autocast leva a gradientes `nan`.

Para caber em pouca VRAM: reduzir batch size e compensar com gradient accumulation; `model.gradient_checkpointing_enable()` (recalcula activations no backward em troca de memória, ~20–30% mais lento); reduzir `max_length`; QLoRA (Semana 20).

> **No mundo real.** A receita do paper do **QLoRA** é a mais reproduzida: learning rate de `2e-4` para modelos de até 13 bilhões de parâmetros e de `1e-4` para os maiores, rank 64, alpha 16, dropout de 0,1 nos modelos menores e schedule constante. Esses números aparecem, com pequenas variações, em praticamente todo tutorial e script de fine-tuning de LLMs abertos publicado desde então.

### 7. Lendo o treino

> **Em palavras simples.** Durante o treino, acompanham-se duas curvas: o erro nos dados de treino e o erro em dados separados para avaliação. O sinal de alerta é o erro de treino continuar caindo enquanto o de avaliação sobe: o modelo começou a decorar. Em fine-tuning de LLMs, isso pode acontecer muito cedo, às vezes já na segunda passada pelos dados.

| Train loss | Validation loss | Diagnóstico |
|---|---|---|
| Cai | Acompanha | Saudável |
| Vai a quase zero | Sobe | Memorização: reduza epochs, aumente dropout ou dados |
| Não cai | Não cai | LR, `target_modules` errados, ou labels todos mascarados |
| `nan` | — | Precisão (fp16 sem autocast), LR alta, exemplo vazio |

A loss sozinha não basta: avalie a **tarefa** (JSON válido? campos corretos?), antes e depois. A loss pode cair por o modelo aprender o formato enquanto continua errando o conteúdo.

Loss inicial muito alta em um modelo pré-treinado é sinal de chat template errado: o modelo está vendo um formato que nunca viu.

> **No mundo real.** Jeremy Howard e Jonathan Whitaker (fast.ai, 2023) relataram um padrão curioso no fine-tuning de LLMs: a curva de treino descia em **degraus**, com uma queda brusca exatamente no início de cada epoch. A investigação indicou que o modelo memorizava exemplos vistos uma única vez. A recomendação prática é desconfiar de mais de uma ou duas epochs e sempre observar a curva de avaliação, não a de treino.

### 8. O que é salvo

> **Em palavras simples.** Ao fim do treino com LoRA, o que se salva não é o modelo inteiro: são só os adaptadores, um arquivo pequeno, e um arquivo de configuração que diz a qual modelo base eles pertencem. Para usar, carrega-se o modelo base e, por cima, o adaptador. O adaptador sozinho não serve para nada.

`model.save_pretrained("adapter")` grava apenas `adapter_model.safetensors` e `adapter_config.json`: megabytes. Para usar:

```python
base = AutoModelForCausalLM.from_pretrained(BASE)
model = PeftModel.from_pretrained(base, "adapter")
```

O `adapter_config.json` registra o `base_model_name_or_path`. O adapter só é válido para aquele modelo exato. Salve o tokenizer junto se você adicionou tokens ou alterou o chat template.

> **No mundo real.** O **Guanaco**, modelo publicado junto com o paper do QLoRA, foi distribuído exatamente assim: um repositório no Hugging Face Hub com o `adapter_model` e o `adapter_config.json`, que aponta para o modelo LLaMA base. Para a versão de 7 bilhões de parâmetros, o adaptador tem poucas centenas de megabytes, contra mais de 13 GB do modelo base. Por isso o Hub consegue hospedar milhares de variações do mesmo modelo.

### 9. `Trainer` e `SFTTrainer`

> **Em palavras simples.** O `Trainer` é o loop de treino genérico da Hugging Face. O `SFTTrainer`, da biblioteca TRL, é uma versão especializada em fine-tuning de conversas: aplica o chat template, monta os labels e integra o PEFT com poucas linhas. Ele poupa trabalho, mas é importante saber o que ele faz por baixo, que é o que você implementa nesta semana.

Em projetos reais usa-se o `Trainer` do `transformers` ou o `SFTTrainer` da biblioteca TRL, que recebe um dataset no formato `messages`, aplica o chat template, faz o masking e cuida de logging, checkpoints, evaluation e schedules. Cada argumento de `TrainingArguments` corresponde a algo que você escreveu à mão nesta semana e na Semana 10. Ferramentas como Unsloth e Axolotl são camadas de conveniência e otimização sobre os mesmos conceitos.

> **No mundo real.** A maior parte dos guias oficiais de fine-tuning publicados para modelos abertos, como os da Hugging Face para as famílias Llama, Mistral e Gemma, usa o `SFTTrainer`. O benefício é a padronização; o risco conhecido é que os valores padrão mudam entre versões da biblioteca, por exemplo no tratamento do chat template e do EOS. Fixar as versões no `requirements.txt` evita que o mesmo script produza modelos diferentes.

### 10. Antes e depois

> **Em palavras simples.** A forma mais direta de saber se o fine-tuning funcionou é comparar o modelo antes e depois, nas mesmas entradas, que não estavam no treino. Olham-se os exemplos lado a lado e mede-se o que importa para a tarefa: formato correto, campos certos, tom adequado. Se o modelo ajustado não é claramente melhor que o original com um bom prompt, o fine-tuning não se justificou.

A comparação honesta é com o mesmo prompt, o mesmo conjunto de validation e as mesmas métricas de tarefa, antes e depois. E a comparação que justifica o projeto é com zero-shot e few-shot bem feitos (Semana 21): fine-tuning que só empata com um bom prompt não se paga.

> **No mundo real.** A demonstração original do **Alpaca** foi exatamente essa comparação: o LLaMA de 7 bilhões de parâmetros, sem ajuste, tendia a continuar o texto da instrução em vez de obedecê-la; depois do fine-tuning com 52 mil exemplos (cerca de três horas em oito GPUs, por menos de 100 dólares, segundo os autores), passava a responder de forma semelhante a um modelo comercial da época em avaliações cegas preliminares.

## Erros comuns

- Esquecer o EOS no fim da resposta de treino.
- Não mascarar o padding nos labels quando `pad_token = eos_token`.
- Formatar o treino com um chat template e inferir com outro, ou com system prompt diferente.
- Tokenizar o prompt já formatado com `add_special_tokens=True` e duplicar o BOS.
- `target_modules` que não existem na arquitetura.
- Truncar exemplos em `max_length` e cortar a resposta (e o EOS).
- Avaliar só pela loss.

## Leitura e vídeos (fim de semana)

- Hugging Face PEFT documentation: *Quicktour*; *LoRA* (conceptual guide e developer guide); *Troubleshooting*.
- TRL documentation: *SFTTrainer* (leia para reconhecer os mesmos passos em forma de configuração, incluindo o treino só nas completions).
- Hugging Face *LLM Course*: capítulo sobre *Supervised Fine-Tuning* / instruction tuning e chat templates.
- Philipp Schmid, *How to Fine-Tune Open LLMs* (guia prático atualizado a cada ano).
- `transformers` documentation: *Trainer* e a seção *Gradient checkpointing* de *Methods and tools for efficient training on a single GPU*.
- Opcional: Maxime Labonne, *Fine-Tune Your Own Llama Model* (LLM Course no GitHub).

## Perguntas de autoavaliação

1. Para que serve o valor `-100` em `labels`?
2. O modelo ajustado não para de gerar. Qual a causa mais provável?
3. Como descobrir os nomes corretos para `target_modules`?
4. O que exatamente está dentro do arquivo do adapter?
5. Por que carregar em fp32 e usar autocast, em vez de carregar em fp16, na T4?
6. A loss inicial de um modelo pré-treinado veio muito alta. O que você verifica primeiro?

## Perguntas do entrevistador

1. Durante o fine-tuning instrucional com LoRA, você esqueceu de aplicar label masking nos tokens do system prompt e da pergunta do usuário. Que impacto isso tem na loss, no comportamento do modelo em inferência e no risco de memorização?

## Entregável da semana

Notebook resolvido, o adapter salvo e uma tabela antes/depois (JSON válido, categoria correta, prioridade correta) em `docs/`.

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/04-Phase-4-Fine-Tuning/notebooks/semana_19_exercicio.ipynb) · [`semana_19_exercicio.ipynb`](../notebooks/semana_19_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/04-Phase-4-Fine-Tuning/notebooks/semana_19_gabarito.ipynb) · [`semana_19_gabarito.ipynb`](../notebooks/semana_19_gabarito.ipynb) |

[← Semana 18](/04-Phase-4-Fine-Tuning/docs/Semana-18-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 20 →](/04-Phase-4-Fine-Tuning/docs/Semana-20-Conceitos.md)
