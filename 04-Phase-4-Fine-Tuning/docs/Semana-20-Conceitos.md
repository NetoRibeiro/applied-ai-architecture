# Semana 20 — QLoRA: 4-bit Quantization

**Fase 4: Parameter-Efficient Fine-Tuning (PEFT)** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Entender quantization (representar pesos com menos bits) e aplicar QLoRA: carregar o base model em 4-bit e treinar LoRA adapters sobre ele. É a técnica que permite ajustar modelos de vários bilhões de parâmetros em uma única GPU comum.

## Por que isso importa para um AI Architect

VRAM é o recurso mais caro e mais disputado de um projeto de LLM. Quantization decide qual modelo cabe em qual hardware, tanto no treino quanto na inference, e o tradeoff entre bits e qualidade é uma decisão de custo que cabe a você justificar com medições.

## Conceitos

### 1. O que é quantizar

> **Em palavras simples.** **Quantizar** é guardar cada número do modelo com menos precisão, como arredondar preços para o real mais próximo em vez de registrar os centavos. Em vez de 16 ou 32 bits por parâmetro, usam-se 8 ou 4. O modelo ocupa uma fração da memória e perde um pouco de qualidade, em geral bem menos do que se imaginaria.

Mapear números de ponto flutuante de 32 ou 16 bits para um conjunto pequeno de níveis. Com 4 bits há apenas `2⁴ = 16` valores possíveis por peso. Guarda-se o índice do nível e, por bloco de pesos, as constantes para reconstruir o valor (`scale` e, às vezes, `zero_point`).

Quantization uniforme (affine): `q = round(x / scale) + zero_point` e, na volta, `x̂ = (q − zero_point) · scale`. A diferença `x − x̂` é o **quantization error**.

| Precisão | Bytes por parâmetro | Modelo de 7B |
|---|---|---|
| fp32 | 4 | ~28 GB |
| fp16 / bf16 | 2 | ~14 GB |
| int8 | 1 | ~7 GB |
| 4-bit | ~0,5 | ~4 a 5 GB |

Granularidade: as constantes podem ser por tensor, por canal ou por bloco de pesos. Blocos menores erram menos e gastam mais com constantes.

> **No mundo real.** Em março de 2023, poucos dias após o vazamento dos pesos do LLaMA, Georgi Gerganov publicou o **llama.cpp**, que rodava o modelo de 7 bilhões de parâmetros quantizado em 4 bits em um MacBook comum, ocupando cerca de 4 GB. Até então, rodar um LLM exigia uma GPU de servidor. Foi a quantization que abriu caminho para os modelos locais, e o projeto se tornou a base de ferramentas como o Ollama.

### 2. Uniform vs. NF4

> **Em palavras simples.** A quantization **uniform** divide a faixa de valores em degraus iguais. Mas os pesos de uma rede não se distribuem por igual: a maioria fica perto de zero e poucos ficam nos extremos. O **NF4** coloca mais degraus onde há mais valores, perto do zero, e menos nos extremos. Com os mesmos 4 bits, o erro de arredondamento fica menor.

- **Uniform:** níveis igualmente espaçados entre o mínimo e o máximo. Um único outlier estica a faixa e desperdiça níveis em regiões onde quase não há pesos.
- **NF4 (4-bit NormalFloat):** os pesos de redes neurais seguem aproximadamente uma distribuição normal centrada em zero. O NF4 posiciona os 16 níveis nos **quantiles** dessa distribuição: mais níveis perto de zero, onde estão quase todos os pesos. Cada nível é usado com frequência parecida, o que é a alocação ótima em termos de informação.

O notebook implementa as duas e mostra o erro e o histograma de uso dos níveis.

> **No mundo real.** O **NF4** (*4-bit NormalFloat*) foi proposto no paper do QLoRA a partir da observação de que os pesos de redes pré-treinadas seguem aproximadamente uma distribuição normal. Nos experimentos dos autores, o NF4 obteve resultados melhores que os formatos de 4 bits com degraus uniformes em todas as comparações. É a opção `bnb_4bit_quant_type="nf4"` da biblioteca `bitsandbytes`.

### 3. PTQ e QAT

> **Em palavras simples.** Há dois momentos para quantizar. **PTQ** (*post-training quantization*) comprime um modelo já treinado: é rápido e não exige novo treino. **QAT** (*quantization-aware training*) simula a compressão durante o treino, para que o modelo aprenda a conviver com ela: dá resultados melhores em precisões muito baixas, e custa um treino completo.

- **Post-Training Quantization (PTQ):** quantiza um modelo já treinado. Pode usar um pequeno calibration set para escolher as constantes. É o caso de bitsandbytes, GPTQ, AWQ e GGUF.
- **Quantization-Aware Training (QAT):** simula a quantization durante o treino para que o modelo aprenda a tolerá-la. Melhor qualidade, muito mais caro.

O que importa no fim não é o erro em cada peso, e sim o erro na **saída** da layer (`W x`). Métodos como GPTQ e AWQ minimizam esse erro usando dados de calibração, e por isso preservam mais qualidade que o arredondamento simples.

> **No mundo real.** O paper de Jacob et al. (Google, 2018) introduziu o QAT usado no **TensorFlow Lite**, que permitiu rodar modelos de visão como o MobileNet em celulares usando apenas aritmética de inteiros de 8 bits, com perda pequena de accuracy. Em LLMs, o custo de retreinar fez o PTQ predominar: métodos como **GPTQ** e **AWQ** quantizam um modelo de bilhões de parâmetros em poucas horas, usando uma pequena amostra de textos para calibração.

### 4. Os três ingredientes do QLoRA

> **Em palavras simples.** O **QLoRA** combina três ideias para treinar modelos grandes em GPUs modestas: guardar o modelo base em 4 bits com NF4; comprimir também as constantes da própria quantization (**double quantization**); e usar a memória comum do computador como reserva quando a da GPU chega ao limite (**paged optimizers**). Sobre esse modelo comprimido e congelado, treinam-se adaptadores LoRA em precisão normal.

1. **Base model em 4-bit NF4, frozen.** Os pesos quantizados não recebem gradiente.
2. **Double quantization:** as próprias constantes de escala são quantizadas, economizando cerca de 0,4 bit por parâmetro.
3. **LoRA adapters em precisão maior:** só eles são treinados. No forward, cada bloco é desquantizado para o **compute dtype** (`bnb_4bit_compute_dtype`: fp16 ou bf16), multiplicado, e somado à saída do adapter.

O paper também introduz paged optimizers para absorver picos de memória. O resultado relatado: qualidade equivalente à do fine-tuning em 16 bits, com uma fração da memória.

O gradiente passa **através** dos pesos quantizados (eles participam do forward), mas só `A` e `B` são atualizados. Em certo sentido, o adapter aprende também a compensar o quantization error do base model.

> **No mundo real.** O paper (Dettmers et al., 2023) mostrou o fine-tuning de um modelo de **65 bilhões** de parâmetros em uma única GPU de 48 GB; o full fine-tuning em 16 bits exigiria mais de 780 GB. O modelo resultante, o **Guanaco**, treinado em 24 horas, alcançou 99,3% do desempenho do ChatGPT no benchmark Vicuna. O fine-tuning de modelos grandes deixou de ser exclusivo de quem tinha um cluster.

### 5. Configuração típica

> **Em palavras simples.** A configuração usual do QLoRA cabe em quatro linhas: carregar em 4 bits, usar o formato NF4, ligar a double quantization e fazer as contas em 16 bits (`bfloat16`). Esse último ponto importa: os pesos ficam guardados em 4 bits, mas cada cálculo é feito com mais precisão, o que mantém o treino estável.

```python
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map={"": 0})
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(...))
```

`prepare_model_for_kbit_training` congela o base model, converte algumas layers (como as de normalization) para fp32 por estabilidade e habilita gradient checkpointing. `bitsandbytes` exige GPU CUDA.

> **No mundo real.** Os números do paper ajudam a entender cada linha. A **double quantization** economiza em média cerca de 0,37 bit por parâmetro, o que parece pouco e representa aproximadamente 3 GB em um modelo de 65 bilhões de parâmetros. É a diferença entre caber e não caber em uma GPU de 48 GB. Essa configuração (`BitsAndBytesConfig`) é reproduzida quase sem alteração nos notebooks de fine-tuning do Colab.

### 6. Custos e cuidados

> **Em palavras simples.** O QLoRA economiza memória e cobra em outros pontos. O treino fica mais lento, porque os pesos precisam ser convertidos a cada cálculo. A qualidade do modelo base cai um pouco. E juntar o adaptador ao modelo base exige cuidado: ele foi treinado sobre a versão comprimida, e somá-lo à versão em precisão cheia dá um resultado ligeiramente diferente do que foi avaliado.

- **Velocidade:** treinar sobre base quantizado é **mais lento** por step (há dequantization a cada forward). Ganha-se memória, não tempo.
- **Memória de treino ≠ memória dos pesos:** activations e gradients crescem com batch size e sequence length. Combine com gradient checkpointing e gradient accumulation.
- **Merging:** fundir um adapter diretamente em um base de 4-bit degrada a qualidade, porque a soma precisa ser requantizada. O caminho recomendado: recarregar o base em fp16, aplicar o adapter, `merge_and_unload()`, e então quantizar o modelo final para o formato de serving.
- **Train/serve mismatch:** um adapter treinado sobre o base em 4-bit e servido sobre o base em fp16 (ou em outra quantization) se comporta de forma ligeiramente diferente. Meça.

> **No mundo real.** A documentação das bibliotecas `peft` e `bitsandbytes` registra essas limitações, e os relatos de quem treina confirmam que um passo de treino em 4 bits costuma ser sensivelmente mais lento que o equivalente em 16 bits na mesma GPU. A recomendação prática decorre daí: usar QLoRA quando o modelo **não cabe** de outra forma; se ele cabe em 16 bits, o LoRA comum treina mais rápido.

### 7. Quantization para inference

> **Em palavras simples.** Quantizar para **inference** é comprimir o modelo final para servi-lo com menos memória e, muitas vezes, mais velocidade. Existem vários formatos, cada um ligado a um tipo de servidor: **GGUF** para CPU e computadores pessoais, **AWQ** e **GPTQ** para GPU. O nível de compressão é escolhido conforme o equipamento e a perda de qualidade aceitável.

| Formato | Ecossistema | Uso |
|---|---|---|
| **GGUF** | llama.cpp, Ollama, LM Studio | CPU e GPU; laptops e servidores pequenos |
| **AWQ**, **GPTQ** | vLLM, TGI | Servidores de GPU de alta vazão |
| **bitsandbytes** | `transformers` | Experimentação e QLoRA |
| **EXL2** | ExLlamaV2 | GPU única, bitrate flexível |

A nomenclatura GGUF (`Q4_K_M`, `Q5_K_M`, `Q8_0`) indica bits e variante: `K` são os esquemas mistos que dão mais bits às layers mais sensíveis; `S`/`M`/`L` indicam tamanho.

Regras práticas, a serem confirmadas na sua tarefa:

- 8-bit é quase indistinguível do fp16.
- 4-bit preserva a maior parte da qualidade e é o ponto de equilíbrio usual.
- Abaixo de 4 bits a queda acelera.
- **Um modelo maior em 4-bit costuma superar um modelo menor em fp16 com a mesma memória.**
- Tarefas de raciocínio longo, matemática e structured output são mais sensíveis que conversação.

Quantization reduz memória sempre; latência só melhora se o hardware e o runtime tiverem kernels otimizados para o formato.

> **No mundo real.** O **Ollama**, a ferramenta mais popular para rodar LLMs localmente, distribui por padrão versões em 4 bits no formato GGUF: ao baixar um modelo de 7 ou 8 bilhões de parâmetros, o arquivo tem cerca de 4 a 5 GB em vez dos 15 GB originais. Os sufixos no nome dos arquivos, como `Q4_K_M` ou `Q8_0`, indicam o nível de compressão escolhido.

### 8. Como avaliar o impacto

> **Em palavras simples.** O efeito da quantization só se conhece medindo. Compara-se o modelo original com o quantizado nas mesmas tarefas: a perplexity, a accuracy na sua tarefa, o consumo de memória e a velocidade. Em geral, 8 bits quase não alteram nada, 4 bits custam pouco, e abaixo disso a qualidade cai depressa. Mas depende do modelo e da tarefa.

Perplexity em um texto de referência é a triagem; a decisão sai das métricas de tarefa (Semana 21) medidas no modelo quantizado que **será servido**, não no modelo de treino.

> **No mundo real.** O estudo *The Case for 4-bit Precision* (Dettmers e Zettlemoyer, 2022) rodou mais de 35 mil experimentos com várias famílias de modelos e concluiu que, para um orçamento fixo de memória, **4 bits** é quase sempre o ponto ótimo: vale mais usar um modelo maior em 4 bits do que um menor em 8 ou 16. É a regra que orienta a escolha de modelos para hardware limitado.

## Erros comuns

- Esperar que QLoRA acelere o treino.
- Fundir o adapter no base de 4-bit.
- Avaliar em fp16 e servir em 4-bit sem medir a diferença.
- Esquecer `prepare_model_for_kbit_training`.
- `bnb_4bit_compute_dtype` em fp32 sem necessidade (lento) ou em bf16 em GPU sem suporte.
- Comparar quantizations por perplexity e ignorar a métrica da tarefa.

## Leitura e vídeos (fim de semana)

- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023): Sections 1 a 3.
- `transformers` documentation: *Quantization* → *bitsandbytes*, e a visão geral comparando os métodos.
- Hugging Face blog: *Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA*.
- Maarten Grootendorst, *A Visual Guide to Quantization*.
- Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* e Lin et al., *AWQ: Activation-aware Weight Quantization* (abstracts).
- llama.cpp documentation: a tabela de quantization types do GGUF.
- Opcional: Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference* (a referência clássica de affine quantization e QAT).

## Perguntas de autoavaliação

1. Por que o NF4 erra menos que a uniform quantization para pesos de redes neurais?
2. No QLoRA, o que é treinado e em que precisão? O que é o compute dtype?
3. Por que não é recomendável fundir o adapter diretamente no base de 4-bit?
4. Um modelo de 13B cabe em uma GPU de 16 GB para inference? Em que precisão?
5. Qual a diferença entre PTQ e QAT?
6. Com 12 GB de VRAM, você escolheria um modelo de 7B em fp16 ou um de 13B em 4-bit? Como decidiria?

## Perguntas do entrevistador

1. Você precisa fazer fine-tuning de um modelo de 7B a 14B parâmetros em uma única GPU de 16 GB ou 24 GB sem degradar a qualidade. Como a quantização NF4 com double quantization e paged optimizers funciona no QLoRA, e quando a economia de memória traz perda de precisão inaceitável?

## Entregável da semana

Notebook resolvido e uma tabela em `docs/`: memória e accuracy da tarefa para base fp32 + LoRA (Semana 19) e base 4-bit + LoRA.

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/04-Phase-4-Fine-Tuning/notebooks/semana_20_exercicio.ipynb) · [`semana_20_exercicio.ipynb`](../notebooks/semana_20_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/04-Phase-4-Fine-Tuning/notebooks/semana_20_gabarito.ipynb) · [`semana_20_gabarito.ipynb`](../notebooks/semana_20_gabarito.ipynb) |

[← Semana 19](/04-Phase-4-Fine-Tuning/docs/Semana-19-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 21 →](/04-Phase-4-Fine-Tuning/docs/Semana-21-Conceitos.md)
