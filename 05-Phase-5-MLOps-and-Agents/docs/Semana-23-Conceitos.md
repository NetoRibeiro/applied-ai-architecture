# Semana 23 — Model Serving: Latency, Throughput e Batching

**Fase 5: MLOps, Deployment e Agent-First Products** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Entender o que determina a latência e o throughput de um LLM em produção: time to first token, tokens per second, KV cache e batching. Você vai medir cada um e simular uma fila de requisições para fazer capacity planning.

## Por que isso importa para um AI Architect

Servir LLMs é um problema de filas sobre um recurso caro. As perguntas que chegam ao arquiteto são: quantas GPUs preciso para N usuários com p95 abaixo de X segundos? Vale mais um modelo maior ou duas réplicas de um menor? Self-hosting ou API? Sem os conceitos desta semana, a resposta é chute.

## Conceitos

### 1. As métricas de um serviço de LLM

> **Em palavras simples.** Um serviço de LLM se mede com poucos números. **TTFT** (*time to first token*) é o tempo até a primeira palavra aparecer, o que o usuário percebe como rapidez. **Tokens por segundo** é a velocidade com que o texto é escrito. **Throughput** é quantos pedidos o sistema atende ao mesmo tempo. E os **percentis** (p95, p99) mostram a experiência dos usuários mais azarados, que a média esconde.

| Métrica | Significado | Depende principalmente de |
|---|---|---|
| **TTFT** (time to first token) | Espera até a resposta começar | Queue time + prompt length |
| **TPOT / ITL** (time per output token / inter-token latency) | Intervalo entre tokens | Modelo, hardware, batch |
| **TPS** (tokens per second) | Inverso do TPOT; por requisição ou agregado | Idem |
| **End-to-end latency** | `TTFT + output tokens × TPOT` | Tamanho da resposta |
| **Throughput** | Tokens ou requisições por segundo no servidor | Batching |
| **p50 / p95 / p99** | Percentis de latência | Carga e variabilidade |

Em chat com streaming, o usuário percebe o TTFT e a fluidez; em batch processing, importa o throughput e o custo por milhão de tokens. Defina o **SLO** (ex.: p95 de TTFT < 1 s) antes de dimensionar.

Sempre percentis, nunca a média: a latência de LLMs tem cauda longa, porque o tamanho das respostas varia muito.

> **No mundo real.** Sites independentes como o **Artificial Analysis** medem continuamente esses números para as APIs comerciais e os publicam lado a lado: TTFT, tokens por segundo e preço por milhão de tokens de cada modelo em cada provedor. As diferenças são grandes: o mesmo modelo aberto pode ser várias vezes mais rápido em um provedor do que em outro, conforme o hardware e o servidor de inference usados.

### 2. Prefill e decode

> **Em palavras simples.** Uma resposta tem duas fases. No **prefill**, o modelo lê o prompt inteiro de uma vez, em paralelo: é rápido por token. No **decode**, escreve a resposta um token por vez, e cada token exige uma passagem completa pelo modelo: é lento por token. Por isso um prompt longo custa pouco tempo e uma resposta longa custa muito.

- **Prefill (prompt processing):** o prompt inteiro é processado em paralelo, em um único forward. Custo proporcional ao input; é **compute-bound**.
- **Decode:** cada output token exige um forward. É sequencial, domina o tempo em respostas longas e é **memory-bandwidth-bound**: a cada token, todos os pesos do modelo são lidos da memória.

Consequência: output tokens são muito mais caros em tempo que input tokens. É também por isso que APIs cobram mais pelo output.

> **No mundo real.** A diferença aparece nas tabelas de preço: as APIs cobram os tokens de saída bem mais caro que os de entrada. Em 2024, o GPT-4o custava 2,50 dólares por milhão de tokens de entrada e 10 dólares por milhão de saída; o Claude 3.5 Sonnet, 3 e 15 dólares. A razão de 4 a 5 vezes reflete o custo computacional real de gerar em vez de ler.

### 3. KV cache

> **Em palavras simples.** Ao escrever cada novo token, o modelo precisaria reler todo o texto anterior. O **KV cache** guarda os resultados intermediários já calculados para os tokens anteriores, de modo que só o token novo é processado. Troca-se memória por velocidade: a geração fica muito mais rápida, e cada conversa em andamento ocupa espaço na GPU.

Na geração, as keys e values de attention dos tokens já processados não mudam (Semana 7). Guardá-los evita reprocessar todo o prefixo a cada novo token: o custo por token deixa de crescer com o comprimento da sequência.

O preço é VRAM: `2 × n_layers × n_kv_heads × head_dim × tokens × bytes`, por sequência. Para um modelo de 7B com contexto de 8k em fp16, cada sequência ocupa de 1 GB (com GQA) a 4 GB (sem). É o KV cache, não os pesos, que limita quantas requisições simultâneas cabem na GPU. Grouped-Query Attention (Semana 8) existe para reduzi-lo.

**Prefix / prompt caching:** reaproveitar o KV cache de um prefixo comum (system prompt, few-shot examples, documento longo) entre requisições. Reduz muito TTFT e custo quando o prefixo é estável.

> **No mundo real.** As APIs transformaram esse mecanismo em recurso comercial, o **prompt caching**. Ao lançá-lo em 2024, a Anthropic informou que a parte do prompt já guardada em cache é cobrada a 10% do preço normal de entrada, e que a latência de prompts longos podia cair em até 85%. Para aplicações que repetem um contexto grande, como um manual ou um código-fonte, a economia é direta.

### 4. Batching

> **Em palavras simples.** **Batching** é processar vários pedidos juntos, aproveitando que a GPU faz muitas contas em paralelo. O **continuous batching** vai além: em vez de esperar um lote inteiro terminar, o servidor encaixa novos pedidos assim que uma resposta acaba, a cada token gerado. A GPU fica sempre ocupada, e ninguém espera pelos pedidos mais demorados dos outros.

Uma GPU processa várias sequências no mesmo forward quase pelo custo de uma, até saturar.

- **No batching:** uma requisição por vez. Throughput mínimo.
- **Static batching:** espera juntar `B` requisições ou estourar um timeout. Aumenta o throughput; todas esperam a mais longa terminar, e a GPU processa padding.
- **Dynamic batching:** forma batches com o que chegou em uma janela curta de tempo (Triton).
- **Continuous batching (iteration-level scheduling):** a cada decode step, requisições que terminaram saem do batch e novas entram. Sem espera por batch cheio e sem padding desperdiçado. É o que vLLM e TGI fazem, e a razão de entregarem várias vezes o throughput de um loop ingênuo.

O tradeoff fundamental: batches maiores aumentam o throughput do servidor e a latência de cada requisição.

> **No mundo real.** Um teste publicado pela Anyscale em 2023 comparou formas de servir o mesmo modelo na mesma GPU. Com continuous batching no **vLLM**, o throughput chegou a ser **23 vezes** maior do que processando um pedido por vez, com latência mediana menor. A técnica vem do sistema Orca (OSDI 2022) e hoje está em todos os servidores de inference relevantes.

### 5. Um pouco de queueing theory

> **Em palavras simples.** Quando os pedidos chegam mais rápido do que o sistema atende, forma-se fila. A **queueing theory** mostra algo pouco intuitivo: o tempo de espera não cresce aos poucos, ele dispara quando o sistema se aproxima do limite. A 70% de ocupação, tudo flui; a 95%, a espera fica várias vezes maior. Por isso se planeja folga.

Utilização `ρ = arrival rate / service rate`. Em uma fila simples, o tempo de espera cresce como `ρ / (1 − ρ)`: a 50% de utilização a espera é igual ao tempo de serviço; a 90%, nove vezes; perto de 100%, explode. O gráfico de p95 × carga do notebook é essa curva (o "hockey stick").

Regras que saem daí: dimensione para o **pico**, não para a média; mantenha folga (operar a 60–70% da capacidade); e meça com **load testing**, porque chegadas reais vêm em rajadas.

> **No mundo real.** O artigo *The Tail at Scale* (Dean e Barroso, Google, 2013) deu o exemplo que virou referência: se um servidor é lento em apenas 1% das vezes e um pedido depende de 100 servidores, **63%** dos pedidos serão lentos. Latências raras em um componente tornam-se comuns no sistema. Daí a prática de acompanhar p99, e não a média, em qualquer serviço.

### 6. Inference engines

> **Em palavras simples.** Um **inference engine** é o programa especializado em servir o modelo: administra a memória da GPU, agrupa os pedidos, mantém os caches e expõe uma API. A diferença entre um bom engine e o código de um notebook pode ser de dez vezes ou mais em capacidade, com o mesmo modelo e a mesma GPU.

| Engine | Uso típico |
|---|---|
| **vLLM** | Servidor de GPU de alto throughput; OpenAI-compatible API; continuous batching; PagedAttention; suporta LoRA adapters e structured outputs |
| **TGI** (Text Generation Inference) | Servidor de produção da Hugging Face |
| **NVIDIA Triton** (com TensorRT-LLM ou vLLM backend) | Plataforma multi-model e multi-framework; comum em ambientes corporativos |
| **llama.cpp / Ollama** | CPU, laptops e GPU única; formato GGUF |
| **SGLang**, **LMDeploy** | Alternativas de alto desempenho |

**PagedAttention** (vLLM) aloca o KV cache em blocos, como páginas de memória virtual, evitando fragmentação e reserva antecipada. É o que permite muitas sequências simultâneas.

Subir um modelo: `vllm serve <model> --max-model-len 4096`; o cliente fala com `http://host:8000/v1` usando o SDK da OpenAI.

> **No mundo real.** O **vLLM** nasceu de um paper de Berkeley (Kwon et al., SOSP 2023) que introduziu o **PagedAttention**: a memória do KV cache passa a ser administrada em páginas, como a memória virtual de um sistema operacional, o que elimina desperdício. O paper reportou throughput de 2 a 4 vezes maior que os sistemas de referência da época, com a mesma latência. Tornou-se o servidor aberto mais usado para LLMs em GPU.

### 7. Outras otimizações

> **Em palavras simples.** Há outras formas de acelerar. **Speculative decoding** usa um modelo pequeno para adiantar vários tokens, que o modelo grande apenas confere. **Quantization** reduz a memória. **Prompt caching** evita reprocessar o que se repete. **Streaming** não acelera nada, mas mostra o texto enquanto ele é escrito, e a espera parece menor.

Quantization (Semana 20); Flash Attention; compilação com TensorRT; **speculative decoding** (um modelo pequeno propõe vários tokens e o grande os verifica em um só forward); multi-GPU por tensor parallelism; e, no nível da aplicação, limitar `max_tokens`, encurtar prompts, response caching e semantic caching.

> **No mundo real.** O paper *Fast Inference from Transformers via Speculative Decoding* (Leviathan et al., Google, 2023) demonstrou aceleração de 2 a 3 vezes em um modelo T5 grande, com uma propriedade importante: a saída segue exatamente a mesma distribuição do modelo original. Ou seja, ganha-se velocidade sem perda de qualidade. O Google relatou depois o uso da técnica em produtos de busca.

### 8. Capacity planning

> **Em palavras simples.** **Capacity planning** é calcular quantas GPUs são necessárias para atender à demanda prevista dentro da latência prometida. Parte-se dos números: quantos pedidos por segundo no pico, quantos tokens por pedido, quantos tokens por segundo cada GPU entrega. Acrescenta-se folga para picos e falhas. Errar para menos gera filas; errar para mais gera custo ocioso.

1. Estime a carga: requisições por segundo no pico, input e output tokens médios e no p95.
2. Meça o throughput de uma réplica (tokens/s) no nível de concorrência que respeita o SLO.
3. `réplicas ≈ (req/s × output tokens por req) ÷ throughput por réplica`, com folga de 30 a 50%.
4. Valide com load test (ferramentas: GenAI-Perf, `llmperf`, Locust, k6) usando uma distribuição realista de tamanhos.
5. Planeje **autoscaling** com base em queue length ou em requisições em andamento, lembrando do **cold start**: carregar um modelo leva de dezenas de segundos a minutos.

> **No mundo real.** Nos primeiros meses de 2023, o ChatGPT exibia com frequência a mensagem *ChatGPT is at capacity right now*. Um relatório do banco UBS estimou que o serviço chegara a 100 milhões de usuários em cerca de dois meses, um crescimento que nenhum plano de capacidade previa. Mesmo com o apoio da Microsoft, a oferta de GPUs não acompanhou a demanda, e a OpenAI chegou a pausar novas assinaturas pagas no fim daquele ano.

### 9. Self-hosting vs. API

> **Em palavras simples.** Usar a **API** de um fornecedor é pagar por token: sem custo fixo, fácil de começar, o preço acompanha o uso. **Self-hosting** é manter as próprias GPUs: custo fixo alto, mais controle sobre dados e modelo, e custo por token menor apenas se o volume for grande e constante. A conta depende do volume, e deve incluir as pessoas que operam o sistema.

| | API gerenciada | Self-hosted |
|---|---|---|
| Custo | Por token; zero quando ocioso | Por GPU-hora; alto quando ocioso |
| Break-even | Baixo volume | Alto volume com utilização constante |
| Dados | Saem do ambiente | Ficam no ambiente |
| Controle | Modelo e versão do fornecedor | Total (adapters, quantization, versão fixa) |
| Operação | Nenhuma | Time, on-call, upgrades |

Custo por milhão de tokens em self-hosting: `preço da GPU por hora ÷ (tokens/s × 3600) × 10⁶`, dividido pela utilização real. Uma GPU ociosa metade do tempo dobra o custo efetivo.

> **No mundo real.** Uma conta com preços públicos: uma instância `g5.xlarge` da AWS, com uma GPU A10G de 24 GB, custava cerca de 1 dólar por hora sob demanda, ou aproximadamente 730 dólares por mês ligada continuamente, **atenda ela um pedido ou um milhão**. Com pouco tráfego, a API sai muito mais barata. Só com a GPU ocupada a maior parte do tempo o custo por token do self-hosting se aproxima do de um provedor.

### 10. Routing e model cascades

> **Em palavras simples.** Nem toda pergunta precisa do modelo mais caro. **Routing** é encaminhar cada pedido ao modelo adequado: perguntas simples vão a um modelo pequeno e barato, as difíceis a um grande. Em uma **cascade**, tenta-se primeiro o modelo barato e só se recorre ao caro se a primeira resposta não inspirar confiança.

Nem toda requisição precisa do modelo maior. Um **router** envia o fácil para um modelo pequeno e o difícil para o grande; uma **cascade** tenta o pequeno e escala se a confiança for baixa. O classifier da Semana 10 e o modelo ajustado da Fase 4 são peças desse desenho.

> **No mundo real.** O paper **FrugalGPT** (Chen, Zaharia e Zou, Stanford, 2023) testou cascatas com APIs comerciais reais. Em vários datasets, a cascata igualou o desempenho do GPT-4 com redução de custo de até **98%**, pois a maior parte das perguntas era resolvida por modelos mais baratos. Em alguns casos a combinação superou o GPT-4 sozinho, porque os modelos erram em perguntas diferentes.

## Erros comuns

- Dimensionar pela média de carga e de latência.
- Medir throughput com uma requisição por vez e extrapolar.
- Ignorar o KV cache ao calcular quantas requisições cabem na GPU.
- Não limitar `max_tokens` e deixar uma requisição ocupar a GPU por minutos.
- Left padding esquecido em batch manual.
- Comparar self-hosting e API assumindo 100% de utilização.
- Autoscaling sem considerar cold start.

## Leitura e vídeos (fim de semana)

- vLLM documentation: *Quickstart*, *OpenAI-Compatible Server* e *Optimization and Tuning*.
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (abstract e Sections 1–3).
- Anyscale, *How continuous batching enables 23x throughput in LLM inference while reducing p50 latency*.
- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models* (a origem do iteration-level scheduling; abstract).
- NVIDIA, *LLM Inference Benchmarking: Fundamental Concepts* (definições de TTFT, ITL, TPS).
- `transformers` documentation: *Best practices for generation with cache*.
- Databricks, *LLM Inference Performance Engineering: Best Practices*.

## Perguntas de autoavaliação

1. Por que o TTFT cresce com o tamanho do prompt e o TPOT (com cache) quase não?
2. O que o KV cache troca por quê? Por que ele limita a concorrência?
3. Qual a vantagem do continuous batching sobre o static?
4. 20 req/s no pico, 300 output tokens por requisição, 2.500 tokens/s por réplica. Quantas réplicas?
5. Por que a latência explode quando a utilização se aproxima de 100%?
6. Em que condições self-hosting sai mais barato que uma API?

## Perguntas do entrevistador

1. Seu serviço de LLM tem TTFT (Time to First Token) de 4 segundos no pico de tráfego com static batching, mesmo sem saturar a memória da GPU. Como continuous batching, KV cache com paginação e separação de capacidade entre prefill e decode resolvem o gargalo?

## Entregável da semana

Notebook resolvido e uma página em `docs/` com o capacity planning de um serviço hipotético (carga, SLO, réplicas, custo mensal estimado).

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/05-Phase-5-MLOps-and-Agents/notebooks/semana_23_exercicio.ipynb) · [`semana_23_exercicio.ipynb`](../notebooks/semana_23_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/05-Phase-5-MLOps-and-Agents/notebooks/semana_23_gabarito.ipynb) · [`semana_23_gabarito.ipynb`](../notebooks/semana_23_gabarito.ipynb) |

[← Semana 22](/04-Phase-4-Fine-Tuning/docs/Semana-22-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 24 →](/05-Phase-5-MLOps-and-Agents/docs/Semana-24-Conceitos.md)
