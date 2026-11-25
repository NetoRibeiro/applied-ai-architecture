# Semana 21 — Avaliando o Fine-Tuned Model

**Fase 4: Parameter-Efficient Fine-Tuning (PEFT)** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Avaliar um modelo ajustado como se avalia qualquer mudança de arquitetura: contra alternativas mais baratas (zero-shot e few-shot), com task metrics, custo por requisição, teste out-of-distribution, verificação de forgetting e, por fim, o merge do adapter para deployment.

## Por que isso importa para um AI Architect

"O fine-tuning melhorou a loss" não justifica colocar um modelo em produção. A pergunta de arquitetura é: o ganho sobre um bom prompt paga o custo de treinar, versionar, servir e manter um modelo? E o modelo generaliza, ou decorou os templates do dataset?

## Conceitos

### 1. A escada de alternativas

> **Em palavras simples.** Antes de concluir que o fine-tuning valeu a pena, é preciso compará-lo com as alternativas mais baratas: o modelo original com um prompt simples, com um prompt bem trabalhado, com exemplos no prompt, com RAG. Cada degrau custa mais que o anterior. Se um degrau mais baixo já resolve, subir é desperdício.

Compare sempre, em ordem de custo crescente:

1. **Zero-shot:** só a instrução.
2. **Few-shot (in-context learning):** instrução + exemplos no prompt. Sem treino; paga input tokens em toda requisição e depende de quais exemplos são escolhidos.
3. **Fine-tuning:** custo de treino e de manutenção; prompt curto e comportamento consistente.

O fine-tuning se justifica quando ganha em qualidade **ou** quando entrega a mesma qualidade com menos tokens, menor latência ou um modelo menor, em volume suficiente para pagar o investimento. Faça a conta de **break-even**: custo de treinar e manter dividido pela economia por requisição.

Custos que costumam ser esquecidos: retreinar quando o base model for atualizado ou descontinuado, manter o dataset, e operar o serving de um modelo próprio.

> **No mundo real.** O estudo **Medprompt** (Nori et al., Microsoft, 2023) mostrou a força dos degraus baixos: o GPT-4, sem nenhum fine-tuning, usando apenas técnicas de prompt (escolha dinâmica de exemplos, raciocínio passo a passo e votação entre respostas), superou no benchmark de questões médicas MedQA modelos que haviam sido especialmente ajustados para medicina, passando de 90% de acerto. Prompt bem feito é uma baseline séria.

### 2. Métricas por tipo de tarefa

> **Em palavras simples.** Cada tipo de tarefa pede uma métrica. Para classificação e extração, há resposta certa e pode-se contar acertos. Para texto livre, como resumos, contar palavras em comum com uma referência diz pouco: dois bons resumos podem não compartilhar quase nenhuma palavra. Nesses casos, usa-se avaliação humana ou um LLM como juiz, com critérios escritos.

| Tarefa | Métricas |
|---|---|
| Classification / structured extraction | Parse rate (output válido), accuracy por campo, classification report, exact match |
| Generation com referência | ROUGE, BLEU, BERTScore, embedding similarity (fracas; use como triagem) |
| Open-ended generation | Human evaluation, LLM-as-a-judge com rubric, pairwise comparison |
| Language modeling | Loss / perplexity em held-out data |

Sobre as métricas de overlap: **ROUGE** mede sobreposição de n-grams com a referência, com ênfase em recall (sumarização); **BLEU**, com ênfase em precision (tradução). Ambas punem paráfrases corretas e premiam cópias erradas. **BERTScore** compara embeddings de tokens e tolera paráfrase. Nenhuma verifica fatos.

Para structured output, o **parse rate** é a primeira métrica: um modelo que acerta o conteúdo e quebra o formato quebra o sistema que o consome (Semana 26). Depois vêm a validade contra o schema e a accuracy por campo. Exact match do objeto inteiro é a mais rígida.

> **No mundo real.** O estudo *To Ship or Not to Ship* (Kocmi et al., Microsoft, 2021) comparou métricas automáticas de tradução com julgamentos humanos em milhares de pares de sistemas. A métrica tradicional **BLEU**, baseada em palavras em comum, foi a que menos concordou com os humanos na decisão de qual sistema era melhor; métricas baseadas em modelos, como o COMET, concordaram bem mais. Os autores recomendaram abandonar o BLEU como critério de decisão.

### 3. Pairwise comparison

> **Em palavras simples.** Dar uma nota absoluta a uma resposta é difícil; dizer qual de duas respostas é melhor é fácil. A **pairwise comparison** mostra as respostas de dois modelos para a mesma entrada, sem identificar qual é qual, e pergunta qual é a melhor. Somando muitas comparações, obtém-se uma taxa de vitória ou um ranking.

Para geração aberta, pedir a um avaliador (humano ou LLM) que escolha entre a resposta A e a B é mais confiável do que pedir uma nota absoluta. O **win rate** entre dois modelos, com as posições A/B embaralhadas para evitar position bias, é uma métrica simples e sensível. É também o formato de dado que alimenta o DPO (Semana 22).

> **No mundo real.** A **Chatbot Arena**, criada pelo grupo LMSYS de Berkeley em 2023, funciona assim: o visitante faz uma pergunta, recebe respostas de dois modelos anônimos e vota. Com milhões de votos acumulados, calcula-se um ranking no estilo **Elo**, o sistema do xadrez. Tornou-se uma das referências mais consultadas para comparar LLMs, justamente por se basear em preferências humanas em vez de provas de múltipla escolha.

### 4. In-distribution vs. out-of-distribution

> **Em palavras simples.** Dados **in-distribution** se parecem com os do treino; dados **out-of-distribution** são diferentes: outro período, outro tipo de cliente, outro jeito de escrever. Um modelo ajustado costuma ir muito bem nos primeiros e pior nos segundos. Como o mundo real sempre traz casos novos, é preciso testar com os dois.

O test set da Semana 17 vem dos mesmos templates do treino: mede o **melhor caso**. Um conjunto **out-of-distribution (OOD)**, com frases escritas de outro jeito, por outra pessoa, em outra data, mede o que acontecerá em produção. O **generalization gap** entre os dois é a medida de quanto o modelo decorou.

No notebook, espere ver um campo generalizar melhor que o outro: aquele cuja pista no treino era uma frase fixa (prioridade ligada a expressões de urgência específicas) tende a quebrar quando a frase muda. Isso é **shortcut learning**: o modelo aprendeu a correlação superficial, não o conceito.

Como reduzir o gap: mais diversidade no dataset (paráfrases, vários autores), dados reais em vez de templates, e regularization.

> **No mundo real.** O estudo *Do ImageNet Classifiers Generalize to ImageNet?* (Recht et al., 2019) construiu um novo test set seguindo com todo o cuidado o mesmo processo do original. Mesmo assim, todos os modelos testados perderam de 11 a 14 pontos percentuais de accuracy. Pequenas diferenças na coleta dos dados, imperceptíveis para pessoas, bastaram. A queda fora da distribuição de treino é a regra, não a exceção.

### 5. Robustness

> **Em palavras simples.** **Robustness** é o modelo continuar funcionando quando a entrada muda de forma irrelevante: um erro de digitação, uma ordem diferente das frases, letras maiúsculas, uma instrução reformulada. Modelos ajustados podem ficar frágeis, por terem visto sempre o mesmo formato. O teste consiste em perturbar as entradas e ver se a resposta se mantém.

Além do OOD, vale testar: erros de digitação, textos muito curtos e muito longos, outro idioma, entradas vazias ou fora do escopo (o modelo deveria recusar ou devolver uma categoria "outros"?), e **adversarial inputs** (o texto do chamado contém instruções). Um extractor ajustado ainda é um LLM e ainda é suscetível a prompt injection.

> **No mundo real.** O paper **CheckList** (Ribeiro et al., melhor paper do ACL 2020) aplicou testes desse tipo a serviços comerciais de análise de sentimento da Microsoft, do Google e da Amazon, e a modelos de pesquisa de ponta. Todos falharam em casos elementares, como frases com negação (*não é um voo ruim*) ou a simples troca de um nome de pessoa ou de lugar, apesar das notas altas nos benchmarks.

### 6. Forgetting

> **Em palavras simples.** Ao aprender a tarefa nova, o modelo pode piorar no que fazia bem antes: responder a perguntas gerais, seguir instruções fora do domínio, escrever em outro idioma. Para detectar esse **forgetting**, guarda-se um pequeno conjunto de tarefas gerais e mede-se o modelo nele antes e depois do fine-tuning.

Verifique se o modelo ajustado ainda responde bem a pedidos gerais. Em LoRA o risco é menor e, com o adapter separado, basta desligá-lo (`with model.disable_adapter():`). Em modelos fundidos e em full fine-tuning é um teste obrigatório. Para uso em produção, rode um pequeno benchmark geral antes e depois.

Quando o modelo ajustado é usado **só** para a tarefa estreita, algum forgetting é aceitável. Quando ele substitui o modelo geral, não.

> **No mundo real.** O estudo *An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning* (Luo et al., 2023) ajustou vários LLMs abertos em sequências de tarefas e mediu o conhecimento geral, o raciocínio e a compreensão de leitura depois de cada etapa. O esquecimento apareceu em todos os modelos testados e, nos experimentos, tendeu a ser mais severo nos modelos maiores da mesma família.

### 7. Custo por requisição

> **Em palavras simples.** Um modelo ajustado tem um custo por uso que precisa entrar na conta: o tempo de GPU, se for hospedado pela empresa, ou o preço por token, se for hospedado pelo fornecedor. Em compensação, costuma permitir prompts mais curtos, sem longas instruções e exemplos. A comparação justa é o custo total por requisição, dos dois lados.

`custo ≈ input tokens × preço_in + output tokens × preço_out`

`latência ≈ TTFT (cresce com o input) + output tokens ÷ tokens por segundo`

Few-shot com dez exemplos pode multiplicar os input tokens por dez. **Prompt caching** (reaproveitar o processamento de um prefixo fixo) reduz esse custo em servidores que o suportam, e muda a conta a favor do few-shot. Um modelo ajustado pode dispensar o system prompt longo e os exemplos.

> **No mundo real.** Na tabela de preços da OpenAI de 2024, usar um `gpt-4o-mini` ajustado custava o **dobro** por token em relação ao modelo padrão (0,30 contra 0,15 dólar por milhão de tokens de entrada). O fine-tuning só se pagava se o prompt encurtasse para menos da metade ou se permitisse substituir um modelo maior e mais caro. Os valores mudam, mas a estrutura da conta permanece.

### 8. Significância

> **Em palavras simples.** Se o modelo ajustado acertou 84% e a baseline acertou 81% em 200 exemplos, a diferença pode ser acaso. Um teste de **significância** ou um intervalo de confiança mostra se o ganho resiste. Com conjuntos de teste pequenos, só diferenças grandes são confiáveis. É melhor saber disso antes de anunciar a melhoria.

Com 40 a 200 exemplos de teste, diferenças de poucos pontos são ruído. Use o bootstrap (Semana 15) ou, para taxas, um intervalo de confiança binomial. Treine com duas ou três seeds se o resultado for decisivo.

> **No mundo real.** O paper *With Little Power Comes Great Responsibility* (Card et al., 2020) analisou a literatura de processamento de linguagem e concluiu que muitos experimentos têm conjuntos de teste pequenos demais para detectar as melhorias que anunciam. Em tarefas populares do benchmark GLUE, por exemplo, diferenças típicas de um ponto estavam abaixo do que o tamanho do teste permitia distinguir com segurança.

### 9. Merge e deployment

> **Em palavras simples.** Para colocar o modelo ajustado em uso, há duas opções. Fundir o adaptador no modelo base gera um modelo único, simples de servir. Manter o adaptador separado permite que um só modelo base atenda várias tarefas, carregando o adaptador de cada uma conforme o pedido. A segunda opção economiza muita GPU quando há vários ajustes.

`merge_and_unload()` incorpora o adapter aos pesos e devolve um modelo comum, que pode ser salvo, quantizado (Semana 20) e servido como qualquer outro (Semana 23).

| | Adapter separado | Merged |
|---|---|---|
| Latência | Pequeno overhead (ou nenhum, se o servidor fizer o merge em memória) | Igual ao base |
| Várias especializações | Um base, vários adapters | Um modelo completo por tarefa |
| Storage | Megabytes | Gigabytes |
| Quantization para serving | Depende do servidor | Direta |

Confirme sempre que o modelo fundido produz a mesma saída que o base + adapter, como o notebook faz.

> **No mundo real.** O projeto **LoRA Land** (Predibase, 2024) publicou mais de 25 adaptadores do Mistral 7B, cada um ajustado para uma tarefa, todos servidos a partir de **uma única GPU** com um único modelo base. Segundo os autores, vários deles superavam o GPT-4 na respectiva tarefa. Servidores de inference como o vLLM oferecem o mesmo recurso: o adaptador é escolhido por requisição.

### 10. Model card e registro

> **Em palavras simples.** O **model card** é a ficha técnica do modelo: para que serve, com quais dados foi treinado, como foi avaliado, quais são os limites conhecidos e para que não deve ser usado. Junto com o registro da versão dos dados e do código, é o que permite a outra pessoa confiar no modelo, reproduzi-lo e saber quando não usá-lo.

Guarde junto do modelo: base model e versão, versão do dataset, hyperparameters, métricas in-distribution e OOD, comparação com zero-shot e few-shot, uso pretendido, usos fora do escopo, limitações e riscos conhecidos. É o conteúdo de um **model card** e a entrada do Model Registry (Semana 24). Um auditor, ou você mesmo em seis meses, deve conseguir responder "por que este modelo está em produção?" lendo esse documento.

> **No mundo real.** O conceito vem do paper *Model Cards for Model Reporting* (Mitchell et al., Google, 2019), que propôs relatar o desempenho do modelo separado por grupos relevantes, e não só a média. No **Hugging Face Hub**, o model card é o `README.md` de cada repositório, com campos padronizados. A documentação técnica exigida pelo **EU AI Act** para sistemas de alto risco cobre, em boa parte, os mesmos itens.

### 11. Regression testing

> **Em palavras simples.** **Regression testing** é manter um conjunto fixo de casos que o modelo precisa continuar acertando e rodá-lo a cada nova versão. Um ajuste que melhora a média pode quebrar casos que antes funcionavam. Sem esse teste, a quebra só aparece quando um usuário reclama.

O conjunto de avaliação desta semana vira um teste automático: toda nova versão do modelo (novo dataset, novo base model, nova quantization) precisa igualar ou superar a anterior nas mesmas métricas antes de ser promovida. É o equivalente, para modelos, do golden set do RAG (Semana 16).

> **No mundo real.** Em março de 2023, a OpenAI abriu o código do **OpenAI Evals**, o framework que usa internamente para verificar se uma nova versão de modelo regrediu em tarefas conhecidas, e convidou a comunidade a contribuir com casos de teste. A motivação declarada foi que, sem avaliações escritas, é muito difícil saber se uma mudança de modelo ou de prompt melhorou ou piorou uma aplicação.

## Erros comuns

- Comparar o modelo ajustado só com o base em zero-shot, sem um few-shot bem feito.
- Avaliar só in-distribution.
- Reportar ROUGE como se medisse correção.
- Ignorar o parse rate.
- Não verificar a equivalência entre o modelo fundido e o base + adapter.
- Comparar custos sem considerar prompt caching e volume.
- Promover um modelo sem model card.

## Leitura e vídeos (fim de semana)

- Hugging Face PEFT documentation: *Merging adapters* / `merge_and_unload` e *Model merging*.
- Mitchell et al., *Model Cards for Model Reporting* (Sections 1 a 4) e os *Model Cards* do Hugging Face Hub.
- Hamel Husain, *Your AI Product Needs Evals* (releitura, agora pensando em modelos ajustados).
- Geirhos et al., *Shortcut Learning in Deep Neural Networks* (abstract e figuras).
- Lin, *ROUGE: A Package for Automatic Evaluation of Summaries* e Zhang et al., *BERTScore* (abstracts), para saber o que essas métricas medem.
- Hugging Face `evaluate` documentation (ROUGE, exact match).
- Opcional: Ribeiro et al., *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList*.

## Perguntas de autoavaliação

1. Em que cenário few-shot é melhor escolha que fine-tuning?
2. O test set dá 98% e o conjunto OOD dá 70%. O que isso indica e o que você faria?
3. Por que o parse rate vem antes da accuracy?
4. Quando manter o adapter separado em vez de fundi-lo?
5. Por que ROUGE e BLEU são medidas fracas de qualidade para respostas abertas?
6. Como você calcularia o break-even de um projeto de fine-tuning?

## Perguntas do entrevistador

1. Antes de aprovar um modelo fine-tuned para produção, você precisa justificar o investimento para o CTO. Como você desenha o benchmark contra zero-shot/few-shot de modelos frontier, calcula o break-even considerando treino + serving e valida contra regressão em tarefas gerais?

## Entregável da semana

Notebook resolvido e um `MODEL_CARD.md` em `docs/` com a tabela comparativa (zero-shot, few-shot, fine-tuned), custo em tokens, resultado out-of-distribution e limitações.

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/04-Phase-4-Fine-Tuning/notebooks/semana_21_exercicio.ipynb) · [`semana_21_exercicio.ipynb`](../notebooks/semana_21_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/04-Phase-4-Fine-Tuning/notebooks/semana_21_gabarito.ipynb) · [`semana_21_gabarito.ipynb`](../notebooks/semana_21_gabarito.ipynb) |

[← Semana 20](/04-Phase-4-Fine-Tuning/docs/Semana-20-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 22 →](/04-Phase-4-Fine-Tuning/docs/Semana-22-Conceitos.md)
