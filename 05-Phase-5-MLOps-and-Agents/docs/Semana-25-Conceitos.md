# Semana 25 — Observability e Guardrails

**Fase 5: MLOps, Deployment e Agent-First Products** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Instrumentar uma aplicação de LLM para que cada requisição seja observável (traces e spans), construir um input guardrail contra prompt injection avaliado com as métricas da Semana 1 e detectar drift no comportamento do sistema ao longo do tempo.

## Por que isso importa para um AI Architect

Sistemas de LLM falham de modo silencioso: nenhuma exceção é lançada quando o modelo alucina, quando o retriever traz o chunk errado ou quando um usuário contorna as instruções. Sem observability você descobre o problema pela reclamação do cliente; sem guardrails, pelo jornal.

## Conceitos

### 1. Monitoring vs. observability

> **Em palavras simples.** **Monitoring** é acompanhar indicadores definidos de antemão e avisar quando passam do limite: o painel do carro. **Observability** é ter registro suficiente para investigar um problema que ninguém previu: a caixa-preta do avião. Sistemas com LLM falham de maneiras inesperadas, e por isso precisam dos dois.

**Monitoring** responde a perguntas previstas ("a latência passou de 2 s?"). **Observability** permite investigar perguntas não previstas ("por que esta resposta específica saiu errada?"). Os três pilares clássicos são logs, metrics e traces; em aplicações de LLM, o trace é o mais importante, porque uma requisição passa por várias etapas não determinísticas.

> **No mundo real.** O livro *Site Reliability Engineering*, do Google (2016), consolidou os **four golden signals** que todo serviço deveria monitorar: latência, tráfego, erros e saturação. Em aplicações com LLM, esses quatro continuam necessários e não bastam: um serviço pode estar rápido, sem erros e com folga, e ainda assim estar respondendo errado. Daí a necessidade de registrar também o conteúdo de cada interação.

### 2. Trace e span

> **Em palavras simples.** Um **trace** é o registro completo do caminho de um pedido pelo sistema. Cada etapa do caminho é um **span**: a busca nos documentos, a chamada ao LLM, a execução de uma ferramenta. Cada span guarda quando começou, quanto durou, o que entrou e o que saiu. Com o trace, vê-se onde o tempo foi gasto e em que ponto a resposta desandou.

Um **trace** é o registro de uma requisição de ponta a ponta. Ele é uma árvore de **spans**: cada span é uma etapa (query understanding, retrieval, reranking, generation, tool call) com:

- `trace_id`, `span_id`, `parent_span_id` (a estrutura da árvore);
- start time e duration;
- inputs e outputs;
- **attributes**: modelo, temperature, input/output tokens, custo, scores de retrieval, versão do prompt.

Com traces você responde: qual etapa é o gargalo? Quanto custa cada requisição? O que exatamente o modelo recebeu quando respondeu errado?

O padrão aberto é o **OpenTelemetry (OTel)**, que define o formato e a propagação de contexto entre serviços; há semantic conventions específicas para GenAI (nomes padronizados de atributos como modelo e tokens). Ferramentas como **LangSmith**, **Arize Phoenix**, **Langfuse** e o **MLflow Tracing** recebem traces, exibem a árvore e rodam evals sobre eles. Instrumentar com OTel evita ficar preso a uma delas.

No notebook você escreve um decorador `@span` com `contextvars` para entender a mecânica; em produção, use o SDK.

> **No mundo real.** O conceito vem do paper **Dapper** (Google, 2010), que descreveu como rastrear um pedido através de centenas de serviços internos. Ele deu origem a ferramentas abertas e, depois, ao padrão **OpenTelemetry**. Ferramentas voltadas a LLMs, como LangSmith, Langfuse e Arize Phoenix, usam o mesmo modelo de trace e span, acrescentando o prompt, a resposta e o número de tokens em cada etapa.

### 3. O que monitorar

> **Em palavras simples.** Em uma aplicação com LLM, acompanham-se quatro grupos de indicadores. **Operação**: latência, erros, disponibilidade. **Custo**: tokens de entrada e de saída por pedido. **Qualidade**: avaliações automáticas por amostragem, taxa de recusas, feedback dos usuários. **Segurança**: tentativas de abuso e bloqueios dos guardrails. Cada grupo responde a uma pergunta diferente.

| Camada | Sinais |
|---|---|
| **Operations** | Latência p50/p95 por span, error rate, tokens, custo, rate limits |
| **Quality** | Faithfulness e relevance em amostra (Semana 16), refusal rate, user feedback |
| **Data** | Distribuição dos retrieval scores, tamanho e tópico das queries, idioma |
| **Security** | Tentativas de injection bloqueadas, PII na saída, tool calls anômalas |

**Online evaluation:** rodar as métricas reference-free da Semana 16 em uma amostra do tráfego (1 a 10%), de forma assíncrona. **User feedback** explícito (thumbs) é esparso e enviesado; sinais implícitos (reformular a pergunta, copiar a resposta, escalar para humano) são mais abundantes.

Cuidado com o que se grava: prompts e respostas contêm dados pessoais e confidenciais. Defina retenção, mascaramento e controle de acesso aos traces.

> **No mundo real.** O projeto **OpenTelemetry** mantém as *semantic conventions for generative AI*, que padronizam os nomes dos atributos registrados em cada chamada, como `gen_ai.request.model`, `gen_ai.usage.input_tokens` e `gen_ai.usage.output_tokens`. Com nomes comuns, os dados de aplicações e fornecedores diferentes podem ser analisados na mesma ferramenta, e o custo por funcionalidade ou por cliente sai de uma consulta simples.

### 4. Riscos de segurança próprios de LLMs

> **Em palavras simples.** LLMs trazem riscos que os sistemas tradicionais não tinham. O principal é a **prompt injection**: um texto malicioso, digitado pelo usuário ou escondido em um documento, que faz o modelo ignorar as suas instruções. Outros são o vazamento de dados confidenciais na resposta, a execução de ações indevidas por agents e a exposição do system prompt.

A referência do setor é a lista *OWASP Top 10 for LLM Applications*. Os que mais afetam RAG e agentes:

- **Direct prompt injection:** o usuário escreve instruções que tentam sobrepor as do sistema ("ignore as instruções anteriores").
- **Indirect prompt injection:** a instrução maliciosa está em um documento recuperado, uma página web, um e-mail ou o resultado de uma tool. O usuário pode ser inocente. É a mais perigosa, porque o conteúdo externo entra no contexto com a mesma "autoridade" do resto.
- **Sensitive information disclosure:** vazamento do system prompt, de dados de outros usuários (falha de access control no retriever), de PII, de dados de treino memorizados.
- **Excessive agency:** o agente tem permissão para ações que não deveria executar sem confirmação (enviar e-mail, apagar registros, fazer pagamentos).
- **Improper output handling:** tratar a saída do LLM como confiável (executar SQL ou código gerado, renderizar HTML).
- **Jailbreaks:** contornar as políticas de conteúdo do modelo.

A raiz do problema: um LLM não separa instruções de dados. Tudo é texto no mesmo contexto. Não existe, hoje, defesa que elimine prompt injection; existe redução de risco.

> **No mundo real.** Em fevereiro de 2023, poucos dias após o lançamento do chat do Bing, o estudante Kevin Liu fez o sistema revelar as suas instruções internas, inclusive o codinome *Sydney*, com uma única mensagem que pedia para ignorar as instruções anteriores. A Microsoft confirmou a autenticidade. A **OWASP** lista a prompt injection como o risco número 1 em seu *Top 10 for LLM Applications*.

### 5. Guardrails em camadas (defense in depth)

> **Em palavras simples.** Nenhuma proteção isolada basta; por isso se usam camadas (**defense in depth**). Na entrada, filtra-se o que o usuário envia. No meio, limita-se o que o modelo pode acessar e fazer. Na saída, verifica-se a resposta antes de entregá-la. E ações de consequência exigem confirmação humana. Se uma camada falha, a seguinte ainda segura.

1. **Input:** classifier de injection, topic filter, limite de tamanho, rate limiting.
2. **Context:** tratar conteúdo recuperado como dado; delimitá-lo claramente; access control no retriever (Semana 12); sanitizar documentos na ingestão.
3. **Output:** verificação de faithfulness, PII redaction, validação de schema (Semana 26), filtro de conteúdo.
4. **Action:** least privilege para tools; allowlists; **human-in-the-loop** em ações irreversíveis; sandbox para execução de código.

Nenhum classifier de injection é perfeito. A defesa real é limitar o **blast radius**: o que o modelo **pode fazer** quando for enganado. Um assistente que só lê documentos que o próprio usuário já pode ver, e não executa ações, tem pouco a perder.

Ferramentas: Llama Guard e Prompt Guard (classifiers), NeMo Guardrails e Guardrails AI (frameworks), Presidio (PII).

> **No mundo real.** Em dezembro de 2023, o chatbot do site de uma concessionária Chevrolet na Califórnia, baseado em um LLM sem restrições adicionais, foi levado por um usuário a *concordar* em vender uma Tahoe por 1 dólar, com a frase de que a oferta era juridicamente vinculante. As capturas de tela circularam pelo mundo. Faltavam todas as camadas: limite de assunto na entrada, verificação da saída e qualquer regra sobre o que o bot podia prometer.

### 6. Um guardrail é um classifier

> **Em palavras simples.** Um **guardrail** decide se um texto passa ou é bloqueado: é um classifier. Tudo o que foi visto na Semana 1 se aplica. Um limite rígido bloqueia mais ataques e também mais pedidos legítimos; um limite frouxo, o contrário. Mede-se com precision e recall, e o ponto de equilíbrio depende do custo de cada tipo de erro.

E portanto volta à Semana 1:

- **False negative:** ataque que passou. Custo: incidente de segurança.
- **False positive:** usuário legítimo bloqueado. Custo: produto frustrante, e pressão para desligar o guardrail.

A escolha do threshold é uma decisão de negócio explícita, em geral priorizando recall, com um false positive rate medido e aceito.

O notebook mostra um ponto importante: um classifier treinado em templates atinge métricas perfeitas no test set e deixa passar ataques escritos de outra forma. É o generalization gap da Semana 21 em um contexto **adversarial**: o atacante procura ativamente o que o seu test set não cobre. Por isso se faz **red teaming** contínuo, e por isso guardrails por regex ou por classifier simples são uma camada, nunca a defesa.

Latência: cada guardrail é mais um modelo no caminho. Rode os de input em paralelo com o retrieval; rode os de output em streaming quando possível.

> **No mundo real.** O **Llama Guard** (Meta, 2023) é um LLM ajustado para classificar prompts e respostas como seguros ou inseguros segundo uma lista de categorias; a OpenAI oferece a **Moderation API** com a mesma função. Ambos devolvem, além da decisão, uma pontuação por categoria, e cabe a quem integra escolher o threshold. Os papers e a documentação reportam precision e recall, exatamente como em qualquer classificador.

### 7. Drift

> **Em palavras simples.** **Drift** é o mundo mudar depois que o sistema foi construído. Os usuários passam a perguntar sobre outros assuntos, os documentos ficam desatualizados, o fornecedor atualiza o modelo. Nada quebra de forma visível: a qualidade apenas cai aos poucos. Detecta-se comparando, ao longo do tempo, a distribuição das entradas e os indicadores de qualidade.

O mundo muda depois do deploy:

- **Data drift (covariate shift):** as queries mudam (novos produtos, novo vocabulário, novo idioma).
- **Concept drift:** a resposta correta muda (a política foi alterada e a base não).
- **Model drift:** o fornecedor atualizou o modelo por trás da API; o comportamento muda sem que você tenha feito deploy. Fixe a versão do modelo.

Um sinal barato e eficaz em RAG é a distribuição do **top-1 retrieval score**: se ela cai, os usuários estão perguntando coisas que a base não cobre. Testes para comparar a distribuição atual com a de referência: **Kolmogorov–Smirnov** (KS), **Population Stability Index** (PSI; acima de 0,2 costuma indicar mudança relevante), e distância entre os centroides dos embeddings das queries.

Com amostras grandes, qualquer diferença minúscula é "estatisticamente significativa": alerte pelo **tamanho do efeito**, não só pelo p-valor.

> **No mundo real.** Em março de 2020, com o início da pandemia, o comportamento de compra mudou em dias. A **Instacart** relatou que a accuracy do modelo que previa a disponibilidade de produtos nas lojas caiu de 93% para 61%, porque os padrões aprendidos (papel higiênico quase sempre disponível) deixaram de valer. A empresa precisou encurtar a janela de dados usada no treino, de semanas para dias.

### 8. Alerting e resposta

> **Em palavras simples.** Um **alert** só é útil se pede uma ação. Alertas demais são ignorados; por isso se alerta para sintomas que afetam o usuário, com um responsável e um roteiro do que fazer. Depois de um incidente, escreve-se um **postmortem** sem busca de culpados: o que aconteceu, por que, e o que muda para não se repetir.

Um painel sem alertas não é monitoramento. Para cada métrica: threshold, janela, quem é avisado e o que fazer (**runbook**). Tenha um **kill switch**: a capacidade de desligar uma feature ou voltar ao modelo/prompt anterior sem deploy (Semana 24).

> **No mundo real.** Em 20 de março de 2023, a OpenAI tirou o ChatGPT do ar por algumas horas depois que usuários viram títulos de conversas de outras pessoas. O postmortem publicado dias depois explicou a causa, um bug em uma biblioteca aberta de acesso ao Redis, informou que dados de pagamento de cerca de 1,2% dos assinantes do ChatGPT Plus podiam ter sido expostos e listou as correções adotadas.

### 9. O ciclo de melhoria

> **Em palavras simples.** A operação alimenta a melhoria. Os casos em que o sistema falhou em produção são coletados, analisados e transformados em novos itens do conjunto de avaliação. Corrige-se o prompt, a busca ou o modelo; roda-se a avaliação para confirmar que melhorou sem quebrar o resto; e publica-se. Esse ciclo, repetido, é o que faz o sistema amadurecer.

Traces de produção → amostragem e revisão de erros (error analysis) → novos casos no golden set (Semana 16) → correção → avaliação offline → deploy. Observability é o que alimenta a avaliação. Times que melhoram rápido são os que **leem traces** toda semana.

> **No mundo real.** A equipe do **GitHub Copilot** publicou um estudo (Ziegler et al., 2022) com dados de uso de milhares de desenvolvedores. Entre as métricas coletadas, a que melhor previa a produtividade percebida pelos usuários era a **taxa de aceitação** das sugestões, mais do que a quantidade de código que permanecia no arquivo. A métrica passou a guiar as melhorias do produto: é um sinal barato, colhido em produção, que orienta o ciclo.

## Erros comuns

- Logar só a resposta final, sem as etapas intermediárias.
- Gravar PII nos traces sem política de retenção e de acesso.
- Confiar em um classifier de injection como única defesa.
- Dar a um agente permissões que o usuário final não tem.
- Tratar conteúdo recuperado como instrução.
- Medir drift só pelo p-valor.
- Não fixar a versão do modelo da API.

## Leitura e vídeos (fim de semana)

- *OWASP Top 10 for LLM Applications* (Prompt Injection, Sensitive Information Disclosure, Excessive Agency, Improper Output Handling).
- OpenTelemetry documentation: *Traces* (trace, span, context propagation) e as *Semantic Conventions for Generative AI*.
- Arize Phoenix ou Langfuse documentation: *Tracing* (conceitos) e *Evaluation*.
- Simon Willison, série sobre *prompt injection*, em especial *Prompt injection: What's the worst that can happen?* e os textos sobre indirect injection e o *lethal trifecta*.
- Greshake et al., *Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection* (abstract e seção 2).
- Shreya Shankar et al., *Who Validates the Validators?* ou Hamel Husain, *A Field Guide to Rapidly Improving AI Products* (sobre error analysis a partir de traces).
- Opcional: NIST *AI Risk Management Framework: Generative AI Profile*.

## Perguntas de autoavaliação

1. Qual a diferença entre trace e span? Que atributos você colocaria em um span de generation?
2. Por que a indirect prompt injection é mais difícil de tratar que a direta?
3. Seu guardrail tem recall 0,99 e bloqueia 8% dos usuários legítimos. É aceitável? De que depende?
4. Que sinal barato indica que a base de conhecimento deixou de cobrir as perguntas dos usuários?
5. O que significa limitar o blast radius e por que isso é mais confiável que detectar ataques?
6. Qual a diferença entre data drift, concept drift e model drift?

## Entregável da semana

Notebook resolvido e um documento curto em `docs/`: o painel mínimo de monitoramento que você exigiria para aprovar um RAG em produção (métricas, thresholds de alerta, responsável).

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/ai-architect-foundations/blob/main/05-Phase-5-MLOps-and-Agents/notebooks/semana_25_exercicio.ipynb) · [`semana_25_exercicio.ipynb`](../notebooks/semana_25_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/ai-architect-foundations/blob/main/05-Phase-5-MLOps-and-Agents/notebooks/semana_25_gabarito.ipynb) · [`semana_25_gabarito.ipynb`](../notebooks/semana_25_gabarito.ipynb) |

[← Semana 24](/05-Phase-5-MLOps-and-Agents/docs/Semana-24-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 26 →](/05-Phase-5-MLOps-and-Agents/docs/Semana-26-Conceitos.md)
