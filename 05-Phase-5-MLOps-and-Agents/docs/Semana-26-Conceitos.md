# Semana 26 — Agent-First Analytics: Structured Outputs para Agents

**Fase 5: MLOps, Deployment e Agent-First Products** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Projetar produtos analíticos cujo consumidor é um AI agent, não uma pessoa olhando um dashboard: outputs em schemas rígidos, tools com contrato explícito, structured errors e um agent loop que valida e se recupera. No final, você automatiza parte do próprio trabalho de Data Science.

## Por que isso importa para um AI Architect

A análise de dados está deixando de terminar em um gráfico. Cada vez mais, um pipeline de GenAI chama uma função analítica, lê o resultado e decide o próximo passo sozinho. Uma tabela bonita não serve para isso; um objeto tipado, com unidades, incerteza e avisos, serve. Projetar "agent-first" é o que diferencia um time de dados moderno, e é uma habilidade de design de API tanto quanto de estatística.

## Conceitos

### 1. Human-first vs. agent-first

> **Em palavras simples.** Ferramentas feitas para pessoas dependem de telas, mensagens livres e bom senso de quem lê. Ferramentas feitas para **agents** (programas guiados por LLM) precisam de outra coisa: entradas e saídas com formato exato, descrições claras do que cada operação faz e erros que expliquem o que corrigir. **Agent-first** é projetar pensando nesse leitor que só entende o que está escrito.

| | Para pessoas | Para agents |
|---|---|---|
| Formato | Tabela, gráfico, texto livre | JSON validado por schema |
| Ambiguidade | Tolerada (a pessoa interpreta) | Proibida (o agent age sobre o literal) |
| Unidades e definições | No título do gráfico | Em campos explícitos |
| Incerteza | "Parece significativo" | Confidence interval, p-value, sample size |
| Erro | Mensagem na tela | Error object com código e ação sugerida |
| Decisão | Fica com quem lê | Campo de recomendação com valores enumerados |
| Volume | Tudo o que couber na tela | O mínimo necessário: context window é recurso escasso |

A análise continua sendo a mesma; muda a interface.

> **No mundo real.** O paper **SWE-agent** (Yang et al., Princeton, 2024) cunhou o termo *agent-computer interface*. Os autores mostraram que dar ao LLM um terminal comum funcionava mal; ao criar comandos próprios para o agent, como abrir um arquivo mostrando só uma janela de linhas e editar com verificação de sintaxe, a taxa de resolução de problemas reais do GitHub (benchmark SWE-bench) subiu de forma expressiva, com o mesmo modelo.

### 2. Schemas como contrato

> **Em palavras simples.** Um **schema** é a descrição formal de um dado: quais campos existem, de que tipo é cada um, quais são obrigatórios, quais valores são aceitos. Ele funciona como um contrato entre quem envia e quem recebe. Se o dado não segue o contrato, é recusado na entrada, com uma explicação, em vez de causar um erro obscuro mais adiante.

Com Pydantic, o schema é código: tipos, valores permitidos (`Literal`, `Enum`), limites (`Field(ge=0, le=1)`) e descriptions. `model_json_schema()` exporta o **JSON Schema**, que é exatamente o que as APIs de tool calling / function calling e o protocolo MCP usam para descrever tools a um LLM.

Boas práticas:

- Um campo, um significado. Nada de `resultado: str` contendo três coisas.
- Enumerar decisões: `Literal["lancar", "nao_lancar", "continuar_teste"]`, não texto livre.
- Nomes autoexplicativos e com unidade: `latencia_p95_ms`, não `lat`.
- `description` em cada campo: o modelo lê.
- Metadata: `schema_version`, período dos dados, definição da métrica.
- `warnings`: o que um analista experiente comentaria ao lado do número (sample ratio mismatch, amostra pequena, dados incompletos).
- Campos opcionais explícitos (`Optional`) em vez de strings vazias ou `-1`.

**Schema evolution:** agentes e pipelines dependem do contrato. Adicionar campos opcionais é compatível; renomear ou remover exige nova versão.

> **No mundo real.** Quando a OpenAI lançou o **function calling**, em junho de 2023, escolheu o **JSON Schema** para descrever os parâmetros de cada função, o mesmo padrão já usado pela especificação OpenAPI para descrever APIs web. Os demais fornecedores seguiram o mesmo caminho. Na prática, bibliotecas como o **Pydantic** geram o schema a partir de classes Python e validam a resposta do modelo com o mesmo código.

### 3. Anatomia de uma tool

> **Em palavras simples.** Uma **tool** é uma função que o modelo pode pedir para executar. Ela tem um nome, uma descrição em linguagem natural, os parâmetros com os seus tipos e o formato do resultado. A descrição é lida pelo modelo para decidir quando e como usar a tool; por isso é escrita com o cuidado de um bom prompt, com exemplos e limites.

Nome, description, input schema, output schema e a função. A description é parte do prompt: é por ela que o modelo decide **quando** e **como** usar a tool.

Uma boa description diz o que a tool faz, quando usar e quando **não** usar, o formato dos argumentos com exemplo (ex.: "taxa entre 0 e 1; envie 0.05 para 5%") e o que ela devolve.

Propriedades desejáveis da função: determinística; **idempotent** (chamar duas vezes não causa dano); rápida; sem side effects não declarados; com paginação ou limite de tamanho no output. Tools de escrita (que alteram o mundo) merecem separação clara das de leitura, e confirmação humana.

Granularidade: poucas tools de alto nível, alinhadas a tarefas ("analisar um A/B test"), funcionam melhor do que muitas tools de baixo nível que o agent precisa encadear ("somar", "dividir", "calcular z").

> **No mundo real.** No texto *Building Effective Agents* (2024), a Anthropic conta que, ao construir o seu agent para o SWE-bench, gastou mais tempo otimizando as tools do que o prompt geral. Um exemplo: o modelo errava ao usar caminhos de arquivo relativos depois de mudar de diretório; a correção foi exigir caminhos absolutos na definição da tool, e o erro desapareceu. O conserto estava na interface, não no modelo.

### 4. Structured errors

> **Em palavras simples.** Quando uma tool falha, a mensagem de erro é a única pista que o agent tem para se corrigir. Um erro vago, como *falha na operação*, leva a tentativas às cegas. Um **structured error** informa o tipo do erro, o campo com problema, o valor recebido, o que era esperado e se vale a pena tentar de novo.

```json
{"ok": false, "erro": "ARGUMENTO_INVALIDO",
 "detalhe": "taxa_base: Input should be less than 1",
 "sugestao": "Envie a taxa entre 0 e 1 (0.05 para 5%)."}
```

Isso permite ao agent se corrigir no passo seguinte. Um stack trace Python, não. Distinga erros **recuperáveis** (argumento inválido: tente de novo) de **não recuperáveis** (sem permissão: pare e informe).

> **No mundo real.** A API da **Stripe** é a referência citada com mais frequência: cada erro traz `type`, `code`, `param` (o campo que causou o problema), uma mensagem legível e um link para a documentação. A mesma ideia foi padronizada pela IETF na **RFC 9457** (*Problem Details for HTTP APIs*). APIs com erros nesse formato, feitas para desenvolvedores humanos, revelaram-se também as mais fáceis de usar por agents.

### 5. Tool calling

> **Em palavras simples.** No **tool calling**, o modelo não executa nada. Ele recebe a lista de tools disponíveis e, quando julga necessário, responde com um pedido estruturado: chame esta função com estes argumentos. O programa em volta valida o pedido, executa a função e devolve o resultado ao modelo, que então continua. O controle fica sempre com o programa.

O fluxo padronizado pelas APIs de LLM:

1. A aplicação envia as mensagens e a lista de tools (nome, description, JSON Schema dos parâmetros).
2. O modelo responde com texto **ou** com um pedido de tool call: nome e argumentos em JSON.
3. A aplicação **valida** os argumentos, executa a função e devolve o resultado como uma mensagem de papel `tool`.
4. O modelo continua, podendo pedir outra tool ou dar a resposta final.

O modelo nunca executa nada: ele pede. Quem executa, e portanto quem é responsável por validar e autorizar, é a aplicação.

> **No mundo real.** O **Berkeley Function-Calling Leaderboard** (BFCL), do projeto Gorilla, avalia os modelos nessa habilidade com milhares de funções reais, em várias linguagens e cenários, incluindo casos em que nenhuma função deve ser chamada. Os resultados mostram diferenças grandes entre modelos e um erro recorrente: chamar uma função quando deveria responder diretamente ou pedir a informação que falta.

### 6. O agent loop

> **Em palavras simples.** O **agent loop** é o ciclo básico de um agent: o modelo pensa, pede uma ação, recebe o resultado, pensa de novo, e repete até concluir a tarefa ou atingir um limite. É simples de programar. A dificuldade está em evitar que os erros se acumulem ao longo dos passos, e por isso há limite de iterações, validação a cada passo e registro de tudo.

Um agent é um LLM em loop com tools: observa, decide, age, observa o resultado, repete, até concluir ou atingir um limite. O padrão **ReAct** (reason + act) intercala raciocínio e ação.

Controles obrigatórios: `max_steps`; timeout e orçamento de tokens; validação de toda saída do modelo (LLM output é input não confiável); logging de cada passo como span (Semana 25); least privilege (Semana 25).

**Workflows vs. agents.** Um **workflow** tem o caminho definido no código (prompt chaining, routing, parallelization), com o LLM em passos específicos. Um **agent** decide o caminho. Workflows são mais previsíveis, baratos e testáveis; use agents quando o caminho realmente não pode ser previsto. A maioria dos problemas corporativos é resolvida por workflows.

> **No mundo real.** O benchmark **SWE-bench** (Jimenez et al., 2023) pede que o sistema resolva problemas reais de repositórios Python no GitHub. Na publicação original, o melhor modelo, usado com busca simples e sem loop de ações, resolveu apenas **1,96%** dos casos. Pouco mais de um ano depois, agents que executam, testam e corrigem em ciclo passavam da metade na versão verificada do benchmark. O salto veio do loop e das tools, tanto quanto dos modelos.

### 7. Structured output de LLMs

> **Em palavras simples.** Para um programa consumir a resposta de um LLM, ela precisa vir em formato exato. Pedir no prompt ajuda e não garante. O **constrained decoding** garante: a cada token, o sistema só permite que o modelo escolha entre as opções que mantêm a saída válida segundo o schema. O resultado é sempre um JSON bem formado, com os campos corretos.

Três níveis de garantia:

1. **Prompting:** pedir JSON e torcer. Frágil.
2. **Validate and retry:** validar com Pydantic e, em caso de erro, devolver a mensagem de erro ao modelo para nova tentativa. Robusto, com custo extra de tokens. Bibliotecas como `instructor` implementam o padrão.
3. **Constrained decoding (grammar / schema-guided):** a cada token, o servidor mascara os logits dos tokens que tornariam a saída inválida. A saída é sintaticamente válida por construção. Disponível em vLLM, llama.cpp, Outlines e nas APIs comerciais ("structured outputs").

Constrained decoding garante a **sintaxe**, não a **semântica**: o JSON é válido, o valor pode estar errado. Fine-tuning (Fase 4) eleva o parse rate e a correção ao mesmo tempo.

> **No mundo real.** A biblioteca aberta **Outlines** (Willard e Louf, 2023) mostrou como fazer isso com eficiência: converte o schema em uma máquina de estados e, a cada passo, anula a probabilidade dos tokens inválidos. O **llama.cpp** oferece o mesmo recurso por meio de gramáticas, e as APIs comerciais o incorporaram em 2024 com o nome de *structured outputs*. A validação com Pydantic continua necessária para as regras que o schema não expressa.

### 8. MCP (Model Context Protocol)

> **Em palavras simples.** O **MCP** (*Model Context Protocol*) é um padrão aberto para ligar aplicações de IA a ferramentas e fontes de dados. Antes dele, cada integração era feita sob medida para cada aplicação. Com o protocolo, uma ferramenta é exposta uma única vez, por um **MCP server**, e qualquer aplicação compatível consegue usá-la. A comparação usual é com o conector USB.

Um protocolo aberto que padroniza como aplicações de LLM descobrem e chamam tools, leem resources e usam prompts de servidores externos. Um time de dados pode expor suas ferramentas analíticas como um **MCP server**, e qualquer cliente compatível (assistentes, IDEs, agents) passa a usá-las sem integração específica. Para um produto analítico agent-first, é o canal de distribuição natural. Trate um MCP server como qualquer API: autenticação, autorização por usuário, rate limiting, auditoria.

> **No mundo real.** O protocolo foi publicado pela Anthropic em novembro de 2024 como padrão aberto. Em 2025, OpenAI e Google anunciaram suporte em seus produtos, e IDEs como VS Code e Cursor passaram a aceitar MCP servers. Surgiram milhares de servidores públicos, para GitHub, bancos de dados, Slack e sistemas de arquivos. O mesmo servidor que você escreve no notebook desta semana pode ser usado por vários clientes diferentes.

### 9. Avaliando ferramentas e agents

> **Em palavras simples.** Avaliar um agent é diferente de avaliar uma resposta. Importa se a tarefa foi concluída, e também por qual caminho: quantos passos, quais tools, quanto custou, se violou alguma regra. E importa a **consistência**: um agent que acerta em uma tentativa e erra na seguinte, na mesma tarefa, não é confiável. Por isso se repete cada tarefa várias vezes.

- **Tool-level:** testes unitários da função e do schema.
- **Tool selection accuracy:** dado um pedido, o modelo escolhe a tool certa com os argumentos certos?
- **Task success rate:** o objetivo final foi alcançado?
- **Trajectory:** quantos passos, quantos erros recuperados, quanto custou.

Um golden set de tarefas, com o resultado esperado, cumpre aqui o papel que o golden set de perguntas cumpre no RAG.

> **No mundo real.** O benchmark **τ-bench** (Sierra, 2024) simula atendimento ao cliente em varejo e companhias aéreas, com regras de negócio a respeitar, e introduziu a métrica **pass^k**: a taxa de tarefas resolvidas em **todas** as `k` tentativas. Os melhores modelos da época resolviam menos da metade das tarefas em uma tentativa, e a taxa caía para menos de 25% quando se exigiam oito acertos seguidos no domínio de varejo.

### 10. AI-powered workflows para Data Science

> **Em palavras simples.** Agents também mudam o trabalho de quem faz ciência de dados: um assistente pode explorar um dataset, escrever e rodar código, treinar modelos e redigir o relatório. O papel da pessoa se desloca para definir o problema, fornecer boas ferramentas e, principalmente, **verificar**: conferir se não houve leakage, se a métrica é a correta, se a conclusão se sustenta.

Tarefas que agents já assumem bem, com revisão humana:

- **Experiment design review:** tamanho de amostra, duração, unidade de randomização, número de métricas primárias, risco de spillover (a tool `revisar_experimento` do notebook).
- **Code review** de análises: leakage, comparações múltiplas, joins que duplicam linhas.
- **Documentação:** gerar o relatório do teste a partir dos resultados estruturados.
- **Metric anomaly triage:** investigar quedas em métricas decompondo por dimensão.
- **Text-to-SQL** sobre uma semantic layer com métricas definidas.

O padrão é sempre o mesmo: **tools determinísticas fazem a conta; o LLM orquestra e redige.** Nunca peça ao LLM para calcular um p-value de cabeça.

> **No mundo real.** O benchmark **MLE-bench** (OpenAI, 2024) reúne 75 competições reais do Kaggle e pede a um agent que faça tudo sozinho: entender o problema, preparar os dados, treinar e enviar as previsões. A melhor configuração testada alcançou resultado equivalente a medalha em cerca de **17%** das competições. É um desempenho notável e, ao mesmo tempo, a medida de quanto julgamento humano ainda é necessário no restante.

## Erros comuns

- Devolver texto livre ou uma tabela formatada como output de uma tool.
- Descriptions vagas ("analisa dados").
- Confiar nos argumentos gerados pelo modelo sem validar.
- Agent loop sem `max_steps`.
- Deixar o LLM fazer aritmética em vez de chamar uma função.
- Tools demais, com responsabilidades sobrepostas.
- Outputs enormes que consomem a context window.
- Tools de escrita sem confirmação nem idempotency.

## Leitura e vídeos (fim de semana)

- Pydantic documentation: *JSON Schema* e *Fields*.
- Anthropic, *Building effective agents* e *Writing effective tools for agents* (engineering blog).
- *Model Context Protocol* specification: overview e a seção *Tools*.
- vLLM documentation, *Structured Outputs*, ou Outlines documentation.
- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (abstract e figura 1).
- Willard & Louf, *Efficient Guided Generation for Large Language Models* (a base do constrained decoding; abstract).
- OpenAI ou Anthropic documentation: *Function calling / Tool use* (o fluxo de mensagens).
- Opcional: Jason Liu, documentação do `instructor` (*validate and retry*).

## Perguntas de autoavaliação

1. Por que `recomendacao: str` é pior que `recomendacao: Literal[...]` para um consumidor agent?
2. O que deve conter um erro para que um agent consiga se corrigir?
3. Por que a description de uma tool faz parte do prompt?
4. Quais os três níveis de garantia de structured output e o que o constrained decoding não garante?
5. Qual a diferença entre workflow e agent, e quando preferir cada um?
6. No tool calling, quem executa a função e por que isso importa para segurança?

## Entregável da semana

Notebook resolvido e, em `docs/`, a especificação de uma tool analítica do seu contexto de trabalho: nome, description, input schema, output schema e três erros possíveis.

### Checkpoint da Fase 5

Você deve conseguir, sem consultar: definir TTFT e explicar continuous batching; descrever o ciclo champion/challenger; desenhar um trace de RAG com seus spans; listar as quatro camadas de guardrails; especificar uma tool agent-first.

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/05-Phase-5-MLOps-and-Agents/notebooks/semana_26_exercicio.ipynb) · [`semana_26_exercicio.ipynb`](../notebooks/semana_26_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/05-Phase-5-MLOps-and-Agents/notebooks/semana_26_gabarito.ipynb) · [`semana_26_gabarito.ipynb`](../notebooks/semana_26_gabarito.ipynb) |

[← Semana 25](/05-Phase-5-MLOps-and-Agents/docs/Semana-25-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 27 →](/06-Phase-6-Causal-Inference/docs/Semana-27-Conceitos.md)
