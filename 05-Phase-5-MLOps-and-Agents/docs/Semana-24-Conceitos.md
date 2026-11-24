# Semana 24 — Experiment Tracking e Model Registry com MLflow

**Fase 5: MLOps, Deployment e Agent-First Products** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Tornar o trabalho de modelagem rastreável: registrar parameters, metrics, artifacts e models de cada execução com o MLflow, comparar runs, promover um modelo no Model Registry e carregá-lo por um nome estável.

## Por que isso importa para um AI Architect

"Qual versão está em produção? Com que dados foi treinada? Por que trocamos a anterior?" Se a resposta depende da memória de alguém ou de um notebook perdido, não há governança. Experiment tracking e model registry são a base de auditoria, rollback e reproducibility, e são o primeiro degrau de maturidade em MLOps.

## Conceitos

### 1. O problema

> **Em palavras simples.** Em um projeto de machine learning, o código é a menor parte. Há dados que mudam, dezenas de experimentos com ajustes diferentes, modelos gerados em datas diferentes. Sem registro, em poucas semanas ninguém sabe dizer qual combinação de dados, código e ajustes produziu o modelo que está em produção. **Experiment tracking** é a disciplina de anotar isso automaticamente.

O código de ML é uma fração pequena de um sistema de ML. O resto é dado, configuração, infraestrutura, monitoramento. O paper *Hidden Technical Debt in Machine Learning Systems* descreve os custos escondidos: dependências de dados não versionadas, configuração espalhada, experimentos irreproduzíveis, pipelines frágeis. MLOps é a disciplina que trata disso.

> **No mundo real.** O paper *Hidden Technical Debt in Machine Learning Systems* (Sculley et al., Google, 2015) trouxe a figura mais reproduzida da área: uma caixinha chamada *ML code* cercada por caixas muito maiores, de coleta de dados, verificação, configuração, monitoramento e infraestrutura. Os autores, com base em anos de sistemas em produção no Google, argumentam que o custo de manter um sistema de ML está quase todo fora do modelo.

### 2. Os objetos do MLflow

> **Em palavras simples.** O **MLflow** organiza o trabalho em poucos conceitos. Um **experiment** é uma pasta para um problema. Cada **run** é uma tentativa, com os seus **parameters** (o que foi configurado), **metrics** (o que foi medido) e **artifacts** (os arquivos gerados, como o modelo e os gráficos). Com isso, qualquer tentativa pode ser encontrada, comparada e repetida.

| Objeto | O que guarda |
|---|---|
| **Experiment** | Um conjunto de runs de um mesmo problema |
| **Run** | Parameters, metrics, tags, artifacts e código de uma tentativa |
| **Registered Model** | Um nome estável (`detector-fraude`) com versões numeradas |
| **Model Version** | Um modelo específico, ligado ao run que o gerou |
| **Alias** | Um ponteiro móvel para uma versão (`@champion`, `@challenger`) |

Dois armazenamentos: o **backend store** (metadata: parâmetros, métricas, registry; um banco SQL) e o **artifact store** (arquivos: modelos, figuras; filesystem ou object storage como S3).

> **No mundo real.** O MLflow foi lançado pela Databricks em 2018 como projeto aberto, justamente para oferecer a qualquer empresa o que os grandes tinham internamente (como o FBLearner, da Meta, e o Michelangelo, da Uber). Hoje faz parte da Linux Foundation, tem dezenas de milhões de downloads mensais e é oferecido como serviço gerenciado pelas principais nuvens. A sua API virou um padrão de fato para registro de experimentos.

### 3. O que registrar em cada run

> **Em palavras simples.** Em cada run, registra-se o necessário para reproduzi-lo e entendê-lo: os hyperparameters, a versão do código, a versão dos dados, a **seed** do sorteio, as métricas ao longo do treino e ao final, e o ambiente (versões das bibliotecas). A regra prática: se alguém perguntar daqui a seis meses como este modelo foi feito, o run responde sozinho.

- **Parameters:** hyperparameters, base model, versão do dataset, seed. São valores de entrada, imutáveis no run.
- **Metrics:** as de negócio (recall, precision, custo; Semana 1), não só a loss. Podem ter um `step`, o que gera curvas (loss por epoch).
- **Artifacts:** confusion matrix, relatório, amostras de erros, o modelo.
- **Tags:** git commit, autor, motivo do run, ticket.
- **Dataset:** hash ou versão, e de onde veio (`mlflow.log_input`).
- **Model signature e input example:** o contrato do modelo (colunas, tipos, shapes). Com signature, o MLflow valida o input em inference e rejeita schemas errados.

Regra: se a informação seria necessária para refazer ou explicar o resultado daqui a seis meses, ela entra.

`mlflow.autolog()` captura automaticamente parâmetros e métricas de várias bibliotecas (scikit-learn, PyTorch Lightning, `transformers`). É um bom começo; registre explicitamente o que é específico do seu negócio.

> **No mundo real.** O paper *Deep Reinforcement Learning that Matters* (Henderson et al., 2018) mostrou por que a seed entra na lista: o mesmo algoritmo, com o mesmo código e os mesmos hyperparameters, produziu resultados estatisticamente diferentes quando rodado com dois conjuntos distintos de seeds. Diferenças anunciadas entre algoritmos em vários papers eram menores do que essa variação. Sem registrar a seed e repetir runs, não se sabe o que é ganho e o que é sorte.

### 4. Model flavors e `pyfunc`

> **Em palavras simples.** Modelos são feitos com bibliotecas diferentes, e cada uma salva de um jeito. Um **flavor** do MLflow é o formato de empacotamento de cada biblioteca. O `pyfunc` é o formato universal: qualquer modelo, de qualquer biblioteca, é exposto com a mesma função `predict`. Quem vai usar o modelo não precisa saber como ele foi construído.

O MLflow salva modelos em **flavors** (`mlflow.sklearn`, `mlflow.pytorch`, `mlflow.transformers`). Todo modelo também é exposto pelo flavor genérico **`pyfunc`**, com um método `predict` uniforme: o código de serving não precisa saber de que biblioteca o modelo veio. Junto do modelo ficam o `MLmodel` (metadata), `conda.yaml` / `requirements.txt` (ambiente) e a signature.

Atenção à serialização: formatos baseados em pickle executam código ao carregar; só carregue modelos de origem confiável. Em versões recentes, o MLflow é mais restritivo por padrão com modelos scikit-learn; no notebook indicamos o formato explicitamente.

> **No mundo real.** Essa interface comum é o que permite a plataformas como o **Databricks Model Serving**, o **Amazon SageMaker** e o **Azure Machine Learning** implantar como serviço web um modelo registrado no MLflow sem código específico para cada biblioteca. O pacote inclui, além dos pesos, a lista de dependências e a assinatura de entrada e saída, o que evita o problema clássico do modelo que só funciona na máquina de quem o treinou.

### 5. Tracking em fine-tuning de LLMs

> **Em palavras simples.** No fine-tuning de LLMs, há mais itens a registrar: o modelo base e a sua versão, a configuração do LoRA, o chat template, a versão do dataset, as curvas de loss, e exemplos de respostas geradas antes e depois. Guardar exemplos de texto é tão importante quanto guardar números: muitos problemas só se veem lendo.

Mesma ideia, outros campos:

- **Parameters:** base model e revision, `r`, `lora_alpha`, `target_modules`, learning rate, epochs, max length, versão do instruction dataset, quantization.
- **Metrics:** train/validation loss por step; parse rate e accuracy por campo; resultado OOD (Semana 21); pico de VRAM; tempo de treino.
- **Artifacts:** o adapter, o model card, amostras de geração antes e depois.

Para aplicações de GenAI, o MLflow também registra **prompts** como artefatos versionados e **traces** (Semana 25). Weights & Biases é a alternativa mais usada, com os mesmos conceitos.

> **No mundo real.** O `Trainer` da biblioteca `transformers` traz a integração pronta: com `report_to="mlflow"`, cada treino registra automaticamente os hyperparameters e as curvas de loss de treino e de avaliação. Desde a versão 2.x, o MLflow também tem recursos específicos para LLMs, como o registro de prompts e a avaliação de respostas geradas (`mlflow.evaluate`), acompanhando a mudança no tipo de modelo que as empresas treinam.

### 6. Comparando runs

> **Em palavras simples.** Comparar runs é pôr as tentativas lado a lado e perguntar o que mudou e que efeito teve. Para a comparação ser justa, só uma coisa deve mudar por vez, o conjunto de avaliação deve ser o mesmo, e as diferenças pequenas devem ser confirmadas repetindo o treino. A interface do MLflow facilita, mas a disciplina é de quem conduz os experimentos.

`mlflow.search_runs(order_by=["metrics.custo ASC"])` devolve um DataFrame. Compare pela métrica de **negócio** definida antes do experimento, não pela que ficou mais bonita depois. Runs de um hyperparameter sweep podem ser agrupados como child runs (`nested=True`) de um parent run.

> **No mundo real.** O estudo *Are GANs Created Equal?* (Lucic et al., Google Brain, 2018) comparou vários algoritmos de geração de imagens que os respectivos papers apresentavam como avanços. Dando a todos o mesmo orçamento de busca de hyperparameters e várias seeds, a maioria atingiu resultados semelhantes ao do método original. Boa parte das melhorias publicadas vinha de mais tentativas de ajuste, não do algoritmo. Só uma comparação controlada revelou isso.

### 7. Model Registry e o ciclo de promoção

> **Em palavras simples.** O **Model Registry** é o catálogo oficial dos modelos da empresa. Cada modelo tem versões numeradas e um estado: em avaliação, em produção, aposentado. Promover uma versão a produção passa a ser um ato registrado, com responsável e data, e voltar à versão anterior é imediato. O sistema que usa o modelo pede o que está em produção, sem conhecer o arquivo.

1. Um run bom gera uma **version** do registered model.
2. A candidata recebe o alias `@challenger` e passa por validação: métricas no test set, teste OOD, checagem de fairness e de latência, revisão humana.
3. Aprovada, recebe o alias `@champion`. O serviço de inference carrega sempre `models:/nome@champion`.
4. **Rollback** é mover o alias de volta: nenhuma mudança de código, nenhum redeploy.

Separar "qual versão" (alias, no registry) de "como carregar" (código) é o que permite promover e reverter com segurança. Os antigos stages (`Staging`, `Production`) foram substituídos por aliases e tags, que são mais flexíveis.

Estratégias de rollout que se apoiam nisso: **shadow deployment** (o challenger recebe o tráfego em paralelo, sem responder ao usuário), **canary** (uma fração pequena do tráfego) e **A/B test** (Fase 6).

> **No mundo real.** A Uber descreveu em 2017 a sua plataforma **Michelangelo**, criada depois que as equipes passaram a ter dificuldade para saber quais modelos estavam em produção e como haviam sido gerados. A plataforma guarda cada modelo treinado como um objeto versionado, com dados, configuração e métricas, e permite implantá-lo e revertê-lo de forma padronizada. O relato inspirou os registros de modelos das ferramentas abertas, inclusive o do MLflow.

### 8. Lineage e auditoria

> **Em palavras simples.** **Lineage** é a árvore genealógica de um modelo: de quais dados veio, com qual código, por qual run, quem aprovou e onde está sendo usado. Serve para responder a perguntas de auditoria (por que o modelo tomou esta decisão?) e de operação (se este dataset tinha um erro, quais modelos foram afetados?).

A partir de uma versão em produção, deve ser possível chegar ao run, e do run aos parâmetros, ao hash do dataset e ao commit do código. Essa cadeia é o **lineage**. É o que responde a um auditor, e o que permite investigar um incidente ("o modelo começou a errar depois de qual mudança?").

> **No mundo real.** Nos Estados Unidos, a norma **SR 11-7** do Federal Reserve (2011) exige dos bancos documentação, validação independente e inventário de todos os modelos usados em decisões. Na União Europeia, o **AI Act** (Regulamento 2024/1689) exige, para sistemas de alto risco, documentação técnica e registros automáticos de funcionamento. Em setores regulados, lineage é requisito legal, e não boa prática opcional.

### 9. Reproducibility

> **Em palavras simples.** **Reproducibility** é conseguir obter de novo o mesmo resultado. Depende de fixar tudo o que influencia o treino: código, dados, versões das bibliotecas, seeds e configuração. Mesmo assim, em GPU pode haver pequenas diferenças numéricas. O objetivo prático é que os resultados sejam equivalentes e que qualquer pessoa da equipe consiga refazer o treino.

Um run é reproduzível quando se conhecem: código (commit), dados (versão ou hash), configuração (parâmetros), ambiente (versões das bibliotecas) e seeds. O MLflow guarda os quatro últimos se você os registrar; a Semana 31 fecha o tema com reprodução por um comando.

> **No mundo real.** Em 2019, a conferência **NeurIPS** introduziu um checklist de reproducibility e passou a incentivar o envio de código com os papers. O relatório dos organizadores (Pineau et al., 2021) registra que a proporção de papers aceitos com código subiu de menos da metade, em 2018, para quase 75%, em 2019. O mesmo relatório descreve um desafio em que voluntários tentaram reproduzir papers aceitos, com sucesso apenas parcial em muitos casos.

### 10. Níveis de maturidade em MLOps

> **Em palavras simples.** As empresas passam por estágios na forma de operar modelos. No início, tudo é manual: treina-se em um notebook e copia-se o arquivo. Depois, o treino vira um pipeline automatizado. No estágio avançado, mudanças no código ou nos dados disparam treino, testes e implantação automáticos, com monitoramento. Saber em que estágio a equipe está orienta o próximo investimento.

- **Level 0:** processo manual, notebooks, deploy esporádico.
- **Level 1:** pipeline de treino automatizado, continuous training, tracking e registry.
- **Level 2:** CI/CD completo para pipelines, testes automatizados de dados e de modelo, monitoramento que dispara retreino.

Esta semana cobre o que é necessário para sair do nível 0.

> **No mundo real.** A referência mais usada é o documento do **Google Cloud** *MLOps: Continuous delivery and automation pipelines in machine learning*, que define três níveis: 0 (processo manual), 1 (pipeline de treino automatizado) e 2 (CI/CD completo). A Microsoft publica um modelo equivalente com cinco níveis. Ambos partem da mesma constatação: a maioria das organizações começa no nível mais baixo e deveria avançar um degrau por vez.

### 11. Onde fica o servidor

> **Em palavras simples.** O MLflow precisa de um lugar para guardar os registros e os arquivos. Pode ser uma pasta local, suficiente para estudar; um servidor da equipe, com banco de dados e armazenamento compartilhado; ou um serviço gerenciado na nuvem, que dispensa administração. A escolha define quem consegue ver os experimentos e se eles sobrevivem ao fim de uma sessão.

Localmente, um arquivo SQLite e uma pasta de artifacts. Em equipe, um tracking server com banco e object storage, ou um serviço gerenciado (Databricks, Azure ML, SageMaker). A API de tracking é a mesma; muda só a URI.

> **No mundo real.** No **Google Colab**, o disco é apagado quando a sessão termina, e os runs gravados localmente se perdem, a menos que sejam salvos no Google Drive ou enviados a um servidor remoto. No uso corporativo, o MLflow é oferecido como serviço gerenciado pela Databricks e, desde 2024, pelo Amazon SageMaker, com controle de acesso integrado. Em todos os casos o código muda em uma linha: `mlflow.set_tracking_uri(...)`.

## Erros comuns

- Registrar só a loss e não a métrica de negócio.
- Não registrar a versão ou o hash dos dados.
- Carregar o modelo por número de versão fixo no código de serving.
- Promover um modelo sem validação fora do conjunto em que foi escolhido.
- Registrar tudo em um único experiment sem tags, tornando a busca impossível.
- Escolher o melhor run olhando o test set repetidas vezes (o test set vira validation).
- Guardar credenciais ou dados sensíveis como parameters ou artifacts.

## Leitura e vídeos (fim de semana)

- MLflow documentation: *Tracking* (conceitos e quickstart); *Models* (flavors, signature); *Model Registry* (versions, aliases, tags).
- Sculley et al., *Hidden Technical Debt in Machine Learning Systems*.
- Google Cloud, *MLOps: Continuous delivery and automation pipelines in machine learning* (os níveis 0, 1 e 2).
- Breck et al., *The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction*.
- Chip Huyen, *Designing Machine Learning Systems*, capítulos sobre model development e deployment (se tiver o livro).
- Opcional: MLflow documentation, *MLflow for GenAI* (prompt registry, tracing, evaluation).

## Perguntas de autoavaliação

1. Qual a diferença entre um run e uma model version?
2. Por que o serviço deve carregar o modelo por alias e não por número de versão?
3. Liste cinco coisas necessárias para reproduzir um treino.
4. O que você registraria em um run de fine-tuning com LoRA?
5. Para que serve a model signature?
6. Qual a diferença entre shadow deployment e canary?

## Entregável da semana

Notebook resolvido e uma captura (ou tabela exportada) da comparação de runs em `docs/`, com a justificativa da versão promovida.

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/05-Phase-5-MLOps-and-Agents/notebooks/semana_24_exercicio.ipynb) · [`semana_24_exercicio.ipynb`](../notebooks/semana_24_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/05-Phase-5-MLOps-and-Agents/notebooks/semana_24_gabarito.ipynb) · [`semana_24_gabarito.ipynb`](../notebooks/semana_24_gabarito.ipynb) |

[← Semana 23](/05-Phase-5-MLOps-and-Agents/docs/Semana-23-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 25 →](/05-Phase-5-MLOps-and-Agents/docs/Semana-25-Conceitos.md)
