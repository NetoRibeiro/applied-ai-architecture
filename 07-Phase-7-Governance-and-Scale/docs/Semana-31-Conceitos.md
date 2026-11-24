# Semana 31 — RFCs, ADRs e Reproducibility com um Comando

**Fase 7: Governance e Engineering at Scale** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Praticar duas disciplinas de quem lidera tecnicamente: documentar decisões de arquitetura de forma que sobrevivam a quem as tomou (RFCs e ADRs) e garantir que qualquer pessoa do time reproduza um resultado com um único comando.

## Por que isso importa para um AI Architect

Em nível Staff/Principal, o trabalho deixa de ser "resolver" e passa a ser "fazer o time resolver do mesmo jeito, sempre". Decisões não documentadas são rediscutidas a cada seis meses. Resultados não reproduzíveis não podem ser auditados, comparados nem confiados. Padrões escritos e pipelines reproduzíveis são como a engenharia escala além de uma pessoa: é o que as descrições de vaga chamam de "ser referência" e "institucionalizar".

## Conceitos

### 1. RFC e ADR: papéis diferentes

> **Em palavras simples.** Um **RFC** (*Request for Comments*) é uma proposta escrita, feita **antes** da decisão, para receber críticas de quem será afetado. Um **ADR** (*Architecture Decision Record*) é o registro curto de uma decisão **já tomada**: o que se decidiu e por quê. O primeiro serve para discutir; o segundo, para lembrar. Juntos, evitam que as decisões fiquem apenas na memória das pessoas.

| | RFC (Request for Comments) / Design Doc | ADR (Architecture Decision Record) |
|---|---|---|
| Momento | **Antes** da decisão | **Depois** da decisão |
| Finalidade | Propor, coletar objeções, alinhar | Registrar o que foi decidido e por quê |
| Tamanho | 2 a 10 páginas | 1 página |
| Vida | Discutido, revisado, aprovado ou rejeitado | Imutável; uma decisão nova **supersedes** a antiga |
| Leitor | Pares e stakeholders, agora | Quem entrar no time daqui a dois anos |

Fluxo comum: um RFC é debatido; a conclusão vira um ou mais ADRs. Decisões pequenas e reversíveis não precisam de nenhum dos dois: o custo do processo deve ser proporcional ao custo de errar (as "one-way doors" merecem documento; as "two-way doors", não).

> **No mundo real.** Os RFCs existem desde 1969, quando Steve Crocker escreveu o RFC 1 para documentar a rede que viria a ser a internet; o protocolo HTTP e o e-mail foram definidos assim. O ADR foi proposto por Michael Nygard em 2011, em um texto curto de blog, e foi adotado por organizações como o GitHub e o governo britânico (GOV.UK). O radar de tecnologia da Thoughtworks o classifica como prática a adotar.

### 2. Estrutura de um ADR

> **Em palavras simples.** Um ADR tem poucas seções. **Context**: a situação e as forças em jogo. **Decision**: o que foi decidido, em frase afirmativa. **Consequences**: o que fica mais fácil e o que fica mais difícil a partir de agora. E um **status**: proposto, aceito ou substituído. Cabe em uma página, e não se edita depois: se a decisão muda, escreve-se um novo ADR que substitui o anterior.

1. **Title e número:** `ADR-0007: Usar pgvector como vector database`. O título contém a decisão.
2. **Status:** proposed, accepted, deprecated, superseded by ADR-00XX.
3. **Context:** o problema, as forças em jogo, as restrições. Fatos e medições, não opinião.
4. **Options considered:** com prós e contras. Inclua "não fazer nada".
5. **Decision:** em voz ativa. "Usaremos…"
6. **Consequences:** o que fica mais fácil, o que fica mais difícil, o que passa a ser obrigatório. Inclua as negativas: um ADR sem consequências negativas não foi pensado.
7. **Review trigger:** que evidência faria reabrir a decisão ("se passarmos de 50 milhões de vetores ou o p95 exceder 200 ms").

ADRs vivem no repositório, em Markdown, versionados com o código (`docs/adr/0007-usar-pgvector.md`). O conjunto forma o **decision log** do sistema. Modelos prontos estão em [`templates/`](./templates/); o formato MADR é uma variante popular.

> **No mundo real.** O formato original de Nygard tem exatamente essas partes: título, status, contexto, decisão e consequências, em arquivos de texto numerados e guardados no mesmo repositório do código. A **AWS**, em suas orientações de arquitetura, recomenda o processo e relata o uso em projetos com centenas de decisões registradas. O ganho mais citado é na chegada de novos integrantes, que conseguem entender por que o sistema é como é.

### 3. Estrutura de um RFC

> **Em palavras simples.** Um RFC é mais longo e conduz o leitor pelo raciocínio: qual é o problema, por que resolvê-lo agora, qual é a proposta, quais alternativas foram consideradas e por que foram descartadas, quais são os riscos e o que ainda está em aberto. A seção de alternativas é a mais importante: mostra que a escolha foi pensada.

Summary; motivation e problem statement; goals e **non-goals**; proposta detalhada; alternatives considered; risks e mitigations; rollout plan e rollback plan; success metrics; open questions.

Os **non-goals** são a seção mais útil: delimitam o debate e evitam scope creep. As **alternatives** mostram que a proposta não foi a primeira ideia que apareceu.

Um bom RFC é escrito para ser **criticado**: específico o bastante para alguém apontar onde está errado. O processo importa tanto quanto o documento: prazo definido para comentários, revisores nomeados, e um responsável claro pela decisão final (consenso total não é requisito; "disagree and commit" é legítimo).

> **No mundo real.** A linguagem **Rust** adota desde 2014 um processo público de RFCs para qualquer mudança relevante. O modelo exige seções fixas: *Summary*, *Motivation*, explicação para usuários, explicação técnica, *Drawbacks*, *Rationale and alternatives* e *Unresolved questions*. Há milhares de propostas arquivadas no GitHub, aceitas e rejeitadas, com toda a discussão, o que as torna um bom material de estudo sobre como argumentar tecnicamente.

### 4. Escrevendo para ser lido

> **Em palavras simples.** Um documento técnico só funciona se for lido. Isso pede a conclusão no início, frases diretas, um assunto por parágrafo, números em vez de adjetivos e termos definidos na primeira vez em que aparecem. Quem lê é alguém ocupado, que não participou das conversas: o texto deve bastar sozinho.

- Conclusão primeiro (o resumo cabe em cinco frases).
- Um documento, uma decisão.
- Números no lugar de adjetivos: "p95 de 180 ms com 2M de vetores", não "rápido".
- Diagramas para arquitetura (C4 model: context, container, component); Mermaid funciona direto no GitHub.
- Diga o que você **não** sabe.

> **No mundo real.** Na **Amazon**, as reuniões de decisão não usam slides desde 2004, quando Jeff Bezos os substituiu por memorandos narrativos de até seis páginas. Cada reunião começa com todos lendo o documento em silêncio. Bezos justificou a regra em cartas e entrevistas: escrever em texto corrido obriga o autor a explicitar o raciocínio e as relações entre as ideias, o que uma lista de tópicos permite esconder.

### 5. Decisões defensáveis

> **Em palavras simples.** Uma decisão defensável não é a que deu certo: é a que foi bem tomada com as informações disponíveis. Para isso, registram-se os critérios, as alternativas, as evidências e o que faria a equipe mudar de ideia. Vale também distinguir as decisões reversíveis, que podem ser tomadas rápido, das irreversíveis, que merecem mais análise.

Uma decisão de arquitetura se defende com:

- **Critérios explícitos e ponderados**, definidos antes de pontuar as opções (decision matrix).
- **Evidência medida:** um benchmark ou proof of concept no seu dado vale mais que dez artigos de blog.
- **Sensitivity analysis:** a decisão muda se os pesos mudarem um pouco? Se a opção só vence com um conjunto muito específico de pesos, ela é frágil, e isso deve constar no documento.
- **Custo de reversão:** quanto custa mudar de ideia em um ano.

O notebook implementa a matriz, perturba os pesos com uma distribuição de Dirichlet e calcula em que ponto a decisão vira. Esse ponto alimenta diretamente o review trigger do ADR.

Uma ressalva honesta: uma decision matrix organiza o raciocínio, não o substitui. Notas de 1 a 5 são julgamentos; a transparência sobre elas é o que torna a decisão contestável e, por isso, confiável.

> **No mundo real.** Na carta aos acionistas da Amazon de 2015, Bezos distinguiu as decisões do **tipo 1**, portas de mão única, difíceis ou impossíveis de reverter, das do **tipo 2**, portas de mão dupla, em que se pode voltar atrás. Segundo ele, as grandes empresas tendem a aplicar a todas as decisões o processo pesado do tipo 1, e o resultado é lentidão. A classificação entrou no vocabulário da engenharia de software.

### 6. Padronização: o exemplo dos A/B tests

> **Em palavras simples.** Se cada equipe faz experimentos do seu jeito, os resultados não são comparáveis nem confiáveis. Padronizar é definir em um documento comum: como calcular o tamanho da amostra, quais verificações são obrigatórias, quais métricas todos acompanham, como relatar o resultado. O padrão reduz erros e libera as pessoas para pensar no que é específico de cada teste.

Um RFC típico de um time de dados maduro: "Como padronizaremos experimentos". Define randomization unit por tipo de produto, primary metric e guardrails (Semana 32), cálculo de sample size, duração mínima, verificações obrigatórias (SRM, A/A), critérios de decisão, tratamento de multiple comparisons e o formato do relatório.

A tool `revisar_experimento` da Semana 26 é a versão executável desse padrão. Esse é o caminho da institucionalização: **documento → checklist → automação**. Um padrão que depende de as pessoas lembrarem não é um padrão.

> **No mundo real.** Kohavi e Thomke relataram na *Harvard Business Review* (2017) que empresas como Microsoft, Amazon, Booking e Google realizam, cada uma, mais de **10 mil** experimentos controlados por ano. Nessa escala, isso só funciona com uma plataforma comum, como a ExP da Microsoft, que impõe o mesmo método a todos: sorteio, verificações automáticas e um relatório padrão.

### 7. Os cinco pilares da reproducibility

> **Em palavras simples.** Para que um resultado possa ser refeito, cinco coisas precisam estar sob controle: o **código**, os **dados**, o **ambiente** (versões das bibliotecas), a **configuração** (os ajustes usados) e a **aleatoriedade** (as seeds). Se uma delas não foi registrada, o resultado se torna um evento único, que ninguém consegue verificar.

| Pilar | Como fixar |
|---|---|
| **Code** | Git commit registrado no run; working tree limpa |
| **Data** | Versão ou hash do conjunto exato (DVC, Delta Lake time travel, lakeFS, snapshot imutável) |
| **Configuration** | Arquivo versionado e validado (Pydantic, Hydra); nada de parâmetros digitados à mão |
| **Environment** | Versões travadas (lockfile com `uv` ou `pip-tools`; imagem Docker com digest) |
| **Randomness** | Seeds fixadas e registradas, em **todos** os pontos de aleatoriedade |

Dois níveis de exigência: **repeatability** (mesma pessoa, mesmo ambiente, mesmo resultado) e **reproducibility** (outra pessoa, outro ambiente, mesmo resultado). O segundo é o que importa para um time.

Fontes comuns de não determinismo: split e shuffle sem seed; ordem de iteração de sets e de arquivos em disco; operações de GPU não determinísticas; paralelismo; versões diferentes de bibliotecas numéricas; e dados que mudaram na fonte sem que ninguém percebesse.

> **No mundo real.** Em 2010, os economistas Reinhart e Rogoff publicaram um estudo muito influente segundo o qual países com dívida pública acima de 90% do PIB cresciam muito menos. Em 2013, um estudante de pós-graduação, ao tentar reproduzir o resultado com a planilha original, encontrou um erro de fórmula no Excel que deixava cinco países fora da média, além de outras escolhas discutíveis. Corrigido, o crescimento médio passava de -0,1% para 2,2%.

### 8. Reproduzir com um comando

> **Em palavras simples.** O teste definitivo de reproducibility é alguém de fora obter o mesmo resultado com **um único comando**, em uma máquina limpa. Isso exige que a instalação das dependências, a obtenção dos dados, o treino e a avaliação estejam automatizados e documentados. Se são necessárias instruções orais do autor, o trabalho não é reproduzível.

O alvo é: `make reproduce` (ou `python train.py --config config.json`, `dvc repro`, `mlflow run .`) em uma máquina limpa produz as mesmas métricas.

Na prática isso exige:

- um **entry point** único, sem passos manuais em notebook;
- um **manifest** gravado a cada execução, com os cinco pilares, as métricas e um hash das previsões;
- um **teste automático em CI** que reexecuta um caso pequeno e compara com o manifest.

Bit-for-bit idêntico nem sempre é possível em GPU; define-se uma tolerância para as métricas. Em pipelines com LLMs, fixe `temperature=0`, a versão exata do modelo e do prompt, e saiba que APIs externas não garantem determinismo mesmo assim: guarde os outputs.

Notebooks são ótimos para explorar e ruins para reproduzir (estado oculto, células fora de ordem). O que vai para produção ou para um relatório oficial deve sair de um script ou pipeline versionado.

> **No mundo real.** O site **Papers with Code** criou em 2020 um checklist de cinco itens para repositórios de pesquisa (dependências, código de treino, código de avaliação, modelos pré-treinados e um README com resultados e comandos), adotado como recomendação oficial pela NeurIPS. Uma análise dos repositórios da conferência mostrou que os que cumpriam os cinco itens tinham uma mediana de cerca de 200 estrelas no GitHub, contra menos de 2 dos que não cumpriam nenhum.

### 9. Governança de modelos

> **Em palavras simples.** **Governança de modelos** é o conjunto de regras sobre quem pode colocar um modelo em uso, com quais verificações, quem responde por ele e como ele é acompanhado e retirado. Inclui inventário dos modelos, classificação por risco, aprovação antes do uso, monitoramento e trilha de auditoria. O rigor é proporcional ao dano que um erro pode causar.

Juntando as peças da trilha, o registro mínimo para um modelo em produção: model card (Semana 21), dataset card (Semana 17), run e versão no registry com lineage (Semana 24), relatório de avaliação com golden set (Semanas 16 e 21), ADRs das decisões estruturais e o manifest de reprodução. Regulações como o EU AI Act e frameworks como o NIST AI RMF pedem exatamente esse tipo de documentação para sistemas de maior risco. Quem já trabalha assim não precisa correr quando a auditoria chega.

> **No mundo real.** Na Holanda, a autoridade fiscal usou durante anos um sistema de pontuação de risco para identificar fraudes em benefícios de creche. Cerca de 26 mil famílias foram acusadas indevidamente e obrigadas a devolver valores altos; a nacionalidade era um dos fatores considerados. Sem supervisão nem possibilidade efetiva de contestação, o erro durou anos. Em janeiro de 2021, o governo holandês renunciou por causa do escândalo.

## Erros comuns

- ADR que registra a decisão e omite as alternativas e as consequências negativas.
- Editar um ADR aceito em vez de criar um novo que o substitui.
- RFC sem non-goals e sem prazo para comentários.
- Definir os pesos da decision matrix depois de pontuar as opções.
- Fixar a seed em um lugar e esquecer os outros.
- `requirements.txt` sem versões.
- "Funciona na minha máquina": nunca ter testado a reprodução em ambiente limpo.
- Documentação que ninguém consegue encontrar (fora do repositório, sem índice).

## Leitura e vídeos (fim de semana)

- Michael Nygard, *Documenting Architecture Decisions* (o texto que originou os ADRs).
- `adr.github.io`: modelos (MADR) e exemplos; e *Architecture Decision Records* no ThoughtWorks Technology Radar.
- Gergely Orosz (The Pragmatic Engineer), *Companies Using RFCs or Design Docs and Examples of These* e *Software Engineering RFC and Design Doc Examples and Templates*.
- *Design Docs at Google* (Malte Ubl).
- MLflow documentation: *MLflow Projects*; DVC documentation: *Get Started: Data Pipelines*.
- Pineau et al., *Improving Reproducibility in Machine Learning Research* e o *ML Reproducibility Checklist*.
- Will Larson, *Staff Engineer*: capítulos *Writing engineering strategy* e *Create technical quality*.
- Opcional: Tanya Reilly, *The Staff Engineer's Path*, capítulos sobre big-picture thinking.

## Perguntas de autoavaliação

1. Qual a diferença de finalidade entre RFC e ADR?
2. Por que um ADR aceito não é editado, e sim superseded?
3. O que é um review trigger e por que incluí-lo?
4. Cite os cinco pilares da reproducibility e como fixar cada um.
5. O que uma sensitivity analysis acrescenta a uma decision matrix?
6. Por que notebooks dificultam a reprodução e o que usar no lugar?

## Entregável da semana

Notebook resolvido; um ADR real em `docs/` (sugestão: a escolha do vector database da Semana 12, com os seus pesos); e um RFC de duas páginas: "Como padronizaremos os A/B tests".

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/07-Phase-7-Governance-and-Scale/notebooks/semana_31_exercicio.ipynb) · [`semana_31_exercicio.ipynb`](../notebooks/semana_31_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/07-Phase-7-Governance-and-Scale/notebooks/semana_31_gabarito.ipynb) · [`semana_31_gabarito.ipynb`](../notebooks/semana_31_gabarito.ipynb) |

[← Semana 30](/06-Phase-6-Causal-Inference/docs/Semana-30-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 32 →](/07-Phase-7-Governance-and-Scale/docs/Semana-32-Conceitos.md)
