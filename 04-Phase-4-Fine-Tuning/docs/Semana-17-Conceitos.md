# Semana 17 — Instruction Dataset Engineering (JSONL)

**Fase 4: Parameter-Efficient Fine-Tuning (PEFT)** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Transformar dados corporativos brutos em um instruction dataset de alta qualidade, em JSONL com chat format e validação por schema. É o trabalho que mais determina o resultado de um fine-tuning.

## Por que isso importa para um AI Architect

Na prática, a maior parte do esforço de um projeto de fine-tuning é dado. Um modelo ajustado reproduz fielmente o que há no dataset: labels inconsistentes, duplicatas, PII e formatos quebrados incluídos. E o leakage entre train e test produz métricas ótimas que somem em produção. Decidir **se** fine-tuning é a ferramenta certa também é responsabilidade do arquiteto.

## Conceitos

### 1. Quando fine-tuning é a ferramenta certa

> **Em palavras simples.** **Fine-tuning** muda o comportamento do modelo: o estilo, o formato, o jeito de executar uma tarefa. Não é a melhor forma de dar ao modelo informações novas; para isso existe o RAG. A ordem sensata é tentar primeiro um bom prompt, depois exemplos no prompt, depois RAG, e só então fine-tuning, quando o problema for de comportamento e houver exemplos de qualidade.

| Necessidade | Primeira opção |
|---|---|
| O modelo não conhece os fatos da empresa | RAG |
| O modelo conhece, mas não segue o formato, o tom ou a tarefa | Prompt engineering / few-shot; depois fine-tuning |
| Prompt longo demais ou caro por requisição | Fine-tuning (embute a instrução nos pesos) |
| Latência e custo exigem um modelo pequeno | Fine-tuning de um small model (distillation) |
| Comportamento consistente em tarefa estreita e de alto volume | Fine-tuning |
| Structured output confiável | Fine-tuning e/ou constrained decoding |

Fine-tuning ensina **comportamento**, não fatos atualizáveis. A ordem recomendada de investimento: prompt → few-shot → RAG → fine-tuning. Cada degrau só se justifica quando o anterior foi medido e não bastou.

> **No mundo real.** O estudo *Fine-Tuning or Retrieval?* (Ovadia et al., Microsoft, 2023) comparou as duas formas de dar conhecimento novo a LLMs, em temas como anatomia, astronomia e eventos recentes. O RAG superou de forma consistente o fine-tuning feito com os mesmos textos, tanto para conhecimento que o modelo já tinha visto quanto para conhecimento inédito. A conclusão dos autores: para fatos, buscar funciona melhor que treinar.

### 2. Formatos de dataset

> **Em palavras simples.** O dataset de fine-tuning é uma coleção de exemplos do comportamento desejado, em um formato padronizado. Os dois mais comuns são o de **instruction** (instrução, entrada opcional e resposta) e o de **chat** (lista de mensagens com papéis: sistema, usuário, assistente). O formato precisa ser o mesmo que o modelo verá em uso.

**Chat format (messages)**, o padrão atual, em JSONL (um objeto JSON por linha):

```json
{"messages": [
  {"role": "system", "content": "Você é um classificador de chamados..."},
  {"role": "user", "content": "Não consigo entrar no SAP..."},
  {"role": "assistant", "content": "{\"categoria\": \"acesso\", \"prioridade\": \"alta\", ...}"}
]}
```

O chat template do tokenizer (Semana 9) converte isso no texto exato que o modelo vê. Outros formatos que você encontrará: **Alpaca** (`instruction`, `input`, `output`), **ShareGPT** (`conversations` com `from`/`value`) e **completion** (`prompt`, `completion`). Todos são convertíveis para messages.

Regra de ouro: o formato de treino deve ser idêntico ao de inference, incluindo o system prompt. Treinar com um system prompt e inferir com outro (ou sem) degrada o resultado.

Por que JSONL e não um JSON único: é streamable, appendable, e uma linha corrompida não invalida o arquivo.

> **No mundo real.** O dataset **Alpaca** (Stanford, 2023) fixou o formato `instruction` / `input` / `output` com 52 mil exemplos e virou modelo para centenas de datasets posteriores. Já as APIs de fine-tuning da OpenAI e a maioria das ferramentas atuais usam arquivos **JSONL** em que cada linha contém uma lista `messages`, no mesmo formato de uma chamada de chat. Converter entre os dois é uma das primeiras tarefas de qualquer projeto.

### 3. As etapas do pipeline

> **Em palavras simples.** Preparar dados é uma sequência de etapas: coletar, limpar, remover duplicatas, padronizar o formato, filtrar exemplos ruins, dividir em treino e teste, e registrar o que foi feito. Cada etapa descarta exemplos, e é bom saber quantos e por quê. O resultado do fine-tuning depende mais dessa preparação do que de qualquer ajuste de treino.

1. **Collection e audit:** de onde vêm os dados, quem pode usá-los (licença, consentimento, política interna), que PII contêm.
2. **Cleaning:** normalizar whitespace, encoding (Unicode NFC/NFKC) e labels; remover vazios e lixo de template.
3. **PII redaction:** mascarar e-mails, telefones, documentos, nomes. Regex cobre padrões estruturados; nomes e endereços pedem NER (ex.: Microsoft Presidio). Um modelo ajustado pode **memorizar e reproduzir** dados pessoais do treino (training data extraction).
4. **Deduplication:** exact (hash do texto normalizado) e **near-duplicate** (MinHash/LSH, ou cosine entre embeddings acima de um threshold).
5. **Schema validation:** cada target deve ser um objeto válido (Pydantic). Exemplos inválidos ensinam o modelo a produzir saídas inválidas.
6. **Split** train/validation/test: depois de deduplicar, e sem leakage.
7. **Statistics:** class balance e distribuição de token length.
8. **Dataset card:** origem, tratamento, limitações, versão.

> **No mundo real.** Pouco depois do lançamento do Alpaca, a comunidade notou problemas no dataset original: respostas vazias, instruções impossíveis de cumprir (como comentar uma imagem inexistente), cálculos errados e respostas inventadas. Voluntários publicaram o **alpaca-cleaned**, uma versão corrigida, e modelos treinados nela passaram a se comportar melhor. O episódio mostrou que mesmo um dataset famoso precisa passar por limpeza.

### 4. Train/test leakage

> **Em palavras simples.** **Leakage** acontece quando exemplos do teste, ou versões quase idênticas deles, também estão no treino. O modelo parece ótimo porque já viu as respostas. Em texto, o risco é alto: a mesma pergunta aparece reformulada, o mesmo documento gera vários exemplos. A divisão deve ser feita por grupo (por documento, cliente ou assunto), depois de remover as duplicatas.

Se o mesmo chamado, ou uma variação mínima, aparece nos dois conjuntos, o test set mede memorização. Regras:

- Deduplicar **antes** de dividir.
- **Group split:** agrupar por entidade (cliente, documento de origem, template) e manter o grupo inteiro em um só conjunto (`GroupShuffleSplit`).
- **Temporal split** em dados com data: treinar no passado, testar no futuro.
- Manter um conjunto **out-of-distribution** escrito de outra forma, por outra pessoa (Semana 21).

O notebook mostra o problema: mesmo sem duplicatas exatas, a maioria dos exemplos de teste tem um "quase gêmeo" no treino, gerado do mesmo template.

> **No mundo real.** O estudo *Question and Answer Test-Train Overlap in Open-Domain Question Answering Datasets* (Lewis et al., 2020) examinou três benchmarks populares e encontrou que de 60% a 70% das respostas do teste também apareciam no treino, e cerca de 30% das perguntas do teste tinham uma paráfrase quase idêntica no treino. Nos casos sem sobreposição, o desempenho dos modelos despencava: boa parte da nota vinha de memorização.

### 5. Quality over quantity

> **Em palavras simples.** Poucos exemplos excelentes ensinam mais do que muitos exemplos medianos. O modelo imita o que vê: se parte dos dados tem respostas descuidadas, ele aprende descuido. Mil exemplos revisados, variados e consistentes costumam superar cem mil coletados sem critério. O tempo gasto lendo e corrigindo os dados é o investimento mais rentável do projeto.

Algumas centenas a poucos milhares de exemplos limpos e consistentes costumam superar dezenas de milhares ruidosos. Sinais de problema:

- **Label inconsistency:** duas pessoas rotulariam o mesmo exemplo de formas diferentes. Meça inter-annotator agreement em uma amostra e escreva annotation guidelines.
- **Unsupported targets:** o target contém informação que não está no input. Isso ensina o modelo a inventar.
- **Class imbalance:** uma classe domina e o modelo aprende a prevê-la sempre.
- **Low diversity:** todos os exemplos seguem o mesmo padrão de frase; o modelo aprende o padrão, não a tarefa.
- **Formato inconsistente** do target (às vezes JSON com espaços, às vezes sem; chaves em ordem variável).

Ordens de grandeza: classificação e extração simples, 200 a 1.000 exemplos; mudança de estilo ou formato, 500 a 2.000; tarefas complexas de domínio, alguns milhares. Mais importante que o número é cobrir os casos difíceis e de borda.

> **No mundo real.** O paper **LIMA** (*Less Is More for Alignment*, Zhou et al., Meta, 2023) fez o fine-tuning de um modelo LLaMA de 65 bilhões de parâmetros com apenas **1.000** pares de pergunta e resposta, escolhidos e escritos com cuidado. Em avaliação humana, as respostas do modelo foram consideradas equivalentes ou preferíveis às do GPT-4 em 43% dos casos. Dobrar a quantidade de dados sem aumentar a diversidade e a qualidade não melhorou o resultado.

### 6. Token length

> **Em palavras simples.** Cada exemplo tem um comprimento em tokens, e o treino trabalha com um limite. Exemplos maiores que o limite são cortados, e o corte costuma atingir justamente o fim, onde está a resposta. Antes de treinar, mede-se a distribuição dos comprimentos e escolhe-se um limite que cubra quase todos os exemplos sem desperdiçar memória.

O `max_length` define memória (cresce com `T²` na attention) e tempo de treino. Olhe a distribuição com o **tokenizer do modelo-alvo**: se o p99 é 256 tokens, treinar com `max_length=2048` desperdiça recursos; se 20% dos exemplos são truncados, o modelo aprende respostas cortadas, e pode nunca ver o EOS token.

No notebook estimamos por `caracteres / 3,5` para não depender de download; em um projeto real, tokenize de verdade.

> **No mundo real.** O script do **Alpaca-LoRA**, um dos repositórios de fine-tuning mais copiados de 2023, vinha com `cutoff_len=256` como padrão, adequado aos exemplos curtos do Alpaca. Muita gente reutilizou o script com dados próprios mais longos, sem alterar o valor, e treinou modelos com as respostas cortadas ao meio. O sintoma típico era um modelo que parava de responder no meio da frase.

### 7. Structured targets

> **Em palavras simples.** Quando a saída do modelo será lida por um programa, ela precisa seguir um formato exato, como **JSON** com campos definidos. No dataset, isso significa respostas sempre no mesmo formato, validadas por um **schema** antes do treino. Um único exemplo fora do padrão ensina o modelo que o padrão é opcional.

Quando o output é JSON:

- Serialize sempre do mesmo jeito (mesma ordem de chaves, mesmo espaçamento). `model_dump_json()` do Pydantic garante isso.
- Use valores enumerados (`Literal`) em vez de texto livre para categorias.
- Valide 100% dos targets antes de treinar.
- O mesmo schema será usado na avaliação (Semana 21) e pelo consumidor do output (Semana 26).

> **No mundo real.** Ao lançar o recurso **Structured Outputs** (agosto de 2024), a OpenAI publicou uma avaliação com schemas JSON complexos: o `gpt-4-0613` produzia saídas em conformidade em menos de 40% dos casos; um modelo mais novo, treinado para seguir schemas, chegava a 93%; e só a combinação com **constrained decoding** atingia 100%. O treino melhora muito a aderência ao formato, mas a garantia vem da restrição na geração.

### 8. Synthetic data e distillation

> **Em palavras simples.** **Synthetic data** são exemplos de treino escritos por um modelo em vez de pessoas. Um modelo grande gera perguntas e respostas, e um modelo pequeno é treinado com elas: isso é **distillation**. É rápido e barato, mas os erros e o estilo do modelo professor são herdados, e por isso o material gerado precisa ser filtrado e revisado por amostragem.

LLMs maiores podem gerar ou ampliar datasets: um **teacher model** rotula dados reais não rotulados, ou gera exemplos a partir de seeds, e um **student model** pequeno é treinado neles. Funciona e é hoje prática comum. Cuidados:

- Revise amostras manualmente; filtre com um judge ou com regras.
- Meça diversidade (os exemplos sintéticos tendem a ser parecidos entre si).
- Não avalie em dados gerados pelo mesmo processo do treino: o test set deve ser real.
- Verifique os termos de uso do teacher model quanto a treinar outros modelos com seus outputs.

> **No mundo real.** O método **Self-Instruct** (Wang et al., 2022) partiu de 175 tarefas escritas por pessoas e usou um LLM para gerar dezenas de milhares de novas instruções e respostas, filtrando as repetidas. O Alpaca aplicou a mesma ideia com um modelo da OpenAI e produziu seus 52 mil exemplos por menos de 500 dólares. Os próprios autores do Self-Instruct estimaram, por amostragem, que cerca de metade dos exemplos gerados tinha algum defeito.

### 9. Versioning

> **Em palavras simples.** Um dataset muda ao longo do projeto: exemplos são corrigidos, acrescentados, removidos. **Versioning** é registrar cada versão de forma que se saiba exatamente com quais dados cada modelo foi treinado. Sem isso, não dá para reproduzir um resultado nem explicar por que o modelo novo se comporta diferente do anterior.

Dataset é código: versione (DVC, Hugging Face Datasets com revisions, Delta Lake, ou ao menos um hash do arquivo). Registre a versão do dataset em cada training run (Semana 24). Sem isso, não é possível explicar por que o modelo de março se comporta diferente do de fevereiro.

> **No mundo real.** No **Hugging Face Hub**, cada dataset é um repositório Git: toda alteração gera um commit com identificador único, e `load_dataset(..., revision="<hash>")` carrega exatamente aquela versão. O caso do Alpaca ilustra a utilidade: existem a versão original e várias versões limpas, com resultados diferentes, e um model card que diz apenas *treinado no Alpaca* não permite saber qual foi usada.

## Erros comuns

- Fazer fine-tuning para "ensinar fatos" que mudam toda semana.
- Dividir antes de deduplicar.
- System prompt de treino diferente do de produção.
- Targets com formato inconsistente.
- Estimar token length com outro tokenizer.
- Não remover PII e descobrir depois que o modelo a reproduz.
- Avaliar o modelo em dados sintéticos do mesmo gerador.

## Leitura e vídeos (fim de semana)

- `transformers` documentation: *Chat templates* (seção sobre uso em treino).
- Pydantic documentation: *Models*, *Fields* e *Validators*.
- Gebru et al., *Datasheets for Datasets* (leia a lista de perguntas) e os *Dataset Cards* do Hugging Face Hub.
- OpenAI ou Anthropic, guias de *fine-tuning best practices / data preparation*: as recomendações de formato, volume e qualidade são transferíveis para modelos abertos.
- Carlini et al., *Extracting Training Data from Large Language Models* (abstract), sobre memorização.
- Northcutt et al., *Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks* (abstract), sobre label noise.
- Opcional: Microsoft Presidio documentation (PII detection).

## Perguntas de autoavaliação

1. Um stakeholder pede fine-tuning para o modelo "saber os preços atuais". O que você responde?
2. Por que deduplicar antes de dividir em train e test? O que é um group split?
3. Qual o risco de não fazer PII redaction nos dados de treino?
4. Como você escolhe `max_length` para o treino?
5. Por que a serialização do target JSON precisa ser consistente?
6. Quais os cuidados ao usar synthetic data gerado por um teacher model?

## Entregável da semana

Notebook resolvido, os arquivos `train.jsonl`, `val.jsonl`, `test.jsonl` e um `DATASET_CARD.md` curto.

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/04-Phase-4-Fine-Tuning/notebooks/semana_17_exercicio.ipynb) · [`semana_17_exercicio.ipynb`](../notebooks/semana_17_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/04-Phase-4-Fine-Tuning/notebooks/semana_17_gabarito.ipynb) · [`semana_17_gabarito.ipynb`](../notebooks/semana_17_gabarito.ipynb) |

[← Semana 16](/03-Phase-3-Advanced-RAG/docs/Semana-16-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 18 →](/04-Phase-4-Fine-Tuning/docs/Semana-18-Conceitos.md)
