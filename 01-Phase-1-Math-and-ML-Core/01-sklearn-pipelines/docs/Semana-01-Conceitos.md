# Semana 1 — Scikit-Learn Pipelines e o Accuracy Paradox

**Fase 1: Math Foundations e ML Core** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Recalibrar o que você já domina em ML clássico (pré-processamento, `Pipeline`, métricas de classificação) e reler cada item pela lente de quem vai avaliar LLMs e sistemas RAG. Ao final você deve conseguir olhar para uma confusion matrix e traduzir cada célula em custo de negócio, escolher um threshold com justificativa e explicar por que accuracy é a métrica errada para quase todo problema real.

## Por que isso importa para um AI Architect

Avaliar um LLM é, no fundo, um problema de classificação com classes desbalanceadas: a maioria das respostas está certa, e o que interessa é a minoria errada. Hallucination detection, prompt injection guardrails, retrieval relevance e routing são todos classificadores binários ou multiclasse. Quem não domina precision e recall em um problema tabular não consegue defender um critério de avaliação diante da diretoria, nem discutir de igual para igual com um time de Search.

## Conceitos

### 1. Feature encoding: o que o modelo realmente enxerga

> **Em palavras simples.** Um modelo só faz conta com números. **Feature encoding** é a tradução de categorias (cor, cidade, tipo de loja) para números. A tradução ingênua, numerar as categorias (**label encoding**), cria sem querer uma hierarquia: se Maçã é 0 e Laranja é 2, o modelo entende que Laranja "vale mais". O **one-hot encoding** evita isso dando a cada categoria a sua própria coluna de sim/não, como um formulário com uma caixa de marcar para cada opção. O **embedding** é o passo seguinte: em vez de caixas isoladas, cada categoria ganha uma pequena lista de números que o modelo aprende, e categorias que se comportam de forma parecida terminam com listas parecidas.

**Label encoding** (`Maçã=0, Banana=1, Laranja=2`) introduz uma ordem e uma distância que não existem: para o modelo, Laranja está "duas vezes mais longe" de Maçã do que Banana. Tree-based models toleram, porque só fazem splits (`x <= 0.5`). Modelos baseados em álgebra linear (linear/logistic regression, neural networks) multiplicam o valor por um peso e, portanto, tratam a ordem como informação.

**One-hot encoding** cria um vetor por categoria com um único `1`. Todos os vetores são ortogonais entre si (dot product igual a zero): nenhuma categoria está mais perto de outra. O custo é dimensionalidade: 50 mil categorias viram 50 mil colunas esparsas.

**A ponte para embeddings.** Um embedding resolve o problema do one-hot de propósito: em vez de 50 mil dimensões ortogonais, cada categoria (ou token) recebe um vetor denso de, digamos, 768 dimensões, e o treino aprende quais devem ficar próximas. Matematicamente, uma `nn.Embedding` é uma camada linear aplicada a um vetor one-hot. Você verá isso em código na Semana 6.

Quando usar o quê em dados tabulares: `OneHotEncoder` para baixa cardinalidade; `OrdinalEncoder` só quando a ordem é real (baixo/médio/alto); target encoding ou embeddings para alta cardinalidade, com cuidado redobrado com leakage.

> **No mundo real.** Na competição Kaggle **Rossmann Store Sales** (2015), o objetivo era prever as vendas diárias de 1.115 farmácias na Alemanha. Em vez de one-hot para loja, dia da semana e estado, Guo e Berkhahn treinaram **entity embeddings** para cada variável categórica e ficaram em 3º lugar com pouca engenharia manual de features. O resultado mais citado do paper (*Entity Embeddings of Categorical Variables*): os embeddings aprendidos para os estados alemães, quando desenhados em duas dimensões, reproduzem aproximadamente o mapa da Alemanha. Ninguém informou a geografia ao modelo; ela emergiu do padrão de vendas.

### 2. Feature scaling

> **Em palavras simples.** **Feature scaling** é colocar todas as colunas na mesma régua. Imagine comparar duas casas somando "número de quartos" (3 ou 4) com "preço" (800.000 ou 900.000): o preço domina a conta só porque seus números são maiores, não porque importe mais. O `StandardScaler` reescreve cada coluna como "quantos desvios acima ou abaixo da média", de modo que todas passem a ter o mesmo peso de partida.

- `StandardScaler`: `z = (x − μ) / σ`. Centra em 0, desvio 1. Padrão para modelos treinados por gradient descent.
- `MinMaxScaler`: comprime em `[0, 1]`. Sensível a outliers (um valor extremo esmaga todos os outros).
- `RobustScaler`: usa mediana e IQR; preferível com outliers fortes.

Por que importa: em modelos treinados por gradiente, features em escalas muito diferentes deixam a loss surface alongada como um vale estreito. O gradiente é enorme em uma direção e minúsculo na outra, e nenhuma learning rate serve para as duas (você vai provocar isso na Semana 3). Tree-based models são invariantes a escala; é por isso que o `RandomForestClassifier` do lab funcionaria sem o scaler. Ele está no `Pipeline` porque o hábito é o que importa quando o estimador for uma rede neural.

Em Deep Learning a mesma ideia reaparece dentro da rede, como `LayerNorm` e `BatchNorm`.

> **No mundo real.** O dataset **Wine** (178 vinhos italianos, 13 medidas químicas), incluído no scikit-learn, tem uma coluna, `proline`, com valores na casa das centenas até cerca de 1.680, enquanto outras, como `hue`, ficam em torno de 1. O exemplo oficial *Importance of Feature Scaling* mostra o efeito: sem scaling, o PCA produz um primeiro componente que é praticamente só `proline`, e um classificador treinado sobre ele vai mal; com `StandardScaler`, as três variedades de vinho se separam com clareza e a accuracy sobe de forma expressiva. Mesmos dados, mesmo modelo; só mudou a régua.

### 3. Data leakage e o papel do `Pipeline`

> **Em palavras simples.** **Data leakage** é quando o modelo, durante o treino, tem acesso a uma informação que não existirá na hora de usar de verdade. É como estudar para a prova com o gabarito na mão: a nota do simulado fica ótima e não diz nada sobre o que o aluno sabe. O perigo é que o leakage não dá erro; ele só produz métricas boas demais. O `Pipeline` é a disciplina que impede a forma mais comum: ele garante que toda estatística (média, desvio, vocabulário) seja calculada só com os dados de treino.

Leakage é qualquer informação do conjunto de teste (ou do futuro) que chega ao treino. O caso mais comum e mais sutil:

```python
X_scaled = StandardScaler().fit_transform(X)        # ERRADO: média e desvio calculados com o teste incluído
X_train, X_test = train_test_split(X_scaled, ...)
```

A correção é a ordem: split primeiro, `fit` só no treino, `transform` no teste. O `Pipeline` do scikit-learn garante isso por construção, inclusive dentro de `cross_val_score` e `GridSearchCV`, onde o erro manual é quase inevitável. Outras formas de leakage: features que só existem depois do evento (ex.: "data do chargeback" em um modelo de fraude), duplicatas entre treino e teste, e seleção de features feita no dataset inteiro.

Sintoma típico: métricas offline excelentes que desabam em produção. Você verá o mesmo problema em fine-tuning (Semana 17), com exemplos quase idênticos em train e test.

> **No mundo real.** Na **KDD Cup 2008**, a tarefa era detectar câncer de mama em dados de mamografia da Siemens. A equipe vencedora (IBM Research) descobriu que o **número de identificação do paciente** previa a doença: na faixa de IDs de 0 a 20.000, 36% dos pacientes tinham câncer; na faixa de 100.000 a 500.000, apenas 1%. A explicação provável é que os dados vinham de instituições diferentes, algumas de triagem e outras de tratamento, cada uma com sua faixa de IDs. O ID não tem relação médica com a doença e não ajudaria com um paciente novo. O caso é analisado em *Leakage in Data Mining* (Kaufman, Rosset e Perlich).

### 4. The accuracy paradox

> **Em palavras simples.** **Accuracy** é a porcentagem de acertos. Parece a medida óbvia, e engana sempre que um dos resultados é raro. Um detector de incêndio que nunca toca acerta em 99,9% dos dias, porque quase nunca há incêndio, e é inútil exatamente no dia que importa. Esse é o **accuracy paradox**: em problemas desbalanceados, o modelo que não faz nada parece excelente. Por isso, o primeiro passo é sempre calcular a accuracy desse modelo preguiçoso (o **baseline**).

Com 0,5% de fraudes, o modelo que responde sempre "não é fraude" atinge 99,5% de accuracy e zero utilidade. Accuracy é dominada pela majority class. Sempre calcule esse **baseline trivial** antes de comemorar qualquer número.

**Ponte GenAI:** um assistente que responde certo 95 vezes e inventa uma cláusula contratual 5 vezes tem 95% de "accuracy" e é impossível de colocar em produção. A pergunta certa não é "quantas acertou", é "quais errou e quanto custa cada erro".

> **No mundo real.** O dataset público **Credit Card Fraud Detection** (Kaggle, do grupo de ML da Université Libre de Bruxelles) contém 284.807 transações reais de cartões europeus em dois dias de setembro de 2013. Apenas **492 são fraudes: 0,172%**. Um modelo que responde "não é fraude" para tudo atinge **99,83% de accuracy** e deixa passar as 492. Por isso a própria descrição do dataset recomenda avaliar pela área sob a Precision-Recall curve, e não por accuracy.

### 5. Confusion matrix e as métricas derivadas

> **Em palavras simples.** A **confusion matrix** é uma tabela de quatro casas que separa os acertos e os erros por tipo. Pense em um exame médico: ele pode acertar dizendo "doente" (**true positive**), acertar dizendo "saudável" (**true negative**), alarmar uma pessoa saudável (**false positive**) ou liberar uma pessoa doente (**false negative**). **Precision** responde: "quando o exame diz doente, quantas vezes está certo?". **Recall** responde: "de todos os doentes, quantos o exame encontrou?". As duas medidas puxam em direções opostas, e o **F1-score** é um resumo que só fica alto quando as duas estão altas.

|  | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | TN | FP (false alarm, Type I error) |
| **Actual Positive** | FN (miss, Type II error) | TP |

- **Precision** = TP / (TP + FP). Das vezes em que o modelo disse "positivo", quantas eram verdade?
- **Recall** (sensitivity, true positive rate) = TP / (TP + FN). Dos positivos reais, quantos o modelo encontrou?
- **Specificity** = TN / (TN + FP). **False positive rate** = 1 − specificity.
- **F1-score** = 2 · P · R / (P + R), a média harmônica. Com P = 1,0 e R = 0,1, a média aritmética daria 0,55; o F1 dá 0,18. A média harmônica pune o desequilíbrio.
- **Fβ**: generalização em que β > 1 pesa mais o recall (F2 para detecção de fraude ou segurança) e β < 1 pesa mais a precision.

Exemplo numérico: 20.000 transações, 100 fraudes. O modelo marca 80 como fraude, das quais 60 são fraude de fato. TP = 60, FP = 20, FN = 40, TN = 19.880. Precision = 60/80 = 0,75. Recall = 60/100 = 0,60. Accuracy = 19.940/20.000 = 0,997. A accuracy não deixa ver que 40% das fraudes passaram.

No `classification_report`, repare em três linhas: as métricas **por classe** (a da minority class é a que interessa), `macro avg` (média simples entre classes, trata todas como iguais) e `weighted avg` (ponderada pelo support, volta a ser dominada pela majority class).

| Erro | ML clássico | LLM / RAG |
|---|---|---|
| False Positive | E-mail legítimo vai para o spam | O modelo cita um documento que não diz aquilo (citation hallucination) |
| False Negative | Fraude aprovada | O guardrail deixa passar uma prompt injection |

**Regra do arquiteto:** escolha a métrica pelo erro mais caro. FP caro (jurisprudência inventada para um advogado) pede precision. FN caro (vazamento de dados) pede recall.

> **No mundo real.** O dataset **Breast Cancer Wisconsin (Diagnostic)**, incluído no scikit-learn, tem 569 biópsias reais com 30 medidas de núcleos celulares: 212 malignas e 357 benignas. Um classificador simples passa de 95% de accuracy, mas o número que um médico olha é outro: quantos dos 212 tumores malignos foram classificados como benignos (false negatives). Dois modelos com a mesma accuracy podem diferir justamente aí, um errando para o lado do alarme falso (biópsia extra) e outro para o lado da omissão (câncer não tratado). É um bom dataset para repetir o lab desta semana.

### 6. Decision threshold, PR curve e ROC curve

> **Em palavras simples.** O modelo não responde "sim" ou "não"; ele responde com uma probabilidade, como "73% de chance". O **threshold** é a nota de corte que transforma esse número em decisão. Baixar o corte faz o modelo dizer "sim" com mais facilidade: encontra mais casos reais (recall sobe) e dá mais alarmes falsos (precision cai). Subir faz o contrário. Não existe corte correto em abstrato; existe o corte adequado ao custo de cada erro. A **Precision-Recall curve** é o gráfico que mostra todas as trocas possíveis para você escolher.

`predict()` aplica um threshold de 0,5 sobre `predict_proba()`. Esse 0,5 não tem nada de especial: é uma decisão de negócio que ninguém tomou. Baixar o threshold aumenta o recall e reduz a precision; subir faz o inverso. Nada é retreinado.

- **Precision-Recall curve:** todos os pares (recall, precision) possíveis variando o threshold. O resumo em um número é a **average precision (PR-AUC)**.
- **ROC curve:** true positive rate × false positive rate. O **ROC-AUC** é a probabilidade de o modelo dar score maior a um positivo aleatório do que a um negativo aleatório.

Em classes muito desbalanceadas a ROC engana: como TN é enorme, o false positive rate fica minúsculo mesmo com milhares de false positives, e a curva parece ótima. A PR curve não usa TN e mostra o problema. Use PR quando a classe positiva é rara e é a que importa.

> **No mundo real.** Em 2016, o Google publicou no JAMA (Gulshan et al.) um modelo treinado em 128.175 imagens de retina para detectar retinopatia diabética. O **mesmo modelo** foi reportado com dois thresholds, no conjunto de validação EyePACS-1. No ponto de alta especificidade: sensitivity (recall) de 90,3% e specificity de 98,1%, adequado para confirmar casos com poucos alarmes falsos. No ponto de alta sensibilidade: sensitivity de 97,5% e specificity de 93,4%, adequado para triagem, quando perder um doente é o pior erro. Nada foi retreinado entre um e outro; mudou só a nota de corte.

### 7. Cost-sensitive learning

> **Em palavras simples.** Nem todo erro custa igual. **Cost-sensitive learning** é levar isso em conta de forma explícita: em vez de pedir ao modelo "erre o mínimo possível", pede-se "gaste o mínimo possível", atribuindo um preço a cada tipo de erro. Há três formas: mover o threshold, dar mais peso aos erros da classe rara durante o treino (`class_weight`) ou alterar a proporção das classes nos dados (**resampling**). **Calibration** é a garantia de que, quando o modelo diz 80%, ele acerta em cerca de 80% das vezes; sem isso, a conta de custo não fecha.

Três alavancas para lidar com desbalanceamento, da mais barata para a mais invasiva:

1. **Mover o threshold** com base em uma matriz de custos (`custo = FN · c_fn + FP · c_fp`). É o que o Passo 8 do notebook faz.
2. **`class_weight="balanced"`**: pesa os erros da minority class na loss de forma inversamente proporcional à frequência. Não altera os dados.
3. **Resampling** (oversampling, SMOTE, undersampling): altera a distribuição de treino e, com ela, as probabilidades previstas, que deixam de ser calibradas.

**Calibration:** um modelo é calibrado quando, entre os casos em que ele diz 0,8, cerca de 80% são positivos. Thresholds baseados em custo só fazem sentido com probabilidades calibradas (`CalibratedClassifierCV`, reliability diagram).

> **No mundo real.** A **KDD Cup 1998** usou dados reais de uma organização de veteranos dos EUA: cerca de 95 mil pessoas que poderiam receber uma carta pedindo doação. Enviar cada carta custava **US$ 0,68**; só cerca de 5% respondiam, com doações de valores variados. Um classificador que maximiza accuracy conclui "ninguém doa" e não envia nada. A formulação correta, usada nos trabalhos de Zadrozny e Elkan sobre esse dataset, é enviar a carta quando `probabilidade de doar × valor esperado da doação > 0,68`. A decisão depende de probabilidades calibradas e de custos, não de acertar a classe.

## Erros comuns

- Reportar accuracy sem o baseline da majority class.
- Fazer `fit` do scaler (ou de qualquer transformador) antes do split.
- Esquecer `stratify=y` no `train_test_split` com classes raras: o teste pode ficar com pouquíssimos positivos.
- Escolher o threshold olhando o test set. O threshold é um hiperparâmetro: escolha em validation, reporte em test.
- Ler o `weighted avg` do report e concluir que o modelo vai bem.

## Leitura e vídeos (fim de semana)

- scikit-learn User Guide: *Pipelines and composite estimators*; *Metrics and scoring: classification metrics*; *Common pitfalls and recommended practices* (seção sobre data leakage).
- Google Machine Learning Crash Course: módulo *Classification* (thresholding, confusion matrix, precision/recall, ROC e AUC).
- Saito & Rehmsmeier, *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets* (leia o abstract e as figuras).
- scikit-learn User Guide: *Probability calibration*.

## Perguntas de autoavaliação

1. Por que ajustar o `StandardScaler` antes do `train_test_split` é um erro, mesmo que o resultado numérico mude pouco?
2. Em um detector de prompt injection, qual métrica você maximiza e qual o custo de fazer isso?
3. O que acontece com precision e recall quando o threshold cai de 0,5 para 0,1?
4. Por que o F1 usa média harmônica e não aritmética?
5. Um modelo tem ROC-AUC de 0,98 em um dataset com 0,1% de positivos. Por que isso pode ser pouco informativo?
6. Qual a diferença entre `macro avg` e `weighted avg` e qual delas esconde o problema da minority class?

## Entregável da semana

Notebook resolvido no GitHub e um parágrafo em `docs/impacto-de-negocio.md` descrevendo, para o modelo de fraude, quanto custa cada false positive e cada false negative, qual threshold você recomendaria e como a mesma análise se aplica a um hallucination detector em um sistema RAG.

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/ai-architect-foundations/blob/main/01-Phase-1-Math-and-ML-Core/01-sklearn-pipelines/notebooks/semana_01_exercicio.ipynb) · [`semana_01_exercicio.ipynb`](../notebooks/semana_01_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/ai-architect-foundations/blob/main/01-Phase-1-Math-and-ML-Core/01-sklearn-pipelines/notebooks/semana_01_gabarito.ipynb) · [`semana_01_gabarito.ipynb`](../notebooks/semana_01_gabarito.ipynb) |

[Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 2 →](/01-Phase-1-Math-and-ML-Core/02-linear-algebra-numpy/docs/Semana-02-Conceitos.md)
