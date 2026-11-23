# Guia de Execução (Rascunho Revisado): Fase 1 - Fundamentos Matemáticos e Machine Learning Core

> **Nota:** Para a trilha completa e atualizada de 32 semanas, consulte [`PLANO_DE_ESTUDO.md`](../PLANO_DE_ESTUDO.md) e os documentos `Semana-NN-Conceitos.md`.

**Objetivo da Fase:** Elevar a sólida base de Ciência/Análise de Dados para o nível de Arquitetura de IA Corporativa. A transição real aqui é sair do ecossistema de Machine Learning clássico (dados tabulares, modelos preditivos padrão) para dominar a matemática de Deep Learning (dados não-estruturados, tensores, embeddings e mecanismos de atenção). Construir a ponte técnica para diagnosticar modelos complexos de RAG e Fine-Tuning corporativo no futuro.

**Rotina Semanal:**
*   **Fim de Semana (Sábado e Domingo):** Laptop. Foco 100% em absorção teórica e intuição visual. Nenhuma linha de código.
*   **Dias Úteis (Segunda a Sexta):** Google Colab (CPU ou T4 gratuito). Foco 100% em engenharia. Traduzir a intuição em código vetorizado e documentar no GitHub no fim do dia.

---

## Semana 1: Scikit-Learn & Orquestração (Refresh Estratégico)

**Tema:** Calibrar o conhecimento existente em métricas e pipelines estruturados, mapeando-os diretamente para os desafios de avaliação de LLMs (ex: como Precision/Recall se traduz na avaliação de alucinações em sistemas RAG).

### Sábado e Domingo: Teoria das Métricas (Visão de Arquiteto)
*   **Tópico 1: Transformação de Dados.** Revisar `One-Hot Encoding`, `Label Encoding`, `StandardScaler` e `MinMaxScaler`. O foco mental aqui é lembrar *por que* modelos matemáticos (incluindo redes neurais no futuro) falham sem dados padronizados.
*   **Tópico 2: A Mentira da Acurácia.** Reforçar os perigos de medir apenas a "Acurácia" em datasets desbalanceados.
*   **Tópico 3: O Santo Graal das Métricas.** Revisão profunda da **Matriz de Confusão**. Internalizar Falsos Positivos e Falsos Negativos definindo sempre a classe positiva (`y = 1`): *em um detector de alucinações (`y = 1` = alucinação), deixar passar uma alucinação ao cliente é um Falso Negativo; em um verificador de citação (`y = 1` = citação válida), aprovar uma fonte inventada é um Falso Positivo*. Revisar **Precisão**, **Recall**, **F1-Score** e **Precision-Recall curve**.

### Segunda a Sexta: Prática no Colab (2h / dia)
*   **Passo 1:** No Colab, gere um dataset desbalanceado com `make_classification(..., weights=[0.995])` (evite datasets de regressão como *California Housing* para classificação).
*   **Passo 2:** Crie um objeto `Pipeline` do Scikit-Learn que execute o escalonamento após o `train_test_split`, evitando data leakage.
*   **Passo 3:** Adicione um modelo de classificação (ex: `RandomForestClassifier` ou `LogisticRegression`).
*   **Passo 4:** Divida os dados com `stratify=y`, treine o pipeline completo e gere previsões calibradas por threshold.
*   **Passo 5 (Crucial):** Utilize `classification_report`, `confusion_matrix` e análise de custo por erro. Redija um breve parágrafo (salvo na pasta `/docs/`) sobre o impacto de negócio dos Falsos Positivos e Falsos Negativos.

---

## Semana 2: A Geometria dos Dados (Álgebra Linear)

**Tema:** A matemática pura por trás da similaridade semântica, embeddings e buscas vetoriais (o núcleo dos sistemas RAG corporativos).

### Sábado e Domingo: Intuição Visual (3Blue1Brown)
*   **Ação:** Assistir aos vídeos essenciais da playlist *Essence of Linear Algebra* do canal 3Blue1Brown no YouTube.
*   **Foco Principal:** Entender o que acontece geometricamente quando você multiplica matrizes.
*   **Conceito Chave:** Dominar visualmente o **Produto Escalar (Dot Product)** e a **Similaridade de Cosseno (Cosine Similarity)**. Internalize que a Cosine Similarity mede a similaridade de direção independente do comprimento do vetor — coincidindo no ranking com o Dot Product quando os embeddings estão normalizados (`||v|| = 1`).

### Segunda a Sexta: Prática no Colab (2h / dia)
*   **Passo 1:** Esqueça o Pandas e o Scikit-Learn. Trabalhe puramente com matrizes em `numpy`.
*   **Passo 2:** Crie Arrays do Numpy representando pequenos vetores de características.
*   **Passo 3:** Programe, do zero, uma função que calcula a Similaridade de Cosseno entre dois vetores baseada na fórmula do Produto Escalar.
*   **Passo 4:** Crie uma "mini-base de dados vetorial" em Numpy e escreva um loop que encontre qual vetor é o mais similar à sua consulta. Esta é a engenharia reversa de um Vector DB.

---

## Semana 3: O Motor do Aprendizado (Cálculo)

**Tema:** Como o fine-tuning de um LLM open-source realmente funciona por debaixo dos panos (Backpropagation e Gradient Descent).

### Sábado e Domingo: Intuição Visual (3Blue1Brown)
*   **Ação:** Assistir aos vídeos chave das playlists *Essence of Calculus* e *Neural Networks* do 3Blue1Brown.
*   **Foco Principal:** Entender a **Derivada** como o vetor que aponta a "direção" para reduzir a taxa de erro do modelo.
*   **Conceitos Chave:** Compreender a **Descida do Gradiente (Gradient Descent)** e a **Regra da Cadeia (Chain Rule)**. Saber isso será fundamental para quando formos configurar hiperparâmetros (Learning Rate) no fine-tuning corporativo (PEFT/LoRA).

### Segunda a Sexta: Prática no Colab (2h / dia)
*   **Passo 1:** Usando `numpy`, crie dados fictícios (ex: uma linha reta com ruído `y = 2x + 1`).
*   **Passo 2:** Inicialize dois parâmetros de peso aleatórios.
*   **Passo 3:** Crie o loop de treinamento manual (Epochs).
*   **Passo 4:** Em cada iteração, calcule a previsão, o Erro Quadrático Médio (MSE), o Gradiente manual, e atualize os pesos multiplicando pela Taxa de Aprendizado.
*   **Passo 5:** Registre a queda do erro (Loss) a cada epoch. Você programou a base de uma rede neural do absoluto zero.

---

## Semana 4: A Ponte para Tensores (PyTorch)

**Tema:** O salto final. Sair da computação linear estruturada em CPU (Numpy/Pandas) para tensores de alta dimensionalidade acelerados por GPU.

### Sábado e Domingo: Compreensão de Arquitetura
*   **Ação:** Leitura focada na documentação oficial do PyTorch (`torch.Tensor` e `torch.autograd`).
*   **Foco Principal:** Entender a diferença arquitetural entre Numpy Arrays e PyTorch Tensors.
*   **Conceito Chave:** Entender o `Autograd` – o motor interno do PyTorch que calcula automaticamente todas as derivadas complexas que você fez à mão na Semana 3.

### Segunda a Sexta: Prática no Colab (2h / dia)
*   **Passo 1:** No Google Colab, ative o Runtime de GPU (T4 gratuita).
*   **Passo 2:** Crie tensores de dados no PyTorch e aprenda a movê-los da memória RAM (CPU) para a memória de vídeo (`.to('cuda')`).
*   **Passo 3:** Recrie o pipeline simples de treinamento da Semana 3 utilizando `torch.nn.Linear` e o otimizador nativo do PyTorch (`torch.optim`).
*   **Passo 4:** Avalie a performance e a simplicidade.
*   **Fechamento:** Commit final da Fase 1 no repositório público do GitHub.