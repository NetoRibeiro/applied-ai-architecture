# Laboratório Semana 1 (Rascunho Inicial): Pipelines, Desbalanceamento e Matriz de Confusão

> **Nota:** A versão oficial e executável deste laboratório está em [`01-Phase-1-Math-and-ML-Core/01-sklearn-pipelines/notebooks/semana_01_exercicio.ipynb`](01-Phase-1-Math-and-ML-Core/01-sklearn-pipelines/notebooks/semana_01_exercicio.ipynb).

*Instruções: Abra no Google Colab (runtime CPU). Copie os blocos de texto para células de "Texto" (Markdown) e os blocos de código para células de "Código". Complete as partes indicadas com `[SEU CÓDIGO AQUI]`.*

---

### [Célula de Texto 1]
## Passo 1: Importações e Configuração Inicial
Vamos importar as ferramentas necessárias para orquestração de dados e métricas, além das bibliotecas base.

### [Célula de Código 1]
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Utilitários do Scikit-Learn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
```

---

### [Célula de Texto 2]
## Passo 2: O Desafio do Mundo Real (Dados Desbalanceados)
Para focar na arquitetura do pipeline e não perder tempo baixando CSVs, vamos simular matematicamente o dataset do Kaggle de Fraude de Cartão de Crédito. Vamos gerar 100.000 transações, onde apenas 0.2% são fraudes reais.

### [Célula de Código 2]
```python
# Gerando o dataset simulado de fraudes
X, y = make_classification(
    n_samples=100000, 
    n_features=20, 
    n_informative=15, 
    n_redundant=5, 
    weights=[0.998], # 99.8% transações normais, 0.2% fraudes
    flip_y=0, 
    random_state=42
)

# Convertendo para Pandas DataFrame para visualização do analista
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
df['is_fraud'] = y

print("Distribuição das Classes:")
print(df['is_fraud'].value_counts(normalize=True) * 100)
```

---

### [Célula de Texto 3]
## Passo 3: Engenharia do Pipeline
Agora é com você. Separe os dados em treino e teste (80/20) usando o `train_test_split`. Depois, crie um objeto `Pipeline` do Scikit-Learn. O pipeline deve ter duas etapas:
1. `scaler`: Um `StandardScaler` para padronizar os dados (modelos baseados em árvores como `RandomForestClassifier` são invariantes à escala, mas manter o `StandardScaler` no `Pipeline` evita data leakage e prepara o fluxo para modelos lineares e redes neurais).
2. `classifier`: Um `RandomForestClassifier` com `random_state=42`.

### [Célula de Código 3]
```python
# 1. Separe os dados em X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Construa o Pipeline
pipeline_base = Pipeline([
    # [SEU CÓDIGO AQUI: Adicione o passo do StandardScaler]
    # [SEU CÓDIGO AQUI: Adicione o passo do RandomForestClassifier]
])

# 3. Treine o modelo usando apenas uma linha de código
# [SEU CÓDIGO AQUI: pipeline_base.fit(...)]
```

---

### [Célula de Texto 4]
## Passo 4: A Mentira da Acurácia
Gere as previsões no conjunto de teste e avalie a Matriz de Confusão. Observe como a Acurácia geral pode ser +99%, mas o modelo está deixando passar fraudes (Falsos Negativos).

### [Célula de Código 4]
```python
# 1. Gere as previsões
# [SEU CÓDIGO AQUI: y_pred = pipeline_base.predict(...)]

# 2. Imprima o Relatório de Classificação (Precision, Recall, F1-Score)
print("Relatório de Classificação (Pipeline Base):")
# [SEU CÓDIGO AQUI: print(classification_report(...))]

# 3. Visualize a Matriz de Confusão
print("\nMatriz de Confusão:")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal (0)', 'Fraude (1)'])
disp.plot(cmap='Blues')
plt.show()
```

---

### [Célula de Texto 5]
## Passo 5: A Visão do Arquiteto (Maximizando o Recall)
O modelo base falhou miseravelmente no Recall da classe 1 (Fraude), porque a Random Forest focou na maioria. 
**Desafio:** Crie um `pipeline_balanceado`. Use o mesmo StandardScaler e a mesma Random Forest, mas desta vez, adicione o parâmetro `class_weight='balanced'` dentro do RandomForestClassifier. Treine e avalie novamente. Note como os Falsos Negativos despencam.

### [Célula de Código 5]
```python
# [SEU CÓDIGO AQUI: Construa, treine e avalie o novo pipeline_balanceado]
# Dica: Copie o código do Passo 3 e 4, apenas alterando o parâmetro da árvore.
```