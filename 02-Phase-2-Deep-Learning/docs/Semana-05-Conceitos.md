# Semana 5 — O Training Loop: nn.Module, DataLoader e MLP

**Fase 2: Deep Learning e o Ecossistema Hugging Face** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Sair da regressão de um neurônio para uma rede neural de verdade, organizando o código como o ecossistema inteiro organiza: `Dataset`, `DataLoader`, `nn.Module`, training loop e validation loop. Ao final você saberá ler um par de curvas train/validation e dizer o que está errado.

## Por que isso importa para um AI Architect

O `Trainer` da Hugging Face, o `SFTTrainer` e qualquer script de fine-tuning são embalagens em volta deste mesmo loop. Quem o escreveu à mão sabe ler uma learning curve e decidir se o problema é dado, capacidade, regularização ou learning rate. E a cross-entropy loss desta semana é a mesma usada para treinar LLMs.

## Conceitos

### 1. Por que layers e non-linearities

> **Em palavras simples.** Uma **layer** é uma etapa de contas que transforma os números de entrada. Empilhar etapas que só fazem contas "em linha reta" não adianta: dez réguas coladas continuam sendo uma régua. A **activation function** (como a `ReLU`) é uma pequena dobra colocada entre as etapas, e é com muitas dobras que a rede consegue desenhar formas curvas e complicadas, como a fronteira entre "gato" e "cachorro".

Empilhar camadas lineares sem activation colapsa em uma única camada linear: `W₂(W₁x) = (W₂W₁)x`. A **activation function** entre elas é o que permite aproximar fronteiras curvas.

| Activation | Fórmula | Observação |
|---|---|---|
| `ReLU` | `max(0, x)` | Barata; não satura no lado positivo; pode gerar "dead neurons" |
| `GELU` | `x · Φ(x)` | Versão suave do ReLU; padrão em Transformers |
| `SiLU` / Swish | `x · sigmoid(x)` | Usada em vários LLMs recentes |
| `tanh`, `sigmoid` | — | Saturam; sofrem de vanishing gradient em redes profundas |

Um **MLP** (multi-layer perceptron) é `Linear → activation → Linear → … → Linear`. O *universal approximation theorem* diz que um MLP com uma hidden layer larga o bastante aproxima qualquer função contínua; na prática, profundidade é mais eficiente que largura.

O bloco feed-forward dentro de cada camada de um Transformer é exatamente um MLP de duas camadas.

> **No mundo real.** O paper da **AlexNet** (2012) mediu a diferença: em um teste no dataset CIFAR-10, a rede com `ReLU` chegou a 25% de erro de treino **seis vezes mais rápido** que a mesma rede com `tanh`, a ativação padrão até então. Os autores afirmam que não teriam conseguido treinar uma rede daquele tamanho com as ativações tradicionais. Desde então, `ReLU` e suas variantes (`GELU`, `SiLU`) são o padrão.

### 2. `nn.Module`

> **Em palavras simples.** Um `nn.Module` é um bloco de montar. Cada bloco guarda os seus números ajustáveis e sabe transformar uma entrada em uma saída; blocos pequenos se encaixam para formar blocos maiores, até virar o modelo inteiro. A vantagem é organização: dá para pedir a lista de todos os parâmetros, mover tudo para a GPU ou salvar o modelo com um único comando.

Uma classe com `__init__` (declara as layers, cujos pesos viram `nn.Parameter` automaticamente) e `forward` (define o cálculo). Chama-se `model(x)`, nunca `model.forward(x)`: o `__call__` executa hooks além do forward.

- `model.parameters()`: iterador entregue ao optimizer.
- `model.state_dict()`: dicionário nome → tensor. É o que se salva.
- `nn.Sequential`: atalho para pilhas simples.
- `nn.ModuleList`: lista de módulos que o PyTorch enxerga (uma lista Python comum não registra os parâmetros).

> **No mundo real.** A `resnet50` da biblioteca `torchvision`, um dos modelos de visão mais usados na indústria, é um `nn.Module` feito de blocos repetidos (`Bottleneck`), cada um composto de `Conv2d`, `BatchNorm2d` e `ReLU`. `sum(p.numel() for p in model.parameters())` devolve cerca de **25,6 milhões** de parâmetros. Trocar a última layer por uma nova, com o número de classes do seu problema, é uma linha de código: é assim que a maior parte dos classificadores de imagem corporativos nasce.

### 3. `Dataset` e `DataLoader`

> **Em palavras simples.** O `Dataset` é o arquivo: sabe quantos exemplos existem e como buscar o exemplo de número tal. O `DataLoader` é o garçom: vai ao arquivo, monta bandejas com vários exemplos (os **batches**), embaralha a ordem e entrega uma bandeja por vez ao modelo. Separar as duas coisas permite trocar a fonte dos dados sem mexer no treino.

`Dataset` responde "qual é o exemplo `i`?" (`__len__`, `__getitem__`). `DataLoader` monta batches, embaralha e paraleliza a leitura.

Argumentos que importam: `batch_size`; `shuffle=True` no treino (e `False` na validation); `num_workers` para carregar em paralelo; `pin_memory=True` acelera a cópia para a GPU; `drop_last=True` descarta o último batch incompleto; `collate_fn` define como exemplos individuais viram um batch (você escreverá uma na Semana 10 para fazer dynamic padding).

Separar dados do modelo é o que permite trocar um CSV por um corpus de 100 GB sem tocar no loop.

> **No mundo real.** O conjunto de treino do **ImageNet** tem 1,28 milhão de fotos e ocupa mais de 130 GB em disco: não cabe na memória de uma máquina comum. O `torchvision.datasets.ImageFolder` (um `Dataset`) lê cada foto do disco só quando ela é pedida, e o `DataLoader`, com vários `num_workers`, carrega e redimensiona as imagens em paralelo enquanto a GPU treina. Sem essa divisão de trabalho, a GPU passaria a maior parte do tempo esperando dados.

### 4. Logits, softmax e cross-entropy

> **Em palavras simples.** A rede termina com uma lista de notas, uma por opção possível: os **logits**. A função **softmax** transforma essas notas em porcentagens que somam 100%. A **cross-entropy** mede o quanto o modelo foi surpreendido pela resposta certa: se ele dava 90% de chance a ela, a punição é pequena; se dava 1%, é enorme. Errar com convicção custa caro.

A última layer produz **logits**: números reais sem restrição, um por classe. `softmax` os converte em probabilidades:

`pᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)`

A **cross-entropy loss** é `−log p(classe correta)`. Se o modelo dá probabilidade 0,9 à classe certa, a loss é 0,105; se dá 0,01, é 4,6. Errar com confiança custa caro.

`nn.CrossEntropyLoss` recebe **logits crus** e o índice da classe (`long`). Internamente aplica `log_softmax` de forma numericamente estável (o truque log-sum-exp). Aplicar `softmax` antes de passar à loss é um bug comum: o modelo treina, mas mal.

Para duas classes, `nn.BCEWithLogitsLoss` faz o mesmo com um único logit e `sigmoid`.

**Sanity check da loss inicial:** com `C` classes e pesos aleatórios, o modelo chuta uniformemente e a loss deve começar perto de `ln(C)`. Para 2 classes, 0,693; para um vocabulário de 50.000 tokens, 10,8. Se o primeiro valor estiver muito longe disso, há um bug na inicialização ou nos labels.

Em um language model, as "classes" são os tokens do vocabulário e a mesma loss é aplicada em cada posição da sequência.

> **No mundo real.** O **GPT-2** escolhe a próxima palavra entre 50.257 tokens: é uma classificação com 50.257 classes, repetida a cada posição do texto. Antes de qualquer treino, o modelo chuta ao acaso e a cross-entropy fica em torno de `ln(50257) ≈ 10,8`. O README do nanoGPT usa exatamente esse número como verificação de que o código está certo; ao fim do treino no OpenWebText, a loss cai para cerca de 2,85.

### 5. `model.train()` vs. `model.eval()`

> **Em palavras simples.** Algumas peças da rede se comportam de um jeito durante o treino e de outro na hora de usar. O **dropout**, por exemplo, desliga neurônios ao acaso no treino para a rede não depender de nenhum deles; na hora da prova, todos precisam estar ligados. `model.train()` e `model.eval()` são o interruptor entre os dois modos. Esquecer de virar o interruptor produz resultados piores e instáveis, sem nenhuma mensagem de erro.

- `model.train()`: ativa `Dropout` e a atualização das estatísticas de `BatchNorm`.
- `model.eval()`: desativa ambos. Combine com `torch.no_grad()` para não construir o graph.

São coisas diferentes: `eval()` muda o **comportamento** das layers; `no_grad()` desliga o **registro de gradientes**. Esquecer `eval()` é um bug clássico: as métricas de validation ficam ruidosas e piores sem motivo aparente.

> **No mundo real.** A **AlexNet** usou dropout de 50% nas duas primeiras camadas densas, e o paper registra que, sem ele, a rede sofria overfitting substancial. No teste, todos os neurônios ficam ativos e as saídas são ajustadas para compensar. É a razão histórica de todo framework ter um modo de treino e um modo de avaliação. O mesmo interruptor controla o `BatchNorm`, presente em quase todo modelo de visão.

### 6. Train, validation e test

> **Em palavras simples.** Os dados são divididos em três partes com papéis diferentes. **Train** é o material de estudo. **Validation** é o simulado, usado para ajustar a forma de estudar. **Test** é a prova final, feita uma única vez. Se você ajusta o método olhando a prova final repetidas vezes, ela deixa de medir o que você sabe e passa a medir o quanto você se adaptou a ela.

- **Train:** ajusta os pesos.
- **Validation:** escolhe hiperparâmetros, número de epochs e threshold. O modelo nunca treina nele, mas você decide com base nele.
- **Test:** usado uma única vez, no final. Qualquer decisão tomada olhando o test set o transforma em validation.

> **No mundo real.** Na competição **ImageNet**, os rótulos do test set eram secretos e cada equipe podia enviar poucas respostas por semana ao servidor de avaliação. Em 2015, a equipe da Baidu foi **banida por um ano** depois de fazer cerca de 200 envios em seis meses usando várias contas, o que permitia ajustar o modelo ao test set. O resultado anunciado foi retirado. O episódio virou o exemplo clássico de por que o test set precisa ser protegido.

### 7. Lendo learning curves

> **Em palavras simples.** **Learning curves** são dois gráficos lado a lado: o erro nos dados de estudo e o erro no simulado, ao longo do treino. Se os dois são altos, o modelo ainda não aprendeu (**underfitting**). Se o erro de estudo cai e o do simulado sobe, o modelo está decorando em vez de entender (**overfitting**). Ler esse par de curvas é o exame de sangue de qualquer treino.

| Train loss | Validation loss | Diagnóstico | Ação |
|---|---|---|---|
| Alta | Alta | **Underfitting** | Modelo maior, mais epochs, LR diferente |
| Cai | Cai junto | Saudável | Continue |
| Cai | Para de cair e sobe | **Overfitting** | Regularização, mais dados, early stopping |
| Oscila muito | Oscila muito | LR alta ou batch pequeno | Reduza a LR |

O espaço entre as duas curvas é o **generalization gap**. Esse é o **bias-variance tradeoff** em forma visual: underfitting é bias alto; overfitting é variance alta.

> **No mundo real.** Nas competições do **Kaggle**, o test set é dividido em uma parte pública, cujo placar todos veem durante a competição, e uma parte privada, revelada só no fim. É comum que equipes no topo do placar público despenquem dezenas ou centenas de posições no privado (o *shake-up*): elas ajustaram o modelo ao placar visível. É a mesma curva de validação subindo, em escala de competição.

### 8. Regularization

> **Em palavras simples.** **Regularization** é o conjunto de freios que impedem o modelo de decorar. **Weight decay** puxa os números internos para perto de zero, desestimulando soluções exageradas. **Dropout** desliga partes da rede ao acaso, forçando redundância. **Early stopping** interrompe o treino quando o simulado para de melhorar. Todos trocam um pouco de desempenho no material de estudo por mais desempenho em dados novos.

- **Weight decay:** penaliza pesos grandes, puxando-os para zero a cada step. No `AdamW` é aplicado de forma desacoplada do gradiente (valor típico: 0,01).
- **Dropout:** zera aleatoriamente uma fração `p` das ativações durante o treino, forçando redundância. Na inference é desligado.
- **Early stopping:** interrompe quando a validation loss para de melhorar por `patience` epochs, e restaura o melhor checkpoint.
- **Data augmentation** e **mais dados:** o que mais funciona, quando disponível.

Em fine-tuning de LLMs, overfitting aparece como o modelo decorando o instruction dataset: train loss perto de zero e respostas que só funcionam para os exemplos vistos.

> **No mundo real.** A receita de fine-tuning publicada com o **BERT** (Devlin et al., 2018) usa dropout de 0,1 em todas as camadas e weight decay de 0,01, com apenas 2 a 4 epochs. São valores pequenos, e deliberados: o modelo tem 110 milhões de parâmetros e muitos datasets de fine-tuning têm poucos milhares de exemplos, uma proporção em que decorar seria trivial. Esses mesmos valores são o padrão da biblioteca `transformers` até hoje.

### 9. Initialization

> **Em palavras simples.** Antes do treino, os números internos da rede são sorteados. O tamanho desse sorteio importa: valores grandes demais fazem o sinal crescer sem controle de camada em camada; pequenos demais fazem o sinal sumir. Uma boa **initialization** calibra o sorteio para o sinal atravessar a rede inteira com intensidade estável, como ajustar o volume de cada etapa de uma cadeia de amplificadores.

Pesos começam aleatórios, mas a escala importa: grande demais e as ativações explodem; pequena demais e os gradientes somem. `nn.Linear` usa Kaiming initialization por padrão, calibrada para manter a variância das ativações estável entre layers. Fixe `torch.manual_seed` para comparar experimentos.

> **No mundo real.** O paper *Delving Deep into Rectifiers* (He et al., Microsoft, 2015) propôs a **Kaiming initialization**, calibrada para redes com `ReLU`. Uma rede de 30 camadas que não convergia com a inicialização anterior passou a treinar normalmente. O modelo resultante atingiu 4,94% de erro top-5 no ImageNet, o primeiro resultado publicado abaixo da estimativa de erro humano na tarefa (5,1%). É a inicialização padrão do `nn.Linear` e do `nn.Conv2d` no PyTorch.

### 10. Um roteiro de debugging

> **Em palavras simples.** Quando um treino não funciona, a causa pode ser o modelo, os dados, a loss ou os ajustes, e não há mensagem de erro para dizer qual. O teste mais útil é tentar fazer a rede **decorar um único batch** de poucos exemplos. Se ela não consegue nem isso, há um defeito no código, e não adianta mexer em ajustes finos. Primeiro prova-se que a máquina funciona; depois se calibra.

A prática mais valiosa do texto do Karpathy: antes de treinar de verdade, tente **overfit a single batch**. Pegue 8 exemplos e treine até a loss chegar a zero. Se não chegar, há um bug no modelo, na loss ou nos dados, e não adianta olhar hiperparâmetros. Só depois escale.

> **No mundo real.** O texto *A Recipe for Training Neural Networks* (2019), de Andrej Karpathy, então diretor de IA da Tesla, descreve esse procedimento passo a passo e é provavelmente o guia de depuração mais citado da área. A observação central: redes neurais "falham em silêncio". Um bug raramente gera exceção; ele só faz o modelo treinar um pouco pior. Por isso a ordem recomendada é começar simples, verificar cada etapa e só então aumentar a complexidade.

### 11. Salvar e carregar

> **Em palavras simples.** Salvar um modelo é guardar em arquivo todos os números que ele aprendeu (o `state_dict`), como salvar o progresso de um jogo. Para usar depois, monta-se a mesma estrutura vazia e carregam-se os números. Guardar só os números, e não o programa inteiro, deixa o arquivo portátil e mais seguro.

```python
torch.save(model.state_dict(), "model.pt")
model = MLP(); model.load_state_dict(torch.load("model.pt", map_location=device))
```

Salve o `state_dict`, não o objeto do modelo (que depende do código da classe via pickle). O ecossistema Hugging Face usa o formato `safetensors`, que é seguro contra execução de código e mais rápido de carregar.

> **No mundo real.** O **Hugging Face Hub** hospeda mais de um milhão de modelos públicos. O formato antigo dos arquivos, baseado em `pickle`, permite que um arquivo malicioso execute código na máquina de quem o carrega. Por isso a Hugging Face criou o formato **safetensors**, que guarda apenas tensores, passou por auditoria de segurança independente em 2023 e se tornou o padrão da biblioteca `transformers`.

## Erros comuns

- Aplicar `softmax` antes de `CrossEntropyLoss`.
- Labels com dtype `float` ou shape `(N, 1)` em vez de `long` e `(N,)`.
- Esquecer `model.eval()` na validation ou `model.train()` ao voltar ao treino.
- `shuffle=False` no treino.
- Guardar layers em uma lista Python em vez de `nn.ModuleList`: os parâmetros não são treinados e nenhum erro aparece.
- Ajustar hiperparâmetros olhando o test set.

## Leitura e vídeos (fim de semana)

- PyTorch Tutorials, *Learn the Basics*: *Datasets & DataLoaders*; *Build the Neural Network*; *Save and Load the Model*.
- 3Blue1Brown, *Neural Networks*, Chapter 1 (*But what is a neural network?*).
- Andrej Karpathy, *A Recipe for Training Neural Networks*. Leia inteiro; é o melhor guia prático de debugging de treino.
- Stanford CS231n notes: *Neural Networks Part 1: Setting up the Architecture* e *Part 3: Learning and Evaluation*.
- Opcional: Srivastava et al., *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* (abstract e seção 1).

## Perguntas de autoavaliação

1. Por que `CrossEntropyLoss` recebe logits e não probabilidades?
2. Qual a diferença entre `model.eval()` e `torch.no_grad()`?
3. A train loss cai e a validation loss sobe a partir da epoch 20. Cite três ações possíveis.
4. Para que serve tentar "overfit a single batch" antes de treinar de verdade?
5. Qual deve ser a loss inicial de um classificador de 10 classes? E de um language model com vocabulário de 32.000 tokens?
6. Por que o test set só pode ser usado uma vez?

## Perguntas do entrevistador

1. Ao colocar seu modelo PyTorch em inferência, as previsões para o mesmo input variam entre chamadas e a acurácia caiu em relação à validação. O que provavelmente faltou configurar entre treino e avaliação (`model.eval()`, `torch.no_grad()`, BatchNorm, Dropout) e como você blinda o pipeline contra isso?

## Entregável da semana

Notebook resolvido com as curvas train/validation de dois modelos (sem e com regularization) e duas frases de diagnóstico.

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/02-Phase-2-Deep-Learning/notebooks/semana_05_exercicio.ipynb) · [`semana_05_exercicio.ipynb`](../notebooks/semana_05_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/02-Phase-2-Deep-Learning/notebooks/semana_05_gabarito.ipynb) · [`semana_05_gabarito.ipynb`](../notebooks/semana_05_gabarito.ipynb) |

[← Semana 4](/01-Phase-1-Math-and-ML-Core/04-pytorch-tensors/docs/Semana-04-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 6 →](/02-Phase-2-Deep-Learning/docs/Semana-06-Conceitos.md)
