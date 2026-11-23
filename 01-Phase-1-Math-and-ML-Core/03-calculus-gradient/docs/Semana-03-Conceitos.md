# Semana 3 — O Motor do Aprendizado: Calculus e Gradient Descent

**Fase 1: Math Foundations e ML Core** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Entender a derivative como instrução de ajuste, gradient descent como algoritmo de aprendizado e a chain rule como o mecanismo que torna backpropagation possível. Ao final você terá treinado um modelo calculando cada gradient à mão e saberá diagnosticar uma loss curve.

## Por que isso importa para um AI Architect

Fine-tuning é gradient descent. Quando um treino de LoRA diverge, estagna ou "esquece" o que sabia, o diagnóstico está em quatro coisas que você vai manipular nesta semana: a loss function, o gradient, a learning rate e o batch size. Os mesmos nomes aparecem como hiperparâmetros em qualquer `TrainingArguments` da Hugging Face.

## Conceitos

### 1. Loss function

> **Em palavras simples.** A **loss function** é a nota do modelo, só que ao contrário: quanto maior, pior. Ela resume em um único número o quanto as previsões estão longe das respostas certas. Treinar um modelo é, literalmente, procurar os ajustes internos que deixam esse número o menor possível. O **Mean Squared Error** eleva cada erro ao quadrado antes de tirar a média, o que faz erros grandes pesarem muito mais do que erros pequenos.

Um número que mede o quanto o modelo está errado; treinar é minimizá-lo. Para regressão, o **Mean Squared Error (MSE)**:

`L(w, b) = (1/N) Σ (ŷᵢ − yᵢ)²`, com `ŷᵢ = w·xᵢ + b`

Para classificação, e para language models, usa-se **cross-entropy** (Semana 5). A loss precisa ser diferenciável em relação aos parâmetros; é por isso que não se treina diretamente com accuracy, que é uma função em degraus.

> **No mundo real.** O **Netflix Prize** definiu o problema inteiro por uma loss: o **RMSE** (a raiz do MSE) entre as notas previstas e as notas reais dos usuários, em uma escala de 1 a 5 estrelas. O sistema da própria Netflix, o Cinematch, tinha RMSE em torno de 0,95. O prêmio de US$ 1 milhão iria para quem reduzisse esse número em 10%, e foi pago em 2009 à equipe BellKor's Pragmatic Chaos, depois de quase três anos de competição. Milhares de equipes otimizaram o mesmo número.

### 2. Derivative e gradient

> **Em palavras simples.** A **derivative** responde a uma pergunta de sensibilidade: "se eu girar este botão um pouquinho, o erro sobe ou desce, e quanto?". Um modelo tem muitos botões (os **parâmetros**). O **gradient** é a lista com a resposta para todos os botões ao mesmo tempo: uma seta que aponta para a direção em que o erro mais cresce. Para melhorar o modelo, anda-se no sentido contrário ao da seta.

`dL/dw` responde: "se eu aumentar `w` um pouquinho, quanto `L` muda?". Sinal positivo significa que aumentar `w` piora a loss; então devemos diminuir `w`.

O **gradient** `∇L` é o vetor com as partial derivatives em relação a todos os parâmetros. Ele aponta para a direção em que a loss **cresce** mais rápido; andamos no sentido oposto.

Derivando o MSE linear com a chain rule (o `2` vem da potência, o `xᵢ` vem de `∂ŷ/∂w`):

- `∂L/∂w = (2/N) Σ (ŷᵢ − yᵢ) · xᵢ`
- `∂L/∂b = (2/N) Σ (ŷᵢ − yᵢ)`

Leitura intuitiva: o gradiente em relação a `w` é o erro médio ponderado pela entrada. Exemplos com `x` grande e erro grande empurram `w` com mais força.

> **No mundo real.** A **AlexNet** (Krizhevsky, Sutskever e Hinton, 2012), a rede que iniciou a era atual do Deep Learning ao vencer a competição ImageNet, tem cerca de **60 milhões de parâmetros**. A cada passo do treino, calcula-se o gradient: 60 milhões de números, um para cada parâmetro, dizendo para que lado ajustá-lo. Fazer isso milhões de vezes, sobre 1,2 milhão de imagens, só se tornou viável com GPUs; foi essa combinação que derrubou o erro da competição de cerca de 26% para cerca de 15% em um ano.

### 3. Gradient descent

> **Em palavras simples.** **Gradient descent** é descer uma montanha no nevoeiro: você não enxerga o vale, mas sente a inclinação sob os pés e dá um passo ladeira abaixo, repetidas vezes. A **learning rate** é o tamanho do passo. Passos minúsculos chegam ao vale, mas demoram uma eternidade; passos enormes fazem você pular por cima do vale e subir do outro lado, cada vez mais longe. É o ajuste mais importante de qualquer treino.

`θ ← θ − η · ∇L(θ)`

`η` (eta) é a **learning rate (LR)**, o hiperparâmetro mais importante de qualquer treino.

| LR | Efeito na loss curve |
|---|---|
| Muito pequena | Cai, mas devagar demais; parece estagnada |
| Adequada | Queda suave e rápida |
| Grande | Oscila em volta do mínimo |
| Muito grande | Diverge: a loss cresce até `inf` ou `nan` |

Para uma loss quadrática existe um limite exato: o treino só converge se `η < 2 / λ_max`, onde `λ_max` é a maior curvatura da loss surface. No notebook, com `x` entre −3 e 3, esse limite fica em torno de 0,33: por isso `lr=0.3` oscila e `lr=0.36` diverge.

**Ordens de grandeza para guardar:** treino do zero com Adam, `1e-3`; full fine-tuning de um Transformer, `1e-5` a `5e-5`; LoRA, `1e-4` a `3e-4`.

> **No mundo real.** No treino da **ResNet** no ImageNet (He et al., 2015), a receita publicada foi: learning rate inicial de 0,1, **dividida por 10 cada vez que o erro de validação parava de cair**. No gráfico de treino do paper, cada divisão aparece como um degrau em que o erro despenca de repente. Passos grandes no início para avançar rápido, passos pequenos no fim para se acomodar no fundo do vale. Essa ideia, o **learning rate schedule**, é usada até hoje em praticamente todo treino.

### 4. Batch, mini-batch e stochastic gradient descent

> **Em palavras simples.** Calcular o gradient usando todos os dados a cada passo seria como consultar todos os eleitores do país antes de cada pequena decisão. O **mini-batch** é a pesquisa por amostragem: sorteia-se um grupo pequeno de exemplos (o **batch**), calcula-se a direção com ele e dá-se o passo. A direção é um pouco imprecisa, mas dá para dar milhares de passos no tempo de um. Uma **epoch** é uma passada completa por todos os dados; um **step** é um único passo.

| Variante | Dados por step | Característica |
|---|---|---|
| Batch GD | Todos | Gradiente exato; inviável em datasets grandes |
| SGD | 1 exemplo | Barato e muito ruidoso |
| **Mini-batch SGD** | Um batch (8 a 512) | O padrão; ruído moderado, paralelizável em GPU |

Vocabulário: um **step** (ou iteration) é uma atualização de pesos; uma **epoch** é uma passagem completa pelo dataset. Com 10.000 exemplos e batch size 32, uma epoch tem 313 steps.

O ruído do mini-batch não é só um custo: ele ajuda a escapar de regiões ruins e funciona como regularização. Batch size e learning rate estão ligados: batch maior produz gradiente menos ruidoso e costuma permitir (e pedir) LR maior.

> **No mundo real.** O conjunto de treino do **ImageNet** tem 1.281.167 imagens. A ResNet foi treinada com **batch size de 256**: cada epoch tem, portanto, cerca de 5.000 steps (1.281.167 ÷ 256), e o treino completo usou por volta de 600 mil steps. Se cada step exigisse olhar as 1,28 milhão de imagens, o mesmo treino levaria milhares de vezes mais tempo. Todo LLM atual é treinado assim, com batches de sequências de texto.

### 5. Optimizers: além do SGD

> **Em palavras simples.** Um **optimizer** é a regra que decide como dar o passo a partir do gradient. O **SGD** simples dá o passo direto na direção indicada. O **momentum** acrescenta inércia, como uma bola que ganha velocidade na descida e não muda de rumo a cada pedrinha. O **Adam** vai além e ajusta o tamanho do passo separadamente para cada parâmetro, dando passos maiores onde o terreno é plano e menores onde é íngreme. O **AdamW** é o Adam com uma correção e é o padrão em modelos de linguagem.

- **Momentum:** acumula uma média móvel dos gradientes, como uma bola que ganha inércia. Atravessa vales estreitos com menos oscilação.
- **Adam:** mantém, por parâmetro, médias móveis do gradiente e do gradiente ao quadrado, e normaliza o step por elas. Cada parâmetro ganha uma LR efetiva própria. É pouco sensível à escala do gradiente, o que o torna o padrão.
- **AdamW:** Adam com weight decay aplicado corretamente (desacoplado do gradiente). É o optimizer de praticamente todo Transformer.

Dois complementos que você usará a partir da Semana 10: **learning rate schedule** (warmup seguido de decay) e **gradient clipping** (limitar a norma do gradiente para evitar um step catastrófico).

> **No mundo real.** O **BERT** (Devlin et al., Google, 2018) foi pré-treinado com Adam, learning rate de 1e-4, weight decay de 0,01 e **warmup** nos primeiros 10.000 steps, por 1 milhão de steps no total. Esses valores viraram ponto de partida para toda uma geração de modelos. O próprio paper do Adam (Kingma e Ba, 2014) sugere os valores padrão que você encontra em qualquer biblioteca: learning rate de 0,001, `beta1 = 0.9` e `beta2 = 0.999`.

### 6. Chain rule e backpropagation

> **Em palavras simples.** Uma rede neural é uma linha de montagem: a entrada passa por várias etapas até virar a resposta. Quando a resposta sai errada, é preciso saber quanto cada etapa contribuiu para o erro. A **chain rule** é a regra matemática que permite repassar a "culpa" de trás para frente, etapa por etapa. **Backpropagation** é esse repasse feito de forma organizada: uma passada para frente calcula a resposta, uma passada para trás distribui a responsabilidade pelo erro a cada parâmetro.

Se `L` depende de `ŷ`, e `ŷ` depende de `w`, então `dL/dw = dL/dŷ · dŷ/dw`. Uma rede neural é uma composição de funções, `L(f₃(f₂(f₁(x))))`, e o gradiente em relação a um peso do começo é o produto das derivadas locais de todas as funções no caminho até a loss.

**Backpropagation** é a chain rule aplicada de forma eficiente: um **forward pass** calcula e guarda os valores intermediários; um **backward pass** percorre o **computational graph** da saída para a entrada, e cada nó multiplica o gradiente que recebe (upstream gradient) pela sua derivada local. O custo do backward é da mesma ordem do forward, independentemente do número de parâmetros. É isso que torna viável treinar bilhões deles.

No Passo 9 do notebook você faz esse percurso à mão para `ŷ = w₂ · tanh(w₁x + b₁) + b₂`.

> **No mundo real.** Em 1989, Yann LeCun e colegas, nos Bell Labs, treinaram com backpropagation uma rede para ler **CEPs manuscritos** em envelopes reais do Serviço Postal dos EUA (*Backpropagation Applied to Handwritten Zip Code Recognition*). Foi uma das primeiras aplicações práticas do método, proposto em sua forma moderna por Rumelhart, Hinton e Williams em 1986. A linha de redes que saiu desse trabalho chegou a ler uma parcela relevante dos cheques depositados em bancos americanos nos anos 1990.

### 7. Vanishing e exploding gradients

> **Em palavras simples.** No repasse de trás para frente, a "culpa" é multiplicada a cada etapa. Se cada multiplicação reduz o valor um pouco, depois de muitas etapas ele chega a zero e as primeiras camadas param de aprender: é o **vanishing gradient**. Se cada multiplicação aumenta, o valor explode e o treino quebra: é o **exploding gradient**. É o mesmo efeito de juros compostos, para menos ou para mais. Durante anos isso impediu o treino de redes profundas.

Como o gradiente é um **produto** de muitos termos, em redes profundas ele pode encolher para zero (vanishing: as primeiras camadas param de aprender) ou crescer sem limite (exploding: a loss vira `nan`). `tanh` e `sigmoid` saturam e têm derivada próxima de zero nas pontas, o que agrava o vanishing. As soluções que tornaram o Deep Learning moderno possível: `ReLU` e variantes, **residual connections**, normalization layers e inicialização cuidadosa. Você as encontrará montadas no bloco Transformer (Semana 8).

> **No mundo real.** O paper da **ResNet** (He et al., Microsoft Research, 2015) mostrou o problema com dados: no dataset CIFAR-10, uma rede comum de 56 camadas tinha erro **maior** que uma de 20, inclusive no treino. A solução foi a **residual connection**, um atalho que deixa o sinal pular camadas e chegar intacto ao início da rede. Com isso, treinaram redes de 152 camadas e venceram a competição ImageNet de 2015 com 3,57% de erro top-5. Todo Transformer usa o mesmo atalho.

### 8. Convexidade

> **Em palavras simples.** Uma função **convex** tem o formato de uma tigela: há um único ponto mais baixo e, descendo de qualquer lugar, chega-se a ele. A loss de uma rede neural é uma cordilheira, com muitos vales diferentes: dependendo de onde se começa (os pesos iniciais, sorteados pela **seed**), termina-se em um vale diferente. Na prática a maioria dos vales é boa, mas isso explica por que treinar duas vezes o mesmo modelo dá resultados ligeiramente diferentes.

A loss da regressão linear é **convex**: tem um único mínimo global, e gradient descent o encontra. A loss de uma rede neural é non-convex, cheia de local minima e saddle points. Na prática isso atrapalha menos do que se imaginava: em alta dimensão, a maioria dos mínimos locais tem loss parecida e o ruído do SGD ajuda a sair de saddle points. Mas explica por que seeds diferentes dão resultados ligeiramente diferentes.

> **No mundo real.** O estudo *Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping* (Dodge et al., 2020) fez o fine-tuning do BERT centenas de vezes em tarefas do benchmark GLUE, mudando **apenas a seed**. Nos datasets pequenos, a diferença entre a melhor e a pior execução foi de vários pontos de accuracy, o suficiente para separar um resultado medíocre de um que parecia estado da arte. A conclusão prática: reporte a média de várias seeds, não a melhor execução.

### 9. Por que feature scaling importa (ligação com a Semana 1)

> **Em palavras simples.** Se uma coluna dos dados tem valores na casa dos milhões e outra na casa dos décimos, o terreno que o gradient descent precisa descer deixa de ser uma tigela redonda e vira um cânion estreito e comprido. O passo que serve para uma direção é grande demais para a outra, e o treino fica ziguezagueando de uma parede à outra. Padronizar os dados arredonda o terreno e deixa a descida direta.

Se uma feature vale milhões e outra décimos, a loss surface vira um vale estreito: curvatura enorme em uma direção, mínima na outra. A LR precisa ser pequena por causa da primeira e fica lenta demais para a segunda. Padronizar arredonda o vale. O número que mede esse desbalanço é o **condition number** da matriz de curvatura.

> **No mundo real.** O paper de **Batch Normalization** (Ioffe e Szegedy, Google, 2015) levou a ideia para dentro da rede: em vez de padronizar só os dados de entrada, padroniza-se o sinal entre as camadas, a cada batch. Aplicada a um modelo de classificação de imagens no ImageNet, a técnica atingiu a mesma accuracy com **14 vezes menos steps de treino**. As variantes usadas em LLMs (`LayerNorm`, `RMSNorm`) descendem diretamente dessa ideia.

### 10. Gradient checking

> **Em palavras simples.** Como saber se a fórmula do gradient que você programou está certa? **Gradient checking** é a prova real: muda-se um parâmetro um tiquinho para cima e para baixo, mede-se quanto o erro mudou e compara-se com o que a fórmula previa. Se os dois números batem, a fórmula está certa. É lento demais para treinar, mas é o melhor teste que existe para esse tipo de código.

Uma derivada pode ser aproximada por finite differences: `(L(θ + ε) − L(θ − ε)) / 2ε`. Comparar o gradiente analítico com o numérico é a forma clássica de testar uma implementação de backprop. É lento demais para treinar, mas é o melhor teste unitário que existe para código de gradiente.

> **No mundo real.** O **PyTorch** usa exatamente esse método para se testar. A função `torch.autograd.gradcheck` compara, por diferenças finitas, o gradient calculado pelo autograd com o gradient numérico, e a suíte de testes da biblioteca a aplica às suas operações matemáticas. Quem escreve uma operação nova com backward próprio (`torch.autograd.Function`) é orientado pela documentação a validá-la com `gradcheck` antes de usar.

## Erros comuns

- Loss `nan`: quase sempre LR alta demais ou dados não normalizados. Reduza a LR em 10× antes de qualquer outra coisa.
- Loss parada desde o primeiro step: LR baixa demais, gradiente não chegando aos parâmetros, ou bug nos labels.
- Confundir epoch com step ao comparar experimentos com batch sizes diferentes.
- Esquecer de embaralhar os dados a cada epoch em mini-batch SGD.

## Leitura e vídeos (fim de semana)

- 3Blue1Brown, *Essence of Calculus*: Chapter 1 (*The essence of calculus*), 2 (*The paradox of the derivative*), 3 (*Derivative formulas through geometry*) e 4 (*Visualizing the chain rule and product rule*).
- 3Blue1Brown, *Neural Networks*: Chapter 2 (*Gradient descent, how neural networks learn*), 3 (*What is backpropagation really doing?*) e 4 (*Backpropagation calculus*).
- Andrej Karpathy, *The spelled-out intro to neural networks and backpropagation: building micrograd*. Vale as duas horas e meia.
- Sebastian Ruder, *An overview of gradient descent optimization algorithms* (seções sobre Momentum, Adam e AdamW).
- Opcional: Stanford CS231n notes, *Optimization* e *Backpropagation, Intuitions*.

## Perguntas de autoavaliação

1. O gradient em relação a `w` é positivo. Você aumenta ou diminui `w`? Por quê?
2. A loss virou `nan` na epoch 3. Qual a primeira coisa a testar?
3. Qual a diferença entre uma epoch e um step? Quantos steps tem uma epoch com 50.000 exemplos e batch size 64?
4. Explique a chain rule usando `L = (w·x + b − y)²`.
5. Por que o backward pass custa aproximadamente o mesmo que o forward pass?
6. O que o Adam faz de diferente do SGD e por que isso o torna menos sensível à escala das features?

## Entregável da semana

Notebook resolvido e um gráfico de loss curves para pelo menos três learning rates, com uma frase explicando cada curva.

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/ai-architect-foundations/blob/main/01-Phase-1-Math-and-ML-Core/03-calculus-gradient/notebooks/semana_03_exercicio.ipynb) · [`semana_03_exercicio.ipynb`](../notebooks/semana_03_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/ai-architect-foundations/blob/main/01-Phase-1-Math-and-ML-Core/03-calculus-gradient/notebooks/semana_03_gabarito.ipynb) · [`semana_03_gabarito.ipynb`](../notebooks/semana_03_gabarito.ipynb) |

[← Semana 2](/01-Phase-1-Math-and-ML-Core/02-linear-algebra-numpy/docs/Semana-02-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 4 →](/01-Phase-1-Math-and-ML-Core/04-pytorch-tensors/docs/Semana-04-Conceitos.md)
