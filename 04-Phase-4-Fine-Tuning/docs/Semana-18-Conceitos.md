# Semana 18 — LoRA do Zero: a Matemática do Low Rank

**Fase 4: Parameter-Efficient Fine-Tuning (PEFT)** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Entender LoRA (Low-Rank Adaptation) pela matemática e pelo código: por que uma atualização de pesos pode ser representada por duas matrizes pequenas, como isso reduz os trainable parameters em ordens de grandeza e por que o adapter pode ser fundido ao modelo sem custo de inference.

## Por que isso importa para um AI Architect

LoRA é o que torna o fine-tuning de LLMs viável em uma única GPU e o que permite manter dezenas de especializações de um mesmo base model como arquivos de poucos megabytes. As decisões de `r`, `lora_alpha` e `target_modules` são suas; a biblioteca não as toma por você. E a arquitetura "um base model, muitos adapters" muda como se desenha serving multi-tenant.

## Conceitos

### 1. O problema do full fine-tuning

> **Em palavras simples.** No **full fine-tuning**, todos os bilhões de parâmetros do modelo são ajustados. Isso exige muita memória de GPU e, pior, produz uma cópia completa do modelo para cada tarefa ou cliente. Com dez clientes, são dez modelos inteiros para guardar e servir. O custo de armazenamento e de operação cresce depressa.

Ajustar todos os pesos de um modelo de 7B exige memória para pesos, gradients e optimizer states (o Adam guarda dois valores extras por parâmetro): várias vezes o tamanho do modelo. Produz uma cópia completa de dezenas de gigabytes por tarefa. E mexer em tudo favorece catastrophic forgetting (Semana 10).

**PEFT (Parameter-Efficient Fine-Tuning)** é a família de técnicas que congela o base model e treina um número pequeno de parâmetros novos. LoRA é a mais usada.

> **No mundo real.** O paper do **LoRA** (Hu et al., Microsoft, 2021) dá a dimensão do problema com o GPT-3 de 175 bilhões de parâmetros: cada versão ajustada por full fine-tuning gerava um checkpoint de cerca de **350 GB**. Com LoRA, o que se guarda por tarefa cai para cerca de **35 MB**, dez mil vezes menos. Para quem oferece modelos personalizados a muitos clientes, é a diferença entre um serviço inviável e um viável.

### 2. Rank de uma matriz

> **Em palavras simples.** O **rank** de uma tabela de números diz quanta informação independente ela realmente contém. Uma tabela enorme pode ter rank baixo se as suas linhas forem combinações de poucos padrões básicos. Nesse caso, ela pode ser reconstruída a partir de duas tabelas bem menores, o que economiza muito espaço.

O **rank** é o número de direções independentes que uma matriz contém. Uma matriz `d × k` de rank `r` pode ser escrita como o produto de uma `d × r` por uma `r × k`. Se `r` é pequeno, isso é muito mais compacto: `r(d + k)` números em vez de `d · k`.

A **SVD (Singular Value Decomposition)**, `W = U Σ Vᵀ`, ordena as direções por importância: os singular values em `Σ` dizem quanta "energia" há em cada uma. Manter só os `r` maiores dá a melhor aproximação de rank `r` possível (teorema de Eckart–Young). Muitas matrizes do mundo real têm poucos singular values grandes: são **aproximadamente low-rank**.

> **No mundo real.** O trabalho clássico **Eigenfaces** (Turk e Pentland, MIT, 1991) mostrou isso com fotos de rostos: cada imagem tinha dezenas de milhares de pixels, mas podia ser descrita com boa fidelidade pela combinação de poucas dezenas de rostos-base. A variação real entre rostos ocupava um espaço de dimensão muito menor do que o número de pixels. É a mesma ideia de baixa dimensão efetiva que o LoRA explora nos ajustes de um modelo.

### 3. A hipótese do LoRA

> **Em palavras simples.** A hipótese do **LoRA** é que a mudança necessária para adaptar um modelo a uma tarefa é simples: tem rank baixo. Então, em vez de alterar a tabela original de pesos, que fica congelada, treinam-se duas tabelas pequenas cujo produto representa o ajuste. O modelo original fica intacto, e o ajuste é um acessório leve colocado ao lado.

Os pesos de um modelo pré-treinado **não** são low-rank. A hipótese é sobre a **mudança**: o `ΔW` necessário para adaptar o modelo a uma tarefa nova tem **low intrinsic rank**. Então, em vez de aprender `ΔW` inteiro:

`W' = W + ΔW = W + (α / r) · B · A`

- `W` (`d × k`): pesos originais, **frozen**.
- `A` (`r × k`): inicializada com valores aleatórios pequenos (Kaiming).
- `B` (`d × r`): inicializada com **zeros**, para que `ΔW = 0` no início e o modelo comece idêntico ao original. Se as duas começassem em zero, o gradiente de ambas seria zero e nada treinaria.
- `α / r`: scaling factor.

No forward: `h = W x + (α / r) · B (A x)`. Só `A` e `B` recebem gradiente.

> **No mundo real.** A base empírica é o paper *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning* (Aghajanyan et al., Facebook, 2020). Os autores mostraram que um modelo RoBERTa atingia 90% do desempenho do fine-tuning completo na tarefa MRPC treinando apenas **200 parâmetros**, projetados aleatoriamente sobre os milhões do modelo. Se tão poucos graus de liberdade bastam, o ajuste não precisa mexer em tudo.

### 4. A economia

> **Em palavras simples.** A economia do LoRA aparece em três lugares. No treino, só as tabelas pequenas são ajustadas, o que reduz a memória necessária. No armazenamento, guarda-se um arquivo de megabytes em vez de gigabytes. Na operação, um único modelo base pode servir vários ajustes diferentes, trocando só o acessório.

Para uma layer `4096 × 4096`: full fine-tuning treina 16,8 milhões de parâmetros; LoRA com `r = 8` treina `8 × (4096 + 4096) = 65.536`, ou 0,4%. No modelo inteiro, é comum treinar menos de 1% dos parâmetros.

Ganhos concretos:

- **Memória de treino:** só os adapters têm gradients e optimizer states. (O forward ainda passa pelo modelo inteiro, então as activations continuam ocupando memória.)
- **Storage:** adapters de megabytes em vez de checkpoints de gigabytes.
- **Menos forgetting:** os pesos originais ficam intactos; desligar o adapter devolve o base model.
- **Multi-task serving:** um base model em memória, com adapters trocados por requisição.

> **No mundo real.** No ecossistema do **Stable Diffusion**, isso é visível para qualquer usuário: um checkpoint completo do modelo ocupa vários gigabytes, enquanto um LoRA que ensina um estilo ou um personagem costuma ter de poucas dezenas a pouco mais de cem megabytes. Sites como o Civitai hospedam dezenas de milhares desses arquivos, e os usuários combinam vários sobre o mesmo modelo base, algo impraticável com modelos completos.

### 5. Hyperparameters

> **Em palavras simples.** Os principais ajustes do LoRA são três. O **rank** (`r`) define o tamanho das tabelas pequenas, ou seja, a capacidade do ajuste. O **alpha** regula a força com que o ajuste é aplicado. Os **target modules** definem em quais partes do modelo o ajuste é colocado. Aumentar tudo nem sempre melhora: um ajuste grande demais para poucos dados decora.

| Hyperparameter | Papel | Ponto de partida |
|---|---|---|
| `r` | Capacidade do adapter | 8 a 16; suba se a tarefa for complexa ou distante do pre-training |
| `lora_alpha` | Escala da atualização | `2 × r` é convenção comum |
| `target_modules` | Em quais matrizes inserir | `q_proj`, `v_proj` (mínimo); incluir `k_proj`, `o_proj` e as do MLP (`gate_proj`, `up_proj`, `down_proj`) aumenta a capacidade |
| `lora_dropout` | Regularização | 0,05 |
| Learning rate | Maior que no full fine-tuning | 1e-4 a 3e-4 |

Sobre `alpha`: o que importa é a razão `α / r`. Manter `alpha` fixo e aumentar `r` reduz a escala efetiva; por isso a convenção de escalar os dois juntos. Na prática, `alpha` funciona como um multiplicador da learning rate do adapter.

Sobre `r`: há retorno decrescente. Dobrar `r` dobra os parâmetros e raramente dobra o ganho; `r` alto demais em dataset pequeno favorece overfitting. O notebook traça essa curva.

Sobre `target_modules`: o paper original mostrou que adaptar mais matrizes com `r` menor costuma ser melhor que adaptar poucas com `r` grande. Os nomes dependem da arquitetura: inspecione `model.named_modules()`.

> **No mundo real.** O paper do LoRA testou esses ajustes no GPT-3. Dois resultados orientam a prática até hoje: ranks muito baixos, de 1 a 8, já davam resultados competitivos nas tarefas avaliadas; e, com o mesmo número de parâmetros, era melhor aplicar um rank pequeno a **duas** matrizes da attention (query e value) do que um rank grande a uma só. Espalhar o ajuste vale mais que concentrá-lo.

### 6. Merging

> **Em palavras simples.** **Merging** é somar o ajuste do LoRA de volta aos pesos originais, produzindo um modelo único. Depois disso, o modelo roda exatamente na velocidade do original, sem peça adicional. A desvantagem é perder a flexibilidade de trocar de ajuste; por isso há quem mantenha os adaptadores separados e os carregue sob demanda.

Depois do treino, calcula-se `W' = W + (α / r) B A` uma única vez e descarta-se o adapter. O modelo fundido tem exatamente a mesma arquitetura e latência do original: **zero inference overhead**. No PEFT: `merge_and_unload()`.

Alternativa: manter o adapter separado e carregá-lo em runtime, o que permite trocar de especialização sem recarregar o base model. Servidores como o vLLM suportam vários LoRA adapters sobre o mesmo modelo.

> **No mundo real.** Essa propriedade é um dos argumentos centrais do paper do LoRA contra a técnica anterior, os **adapter layers**: como estes acrescentavam camadas ao modelo, a latência de inference aumentava, e o paper mediu acréscimos de até 20% ou 30% em cenários de batch pequeno. O LoRA, uma vez somado aos pesos, tem latência adicional igual a zero, o que foi decisivo para a sua adoção em produção.

### 7. Limitações

> **Em palavras simples.** O LoRA não é equivalente ao full fine-tuning em todos os casos. Quando a tarefa exige aprender muito conteúdo novo, distante do que o modelo já sabe, o ajuste de rank baixo pode não ter capacidade suficiente. Em compensação, por mexer menos no modelo, ele preserva melhor o que o modelo já sabia.

- Tarefas muito distantes do pre-training (uma língua nova, um domínio com vocabulário próprio) podem precisar de `r` alto ou de full fine-tuning.
- O adapter só serve para o **exato** base model em que foi treinado.
- Não reduz o custo do forward pass no treino, só o do backward e do optimizer.
- A interação entre vários adapters combinados não é garantida.

> **No mundo real.** O paper *LoRA Learns Less and Forgets Less* (Biderman et al., 2024) comparou os dois métodos em programação e matemática, com conjuntos de dados grandes. Com os ranks usuais, o LoRA ficou consistentemente abaixo do full fine-tuning no domínio novo. Por outro lado, manteve melhor o desempenho do modelo em tarefas fora desse domínio. O título resume o trade-off: aprende menos e esquece menos.

### 8. Variantes que você vai encontrar

> **Em palavras simples.** Depois do LoRA surgiram variantes que ajustam a receita. **QLoRA** aplica LoRA sobre um modelo comprimido para caber em GPUs menores. **DoRA** separa a direção e a intensidade do ajuste. **rsLoRA** corrige a escala para ranks altos. **AdaLoRA** distribui o rank de forma desigual entre as camadas. Todas mantêm a ideia de congelar o modelo e treinar um acessório pequeno.

- **QLoRA:** LoRA sobre um base model quantizado em 4-bit (Semana 20).
- **DoRA (Weight-Decomposed LoRA):** separa magnitude e direção dos pesos; costuma ganhar um pouco de qualidade com o mesmo `r`.
- **rsLoRA:** escala por `α / √r`, mais estável com `r` alto.
- **AdaLoRA:** distribui o rank entre as layers de forma adaptativa.
- **LoRA+:** learning rates diferentes para `A` e `B`.
- Outras técnicas PEFT, hoje menos usadas: **adapters** (layers pequenas inseridas em série, com latência extra), **prefix tuning** e **prompt tuning** (vetores treináveis no input), **IA³** (vetores de escala).

> **No mundo real.** O **DoRA** (Liu et al., NVIDIA, 2024) reportou ganhos consistentes sobre o LoRA com o mesmo número de parâmetros, em tarefas de linguagem e de visão, e foi aceito com destaque no ICML 2024. A biblioteca `peft`, da Hugging Face, incorporou-o como uma opção de uma linha (`use_dora=True`), assim como o rsLoRA. Na prática, experimentar uma variante custa trocar um parâmetro de configuração.

### 9. O que o notebook demonstra

Com um MLP pré-treinado em dígitos e uma tarefa nova (dígitos rotacionados), você verá: LoRA com cerca de 5% dos parâmetros recupera a maior parte da accuracy do full fine-tuning; a curva accuracy × rank achata; o modelo fundido produz saídas idênticas; e os singular values do `ΔW` do full fine-tuning decaem rápido, que é a evidência empírica da hipótese de low rank.

## Erros comuns

- Inicializar `B` com valores não nulos (o modelo já começa alterado) ou `A` e `B` com zeros (nada treina).
- Mudar `r` sem ajustar `lora_alpha`.
- Esquecer de congelar o base model: vira full fine-tuning sem querer.
- `target_modules` com nomes de outra arquitetura (o PEFT avisa, mas leia a mensagem).
- Aplicar um adapter em um base model diferente (outra versão ou outra quantization) e esperar o mesmo resultado.
- Learning rate de full fine-tuning (`2e-5`) em LoRA: aprende devagar demais.

## Leitura e vídeos (fim de semana)

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021): Sections 1, 2, 4 e 7 (a análise de quais matrizes adaptar e de qual rank).
- Aghajanyan et al., *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning* (abstract), a base teórica da hipótese.
- 3Blue1Brown, *Essence of Linear Algebra*, Chapter 7 (*Inverse matrices, column space and null space*), para a intuição de rank.
- Hugging Face PEFT documentation: conceptual guide *LoRA* e *Adapters*.
- Sebastian Raschka, *Practical Tips for Finetuning LLMs Using LoRA* e *Improving LoRA: Implementing Weight-Decomposed Low-Rank Adaptation (DoRA) from Scratch*.
- Opcional: Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation* (abstract e figura 1).

## Perguntas de autoavaliação

1. Por que `B` é inicializada com zeros e `A` não?
2. Quantos trainable parameters tem um LoRA de `r = 16` sobre uma layer `2048 × 2048`?
3. O que o merging faz e por que elimina a latência extra?
4. Em que situação você aumentaria `r`? E em que situação preferiria full fine-tuning?
5. Qual o papel de `lora_alpha` e por que costuma acompanhar `r`?
6. LoRA reduz a memória do forward pass no treino? Explique.

## Entregável da semana

Notebook resolvido e o gráfico accuracy × rank, com uma frase sobre o ponto de retorno decrescente.

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/ai-architect-foundations/blob/main/04-Phase-4-Fine-Tuning/notebooks/semana_18_exercicio.ipynb) · [`semana_18_exercicio.ipynb`](../notebooks/semana_18_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/ai-architect-foundations/blob/main/04-Phase-4-Fine-Tuning/notebooks/semana_18_gabarito.ipynb) · [`semana_18_gabarito.ipynb`](../notebooks/semana_18_gabarito.ipynb) |

[← Semana 17](/04-Phase-4-Fine-Tuning/docs/Semana-17-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 19 →](/04-Phase-4-Fine-Tuning/docs/Semana-19-Conceitos.md)
