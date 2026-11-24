# Semana 30 — Synthetic Control

**Fase 6: Causal Inference e Experimentation** · Leitura de conceito (fim de semana, sem código) · termos técnicos mantidos em inglês

## Objetivo

Medir o efeito de uma intervenção quando existe **uma única** unidade tratada (um país, uma cidade, um produto) e nenhum grupo de controle natural. O Synthetic Control Method constrói esse controle como uma combinação ponderada de unidades não tratadas e fornece inferência por placebo tests.

## Por que isso importa para um AI Architect

Lançamentos por mercado, mudanças de preço em um país, uma feature ativada para um único cliente grande: são decisões com `N = 1`. Não há A/B possível, nem DiD confiável com um só controle. Synthetic control é o método padrão para esses casos em economia e, cada vez mais, em tecnologia (geo experiments, lançamentos regionais, medição de marketing).

## Conceitos

### 1. A ideia

> **Em palavras simples.** Quando o evento atinge uma única unidade, como um país, um estado ou uma loja, não existe um grupo de controle natural. O **synthetic control** fabrica um: combina várias unidades não afetadas, com pesos, de modo que a mistura reproduza a trajetória da unidade afetada antes do evento. Depois do evento, a diferença entre a unidade real e a sua versão sintética é o efeito estimado.

Não existe "outro país igual à Espanha". Mas talvez `0,5 × Itália + 0,4 × Portugal + 0,1 × Brasil` tenha acompanhado a Espanha de perto durante todo o período anterior ao lançamento. Se essa combinação reproduz bem o passado, ela é uma estimativa razoável do que a Espanha teria feito **sem** a intervenção: o counterfactual.

`efeito(t) = Y_treated(t) − Σⱼ wⱼ · Y_donorⱼ(t)`, para `t` no post-period.

O conjunto de unidades não tratadas disponíveis é o **donor pool**.

> **No mundo real.** O método nasceu no estudo de Abadie e Gardeazabal (2003) sobre o custo econômico do terrorismo no País Basco. Como nenhuma região espanhola era comparável, eles construíram um País Basco sintético, combinando principalmente a Catalunha e Madri, que acompanhava o original até o fim dos anos 1960. Depois da escalada da violência, o PIB per capita real ficou cerca de 10% abaixo do sintético.

### 2. Os pesos

> **Em palavras simples.** Os **pesos** dizem quanto cada unidade de comparação entra na mistura. São escolhidos por um algoritmo para que a mistura se pareça o máximo possível com a unidade tratada antes do evento. Não podem ser negativos e somam 100%. Como são explícitos, qualquer pessoa pode ver de que é feito o controle e discordar da receita.

Escolhem-se os pesos `w` que minimizam a distância entre a unidade tratada e a combinação dos donors **no pre-period**, com duas restrições:

- `wⱼ ≥ 0` (sem pesos negativos);
- `Σ wⱼ = 1`.

As restrições fazem do sintético uma **convex combination** de unidades reais: uma interpolação, nunca uma extrapolação. Sem elas, o problema vira uma regressão comum, que pode usar pesos negativos e extrapolar para fora do que os dados sustentam.

Efeito colateral útil: a solução tende a ser **sparse** (poucos donors com peso positivo), o que torna o resultado interpretável e auditável ("a Espanha sintética é metade Itália").

No método original, a otimização casa também **covariates** (preditores do outcome), com uma matriz `V` que pondera a importância de cada preditor. Na prática, casar os outcomes do pre-period costuma ser suficiente, e é o que o notebook faz com `scipy.optimize.minimize` (SLSQP).

> **No mundo real.** No estudo da **Proposition 99**, que aumentou o imposto sobre cigarros na Califórnia em 1988 (Abadie, Diamond e Hainmueller, 2010), a Califórnia sintética é composta por cinco estados: Utah, Nevada, Montana, Colorado e Connecticut. Os outros 33 estados candidatos receberam peso zero. No ano 2000, o consumo per capita na Califórnia real estava cerca de 26 maços abaixo do da sintética.

### 3. Por que funciona: o modelo de fatores

> **Em palavras simples.** A lógica do método é esta: se a mistura acompanhou a unidade tratada por muitos anos antes do evento, é provável que responda da mesma forma às mesmas forças ocultas, como ciclos econômicos, modas e tendências. Não é preciso identificar essas forças: basta que a mistura as tenha reproduzido por tempo suficiente. Quanto mais longo e fiel o período anterior, mais convincente o argumento.

Suponha que o outcome de cada unidade seja gerado por alguns fatores latentes comuns (tendência global, sazonalidade, ciclo econômico), com sensibilidades (loadings) diferentes por unidade. Se uma combinação de donors reproduz a unidade tratada por um pre-period longo, ela só pode estar fazendo isso por ter loadings parecidos, e então continuará acompanhando no post-period, inclusive diante de choques comuns.

Isso também explica uma observação do notebook: os pesos não são únicos. Com muitos donors correlacionados, várias combinações reproduzem o pre-period igualmente bem. O que se avalia é a qualidade do ajuste, não a identidade dos donors.

> **No mundo real.** No estudo sobre a reunificação alemã (Abadie, Diamond e Hainmueller, 2015), a Alemanha Ocidental sintética, formada principalmente por Áustria, Estados Unidos, Japão, Suíça e Holanda, reproduz o PIB per capita alemão por três décadas, de 1960 a 1990. Depois da reunificação, a série real passa a ficar abaixo da sintética, com uma diferença média de cerca de 1.600 dólares por pessoa ao ano até 2003.

### 4. Relação com o DiD

> **Em palavras simples.** O DiD supõe que os grupos teriam seguido em paralelo e dá o mesmo peso a todas as unidades de controle. O synthetic control escolhe os pesos para que a trajetória anterior coincida, sem supor paralelismo. Um é mais simples e exige a hipótese; o outro é mais flexível e exige um histórico longo. Métodos recentes combinam as duas ideias.

O DiD usa um controle (ou a média simples de vários) e supõe parallel trends. O synthetic control **escolhe** a combinação que melhor satisfaz essa hipótese no passado, e permite que os fatores não observados tenham efeitos que variam no tempo. É uma generalização guiada por dados, com a escolha do controle feita por um procedimento transparente em vez da intuição do analista.

> **No mundo real.** O paper *Synthetic Difference-in-Differences* (Arkhangelsky et al., 2021) propôs essa combinação e a testou reanalisando os mesmos dados da Proposition 99. O estimador usa pesos para as unidades, como o synthetic control, e também para os períodos, e permite uma diferença constante de nível, como o DiD. Nas simulações com dados reais, mostrou-se mais robusto que os dois métodos originais.

### 5. Pre-treatment fit

> **Em palavras simples.** O primeiro teste de um synthetic control é visual: antes do evento, a linha sintética precisa andar colada à linha real (**pre-treatment fit**). Se as duas já divergiam antes, a diferença posterior não significa nada. Um ajuste ruim indica que não existe combinação de unidades capaz de imitar a unidade tratada, e que o método não se aplica ao caso.

O **RMSPE** (root mean squared prediction error) no pre-period mede a qualidade do sintético. Se ele não reproduz o passado, não há por que confiar nele para o futuro. Regra: olhe o gráfico do pre-period antes de olhar o efeito.

Boa prática: reservar a parte final do pre-period como **validation** (ajustar os pesos em uma parte e verificar o ajuste na outra), para detectar overfitting.

> **No mundo real.** Abadie, em seu artigo de revisão de 2021 no *Journal of Economic Literature*, é explícito: o método não deve ser usado quando o ajuste anterior ao evento é ruim ou quando o período anterior é curto. No caso da Califórnia, a série sintética acompanha a real por quase vinte anos, de 1970 a 1988, e essa aderência visível é o principal argumento de credibilidade do estudo.

### 6. Inferência por placebo

> **Em palavras simples.** Com uma única unidade tratada, os testes estatísticos usuais não funcionam. A alternativa é o **placebo test**: aplica-se o mesmo método a cada unidade não tratada, fingindo que ela sofreu o evento. Se a unidade verdadeira mostra um efeito muito maior que os placebos, é improvável que seja coincidência. Se vários placebos mostram efeitos parecidos, o resultado não se sustenta.

Com uma unidade tratada, não há erro-padrão clássico. Usa-se um teste de permutação.

**In-space placebo:**

1. Aplique o mesmo procedimento a cada donor, fingindo que ele foi o tratado.
2. Para cada unidade, calcule a razão `RMSPE post / RMSPE pre`. A razão corrige o fato de algumas unidades terem ajuste ruim desde o início.
3. Se a razão da unidade realmente tratada for a maior (ou uma das maiores), o efeito é incomum. `p-value = posição no ranking / número de unidades`.

Com 15 donors, o menor p-value possível é `1/16 ≈ 0,063`. Poucos donors limitam a inferência: não é possível obter "p < 0,05" com menos de 20 unidades. É uma limitação honesta do desenho com `N = 1`.

**In-time placebo (backdating):** fingir que a intervenção ocorreu antes; o efeito estimado no intervalo falso deve ser nulo.

**Leave-one-out:** remover um donor de cada vez e verificar se o resultado depende de um único país.

> **No mundo real.** No estudo da Proposition 99, o procedimento foi repetido para os 38 estados de controle. Entre os que tinham bom ajuste antes de 1988, nenhum apresentou uma queda comparável à da Califórnia. A razão entre o desvio depois e o desvio antes do evento foi a maior justamente na Califórnia: a chance de isso ocorrer por acaso é de 1 em 39, cerca de 0,026.

### 7. Requisitos e cuidados

> **Em palavras simples.** O método tem requisitos. É preciso um histórico longo antes do evento; unidades de comparação que não foram afetadas, nem pelo evento nem por choques próprios; e que a unidade tratada não seja extrema a ponto de nenhuma mistura alcançá-la. Também é preciso evitar testar muitas receitas e escolher a que dá o resultado desejado.

- **Pre-period longo:** dezenas de períodos. Com pre-period curto e muitos donors, ajusta-se ruído.
- **Donor pool limpo:** unidades também afetadas pela intervenção (spillover) ou por choques próprios grandes no período não podem ser donors. O notebook mostra o viés: aproximadamente `peso × contaminação`.
- **Convex hull:** se a unidade tratada é maior (ou menor) que todos os donors, nenhuma média ponderada a reproduz. Trabalhar com a métrica normalizada (per capita, índice) costuma resolver.
- **Sem antecipação:** se os usuários reagiram antes da data oficial, mova a data de intervenção para trás.
- **Idiosyncratic shocks:** um evento que atinja só a unidade tratada no post-period se confunde com o efeito. Nenhum método resolve isso; registre o que mais aconteceu no período.
- **Não escolha os donors olhando o resultado.** Defina o donor pool e a especificação antes.

> **No mundo real.** Born, Müller, Schularick e Sedláček (2019) aplicaram o método ao referendo do Brexit, de 2016, construindo um Reino Unido sintético a partir de outras economias avançadas. Estimaram que, no fim de 2018, o PIB britânico estava entre 1,7% e 2,5% abaixo do que estaria sem o resultado do referendo. O caso mostra o método em uso para um evento nacional único, e os autores discutem o risco de os países doadores também terem sido afetados.

### 8. Extensões

> **Em palavras simples.** Há variações do método. O **augmented synthetic control** corrige o resultado quando o ajuste anterior não é perfeito. As versões **bayesianas** usam um modelo de séries temporais para prever a trajetória sem o evento e fornecem uma faixa de incerteza. Há também versões para várias unidades tratadas ao mesmo tempo.

- **Augmented Synthetic Control:** corrige o viés quando o ajuste no pre-period é imperfeito, com um modelo de outcome.
- **Synthetic DiD:** combina pesos de unidade (como o SC) e de tempo, com um efeito fixo de unidade (como o DiD). Mais robusto quando há várias unidades tratadas.
- **Generalized SC / matrix completion:** estimam o modelo de fatores diretamente.
- **Bayesian Structural Time Series (CausalImpact):** modela a série tratada em função das séries de controle e fornece intervalos de credibilidade. Popular em marketing e produto.
- **Geo experiments:** quando é possível **escolher** quais regiões tratar, combina-se o desenho experimental (sortear ou parear regiões) com a análise por synthetic control. Sempre que puder escolher, sorteie.

> **No mundo real.** O **Google** publicou o método **CausalImpact** (Brodersen et al., 2015) e o pacote aberto de mesmo nome. Ele usa um modelo bayesiano de séries temporais, alimentado por séries de controle, para estimar o que teria acontecido sem a intervenção, por exemplo o efeito de uma campanha publicitária em um país sobre os cliques. É uma das ferramentas mais usadas em empresas para medir efeitos sem A/B test.

### 9. Qual método quando: resumo da Fase 6

| Situação | Método | Hipótese-chave (não testável) |
|---|---|---|
| Você controla a atribuição, usuários independentes | A/B test | SUTVA |
| Você controla a atribuição, usuários interagem | Cluster-randomized / switchback | Sem interference entre clusters |
| Tratamento por grupo/região, com controle comparável e painel | DiD | Parallel trends |
| Adoção voluntária, confounders medidos | PSM / IPW / doubly robust | Unconfoundedness |
| Adoção voluntária, confounder oculto, existe instrumento | IV / encouragement design | Exclusion restriction |
| Uma única unidade tratada, série longa, vários donors | Synthetic control | O sintético continuaria acompanhando |

A ordem de preferência é a da tabela: quanto mais abaixo, mais a conclusão depende de hipóteses que os dados não podem confirmar. Experimento sempre que possível.

## Erros comuns

- Olhar o efeito sem verificar o pre-treatment fit.
- Pre-period curto com muitos donors.
- Incluir donors contaminados pela intervenção.
- Escolher o donor pool depois de ver o resultado.
- Reportar um efeito sem placebo tests.
- Esperar p < 0,05 com dez donors.
- Usar a métrica em nível quando a unidade tratada está fora do envelope dos donors.

## Leitura e vídeos (fim de semana)

- Matheus Facure, *Causal Inference for the Brave and True*: Chapters *Synthetic Control* e *Synthetic Difference-in-Differences*.
- Abadie, Diamond & Hainmueller, *Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California's Tobacco Control Program* (Introduction, Sections 2 e 3).
- Abadie, *Using Synthetic Controls: Feasibility, Data Requirements, and Methodological Aspects* (o guia prático; leia inteiro).
- Brodersen et al., *Inferring causal impact using Bayesian structural time-series models* (CausalImpact; abstract e introdução).
- Arkhangelsky et al., *Synthetic Difference-in-Differences* (abstract e figura 1).
- Scott Cunningham, *Causal Inference: The Mixtape*, Chapter *Synthetic Control*.
- Bibliotecas: `pysyncon`, `SyntheticControlMethods`, `causalimpact`, `CausalPy`.

## Perguntas de autoavaliação

1. Por que os pesos são restritos a não negativos e com soma 1?
2. O que o RMSPE do pre-period informa?
3. Como se obtém um p-value com uma única unidade tratada? Qual o menor valor possível com 12 donors?
4. Por que um país vizinho que recebeu parte da campanha não pode ser donor?
5. Em que o synthetic control generaliza o DiD?
6. Por que os pesos encontrados podem diferir dos "verdadeiros" sem que o resultado esteja errado?

## Entregável da semana

Notebook resolvido e um relatório curto em `docs/`: gráfico treated × synthetic, pesos, efeito estimado, placebo p-value e as hipóteses.

### Checkpoint da Fase 6

Você deve conseguir, sem consultar: explicar por que a randomization identifica causalidade; dizer quando usar A/B por cluster, DiD, PSM, IV e synthetic control; nomear a hipótese não testável de cada método; descrever como o spillover enviesa um teste.

---

## Prática da semana (segunda a sexta, Colab)

| Notebook | Abrir |
|---|---|
| Exercícios | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/06-Phase-6-Causal-Inference/notebooks/semana_30_exercicio.ipynb) · [`semana_30_exercicio.ipynb`](../notebooks/semana_30_exercicio.ipynb) |
| Gabarito | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/06-Phase-6-Causal-Inference/notebooks/semana_30_gabarito.ipynb) · [`semana_30_gabarito.ipynb`](../notebooks/semana_30_gabarito.ipynb) |

[← Semana 29](/06-Phase-6-Causal-Inference/docs/Semana-29-Conceitos.md) · [Plano de Estudo](/PLANO_DE_ESTUDO.md) · [Semana 31 →](/07-Phase-7-Governance-and-Scale/docs/Semana-31-Conceitos.md)
