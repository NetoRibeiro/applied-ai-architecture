# Como usar este livro {#apresentacao .unnumbered}

Este livro reúne as leituras de conceito da **Fase 1 — Math Foundations e ML Core**, a primeira de sete fases da trilha *Enterprise AI Architect — Foundations*: 32 semanas para sair de Data Science e chegar a AI Architecture, da matemática por trás dos LLMs até RAG, fine-tuning, MLOps, causal inference e governança.

A Fase 1 recalibra o ML clássico e constrói a base matemática (vectors, gradients, tensors) sobre a qual todo o resto se apoia. As fases seguintes assumem dot product, gradient descent e o training loop na ponta dos dedos; por isso, mesmo que pareça básica, não pule esta parte.

O texto é em português, e os termos técnicos, nomes de funções e de bibliotecas ficam em inglês, como aparecem na documentação, nos papers e no dia a dia de trabalho.

## As quatro semanas

| Semana | Tema | Ambiente |
|---|---|---|
| 1 | Scikit-Learn Pipelines e o Accuracy Paradox | CPU |
| 2 | A Geometria dos Dados: Linear Algebra com NumPy | CPU |
| 3 | O Motor do Aprendizado: Calculus e Gradient Descent | CPU |
| 4 | A Ponte para Tensors: PyTorch e Autograd | GPU T4 |

## Como cada capítulo funciona

Cada capítulo corresponde a uma semana e segue sempre a mesma ordem: o objetivo, por que o tema importa para um AI Architect, os conceitos, os erros comuns, a leitura externa recomendada, as perguntas de autoavaliação e o entregável da semana.

Dentro dos conceitos, dois tipos de destaque ajudam a fixar o conteúdo:

> **Em palavras simples.** Uma analogia, sem jargão, para entender a ideia antes da explicação técnica.

> **No mundo real.** Um caso concreto, com números, de onde a ideia aparece em um paper, um modelo ou uma empresa.

## Livro e notebooks

A prática de cada semana acontece em notebooks no Google Colab, que rodam no plano gratuito. Ao fim de cada capítulo há os links para o notebook de exercícios, com trechos marcados `# [SEU CÓDIGO AQUI]` para você completar, e para o gabarito. O código dos gabaritos também está no apêndice deste livro, para consulta **depois** de tentar.

O ciclo sugerido para cada semana:

1. **Leia** o capítulo e o material externo indicado nele, sem código (fim de semana, cerca de 4 horas).
2. **Responda** às perguntas de autoavaliação, de preferência por escrito.
3. **Resolva** o notebook de exercícios, dois ou três passos por dia (segunda a quinta, cerca de 2 horas por dia).
4. **Compare** com o gabarito. Diferença de estilo é normal; diferença de resultado merece investigação.
5. **Escreva** o entregável da semana.

Tente antes de olhar o gabarito: travar por vinte minutos em um exercício ensina mais do que ler a solução em dois.

## Checkpoint de saída

Ao final da fase você deve conseguir, sem consultar: escrever a fórmula de cosine similarity, explicar o que `loss.backward()` faz e ler uma confusion matrix como custo de negócio. O capítulo da Semana 4 detalha esse checkpoint.

O repositório completo, com todas as fases e notebooks, está em <https://github.com/NetoRibeiro/ai-architect-foundations>, sob a licença Apache 2.0.
