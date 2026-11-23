# RFC: [Título]

- **Autor:** [nome]
- **Status:** rascunho | em revisão | aprovado | rejeitado | substituído
- **Criado em:** AAAA-MM-DD
- **Prazo para comentários:** AAAA-MM-DD
- **Revisores:** [nomes ou papéis]

## Resumo

[Três a cinco frases: o que está sendo proposto e por quê. Quem ler só esta seção deve entender a proposta.]

## Motivação

[O problema atual, com evidência: incidentes, custo, tempo perdido, decisões inconsistentes. Por que agora.]

## Objetivos

- [resultado mensurável 1]
- [resultado mensurável 2]

## Não-objetivos

- [o que esta proposta deliberadamente não resolve]

## Proposta

[O desenho em detalhe suficiente para ser criticado: componentes, fluxos, interfaces, padrões, responsabilidades. Diagramas ajudam.]

## Alternativas consideradas

| Alternativa | Vantagens | Desvantagens | Por que não |
|---|---|---|---|
| Não fazer nada | | | |
| [alternativa] | | | |

## Riscos e mitigação

| Risco | Probabilidade | Impacto | Mitigação |
|---|---|---|---|
| | | | |

## Plano de implantação

1. [fase 1, com critério de saída]
2. [fase 2]

**Plano de reversão:** [como voltar atrás].

## Métricas de sucesso

[Como saberemos, em 3 e em 6 meses, que funcionou.]

## Perguntas em aberto

- [...]

---

### Exemplo de uso: "Como padronizaremos os testes A/B"

Seções que este RFC específico deve cobrir na Proposta: unidade de randomização por tipo de produto (usuário, cluster, switchback); métrica primária (OEC) e métricas de proteção; cálculo de tamanho de amostra e duração mínima; verificações obrigatórias antes da análise (SRM, teste A/A); regra de decisão e tratamento de comparações múltiplas; formato do relatório; quem aprova o lançamento.
