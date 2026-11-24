# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

An open, self-paced 32-week study track ("Enterprise AI Architect — Foundations") going from Data Science to AI Architecture across 7 phases. It is course content, not an application: Markdown readings plus Jupyter notebooks meant to run on Google Colab (free T4 tier). There is no build system, package manifest, linter, or test suite. Each notebook installs its own non-Colab dependencies (`!pip -q install ...`) in its first cells.

The entry points for humans are `README.md` (public landing page) and `PLANO_DE_ESTUDO.md` (week-by-week schedule, models used, notebook validation status, progress checklist).

## Language convention

All prose is in Brazilian Portuguese, but technical terms, function names, library names and model names stay in English as used in industry (e.g. "Um **MLP** (multi-layer perceptron) é `Linear → activation → Linear`"). Keep this mix when writing or editing content; do not translate terms like `training loop`, `embedding`, `reranking`, `fine-tuning`.

## Structure and the per-week triad

Every week N has exactly three files that must stay in sync:

| File | Role |
|---|---|
| `docs/Semana-NN-Conceitos.md` | Concept reading |
| `notebooks/semana_NN_exercicio.ipynb` | Exercise notebook; gaps marked `# [SEU CÓDIGO AQUI]` |
| `notebooks/semana_NN_gabarito.ipynb` | Same cells, solved |

Layout differs by phase:
- **Phase 1** (`01-Phase-1-Math-and-ML-Core/`): one subfolder per week (`01-sklearn-pipelines/`, `02-linear-algebra-numpy/`, ...), each with its own `docs/` and `notebooks/`.
- **Phases 2–7**: `docs/` and `notebooks/` sit directly in the phase folder.
- Each phase folder has a `README.md` index. ADR/RFC templates live in `07-Phase-7-Governance-and-Scale/docs/templates/`.

Exercise and gabarito notebooks share the same cell sequence (same markdown, same setup cells, same "Passo N" steps); only the solution cells differ. When changing one, mirror the change in the other. Notebooks are committed **without outputs**.

Each notebook starts with a markdown header: week title, phase name, "Notebook de exercícios" or "Gabarito", an "Abrir no Colab" badge, and a link to the week's concept reading. Colab badges point at `NetoRibeiro/applied-ai-architecture` on `main` — if a file is moved or renamed, update its badge URL, the links in `PLANO_DE_ESTUDO.md`, the phase `README.md`, and the root `README.md`.

GPU notebooks carry `"accelerator": "GPU"` in notebook metadata. Models are swapped via a `MODELO` constant near the top of the notebook. All models are open and need no API key or license acceptance (Qwen2.5 0.5B/1.5B Instruct, distilgpt2, multilingual MiniLM, mMiniLM cross-encoder, mDeBERTa NLI, distilbert).

## Concept reading format

`Semana-NN-Conceitos.md` files follow a fixed section order: `Objetivo`, `Por que isso importa para um AI Architect`, `Conceitos` (numbered `### N.` subsections), `Erros comuns`, `Leitura e vídeos (fim de semana)`, `Perguntas de autoavaliação`, `Entregável da semana`, `Prática da semana (segunda a sexta, Colab)`. The last reading of each phase contains the phase **checkpoint**; Week 32 describes the final project.

Concept subsections may carry two blockquote callouts:
- `> **Em palavras simples.** ...` — a plain-language analogy for non-specialists, placed right after the subsection heading.
- `> **No mundo real.** ...` — a concrete, factual real-world example (named paper, model, dataset, or company with numbers), placed after the technical explanation.

Facts in "No mundo real" callouts must be verifiable; don't invent figures.

## Validation status

`PLANO_DE_ESTUDO.md` ("Estado de validação dos notebooks") records which gabaritos were run end-to-end, which only with small substitute models, and which were only syntax-checked (Week 10, Week 20 Part B). Update that section when a notebook's validation status changes.

## Book build (prototype)

`book/` turns Phase 1 into a PDF and EPUB: `book/.venv/Scripts/python book/build.py` (setup in `book/README.md`; output in the git-ignored `book/_output/`). It assembles `book/fase-1/apresentacao.md`, the four readings and the gabarito notebooks, rewrites repo links (readings → internal anchors, notebooks → Colab, rest → GitHub), converts with Pandoc (`gfm+attributes`) and prints the PDF with the installed Chrome via Playwright, in two passes: the first finds each chapter's start page, the second prints the TOC with those page numbers (the build fails if the numbers shift). The book title lives in `BOOK_TITLE` in `build.py` and in `book/fase-1/metadata.yaml`. Let Pandoc write output files itself (`convert_file` with `outputfile`): round-tripping text through Python on Windows produces `\r\r\n` and doubles every line in code blocks.

## Legacy / draft files

`ROADMAP.md` (English, older 5-phase plan), `Phase_01.md`, `01-Phase-1-Math-and-ML-Core/Phase-1.md`, `Plano de Estudo Completo` (no extension) and `setup.txt` are the author's original drafts and early folder sketches. They do not reflect the current 7-phase structure; `PLANO_DE_ESTUDO.md` is authoritative (its section "Onde entrou cada ideia do seu rascunho" maps draft ideas to weeks). `LOGBOOK.md` is the learner's personal log.
