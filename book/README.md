# Livro (protótipo da Fase 1)

Gera um PDF (6 × 9 polegadas) e um EPUB a partir das leituras de conceito e dos gabaritos da Fase 1.

## Preparar o ambiente (uma vez)

```bash
python -m venv book/.venv
book/.venv/Scripts/python -m pip install pypandoc_binary playwright pypdf
```

O PDF é renderizado pelo Google Chrome já instalado na máquina (`channel="chrome"` do Playwright), então não é preciso baixar navegador nem LaTeX.

## Gerar

```bash
book/.venv/Scripts/python book/build.py
```

Saída em `book/_output/` (ignorada pelo Git): `fase-1.pdf`, `fase-1.epub` e os intermediários `fase-1.md` e `fase-1.html`.

## Como funciona

- `fase-1/apresentacao.md`: capítulo de abertura escrito para o livro; `fase-1/metadata.yaml`: título, autor e idioma.
- `build.py` junta a apresentação, as quatro leituras e um apêndice com os gabaritos convertidos dos notebooks. Links entre leituras viram âncoras internas; links para notebooks viram links do Colab; o resto vira link do GitHub.
- `assets/book.css` (PDF), `assets/epub.css` (EPUB) e `assets/cover.html` (capa, também exportada como PNG para o EPUB).
