"""Gera o livro da Fase 1 em PDF e EPUB a partir das leituras e dos gabaritos.

Uso (a partir da raiz do repositório):
    book/.venv/Scripts/python book/build.py
"""
import json
import posixpath
import re
from pathlib import Path

import pypandoc
from playwright.sync_api import sync_playwright
from pypdf import PdfReader, PdfWriter

BOOK = Path(__file__).resolve().parent
REPO = BOOK.parent
OUT = BOOK / "_output"
ASSETS = BOOK / "assets"
GITHUB = "https://github.com/NetoRibeiro/ai-architect-foundations/blob/main/"
COLAB = "https://colab.research.google.com/github/NetoRibeiro/ai-architect-foundations/blob/main/"

PHASE_DIR = "01-Phase-1-Math-and-ML-Core"
WEEKS = {
    1: "01-sklearn-pipelines",
    2: "02-linear-algebra-numpy",
    3: "03-calculus-gradient",
    4: "04-pytorch-tensors",
}
SLUG = "fase-1"
BOOK_TITLE = "Do Analista ou Cientista de Dados ao Arquiteto de IA com Google Colab"
TITLE = "Math Foundations e ML Core"

PDF_MARGIN = {"top": "0.75in", "bottom": "0.8in", "left": "0.7in", "right": "0.7in"}
FOOTER = ('<div style="width:100%;font-size:8px;font-family:Georgia,serif;'
          'color:#777;text-align:center;"><span class="pageNumber"></span></div>')

PANDOC_FROM = "gfm+attributes"


def reading_path(n):
    return f"{PHASE_DIR}/{WEEKS[n]}/docs/Semana-{n:02d}-Conceitos.md"


def notebook_path(n, kind):
    return f"{PHASE_DIR}/{WEEKS[n]}/notebooks/semana_{n:02d}_{kind}.ipynb"


IN_BOOK = {reading_path(n): f"#semana-{n:02d}" for n in WEEKS}


def rewrite_link(target, source):
    """Converte um link relativo do repositório em âncora interna ou URL absoluta."""
    if re.match(r"^(https?:|mailto:|#)", target):
        return target
    path, _, frag = target.partition("#")
    if path.startswith("/"):
        resolved = path.lstrip("/")
    else:
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(source), path))
    if resolved in IN_BOOK:
        return IN_BOOK[resolved]
    if resolved.endswith(".ipynb"):
        return COLAB + resolved
    return GITHUB + resolved + (f"#{frag}" if frag else "")


BADGE = re.compile(r"\[!\[Abrir no Colab\]\([^)]*\)\]\(([^)]*)\)")
LINK = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)\s]+)\)")


def fix_links(text, source):
    text = BADGE.sub(r"[Abrir no Colab](\1)", text)
    return LINK.sub(lambda m: f"[{m.group(1)}]({rewrite_link(m.group(2), source)})", text)


def reading_chapter(n):
    src = reading_path(n)
    lines = (REPO / src).read_text(encoding="utf-8").splitlines()
    out = []
    for i, line in enumerate(lines):
        if i == 0 and line.startswith("# "):
            out.append(f"{line} {{#semana-{n:02d}}}")
            continue
        if line.startswith("**Fase ") and "Leitura de conceito" in line:
            continue  # subtítulo de navegação do repositório
        if line.startswith("[← Semana") or line.startswith("[Plano de Estudo"):
            continue  # barra de navegação entre arquivos
        out.append(line)
    text = "\n".join(out)
    # A tabela "Prática da semana" vira texto: badges não funcionam em papel.
    text = re.sub(
        r"\| Notebook \| Abrir \|\n\|---\|---\|\n((?:\|.*\|\n?)+)",
        lambda m: practice_list(n),
        text,
    )
    text = text.replace("\n---\n\n## Prática da semana", "\n\n## Prática da semana")
    return fix_links(text, src)


def practice_list(n):
    ex, gab = notebook_path(n, "exercicio"), notebook_path(n, "gabarito")
    return (
        f"- **Exercícios:** [`{Path(ex).name}`]({COLAB}{ex}) (abre no Colab)\n"
        f"- **Gabarito:** [`{Path(gab).name}`]({COLAB}{gab}) (abre no Colab); "
        f"o código também está no [apêndice](#gabarito-{n:02d}).\n"
    )


def notebook_chapter(n):
    src = notebook_path(n, "gabarito")
    nb = json.loads((REPO / src).read_text(encoding="utf-8"))
    cells = nb["cells"]
    title = "".join(cells[0]["source"]).splitlines()[0].lstrip("# ").strip()
    title = title.replace(f"Semana {n} — ", "")
    parts = [
        f"# Gabarito da Semana {n} {{#gabarito-{n:02d}}}",
        f"*{title}*. Notebook original: [`{Path(src).name}`]({COLAB}{src}). "
        f"Leitura de conceito: [Semana {n}](#semana-{n:02d}).",
    ]
    for cell in cells[1:]:
        body = "".join(cell["source"]).strip()
        if not body:
            continue
        if cell["cell_type"] == "code":
            parts.append(f"```python\n{body}\n```")
        else:
            body = re.sub(r"^# ", "## ", body, flags=re.M)
            parts.append(fix_links(body, src))
    return "\n\n".join(parts)


def assemble():
    chunks = [(BOOK / SLUG / "apresentacao.md").read_text(encoding="utf-8")]
    chunks += [reading_chapter(n) for n in WEEKS]
    chunks.append("# Apêndice: gabaritos {#apendice .unnumbered .part}\n\n"
                  "O código resolvido de cada semana, para conferir **depois** de tentar o "
                  "notebook de exercícios. Os gráficos e as saídas aparecem quando você "
                  "executa o notebook no Colab.")
    chunks += [notebook_chapter(n) for n in WEEKS]
    md = "\n\n".join(chunks) + "\n"
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{SLUG}.md"
    path.write_text(md, encoding="utf-8", newline="\n")
    return path


def build_html(md_path):
    # O Pandoc grava os arquivos direto: passar o texto pelo Python no Windows
    # transforma "\r\n" em "\r\r\n", e o Chrome dobra as linhas dentro de <pre>.
    path = OUT / f"{SLUG}.html"
    pypandoc.convert_file(
        str(md_path), "html5", format=PANDOC_FROM, outputfile=str(path),
        extra_args=[
            "--standalone", "--toc", "--toc-depth=1", "--section-divs",
            f"--metadata-file={BOOK / SLUG / 'metadata.yaml'}",
            f"--css={ASSETS / 'book.css'}", "--embed-resources",
            "--syntax-highlighting=tango",
            f"--variable=toc-title:Sumário",
        ],
    )
    return path


def cover_html():
    tpl = (ASSETS / "cover.html").read_text(encoding="utf-8")
    page = OUT / "cover.html"
    page.write_text(tpl.replace("{{LIVRO}}", BOOK_TITLE).replace("{{FASE}}", "Fase 1")
                    .replace("{{TITULO}}", TITLE).replace("{{SEMANAS}}", "Semanas 1 a 4"),
                    encoding="utf-8")
    return page


def print_body(page, path):
    page.pdf(path=str(path), width="6in", height="9in", margin=PDF_MARGIN,
             display_header_footer=True, header_template="<span></span>",
             footer_template=FOOTER, print_background=True)


def squash(text):
    return re.sub(r"\s+", "", text)


def chapter_pages(pdf_path, chapters):
    """Página (numeração do rodapé) em que cada capítulo começa.

    Cada capítulo abre em página nova (break-before no CSS), então basta achar a
    página cujo texto começa com o título do capítulo.
    """
    pages = [squash(p.extract_text())[:300] for p in PdfReader(str(pdf_path)).pages]
    found = {}
    for cid, heading in chapters:
        key = squash(heading)
        hits = [i + 1 for i, text in enumerate(pages) if text.startswith(key)]
        if len(hits) != 1:
            raise RuntimeError(f"capítulo '{heading}' achado nas páginas {hits}")
        found[cid] = hits[0]
    return found


TOC_NUMBERS = r"""(numbers) => {
  for (const a of document.querySelectorAll('nav#TOC a')) {
    const id = a.getAttribute('href').slice(1);
    const old = a.querySelector('.pg'); if (old) old.remove();
    const span = document.createElement('span');
    span.className = 'pg'; span.textContent = numbers[id] ?? '';
    a.appendChild(span);
  }
}"""


def render(body_html, cover_page):
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        # Capa em PNG para o EPUB: 1600 × 2560 px, a proporção 1,6 recomendada pela KDP.
        shot = browser.new_page(viewport={"width": 800, "height": 1280}, device_scale_factor=2)
        shot.goto(cover_page.as_uri())
        shot.screenshot(path=str(OUT / "cover.png"))
        page = browser.new_page()
        page.goto(cover_page.as_uri())
        page.pdf(path=str(OUT / "cover.pdf"), width="6in", height="9in", print_background=True)
        # Dois passes: o primeiro descobre onde cada capítulo começa; o segundo
        # imprime o sumário já com os números de página.
        page.goto(body_html.as_uri())
        chapters = page.evaluate(
            "() => [...document.querySelectorAll('section.level1')]"
            ".map(s => [s.id, s.querySelector('h1').textContent])")
        page.evaluate(TOC_NUMBERS, {cid: "000" for cid, _ in chapters})  # reserva o espaço
        print_body(page, OUT / "body.pdf")
        numbers = chapter_pages(OUT / "body.pdf", chapters)
        page.evaluate(TOC_NUMBERS, numbers)
        print_body(page, OUT / "body.pdf")
        if chapter_pages(OUT / "body.pdf", chapters) != numbers:
            raise RuntimeError("o sumário mudou a paginação; rode de novo")
        browser.close()
    writer = PdfWriter()
    for part in ("cover.pdf", "body.pdf"):
        writer.append(str(OUT / part))
    writer.add_metadata({"/Title": f"{BOOK_TITLE} — Fase 1: {TITLE}",
                         "/Author": "Neto Ribeiro"})
    with open(OUT / f"{SLUG}.pdf", "wb") as fh:
        writer.write(fh)
    for tmp in ("cover.pdf", "body.pdf"):
        (OUT / tmp).unlink()


def build_epub(md_path):
    pypandoc.convert_file(
        str(md_path), "epub3", format=PANDOC_FROM, outputfile=str(OUT / f"{SLUG}.epub"),
        extra_args=[
            "--toc", "--toc-depth=2", "--split-level=1",
            f"--metadata-file={BOOK / SLUG / 'metadata.yaml'}",
            f"--css={ASSETS / 'epub.css'}",
            f"--epub-cover-image={OUT / 'cover.png'}",
            "--syntax-highlighting=tango",
        ],
    )


def main():
    md = assemble()
    body = build_html(md)
    render(body, cover_html())
    build_epub(md)
    for f in sorted(OUT.glob(f"{SLUG}.*")):
        print(f"{f.relative_to(REPO)}  {f.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
