import fs from 'node:fs/promises';
import path from 'node:path';
import JSZip from 'jszip';
import { marked } from 'marked';
import type {
  ExportBundlePayload,
  ExportChapter,
  ExportScope,
  NotebookCell,
  PhaseInfo,
  WeekMeta,
} from './src/types.js';

const BOOK_TITLE = 'Do Analista ou Cientista de Dados ao Arquiteto de IA com Google Colab';
const BOOK_AUTHOR = 'Neto Ribeiro';
const GITHUB_BLOB_BASE = 'https://github.com/NetoRibeiro/applied-ai-architecture/blob/main/';
const COLAB_BLOB_BASE =
  'https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/';

marked.setOptions({
  gfm: true,
  breaks: false,
});

export interface ExportOptions {
  scope: ExportScope;
  phase?: number;
  week?: number;
  includeNotebooks: boolean;
}

function escapeXml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

function stripInvalidXmlChars(text: string): string {
  // Remove ASCII control characters forbidden in XML 1.0 (keep \t, \n, \r)
  return text.replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\uFFFE\uFFFF]/g, '');
}

function htmlToXhtml(html: string): string {
  let out = stripInvalidXmlChars(html);
  // Ensure void HTML elements are self-closing for strict XHTML5 parsing in EPUB readers
  out = out.replace(/<(br|hr)\s*>/gi, '<$1 />');
  out = out.replace(/<(img|input|meta|link)([^>]*?)(?<!\/)>/gi, '<$1$2 />');
  // Escape any stray unescaped ampersands (not already a valid XML/HTML entity)
  out = out.replace(
    /&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)/g,
    '&amp;',
  );
  // Replace common named HTML entities not built into strict XML if marked ever emits them
  out = out.replace(/&nbsp;/g, '&#160;').replace(/&mdash;/g, '&#8212;').replace(/&ndash;/g, '&#8211;');
  return out;
}

function rewriteRelativeLink(
  target: string,
  sourceRelPath: string,
  inBookReadingMap: Map<string, string>,
): string {
  if (/^(https?:|mailto:|#)/i.test(target)) {
    return target;
  }
  const hashIndex = target.indexOf('#');
  const rawPath = hashIndex >= 0 ? target.slice(0, hashIndex) : target;
  const frag = hashIndex >= 0 ? target.slice(hashIndex + 1) : '';

  const resolved = rawPath.startsWith('/')
    ? rawPath.replace(/^\/+/, '')
    : path.posix.normalize(path.posix.join(path.posix.dirname(sourceRelPath), rawPath));

  const internalAnchor = inBookReadingMap.get(resolved);
  if (internalAnchor) {
    return internalAnchor;
  }
  if (resolved.endsWith('.ipynb')) {
    return `${COLAB_BLOB_BASE}${resolved}`;
  }
  return `${GITHUB_BLOB_BASE}${resolved}${frag ? `#${frag}` : ''}`;
}

function fixMarkdownLinks(
  markdown: string,
  sourceRelPath: string,
  inBookReadingMap: Map<string, string>,
): string {
  const withoutBadges = markdown.replace(
    /\[!\[Abrir no Colab\]\([^)]*\)\]\(([^)]*)\)/g,
    '[Abrir no Colab]($1)',
  );
  return withoutBadges.replace(
    /(?<!!)\[([^\]]*)\]\(([^)\s]+)\)/g,
    (_full, label: string, url: string) =>
      `[${label}](${rewriteRelativeLink(url, sourceRelPath, inBookReadingMap)})`,
  );
}

function cleanReadingMarkdown(
  rawMarkdown: string,
  meta: WeekMeta,
  inBookReadingMap: Map<string, string>,
  includeNotebooks: boolean,
): string {
  const pad = String(meta.week).padStart(2, '0');
  const lines = rawMarkdown.split(/\r?\n/);
  const filtered: string[] = [];

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    if (line.startsWith('**Fase ') && line.includes('Leitura de conceito')) {
      continue;
    }
    if (line.startsWith('[← Semana') || line.startsWith('[Plano de Estudo')) {
      continue;
    }
    filtered.push(line);
  }

  let text = filtered.join('\n');

  const exName = path.basename(meta.exerciseRelativePath);
  const gabName = path.basename(meta.gabaritoRelativePath);
  const gabaritoRef = includeNotebooks
    ? `; o código resolvido e o teste prático de entrevista também estão no [apêndice](#gabarito-${pad}).`
    : '.';

  const practiceBullets =
    `- **Exercícios:** [\`${exName}\`](${meta.colabExerciseUrl}) (abre no Google Colab)\n` +
    `- **Gabarito:** [\`${gabName}\`](${meta.colabGabaritoUrl}) (abre no Google Colab)${gabaritoRef}\n`;

  text = text.replace(
    /\| Notebook \| Abrir \|\r?\n\|---\|---\|\r?\n(?:\|.*\|\r?\n?)+/g,
    practiceBullets,
  );
  text = text.replace(/\n---\n\n## Prática da semana/g, '\n\n## Prática da semana');

  return fixMarkdownLinks(text, meta.conceptRelativePath, inBookReadingMap);
}

function buildGabaritoMarkdown(
  meta: WeekMeta,
  cells: NotebookCell[],
  inBookReadingMap: Map<string, string>,
): string {
  const pad = String(meta.week).padStart(2, '0');
  const firstHeading =
    cells[0]?.source
      .split(/\r?\n/)[0]
      ?.replace(/^#+\s*/, '')
      .replace(`Semana ${meta.week} — `, '')
      .trim() || meta.title;

  const gabName = path.basename(meta.gabaritoRelativePath);
  const parts: string[] = [
    `# Gabarito da Semana ${pad} — ${meta.title}`,
    `*${firstHeading}*. Notebook original: [\`${gabName}\`](${meta.colabGabaritoUrl}). Leitura de conceito: [Semana ${pad}](#semana-${pad}).`,
  ];

  for (const cell of cells.slice(1)) {
    const body = cell.source.trim();
    if (!body) continue;
    if (cell.cellType === 'code') {
      parts.push(`\`\`\`python\n${body}\n\`\`\``);
    } else {
      const normalized = body.replace(/^# /gm, '## ');
      parts.push(fixMarkdownLinks(normalized, meta.gabaritoRelativePath, inBookReadingMap));
    }
  }

  return parts.join('\n\n');
}

function buildIntroMarkdown(
  scope: ExportScope,
  selectedPhase: PhaseInfo | undefined,
  selectedWeeks: WeekMeta[],
  includeNotebooks: boolean,
): string {
  const lines: string[] = ['# Como usar este material'];

  if (scope === 'full') {
    lines.push(
      `Este volume reúne o conteúdo completo da trilha **Enterprise AI Architect — Foundations**: **32 semanas** organizadas em **7 fases**, cobrindo desde fundamentos matemáticos e pipelines clássicos até Deep Learning, RAG avançado, Fine-Tuning (PEFT/QLoRA/DPO), MLOps, Causal Inference e Governança em escala.`,
    );
  } else if (scope === 'phase' && selectedPhase) {
    lines.push(
      `Este volume reúne as leituras de conceito da **Fase ${selectedPhase.id} — ${selectedPhase.title}** (${selectedPhase.weeksRange}), parte da trilha *Enterprise AI Architect — Foundations*.`,
    );
    lines.push(`${selectedPhase.subtitle}`);
  } else if (selectedWeeks[0]) {
    const w = selectedWeeks[0];
    lines.push(
      `Este caderno reúne o estudo completo da **Semana ${String(w.week).padStart(2, '0')} — ${w.title}** (Fase ${w.phase}: ${w.phaseTitle}), incluindo leitura conceitual, perguntas de autoavaliação, simulação de entrevista técnica e entregável prático.`,
    );
  }

  lines.push(
    'O texto é em português, mantendo termos técnicos, nomes de funções e bibliotecas em inglês — exatamente como aparecem em papers, documentações oficiais e entrevistas técnicas.',
  );

  lines.push('## Roteiro incluído nesta edição');
  lines.push('| Semana | Fase | Tema | Ambiente |');
  lines.push('|---|---|---|---|');
  for (const w of selectedWeeks) {
    lines.push(
      `| Semana ${String(w.week).padStart(2, '0')} | Fase ${w.phase} | ${w.title} | ${w.environment} |`,
    );
  }

  lines.push('## Estrutura de cada capítulo');
  lines.push(
    'Cada semana segue um ciclo consistente: objetivo, relevância para um AI Architect, conceitos centrais, erros comuns, referências recomendadas, perguntas de autoavaliação, perguntas de entrevistador técnico e entregável.',
  );

  if (includeNotebooks) {
    lines.push(
      'Ao final do material, o **Apêndice de Notebooks & Gabaritos** traz o código resolvido e os **Testes Práticos de Entrevista (Live Coding)** de cada semana selecionada.',
    );
  }

  lines.push(
    `Repositório oficial aberto (Apache 2.0): <https://github.com/NetoRibeiro/applied-ai-architecture>.`,
  );

  return lines.join('\n\n');
}

export function buildCoverSvg(bundle: {
  title: string;
  subtitle: string;
  scopeLabel: string;
  weeksRange: string;
  author: string;
  activeWeeks: number[];
}): string {
  const activeSet = new Set(bundle.activeWeeks);

  // Wrap title nicely into <= 4 lines for 800x1280 SVG cover
  const words = bundle.title.split(' ');
  const titleLines: string[] = [];
  let current = '';
  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;
    if (candidate.length > 22 && current) {
      titleLines.push(current);
      current = word;
    } else {
      current = candidate;
    }
  }
  if (current) titleLines.push(current);

  const subWords = bundle.subtitle.split(' ');
  const subLines: string[] = [];
  let subCur = '';
  for (const word of subWords) {
    const candidate = subCur ? `${subCur} ${word}` : word;
    if (candidate.length > 34 && subCur) {
      subLines.push(subCur);
      subCur = word;
    } else {
      subCur = candidate;
    }
  }
  if (subCur) subLines.push(subCur);

  const titleSvg = titleLines
    .map(
      (line, idx) =>
        `<text x="80" y="${210 + idx * 66}" font-family="Georgia, 'Times New Roman', serif" font-size="52" fill="#F5F3EE">${escapeXml(line)}</text>`,
    )
    .join('\n  ');

  const ruleY = 210 + titleLines.length * 66 + 22;
  const scopeY = ruleY + 54;
  const subtitleStartY = scopeY + 46;

  const subSvg = subLines
    .map(
      (line, idx) =>
        `<text x="80" y="${subtitleStartY + idx * 44}" font-family="Georgia, 'Times New Roman', serif" font-size="34" fill="#F5F3EE">${escapeXml(line)}</text>`,
    )
    .join('\n  ');

  const gridSquares: string[] = [];
  const startX = 80;
  const startY = 900;
  const size = 34;
  const gap = 10;
  for (let i = 0; i < 32; i += 1) {
    const row = Math.floor(i / 8);
    const col = i % 8;
    const x = startX + col * (size + gap);
    const y = startY + row * (size + gap);
    const isOn = activeSet.has(i + 1);
    gridSquares.push(
      `<rect x="${x}" y="${y}" width="${size}" height="${size}" fill="${isOn ? '#FCA311' : 'none'}" stroke="${isOn ? '#FCA311' : '#3D5A80'}" stroke-width="2" />`,
    );
  }

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 1280" width="800" height="1280">
  <rect width="800" height="1280" fill="#14213D" />
  <text x="80" y="130" font-family="sans-serif" font-size="18" font-weight="600" letter-spacing="3.5" fill="#FCA311">ENTERPRISE AI ARCHITECT · TRILHA DE 32 SEMANAS</text>
  ${titleSvg}
  <rect x="80" y="${ruleY}" width="144" height="6" fill="#FCA311" />
  <text x="80" y="${scopeY}" font-family="sans-serif" font-size="24" fill="#C9CFDB">${escapeXml(bundle.scopeLabel)} · ${escapeXml(bundle.weeksRange)}</text>
  ${subSvg}
  ${gridSquares.join('\n  ')}
  <text x="80" y="1145" font-family="sans-serif" font-size="22" letter-spacing="1.5" fill="#F5F3EE">${escapeXml(bundle.author)}</text>
</svg>`;
}

export async function buildExportBundle(
  rootDir: string,
  phases: PhaseInfo[],
  weeksMeta: WeekMeta[],
  parseNotebook: (filePath: string) => Promise<NotebookCell[]>,
  options: ExportOptions,
): Promise<ExportBundlePayload> {
  let selectedWeeks: WeekMeta[] = [];
  let selectedPhase: PhaseInfo | undefined;
  let subtitle = '';
  let scopeLabel = '';
  let weeksRange = '';

  if (options.scope === 'week') {
    const targetWeek = options.week ?? 1;
    const found = weeksMeta.find((w) => w.week === targetWeek);
    if (!found) {
      throw new Error(`Semana inválida para exportação: ${targetWeek}`);
    }
    selectedWeeks = [found];
    selectedPhase = phases.find((p) => p.id === found.phase);
    const pad = String(found.week).padStart(2, '0');
    scopeLabel = `Fase ${found.phase} — ${found.phaseTitle}`;
    weeksRange = `Semana ${pad}`;
    subtitle = `Semana ${pad} — ${found.title}`;
  } else if (options.scope === 'phase') {
    const targetPhase = options.phase ?? 1;
    selectedPhase = phases.find((p) => p.id === targetPhase);
    if (!selectedPhase) {
      throw new Error(`Fase inválida para exportação: ${targetPhase}`);
    }
    selectedWeeks = weeksMeta.filter((w) => w.phase === targetPhase);
    scopeLabel = `Fase ${selectedPhase.id}`;
    weeksRange = selectedPhase.weeksRange;
    subtitle = selectedPhase.title;
  } else {
    selectedWeeks = [...weeksMeta];
    scopeLabel = 'Edição Completa · Fases 1 a 7';
    weeksRange = 'Semanas 1 a 32';
    subtitle = 'Trilha Completa de Arquitetura de IA Aplicada';
  }

  const inBookReadingMap = new Map<string, string>();
  for (const w of selectedWeeks) {
    const pad = String(w.week).padStart(2, '0');
    inBookReadingMap.set(w.conceptRelativePath, `#semana-${pad}`);
  }

  const chapters: ExportChapter[] = [];

  const introMd = buildIntroMarkdown(
    options.scope,
    selectedPhase,
    selectedWeeks,
    options.includeNotebooks,
  );
  chapters.push({
    id: 'apresentacao',
    title: 'Como usar este material',
    kind: 'intro',
    html: await marked.parse(introMd),
  });

  for (const w of selectedWeeks) {
    const pad = String(w.week).padStart(2, '0');
    const rawConcept = await fs.readFile(path.join(rootDir, w.conceptRelativePath), 'utf-8');
    const cleanedMd = cleanReadingMarkdown(
      rawConcept,
      w,
      inBookReadingMap,
      options.includeNotebooks,
    );
    chapters.push({
      id: `semana-${pad}`,
      title: `Semana ${pad} — ${w.title}`,
      kind: 'reading',
      week: w.week,
      phase: w.phase,
      html: await marked.parse(cleanedMd),
    });
  }

  if (options.includeNotebooks) {
    const dividerMd =
      '# Apêndice: Notebooks & Gabaritos\n\n' +
      'O código resolvido de cada semana (incluindo os **Testes Práticos de Entrevista / Live Coding**), para conferir **depois** de tentar o notebook de exercícios. Os gráficos e saídas completas aparecem quando você executa o notebook no Google Colab.';

    chapters.push({
      id: 'apendice',
      title: 'Apêndice: Notebooks & Gabaritos',
      kind: 'appendix-divider',
      html: await marked.parse(dividerMd),
    });

    for (const w of selectedWeeks) {
      const pad = String(w.week).padStart(2, '0');
      const cells = await parseNotebook(path.join(rootDir, w.gabaritoRelativePath));
      const gabMd = buildGabaritoMarkdown(w, cells, inBookReadingMap);
      chapters.push({
        id: `gabarito-${pad}`,
        title: `Gabarito da Semana ${pad} — ${w.title}`,
        kind: 'gabarito',
        week: w.week,
        phase: w.phase,
        html: await marked.parse(gabMd),
      });
    }
  }

  return {
    scope: options.scope,
    title: BOOK_TITLE,
    subtitle,
    scopeLabel,
    weeksRange,
    author: BOOK_AUTHOR,
    activeWeeks: selectedWeeks.map((w) => w.week),
    includeNotebooks: options.includeNotebooks,
    generatedAt: new Date().toISOString(),
    chapters,
  };
}

const EPUB_CSS = `body {
  font-family: Georgia, Cambria, "Times New Roman", serif;
  line-height: 1.55;
  color: #1d1d1f;
  margin: 0;
  padding: 0 0.4em;
}
h1 {
  font-family: sans-serif;
  font-weight: 700;
  font-size: 1.55em;
  color: #0f172a;
  border-bottom: 2px solid #fca311;
  padding-bottom: 0.3em;
  margin-top: 0.6em;
  margin-bottom: 0.8em;
}
h2, h3 {
  font-family: sans-serif;
  color: #14213d;
  margin-top: 1.25em;
  margin-bottom: 0.45em;
}
blockquote {
  margin: 1em 0;
  padding: 0.6em 0.9em;
  background: #f4f6fa;
  border-left: 3px solid #3d5a80;
}
blockquote p { margin: 0; }
code {
  font-family: monospace;
  font-size: 0.88em;
  background: #f1f5f9;
  padding: 0.08em 0.25em;
}
pre {
  white-space: pre-wrap;
  word-wrap: break-word;
  font-family: monospace;
  font-size: 0.78em;
  line-height: 1.4;
  background: #f7f6f2;
  padding: 0.65em 0.8em;
  border-left: 3px solid #d9d4c5;
}
pre code { background: transparent; padding: 0; font-size: 1em; }
table {
  border-collapse: collapse;
  width: 100%;
  font-size: 0.85em;
  margin: 1em 0;
}
th, td {
  border-bottom: 1px solid #cbd5e1;
  padding: 0.35em 0.45em;
  text-align: left;
  vertical-align: top;
}
th {
  font-family: sans-serif;
  border-bottom: 1.5px solid #14213d;
}
a { color: #1f4e8c; text-decoration: underline; }
hr { border: none; border-top: 1px solid #cbd5e1; margin: 1.2em 0; }
`;

function rewriteInternalEpubLinks(
  xhtml: string,
  chapterFileById: Map<string, string>,
): string {
  return xhtml.replace(/href="#([^"]+)"/g, (_m, anchor: string) => {
    const targetFile = chapterFileById.get(anchor);
    if (targetFile) {
      return `href="${targetFile}"`;
    }
    return `href="#${anchor}"`;
  });
}

function wrapXhtmlPage(title: string, bodyHtml: string): string {
  return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="pt-BR" lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <title>${escapeXml(title)}</title>
  <link rel="stylesheet" type="text/css" href="styles.css" />
</head>
<body>
${bodyHtml}
</body>
</html>
`;
}

export async function buildEpubBuffer(bundle: ExportBundlePayload): Promise<Buffer> {
  const zip = new JSZip();

  // EPUB spec requires 'mimetype' as the first file and uncompressed (STORE)
  zip.file('mimetype', 'application/epub+zip', { compression: 'STORE' });

  zip.file(
    'META-INF/container.xml',
    `<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml" />
  </rootfiles>
</container>
`,
  );

  const oebps = zip.folder('OEBPS')!;
  oebps.file('styles.css', EPUB_CSS);

  const coverSvg = buildCoverSvg(bundle);
  oebps.file('cover.svg', coverSvg);
  oebps.file(
    'cover.xhtml',
    wrapXhtmlPage(
      `${bundle.title} — Capa`,
      `<div style="text-align:center;margin:0;padding:0;">
  <img src="cover.svg" alt="${escapeXml(bundle.title)}" style="max-width:100%;height:auto;" />
</div>`,
    ),
  );

  const chapterFiles: Array<{ id: string; href: string; title: string }> = [];
  const chapterFileById = new Map<string, string>();

  bundle.chapters.forEach((chapter, idx) => {
    const filename = `ch_${String(idx + 1).padStart(3, '0')}_${chapter.id}.xhtml`;
    chapterFiles.push({
      id: `ch_${idx + 1}`,
      href: filename,
      title: chapter.title,
    });
    chapterFileById.set(chapter.id, filename);
  });

  bundle.chapters.forEach((chapter, idx) => {
    const fileInfo = chapterFiles[idx];
    const xhtmlBody = rewriteInternalEpubLinks(htmlToXhtml(chapter.html), chapterFileById);
    oebps.file(fileInfo.href, wrapXhtmlPage(chapter.title, xhtmlBody));
  });

  const navItems = chapterFiles
    .map(
      (ch) =>
        `      <li><a href="${escapeXml(ch.href)}">${escapeXml(ch.title)}</a></li>`,
    )
    .join('\n');

  oebps.file(
    'nav.xhtml',
    `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="pt-BR" lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <title>Sumário</title>
  <link rel="stylesheet" type="text/css" href="styles.css" />
</head>
<body>
  <nav epub:type="toc" id="toc">
    <h1>Sumário</h1>
    <ol>
${navItems}
    </ol>
  </nav>
</body>
</html>
`,
  );

  const ncxPoints = chapterFiles
    .map(
      (ch, idx) => `    <navPoint id="navPoint-${idx + 1}" playOrder="${idx + 1}">
      <navLabel><text>${escapeXml(ch.title)}</text></navLabel>
      <content src="${escapeXml(ch.href)}" />
    </navPoint>`,
    )
    .join('\n');

  const bookUid = `urn:uuid:applied-ai-arch-${bundle.scope}-${bundle.activeWeeks.join('-')}`;

  oebps.file(
    'toc.ncx',
    `<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <head>
    <meta name="dtb:uid" content="${escapeXml(bookUid)}" />
    <meta name="dtb:depth" content="1" />
    <meta name="dtb:totalPageCount" content="0" />
    <meta name="dtb:maxPageNumber" content="0" />
  </head>
  <docTitle><text>${escapeXml(bundle.title)}</text></docTitle>
  <navMap>
${ncxPoints}
  </navMap>
</ncx>
`,
  );

  const manifestItems = chapterFiles
    .map(
      (ch) =>
        `    <item id="${escapeXml(ch.id)}" href="${escapeXml(ch.href)}" media-type="application/xhtml+xml" />`,
    )
    .join('\n');

  const spineItems = chapterFiles
    .map((ch) => `    <itemref idref="${escapeXml(ch.id)}" />`)
    .join('\n');

  const modifiedTimestamp = new Date().toISOString().replace(/\.\d{3}Z$/, 'Z');
  const fullTitle = `${bundle.title} — ${bundle.subtitle}`;

  oebps.file(
    'content.opf',
    `<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="BookId" xml:lang="pt-BR">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="BookId">${escapeXml(bookUid)}</dc:identifier>
    <dc:title>${escapeXml(fullTitle)}</dc:title>
    <dc:creator>${escapeXml(bundle.author)}</dc:creator>
    <dc:language>pt-BR</dc:language>
    <dc:rights>Apache License 2.0</dc:rights>
    <dc:description>${escapeXml(`${bundle.scopeLabel} (${bundle.weeksRange})`)}</dc:description>
    <meta property="dcterms:modified">${modifiedTimestamp}</meta>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav" />
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml" />
    <item id="css" href="styles.css" media-type="text/css" />
    <item id="cover-img" href="cover.svg" media-type="image/svg+xml" properties="cover-image" />
    <item id="cover" href="cover.xhtml" media-type="application/xhtml+xml" />
${manifestItems}
  </manifest>
  <spine toc="ncx">
    <itemref idref="cover" />
    <itemref idref="nav" />
${spineItems}
  </spine>
</package>
`,
  );

  return zip.generateAsync({
    type: 'nodebuffer',
    compression: 'DEFLATE',
    compressionOptions: { level: 6 },
  });
}

export function buildExportFilename(
  scope: ExportScope,
  ext: 'epub' | 'pdf' | 'html',
  phase?: number,
  week?: number,
): string {
  if (scope === 'week') {
    const pad = String(week ?? 1).padStart(2, '0');
    return `applied-ai-architecture-semana-${pad}.${ext}`;
  }
  if (scope === 'phase') {
    return `applied-ai-architecture-fase-${phase ?? 1}.${ext}`;
  }
  return `applied-ai-architecture-completo-32-semanas.${ext}`;
}
