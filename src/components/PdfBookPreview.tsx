import { useEffect } from 'react';
import { createPortal } from 'react-dom';
import { Download, Printer, X } from 'lucide-react';
import type { ExportBundlePayload } from '../types';

interface PdfBookPreviewProps {
  bundle: ExportBundlePayload;
  onClose: () => void;
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

export default function PdfBookPreview({ bundle, onClose }: PdfBookPreviewProps) {
  const activeSet = new Set(bundle.activeWeeks);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      window.print();
    }, 350);
    return () => window.clearTimeout(timer);
  }, [bundle]);

  const handleDownloadHtml = () => {
    const coverGrid = Array.from({ length: 32 }, (_, idx) => {
      const isOn = activeSet.has(idx + 1);
      return `<span style="display:inline-block;width:24px;height:24px;border:2px solid ${isOn ? '#FCA311' : '#3D5A80'};background:${isOn ? '#FCA311' : 'transparent'};"></span>`;
    }).join('');

    const tocHtml = bundle.chapters
      .map(
        (ch) =>
          `<li style="margin:6px 0;"><a href="#${escapeHtml(ch.id)}" style="color:#14213D;text-decoration:none;">${escapeHtml(ch.title)}</a></li>`,
      )
      .join('\n');

    const chaptersHtml = bundle.chapters
      .map(
        (ch) =>
          `<section id="${escapeHtml(ch.id)}" class="pdf-chapter${ch.kind === 'appendix-divider' ? ' pdf-chapter-appendix' : ''}">\n${ch.html}\n</section>`,
      )
      .join('\n');

    const fullHtml = `<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8" />
<title>${escapeHtml(bundle.title)} — ${escapeHtml(bundle.subtitle)}</title>
<style>
  @page { size: A4; margin: 16mm 16mm 18mm 16mm; }
  body { font-family: Georgia, Cambria, "Times New Roman", serif; color: #1d1d1f; line-height: 1.55; max-width: 820px; margin: 0 auto; padding: 24px; }
  .cover { background: #14213D; color: #F5F3EE; padding: 56px 48px; min-height: 240mm; display: flex; flex-direction: column; justify-content: space-between; break-after: page; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  .toc { break-after: page; }
  .pdf-chapter { break-before: page; }
  h1 { font-family: system-ui, sans-serif; border-bottom: 2px solid #FCA311; padding-bottom: 8px; color: #0F172A; }
  h2, h3 { font-family: system-ui, sans-serif; color: #14213D; break-after: avoid; }
  blockquote { margin: 1em 0; padding: 0.65em 0.95em; background: #F4F6FA; border-left: 3px solid #3D5A80; font-family: system-ui, sans-serif; break-inside: avoid; }
  code { font-family: ui-monospace, monospace; background: #F1EFE8; padding: 0.1em 0.3em; font-size: 0.88em; }
  pre { font-family: ui-monospace, monospace; background: #F7F6F2; border-left: 3px solid #D9D4C5; padding: 0.75em 0.9em; white-space: pre-wrap; word-break: break-word; font-size: 0.82em; }
  pre code { background: transparent; padding: 0; }
  table { border-collapse: collapse; width: 100%; font-family: system-ui, sans-serif; font-size: 0.86em; margin: 1em 0; break-inside: avoid; }
  th, td { border-bottom: 1px solid #E2E8F0; padding: 6px 8px; text-align: left; vertical-align: top; }
  th { border-bottom: 1.5px solid #14213D; }
  a { color: #1F4E8C; }
</style>
</head>
<body>
  <div class="cover">
    <div>
      <div style="font-family:system-ui,sans-serif;letter-spacing:0.18em;text-transform:uppercase;font-size:12px;color:#FCA311;">ENTERPRISE AI ARCHITECT · TRILHA DE 32 SEMANAS</div>
      <h1 style="color:#F5F3EE;border:none;font-family:Georgia,serif;font-weight:400;font-size:38px;line-height:1.14;margin:20px 0 0 0;padding:0;">${escapeHtml(bundle.title)}</h1>
      <div style="width:96px;height:4px;background:#FCA311;margin:28px 0;"></div>
      <div style="font-family:system-ui,sans-serif;font-size:15px;color:#C9CFDB;">${escapeHtml(bundle.scopeLabel)} · ${escapeHtml(bundle.weeksRange)}</div>
      <h2 style="color:#F5F3EE;font-family:Georgia,serif;font-weight:400;font-size:24px;margin:10px 0 0 0;">${escapeHtml(bundle.subtitle)}</h2>
    </div>
    <div>
      <div style="display:grid;grid-template-columns:repeat(8,24px);gap:8px;margin-bottom:24px;">${coverGrid}</div>
      <div style="font-family:system-ui,sans-serif;font-size:14px;letter-spacing:0.06em;">${escapeHtml(bundle.author)}</div>
    </div>
  </div>
  <nav class="toc">
    <h1>Sumário</h1>
    <ol style="font-family:system-ui,sans-serif;font-size:14px;padding-left:20px;">${tocHtml}</ol>
  </nav>
  ${chaptersHtml}
</body>
</html>`;

    const blob = new Blob([fullHtml], { type: 'text/html;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `applied-ai-architecture-${bundle.scope}.html`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  return createPortal(
    <div
      id="pdf-export-portal"
      className="fixed inset-0 z-50 overflow-y-auto bg-slate-900/75 backdrop-blur-xs p-0 sm:p-6"
    >
      {/* Toolbar hidden on print */}
      <div className="no-print sticky top-0 z-10 mx-auto max-w-[900px] mb-4 bg-[#0F172A] text-white rounded-none sm:rounded-xl px-5 py-3.5 shadow-lg flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <div className="text-xs font-mono text-[#FCA311]">
            Pré-visualização para PDF · {bundle.weeksRange}
          </div>
          <div className="text-sm font-semibold truncate">{bundle.subtitle}</div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => window.print()}
            className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-semibold bg-[#FCA311] text-[#0F172A] hover:bg-amber-400 rounded-lg cursor-pointer"
          >
            <Printer className="w-3.5 h-3.5" />
            <span>Salvar como PDF (Imprimir)</span>
          </button>
          <button
            type="button"
            onClick={handleDownloadHtml}
            className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium bg-slate-800 text-slate-200 hover:bg-slate-700 border border-slate-700 rounded-lg cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Baixar HTML</span>
          </button>
          <button
            type="button"
            onClick={onClose}
            aria-label="Fechar pré-visualização"
            className="inline-flex items-center gap-1 px-3 py-2 text-xs font-medium text-slate-300 hover:text-white hover:bg-slate-800 rounded-lg cursor-pointer"
          >
            <X className="w-4 h-4" />
            <span>Fechar</span>
          </button>
        </div>
      </div>

      {/* Printable Sheet */}
      <div
        id="pdf-export-sheet"
        className="mx-auto max-w-[900px] bg-white shadow-2xl sm:rounded-xl overflow-hidden"
      >
        {/* Editorial Cover */}
        <section className="pdf-book-cover p-10 sm:p-14 flex flex-col justify-between min-h-[760px]">
          <div>
            <div className="text-xs font-sans font-semibold tracking-[0.18em] uppercase text-[#FCA311]">
              Enterprise AI Architect · Trilha de 32 semanas
            </div>
            <h1 className="mt-5 font-serif text-3xl sm:text-5xl font-normal leading-tight text-[#F5F3EE]">
              {bundle.title}
            </h1>
            <div className="w-24 h-1 bg-[#FCA311] my-7" />
            <div className="text-sm font-sans text-[#C9CFDB]">
              {bundle.scopeLabel} · {bundle.weeksRange}
            </div>
            <h2 className="mt-2 font-serif text-xl sm:text-2xl font-normal text-[#F5F3EE]">
              {bundle.subtitle}
            </h2>
          </div>

          <div className="mt-14">
            <div className="grid grid-cols-8 gap-2 w-60 mb-6" aria-hidden="true">
              {Array.from({ length: 32 }, (_, idx) => {
                const isOn = activeSet.has(idx + 1);
                return (
                  <span
                    key={idx}
                    className={`aspect-square border-2 ${
                      isOn
                        ? 'bg-[#FCA311] border-[#FCA311]'
                        : 'border-[#3D5A80] bg-transparent'
                    }`}
                  />
                );
              })}
            </div>
            <p className="text-sm font-sans tracking-wider text-[#F5F3EE]">
              {bundle.author}
            </p>
          </div>
        </section>

        {/* Body */}
        <div className="p-8 sm:p-14 pdf-book-document">
          <nav className="pdf-book-toc mb-10">
            <h1>Sumário</h1>
            <ol className="font-sans text-sm space-y-2 list-decimal pl-5 text-slate-800">
              {bundle.chapters.map((ch) => (
                <li key={ch.id}>
                  <a
                    href={`#${ch.id}`}
                    className="!no-underline hover:!underline text-[#14213D]"
                  >
                    {ch.title}
                  </a>
                </li>
              ))}
            </ol>
          </nav>

          {bundle.chapters.map((ch) => (
            <section
              key={ch.id}
              id={ch.id}
              className={`pdf-chapter ${
                ch.kind === 'appendix-divider' ? 'pdf-chapter-appendix' : ''
              }`}
              dangerouslySetInnerHTML={{ __html: ch.html }}
            />
          ))}
        </div>
      </div>
    </div>,
    document.body,
  );
}
