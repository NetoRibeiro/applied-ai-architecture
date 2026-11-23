import { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Copy, Download } from 'lucide-react';
import type { TemplatesPayload } from '../types';

export default function GovernanceView() {
  const [tab, setTab] = useState<'adr' | 'rfc'>('adr');
  const [templates, setTemplates] = useState<TemplatesPayload | null>(null);
  const [adrDraft, setAdrDraft] = useState('');
  const [rfcDraft, setRfcDraft] = useState('');
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    fetch('/api/templates')
      .then((res) => res.json())
      .then((data: TemplatesPayload) => {
        setTemplates(data);
        setAdrDraft(data.adrTemplate);
        setRfcDraft(data.rfcTemplate);
      })
      .catch(() => {
        // handled via empty draft fallback
      });
  }, []);

  const currentMarkdown = tab === 'adr' ? adrDraft : rfcDraft;

  const handleCopy = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // ignore clipboard errors
    }
  };

  const handleDownload = (filename: string, content: string) => {
    const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="mx-auto max-w-[1280px] px-6 py-8 space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-[#0F172A] tracking-tight">
            Governança: ADR &amp; RFC Builders
          </h1>
          <p className="text-sm text-slate-600 mt-1">
            Modelos oficiais da Fase 7 (Semana 31) para documentar decisões arquiteturais e exportar ao seu repositório.
          </p>
        </div>

        <div className="flex items-center gap-1 p-1 bg-slate-100 border border-slate-200 rounded-lg self-start">
          <button
            type="button"
            onClick={() => setTab('adr')}
            className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              tab === 'adr' ? 'bg-white text-[#0F172A] shadow-sm' : 'text-slate-600 hover:text-[#0F172A]'
            }`}
          >
            ADR Builder
          </button>
          <button
            type="button"
            onClick={() => setTab('rfc')}
            className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              tab === 'rfc' ? 'bg-white text-[#0F172A] shadow-sm' : 'text-slate-600 hover:text-[#0F172A]'
            }`}
          >
            RFC Builder
          </button>
        </div>
      </div>

      <div className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3 bg-white border border-slate-200 rounded-xl px-5 py-3.5">
          <span className="text-xs text-slate-600">
            Fonte:{' '}
            <code className="font-mono bg-slate-100 px-1.5 py-0.5 rounded">
              07-Phase-7-Governance-and-Scale/docs/templates/{tab.toUpperCase()}-template.md
            </code>
          </span>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => {
                if (!templates) return;
                if (tab === 'adr') setAdrDraft(templates.adrTemplate);
                else setRfcDraft(templates.rfcTemplate);
              }}
              className="px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-md cursor-pointer"
            >
              Restaurar Modelo
            </button>
            <button
              type="button"
              onClick={() => handleCopy(currentMarkdown)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-md cursor-pointer"
            >
              <Copy className="w-3.5 h-3.5" />
              <span>{copied ? 'Copiado' : 'Copiar Markdown'}</span>
            </button>
            <button
              type="button"
              onClick={() => handleDownload(`${tab}-deliverable.md`, currentMarkdown)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-[#14213D] text-white hover:bg-[#1E3158] rounded-md cursor-pointer"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Baixar .md</span>
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">
          <div className="bg-white border border-slate-200 rounded-xl p-4 flex flex-col">
            <div className="text-xs font-semibold text-slate-700 mb-2">Editor Markdown</div>
            <textarea
              value={currentMarkdown}
              onChange={(e) =>
                tab === 'adr' ? setAdrDraft(e.target.value) : setRfcDraft(e.target.value)
              }
              rows={22}
              className="w-full flex-1 font-mono text-xs leading-relaxed p-3 bg-slate-50 border border-slate-200 rounded-lg text-[#0F172A] focus:outline-none focus:ring-2 focus:ring-[#0284C7]"
            />
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-6 overflow-y-auto max-h-[600px]">
            <div className="text-xs font-semibold text-slate-700 mb-3 pb-2 border-b border-slate-200">
              Pré-visualização
            </div>
            <div className="reading-prose text-sm">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{currentMarkdown}</ReactMarkdown>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
