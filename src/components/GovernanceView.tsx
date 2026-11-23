import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Cloud, Copy, Download, Lock } from 'lucide-react';
import type { TemplatesPayload } from '../types';

interface GovernanceViewProps {
  canPersistToCloud: boolean;
  cloudAdrDraft?: string;
  cloudRfcDraft?: string;
  onSaveDraft: (kind: 'adr' | 'rfc', content: string) => Promise<void>;
  onOpenAccess: () => void;
}

export default function GovernanceView({
  canPersistToCloud,
  cloudAdrDraft,
  cloudRfcDraft,
  onSaveDraft,
  onOpenAccess,
}: GovernanceViewProps) {
  const [tab, setTab] = useState<'adr' | 'rfc'>('adr');
  const [templates, setTemplates] = useState<TemplatesPayload | null>(null);
  const [adrDraft, setAdrDraft] = useState('');
  const [rfcDraft, setRfcDraft] = useState('');
  const [copied, setCopied] = useState(false);
  const [saveState, setSaveState] = useState<'idle' | 'saving' | 'saved'>('idle');
  const saveTimerRef = useRef<number | null>(null);

  useEffect(() => {
    fetch('/api/templates')
      .then((res) => res.json())
      .then((data: TemplatesPayload) => {
        setTemplates(data);
        setAdrDraft((prev) => prev || cloudAdrDraft || data.adrTemplate);
        setRfcDraft((prev) => prev || cloudRfcDraft || data.rfcTemplate);
      })
      .catch(() => {
        // handled via empty draft fallback
      });
  }, [cloudAdrDraft, cloudRfcDraft]);

  useEffect(() => {
    if (canPersistToCloud && cloudAdrDraft !== undefined) {
      setAdrDraft(cloudAdrDraft);
    }
  }, [canPersistToCloud, cloudAdrDraft]);

  useEffect(() => {
    if (canPersistToCloud && cloudRfcDraft !== undefined) {
      setRfcDraft(cloudRfcDraft);
    }
  }, [canPersistToCloud, cloudRfcDraft]);

  useEffect(() => {
    return () => {
      if (saveTimerRef.current) {
        window.clearTimeout(saveTimerRef.current);
      }
    };
  }, []);

  const currentMarkdown = tab === 'adr' ? adrDraft : rfcDraft;

  const scheduleCloudSave = (kind: 'adr' | 'rfc', nextContent: string) => {
    if (!canPersistToCloud) return;
    setSaveState('saving');
    if (saveTimerRef.current) {
      window.clearTimeout(saveTimerRef.current);
    }
    saveTimerRef.current = window.setTimeout(async () => {
      try {
        await onSaveDraft(kind, nextContent);
        setSaveState('saved');
      } catch {
        setSaveState('idle');
      }
    }, 550);
  };

  const handleDraftChange = (value: string) => {
    if (tab === 'adr') {
      setAdrDraft(value);
      scheduleCloudSave('adr', value);
    } else {
      setRfcDraft(value);
      scheduleCloudSave('rfc', value);
    }
  };

  const handleRestore = () => {
    if (!templates) return;
    const restored = tab === 'adr' ? templates.adrTemplate : templates.rfcTemplate;
    handleDraftChange(restored);
  };

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

      {!canPersistToCloud ? (
        <div className="bg-slate-50 border border-slate-200 rounded-xl px-5 py-3.5 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs text-slate-700">
          <div className="flex items-center gap-2">
            <Lock className="w-4 h-4 text-slate-500 shrink-0" />
            <span>
              Modo de visualização local: entre com uma conta aprovada para salvar seus rascunhos de ADR/RFC automaticamente no banco Firestore.
            </span>
          </div>
          <button
            type="button"
            onClick={onOpenAccess}
            className="px-3 py-1.5 font-medium bg-[#14213D] text-white rounded-md hover:bg-[#1E3158] whitespace-nowrap shrink-0 cursor-pointer self-start sm:self-auto"
          >
            Conta &amp; Acesso
          </button>
        </div>
      ) : (
        <div className="bg-white border border-slate-200 rounded-xl px-5 py-2.5 flex items-center justify-between text-xs text-slate-600">
          <div className="flex items-center gap-2">
            <Cloud className="w-4 h-4 text-[#0284C7]" />
            <span>Rascunhos sincronizados na sua conta (Firestore)</span>
          </div>
          <span className="font-mono text-slate-500">
            {saveState === 'saving'
              ? 'Salvando...'
              : saveState === 'saved'
                ? 'Salvo na nuvem'
                : 'Sincronizado'}
          </span>
        </div>
      )}

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
              onClick={handleRestore}
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
              onChange={(e) => handleDraftChange(e.target.value)}
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

