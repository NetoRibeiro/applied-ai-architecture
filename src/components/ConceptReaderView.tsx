import { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  Check,
  ChevronLeft,
  ChevronRight,
  Copy,
  ExternalLink,
  Eye,
  EyeOff,
  FlaskConical,
} from 'lucide-react';
import { WEEKLY_STEPS, countCompletedSteps, createEmptyWeekLog } from './LogbookView';
import type { PhaseInfo, WeekDetail, WeekLogState, WeekMeta, WeeklyStepId } from '../types';

interface ConceptReaderViewProps {
  phases: PhaseInfo[];
  weeks: WeekMeta[];
  selectedWeek: number;
  initialTab: 'concept' | 'notebook';
  completedWeeks: number[];
  weekLogs: Record<number, WeekLogState>;
  onSelectWeek: (week: number) => void;
  onToggleComplete: (week: number) => void;
  onToggleStep: (week: number, step: WeeklyStepId) => void;
  onOpenLab: (labId: string) => void;
}

export default function ConceptReaderView({
  phases,
  weeks,
  selectedWeek,
  initialTab,
  completedWeeks,
  weekLogs,
  onSelectWeek,
  onToggleComplete,
  onToggleStep,
  onOpenLab,
}: ConceptReaderViewProps) {
  const [tab, setTab] = useState<'concept' | 'notebook'>(initialTab);
  const [detail, setDetail] = useState<WeekDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [revealedCells, setRevealedCells] = useState<Record<number, boolean>>({});
  const [revealAll, setRevealAll] = useState(false);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  useEffect(() => {
    setTab(initialTab);
  }, [initialTab]);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    setRevealedCells({});
    setRevealAll(false);

    fetch(`/api/week/${selectedWeek}`)
      .then(async (res) => {
        if (!res.ok) throw new Error('Falha ao carregar os dados desta semana.');
        return (await res.json()) as WeekDetail;
      })
      .then((data) => {
        if (active) {
          setDetail(data);
          setLoading(false);
        }
      })
      .catch((err: unknown) => {
        if (active) {
          setError(err instanceof Error ? err.message : 'Erro desconhecido.');
          setLoading(false);
        }
      });

    return () => {
      active = false;
    };
  }, [selectedWeek]);

  const completedSet = new Set(completedWeeks);
  const isCompleted = completedSet.has(selectedWeek);
  const currentLog = weekLogs[selectedWeek] ?? createEmptyWeekLog();
  const doneSteps = countCompletedSteps(currentLog);

  const handleCopy = async (key: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedKey(key);
      setTimeout(() => setCopiedKey(null), 1600);
    } catch {
      // clipboard unavailable
    }
  };

  return (
    <div className="mx-auto max-w-[1320px] px-6 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
      {/* Left Sidebar: 32-Week Navigator */}
      <aside className="lg:col-span-3 bg-white border border-slate-200 rounded-xl p-4 lg:sticky lg:top-20 max-h-[calc(100vh-7rem)] overflow-y-auto">
        <div className="pb-3 mb-3 border-b border-slate-200 flex items-center justify-between">
          <h2 className="text-xs font-semibold text-slate-700">Índice da Trilha</h2>
          <span className="text-xs font-mono text-slate-500 tabular-nums">
            {completedSet.size}/32
          </span>
        </div>

        <div className="space-y-4">
          {phases.map((phase) => {
            const phaseWeeks = weeks.filter((w) => w.phase === phase.id);
            return (
              <div key={phase.id} className="space-y-1">
                <div className="text-xs font-medium text-slate-500 px-2 py-1">
                  Fase {phase.id} · {phase.title}
                </div>
                {phaseWeeks.map((w) => {
                  const active = w.week === selectedWeek;
                  const done = completedSet.has(w.week);
                  return (
                    <button
                      key={w.week}
                      type="button"
                      onClick={() => onSelectWeek(w.week)}
                      className={`w-full text-left px-2.5 py-2 rounded-lg text-xs flex items-center justify-between gap-2 transition-colors cursor-pointer ${
                        active
                          ? 'bg-[#14213D] text-white font-medium'
                          : 'text-slate-700 hover:bg-slate-100'
                      }`}
                    >
                      <span className="truncate">
                        <span className="font-mono tabular-nums mr-1.5">
                          {String(w.week).padStart(2, '0')}.
                        </span>
                        {w.title}
                      </span>
                      {done && (
                        <Check
                          className={`w-3.5 h-3.5 shrink-0 ${
                            active ? 'text-[#FCA311]' : 'text-[#059669]'
                          }`}
                        />
                      )}
                    </button>
                  );
                })}
              </div>
            );
          })}
        </div>
      </aside>

      {/* Main Content Workspace */}
      <main className="lg:col-span-9 space-y-6">
        {loading && (
          <div className="bg-white border border-slate-200 rounded-xl p-10 text-sm text-slate-500">
            Carregando conteúdo da Semana {selectedWeek}...
          </div>
        )}

        {error && !loading && (
          <div className="bg-white border border-red-200 rounded-xl p-8 text-sm text-red-700">
            {error}
          </div>
        )}

        {detail && !loading && (
          <>
            {/* Week Header & Toolbar */}
            <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-5">
              <div className="flex flex-col md:flex-row md:items-start justify-between gap-4">
                <div className="space-y-1.5">
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-slate-500 tabular-nums">
                    <span className="font-mono font-medium text-slate-700">
                      Semana {String(detail.week).padStart(2, '0')}
                    </span>
                    <span aria-hidden="true">·</span>
                    <span>Fase {detail.phase}: {detail.phaseTitle}</span>
                    <span aria-hidden="true">·</span>
                    <span>{detail.environment}</span>
                    <span aria-hidden="true">·</span>
                    <span>{detail.validationStatus}</span>
                  </div>
                  <h1 className="text-2xl font-bold text-[#0F172A] tracking-tight">
                    {detail.title}
                  </h1>
                  {detail.validationNote && (
                    <p className="text-xs text-slate-500">{detail.validationNote}</p>
                  )}
                </div>

                <button
                  type="button"
                  onClick={() => onToggleComplete(detail.week)}
                  className={`inline-flex items-center gap-2 px-4 py-2 text-xs font-medium rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                    isCompleted
                      ? 'bg-emerald-50 text-[#059669] border border-emerald-200 hover:bg-emerald-100'
                      : 'bg-[#14213D] text-white hover:bg-[#1E3158]'
                  }`}
                >
                  <Check className="w-3.5 h-3.5" />
                  <span>{isCompleted ? 'Semana Concluída' : 'Concluir Todos os Passos'}</span>
                </button>
              </div>

              {/* 6-Step Weekly LogBook Checklist Strip */}
              <div className="pt-4 border-t border-slate-200 space-y-2.5">
                <div className="flex items-center justify-between text-xs text-slate-600 tabular-nums">
                  <span className="font-medium">Controle de Passos da Semana</span>
                  <span className="font-mono">{doneSteps}/6 concluídos</span>
                </div>
                <div className="flex flex-wrap items-center gap-1.5">
                  {WEEKLY_STEPS.map((step) => {
                    const checked = currentLog.steps[step.id];
                    return (
                      <button
                        key={step.id}
                        type="button"
                        onClick={() => onToggleStep(detail.week, step.id)}
                        title={step.fullLabel}
                        className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-md border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                          checked
                            ? 'bg-emerald-50 border-emerald-300 text-[#059669]'
                            : 'bg-slate-50 border-slate-200 text-slate-600 hover:bg-slate-100'
                        }`}
                      >
                        <Check className={`w-3 h-3 ${checked ? 'opacity-100' : 'opacity-25'}`} />
                        <span>{step.shortLabel}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Workspace Controls Row */}
              <div className="pt-4 border-t border-slate-200 flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg">
                  <button
                    type="button"
                    onClick={() => setTab('concept')}
                    className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                      tab === 'concept'
                        ? 'bg-white text-[#0F172A] shadow-sm'
                        : 'text-slate-600 hover:text-[#0F172A]'
                    }`}
                  >
                    Leitura de Conceito
                  </button>
                  <button
                    type="button"
                    onClick={() => setTab('notebook')}
                    className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                      tab === 'notebook'
                        ? 'bg-white text-[#0F172A] shadow-sm'
                        : 'text-slate-600 hover:text-[#0F172A]'
                    }`}
                  >
                    Exercício vs. Gabarito ({detail.exerciseCells.length} células)
                  </button>
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  {detail.hasInteractiveLab && detail.labId && (
                    <button
                      type="button"
                      onClick={() => onOpenLab(detail.labId!)}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-[#0284C7] bg-sky-50 hover:bg-sky-100 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      <FlaskConical className="w-3.5 h-3.5" />
                      <span>Abrir Lab Interativo</span>
                    </button>
                  )}

                  <a
                    href={detail.colabExerciseUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-md transition-colors whitespace-nowrap shrink-0"
                  >
                    <span>Exercício no Colab</span>
                    <ExternalLink className="w-3.5 h-3.5" />
                  </a>

                  <a
                    href={detail.colabGabaritoUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-md transition-colors whitespace-nowrap shrink-0"
                  >
                    <span>Gabarito no Colab</span>
                    <ExternalLink className="w-3.5 h-3.5" />
                  </a>
                </div>
              </div>
            </div>

            {/* Concept Reading Tab */}
            {tab === 'concept' && (
              <article className="bg-white border border-slate-200 rounded-xl p-6 sm:p-10">
                <div className="reading-prose mx-auto">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {detail.conceptMarkdown}
                  </ReactMarkdown>
                </div>
              </article>
            )}

            {/* Notebook Exercise vs Gabarito Tab */}
            {tab === 'notebook' && (
              <div className="space-y-4">
                <div className="bg-white border border-slate-200 rounded-xl px-6 py-4 flex flex-wrap items-center justify-between gap-4">
                  <div className="text-sm text-slate-600">
                    Compare as células de exercício (
                    <code className="text-xs bg-slate-100 px-1.5 py-0.5 rounded"># [SEU CÓDIGO AQUI]</code>
                    ) com o gabarito resolvido antes de executar no Colab.
                  </div>
                  <button
                    type="button"
                    onClick={() => setRevealAll((prev) => !prev)}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-[#14213D] text-white rounded-md hover:bg-[#1E3158] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                  >
                    {revealAll ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                    <span>
                      {revealAll ? 'Ocultar Todos os Gabaritos' : 'Revelar Todos os Gabaritos'}
                    </span>
                  </button>
                </div>

                {detail.exerciseCells.map((exCell, idx) => {
                  const gabCell = detail.gabaritoCells[idx];
                  const isRevealed = revealAll || Boolean(revealedCells[idx]);
                  const hasDiff =
                    exCell.cellType === 'code' &&
                    gabCell &&
                    gabCell.source.trim() !== exCell.source.trim();

                  if (exCell.cellType === 'markdown') {
                    return (
                      <div
                        key={exCell.id}
                        className="bg-white border border-slate-200 rounded-xl px-6 py-5"
                      >
                        <div className="reading-prose max-w-none text-sm">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {exCell.source}
                          </ReactMarkdown>
                        </div>
                      </div>
                    );
                  }

                  return (
                    <div
                      key={exCell.id}
                      className="bg-white border border-slate-200 rounded-xl overflow-hidden"
                    >
                      <div className="px-5 py-3 bg-slate-50 border-b border-slate-200 flex items-center justify-between gap-4">
                        <div className="flex items-center gap-2 text-xs text-slate-600 tabular-nums">
                          <span className="font-mono font-medium">Célula {idx + 1}</span>
                          {exCell.hasTodo && (
                            <>
                              <span aria-hidden="true">·</span>
                              <span className="text-[#D97706] font-medium">
                                Contém # [SEU CÓDIGO AQUI]
                              </span>
                            </>
                          )}
                        </div>

                        <div className="flex items-center gap-2">
                          {hasDiff && (
                            <button
                              type="button"
                              onClick={() =>
                                setRevealedCells((prev) => ({
                                  ...prev,
                                  [idx]: !prev[idx],
                                }))
                              }
                              className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium text-[#0284C7] bg-sky-50 hover:bg-sky-100 rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                            >
                              {isRevealed ? (
                                <EyeOff className="w-3.5 h-3.5" />
                              ) : (
                                <Eye className="w-3.5 h-3.5" />
                              )}
                              <span>{isRevealed ? 'Ocultar Gabarito' : 'Ver Gabarito'}</span>
                            </button>
                          )}
                          <button
                            type="button"
                            onClick={() =>
                              handleCopy(
                                `cell-${idx}`,
                                isRevealed && gabCell ? gabCell.source : exCell.source,
                              )
                            }
                            className="inline-flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-slate-600 bg-white border border-slate-200 hover:bg-slate-100 rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                          >
                            <Copy className="w-3 h-3" />
                            <span>{copiedKey === `cell-${idx}` ? 'Copiado' : 'Copiar'}</span>
                          </button>
                        </div>
                      </div>

                      <div
                        className={
                          isRevealed && hasDiff
                            ? 'grid grid-cols-1 xl:grid-cols-2 divide-y xl:divide-y-0 xl:divide-x divide-slate-800'
                            : ''
                        }
                      >
                        <div className="bg-[#0F172A] text-slate-100 p-4 overflow-x-auto">
                          {isRevealed && hasDiff && (
                            <div className="text-[11px] font-mono text-amber-300 mb-2">
                              Exercício
                            </div>
                          )}
                          <pre className="text-xs leading-relaxed font-mono">
                            <code>{exCell.source}</code>
                          </pre>
                        </div>

                        {isRevealed && hasDiff && gabCell && (
                          <div className="bg-[#0B1324] text-emerald-100 p-4 overflow-x-auto">
                            <div className="text-[11px] font-mono text-emerald-400 mb-2">
                              Gabarito Resolvido
                            </div>
                            <pre className="text-xs leading-relaxed font-mono">
                              <code>{gabCell.source}</code>
                            </pre>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Bottom Week Navigation */}
            <div className="flex items-center justify-between pt-2">
              <button
                type="button"
                disabled={selectedWeek <= 1}
                onClick={() => onSelectWeek(selectedWeek - 1)}
                className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-medium bg-white border border-slate-200 rounded-lg text-slate-700 hover:bg-slate-50 disabled:opacity-40 disabled:pointer-events-none cursor-pointer"
              >
                <ChevronLeft className="w-4 h-4" />
                <span>Semana Anterior</span>
              </button>

              <button
                type="button"
                disabled={selectedWeek >= 32}
                onClick={() => onSelectWeek(selectedWeek + 1)}
                className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-medium bg-white border border-slate-200 rounded-lg text-slate-700 hover:bg-slate-50 disabled:opacity-40 disabled:pointer-events-none cursor-pointer"
              >
                <span>Próxima Semana</span>
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
