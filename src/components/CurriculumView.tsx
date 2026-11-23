import { useEffect, useState } from 'react';
import type { User } from 'firebase/auth';
import { BookOpen, Check, ClipboardList, Cloud, Code2, FlaskConical, LogIn, Search, X } from 'lucide-react';
import { WEEKLY_STEPS, countCompletedSteps } from './LogbookView';
import type { PhaseInfo, SearchHit, UserProfile, WeekLogState, WeekMeta } from '../types';

interface CurriculumViewProps {
  phases: PhaseInfo[];
  weeks: WeekMeta[];
  completedWeeks: number[];
  weekLogs: Record<number, WeekLogState>;
  user: User | null;
  profile: UserProfile | null;
  onSignIn: () => Promise<void>;
  onOpenAccess: () => void;
  onToggleComplete: (week: number) => void;
  onOpenWeek: (week: number, tab?: 'concept' | 'notebook') => void;
  onOpenLab: (labId: string) => void;
  onOpenLogbook: () => void;
}

type TrackFilter = 'all' | 'rag' | 'finetuning' | 'production' | 'causal';

const TRACK_FILTERS: Array<{ id: TrackFilter; label: string; weeks: Set<number> | null; description: string }> = [
  {
    id: 'all',
    label: 'Todas (32)',
    weeks: null,
    description: 'Trilha completa de 32 semanas em 4 ciclos integradores.',
  },
  {
    id: 'rag',
    label: 'Ciclo 3A · RAG & Search',
    weeks: new Set([1, 2, 3, 4, 6, 7, 9, 11, 12, 13, 14, 15, 16]),
    description: 'Base + Embeddings, HNSW, Vector DBs com ACL, Hybrid Search, NDCG e Faithfulness.',
  },
  {
    id: 'finetuning',
    label: 'Ciclo 3B · Fine-Tuning (PEFT)',
    weeks: new Set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22]),
    description: 'Base + Transformers e adaptação de modelos open-weight com LoRA, QLoRA e DPO.',
  },
  {
    id: 'production',
    label: 'Ciclo 4 · Produção & FinOps',
    weeks: new Set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 31, 32]),
    description: 'Serving, KV-cache, MLflow, guardrails calibrados, agents e governança (ADRs/RFCs/OEC).',
  },
  {
    id: 'causal',
    label: 'Ciclo 4 · Causal & Experimentação',
    weeks: new Set([1, 27, 28, 29, 30, 32]),
    description: 'Pode ser antecipado após o Ciclo 1: A/B clusterizado, CUPED, DiD, PSM, IV, Synthetic Control e OEC.',
  },
];

export default function CurriculumView({
  phases,
  weeks,
  completedWeeks,
  weekLogs,
  user,
  profile,
  onSignIn,
  onOpenAccess,
  onToggleComplete,
  onOpenWeek,
  onOpenLab,
  onOpenLogbook,
}: CurriculumViewProps) {
  const [activeTrack, setActiveTrack] = useState<TrackFilter>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchHits, setSearchHits] = useState<SearchHit[]>([]);
  const [isSearching, setIsSearching] = useState(false);

  const completedSet = new Set(completedWeeks);
  const currentTrack = TRACK_FILTERS.find((t) => t.id === activeTrack)!;

  useEffect(() => {
    const trimmed = searchQuery.trim();
    if (trimmed.length < 2) {
      setSearchHits([]);
      setIsSearching(false);
      return;
    }

    const controller = new AbortController();
    setIsSearching(true);
    const timer = setTimeout(async () => {
      try {
        const res = await fetch(`/api/search?q=${encodeURIComponent(trimmed)}`, {
          signal: controller.signal,
        });
        if (res.ok) {
          const data = (await res.json()) as { hits: SearchHit[] };
          setSearchHits(data.hits);
        }
      } catch {
        // ignore abort
      } finally {
        setIsSearching(false);
      }
    }, 180);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [searchQuery]);

  const filteredWeeks = currentTrack.weeks
    ? weeks.filter((w) => currentTrack.weeks!.has(w.week))
    : weeks;

  const totalStepsDone = weeks.reduce(
    (sum, w) => sum + countCompletedSteps(weekLogs[w.week]),
    0,
  );
  const progressPct = Math.round((completedSet.size / Math.max(weeks.length, 1)) * 100);

  return (
    <div className="mx-auto max-w-[1200px] px-6 py-10 space-y-12">
      {/* Hero + 32-Week Matrix */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-stretch">
        <div className="lg:col-span-7 flex flex-col justify-between bg-white border border-slate-200 rounded-xl p-8">
          <div className="space-y-4">
            <p className="text-xs font-medium text-slate-500 tabular-nums">
              32 semanas · 4 ciclos integradores · Open-weight models · Google Colab T4 &amp; CPU
            </p>
            <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-[#0F172A] [text-wrap:balance]">
              Enterprise AI Architect — Foundations
            </h1>
            <p className="text-base leading-relaxed text-slate-600 max-w-2xl">
              Da matemática clássica e pipelines orientados a custo até Self-Attention, RAG com
              controle de acesso, LoRA/QLoRA, FinOps, Inferência Causal e Governança em produção.
              Acompanhe todos os passos da sua jornada pelo LogBook integrado.
            </p>
          </div>

          <div className="mt-8 pt-6 border-t border-slate-200 flex flex-wrap items-center justify-between gap-4">
            <div className="flex flex-wrap items-center gap-4 text-sm text-slate-600 tabular-nums">
              <div>
                <span className="font-semibold text-[#0F172A]">{completedSet.size}/32</span>{' '}
                <span>semanas</span>
              </div>
              <span aria-hidden="true" className="text-slate-300">·</span>
              <div>
                <span className="font-semibold text-[#0F172A]">
                  {totalStepsDone}/{weeks.length * WEEKLY_STEPS.length}
                </span>{' '}
                <span>passos ({progressPct}%)</span>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2.5">
              <button
                type="button"
                onClick={() => onOpenWeek(1, 'concept')}
                className="px-4 py-2 text-sm font-medium bg-[#14213D] text-white rounded-lg hover:bg-[#1E3158] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                Abrir Semana 1
              </button>
              <button
                type="button"
                onClick={onOpenLogbook}
                className="inline-flex items-center gap-1.5 px-4 py-2 text-sm font-medium text-[#0F172A] bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                <ClipboardList className="w-4 h-4" />
                <span>Abrir LogBook</span>
              </button>
            </div>
          </div>
        </div>

        {/* 8x4 32-Week Matrix inspired by book/assets/cover.html */}
        <div className="lg:col-span-5 bg-[#14213D] text-[#F5F3EE] rounded-xl p-7 flex flex-col justify-between">
          <div className="flex items-baseline justify-between gap-4">
            <div>
              <h2 className="text-base font-semibold text-white">Matriz de Progresso</h2>
              <p className="text-xs text-slate-300 mt-1">
                Clique em qualquer semana para ler o conceito ou acompanhar passos.
              </p>
            </div>
            <span className="text-xs font-mono text-[#FCA311] tabular-nums whitespace-nowrap shrink-0">
              {completedSet.size} / 32
            </span>
          </div>

          <div className="grid grid-cols-8 gap-2 my-6" role="grid" aria-label="Matriz de 32 semanas">
            {weeks.map((w) => {
              const isDone = completedSet.has(w.week);
              const stepsDone = countCompletedSteps(weekLogs[w.week]);
              const isInProgress = !isDone && stepsDone > 0;
              const isInTrack = !currentTrack.weeks || currentTrack.weeks.has(w.week);
              return (
                <button
                  key={w.week}
                  type="button"
                  onClick={() => onOpenWeek(w.week, 'concept')}
                  title={`Semana ${w.week}: ${w.title} (${stepsDone}/6 passos)`}
                  className={`aspect-square rounded border text-xs font-mono tabular-nums flex items-center justify-center transition-colors cursor-pointer ${
                    isDone
                      ? 'bg-[#FCA311] border-[#FCA311] text-[#14213D] font-semibold'
                      : isInProgress
                        ? 'bg-[#1E3A5F] border-[#0EA5E9] text-white font-medium'
                        : isInTrack
                          ? 'border-[#3D5A80] text-slate-200 hover:border-[#FCA311] hover:text-white'
                          : 'border-[#253652] text-slate-500 opacity-45 hover:opacity-80'
                  }`}
                >
                  {String(w.week).padStart(2, '0')}
                </button>
              );
            })}
          </div>

          <div className="flex items-center justify-between text-xs text-slate-300 border-t border-[#294066] pt-4">
            <span>Amarelo: concluída</span>
            <span>Azul: em andamento</span>
            <span className="tabular-nums">~14h / semana</span>
          </div>
        </div>
      </section>

      {/* Community Login / Personal Sync Strip */}
      <section className="bg-white border border-slate-200 rounded-xl px-6 py-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        {!user ? (
          <>
            <div className="flex items-start sm:items-center gap-3 text-sm text-slate-700">
              <Cloud className="w-4 h-4 text-[#0284C7] shrink-0 mt-0.5 sm:mt-0" />
              <span>
                <strong>Estude com sua própria conta:</strong> faça login com o Google para ter seu LogBook individual de 192 passos, horas e rascunhos ADR/RFC salvos na nuvem.
              </span>
            </div>
            <button
              type="button"
              onClick={() => void onSignIn()}
              className="inline-flex items-center gap-2 px-4 py-2 text-xs font-medium bg-[#14213D] text-white rounded-lg hover:bg-[#1E3158] transition-colors whitespace-nowrap shrink-0 cursor-pointer self-start sm:self-auto"
            >
              <LogIn className="w-3.5 h-3.5" />
              <span>Entrar com Google</span>
            </button>
          </>
        ) : (
          <>
            <div className="flex items-start sm:items-center gap-3 text-sm text-slate-700">
              <Cloud className="w-4 h-4 text-[#059669] shrink-0 mt-0.5 sm:mt-0" />
              <span>
                Conectado como <strong>{profile?.displayName || user.email}</strong>
                {' · '}
                {profile?.status === 'approved'
                  ? 'Seu plano de estudos e LogBook estão sincronizados no banco de dados.'
                  : profile?.status === 'pending'
                    ? 'Conta aguardando liberação do administrador.'
                    : 'Acesso restrito pelo administrador.'}
              </span>
            </div>
            <div className="flex items-center gap-2 shrink-0 self-start sm:self-auto">
              {profile?.status === 'approved' && (
                <button
                  type="button"
                  onClick={onOpenLogbook}
                  className="px-3.5 py-1.5 text-xs font-medium bg-[#14213D] text-white rounded-lg hover:bg-[#1E3158] whitespace-nowrap cursor-pointer"
                >
                  Meu LogBook
                </button>
              )}
              <button
                type="button"
                onClick={onOpenAccess}
                className="px-3.5 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg whitespace-nowrap cursor-pointer"
              >
                Conta &amp; Acesso
              </button>
            </div>
          </>
        )}
      </section>

      {/* Recommended 4-Cycle Milestones Overview */}
      <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          {
            title: 'Ciclo 1 · Base Matemática',
            weeksLabel: 'Semanas 1–4',
            milestone: 'Marco: Busca vetorial em NumPy + análise financeira de threshold.',
            targetWeek: 1,
          },
          {
            title: 'Ciclo 2 · Anatomia LLMs',
            weeksLabel: 'Semanas 5–10',
            milestone: 'Marco: Shapes de Attention, Mini-GPT e Encoder sem leakage.',
            targetWeek: 5,
          },
          {
            title: 'Ciclo 3 · RAG & PEFT',
            weeksLabel: 'Semanas 11–22',
            milestone: 'Marco: ADR comparando RAG vs. Fine-Tuning com NDCG e custo.',
            targetWeek: 11,
          },
          {
            title: 'Ciclo 4 · Produção & Causal',
            weeksLabel: 'Semanas 23–32',
            milestone: 'Marco: Capstone reproduzível com FinOps, Guardrails e OEC.',
            targetWeek: 23,
          },
        ].map((cycle) => (
          <button
            key={cycle.title}
            type="button"
            onClick={() => onOpenWeek(cycle.targetWeek, 'concept')}
            className="text-left bg-white border border-slate-200 hover:border-slate-300 rounded-xl p-5 transition-colors cursor-pointer space-y-1.5"
          >
            <div className="text-xs font-mono text-slate-500 tabular-nums">{cycle.weeksLabel}</div>
            <div className="text-sm font-semibold text-[#0F172A]">{cycle.title}</div>
            <p className="text-xs text-slate-600 leading-relaxed">{cycle.milestone}</p>
          </button>
        ))}
      </section>

      {/* Search & Track Filter Toolbar */}
      <section className="space-y-4">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          {/* Interactive Track Filter Segmented Control */}
          <div className="flex flex-wrap items-center gap-1 p-1 bg-slate-100 border border-slate-200 rounded-lg">
            {TRACK_FILTERS.map((track) => {
              const active = track.id === activeTrack;
              return (
                <button
                  key={track.id}
                  type="button"
                  onClick={() => setActiveTrack(track.id)}
                  className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                    active
                      ? 'bg-white text-[#0F172A] shadow-sm'
                      : 'text-slate-600 hover:text-[#0F172A]'
                  }`}
                >
                  {track.label}
                </button>
              );
            })}
          </div>

          {/* Real Full-Text Search across 32 Concept Docs */}
          <div className="relative w-full lg:w-80">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
            <input
              type="search"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Buscar conceitos (ex: LoRA, HNSW, OEC)..."
              aria-label="Buscar conceitos nas 32 semanas"
              className="w-full pl-9 pr-8 py-2 text-sm bg-white border border-slate-200 rounded-lg text-[#0F172A] placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-[#0284C7]"
            />
            {searchQuery && (
              <button
                type="button"
                onClick={() => setSearchQuery('')}
                aria-label="Limpar busca"
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>

        <p className="text-sm text-slate-600">{currentTrack.description}</p>

        {/* Search Results Panel */}
        {searchQuery.trim().length >= 2 && (
          <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-3">
            <div className="flex items-center justify-between text-xs text-slate-500">
              <span>
                {isSearching
                  ? 'Buscando nas 32 leituras de conceito...'
                  : `${searchHits.length} semana(s) encontrada(s) para "${searchQuery.trim()}"`}
              </span>
            </div>
            {!isSearching && searchHits.length === 0 && (
              <p className="text-sm text-slate-600">
                Nenhum trecho encontrado. Tente termos técnicos em inglês (ex: attention, gradient, NDCG, spillover).
              </p>
            )}
            <div className="divide-y divide-slate-100">
              {searchHits.map((hit) => (
                <button
                  key={hit.week}
                  type="button"
                  onClick={() => onOpenWeek(hit.week, 'concept')}
                  className="w-full text-left py-3 hover:bg-slate-50 px-2 rounded-md transition-colors cursor-pointer"
                >
                  <div className="flex items-center justify-between gap-4">
                    <span className="text-sm font-semibold text-[#0F172A]">
                      Semana {String(hit.week).padStart(2, '0')} — {hit.title}
                    </span>
                    <span className="text-xs text-slate-500 tabular-nums whitespace-nowrap shrink-0">
                      Fase {hit.phase} · {hit.matchCount} ocorrência(s)
                    </span>
                  </div>
                  <p className="text-xs text-slate-600 mt-1 line-clamp-2">{hit.snippet}</p>
                </button>
              ))}
            </div>
          </div>
        )}
      </section>

      {/* Phases & Weeks List */}
      <section className="space-y-8">
        {phases.map((phase) => {
          const phaseWeeks = filteredWeeks.filter((w) => w.phase === phase.id);
          if (phaseWeeks.length === 0) return null;

          const doneInPhase = phaseWeeks.filter((w) => completedSet.has(w.week)).length;

          return (
            <div
              key={phase.id}
              className="bg-white border border-slate-200 rounded-xl overflow-hidden"
            >
              <div className="px-6 py-5 bg-slate-50/70 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div>
                  <div className="text-xs text-slate-500 tabular-nums">
                    Fase {phase.id} · {phase.weeksRange}
                  </div>
                  <h2 className="text-lg font-semibold text-[#0F172A] mt-0.5">{phase.title}</h2>
                  <p className="text-sm text-slate-600 mt-0.5">{phase.subtitle}</p>
                </div>
                <div className="text-xs font-mono text-slate-600 tabular-nums whitespace-nowrap shrink-0">
                  {doneInPhase}/{phaseWeeks.length} concluídas
                </div>
              </div>

              <div className="divide-y divide-slate-200">
                {phaseWeeks.map((week) => {
                  const isDone = completedSet.has(week.week);
                  const stepsDone = countCompletedSteps(weekLogs[week.week]);
                  return (
                    <div
                      key={week.week}
                      className="px-6 py-4 flex flex-col md:flex-row md:items-center justify-between gap-4 hover:bg-slate-50/60 transition-colors"
                    >
                      <div className="flex items-start gap-4 min-w-0">
                        <button
                          type="button"
                          onClick={() => onToggleComplete(week.week)}
                          aria-label={
                            isDone
                              ? `Desmarcar semana ${week.week} como concluída`
                              : `Marcar semana ${week.week} como concluída`
                          }
                          className={`mt-0.5 w-6 h-6 rounded border flex items-center justify-center shrink-0 transition-colors cursor-pointer ${
                            isDone
                              ? 'bg-[#059669] border-[#059669] text-white'
                              : 'border-slate-300 bg-white text-transparent hover:border-slate-400'
                          }`}
                        >
                          <Check className="w-3.5 h-3.5" />
                        </button>

                        <div className="min-w-0">
                          <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs text-slate-500 tabular-nums">
                            <span className="font-mono font-medium text-slate-700">
                              Semana {String(week.week).padStart(2, '0')}
                            </span>
                            <span aria-hidden="true">·</span>
                            <span>{week.environment}</span>
                            <span aria-hidden="true">·</span>
                            <span>{week.validationStatus}</span>
                            <span aria-hidden="true">·</span>
                            <span className={stepsDone > 0 ? 'text-[#0284C7] font-medium' : ''}>
                              {stepsDone}/6 passos
                            </span>
                            {isDone && (
                              <>
                                <span aria-hidden="true">·</span>
                                <span className="text-[#059669] font-medium">Concluída</span>
                              </>
                            )}
                          </div>
                          <button
                            type="button"
                            onClick={() => onOpenWeek(week.week, 'concept')}
                            className="text-left text-base font-medium text-[#0F172A] hover:text-[#0284C7] transition-colors mt-0.5 cursor-pointer"
                          >
                            {week.title}
                          </button>
                        </div>
                      </div>

                      <div className="flex items-center gap-2 shrink-0 pl-10 md:pl-0">
                        {week.hasInteractiveLab && week.labId && (
                          <button
                            type="button"
                            onClick={() => onOpenLab(week.labId!)}
                            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-[#0284C7] bg-sky-50 hover:bg-sky-100 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                          >
                            <FlaskConical className="w-3.5 h-3.5" />
                            <span>Lab</span>
                          </button>
                        )}
                        <button
                          type="button"
                          onClick={() => onOpenWeek(week.week, 'concept')}
                          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                        >
                          <BookOpen className="w-3.5 h-3.5" />
                          <span>Conceito</span>
                        </button>
                        <button
                          type="button"
                          onClick={() => onOpenWeek(week.week, 'notebook')}
                          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                        >
                          <Code2 className="w-3.5 h-3.5" />
                          <span>Notebook</span>
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </section>
    </div>
  );
}
