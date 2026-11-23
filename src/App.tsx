import { useEffect, useMemo, useState } from 'react';
import CurriculumView from './components/CurriculumView';
import ConceptReaderView from './components/ConceptReaderView';
import InteractiveLabsView from './components/InteractiveLabsView';
import LogbookView, {
  WEEKLY_STEPS,
  countCompletedSteps,
  createEmptyWeekLog,
} from './components/LogbookView';
import GovernanceView from './components/GovernanceView';
import type { LogbookEntry, PhaseInfo, WeekLogState, WeekMeta, WeeklyStepId } from './types';

type ViewMode = 'curriculum' | 'reader' | 'labs' | 'logbook' | 'governance';

const STORAGE_WEEK_LOGS_KEY = 'ai_architect_week_logs_v2';
const STORAGE_COMPLETED_LEGACY_KEY = 'ai_architect_completed_weeks_v1';
const STORAGE_LOGBOOK_KEY = 'ai_architect_logbook_v2';

function loadInitialWeekLogs(): Record<number, WeekLogState> {
  try {
    const rawV2 = localStorage.getItem(STORAGE_WEEK_LOGS_KEY);
    if (rawV2) {
      return JSON.parse(rawV2) as Record<number, WeekLogState>;
    }
    const rawLegacy = localStorage.getItem(STORAGE_COMPLETED_LEGACY_KEY);
    if (rawLegacy) {
      const legacyWeeks = JSON.parse(rawLegacy) as number[];
      const migrated: Record<number, WeekLogState> = {};
      for (const w of legacyWeeks) {
        migrated[w] = {
          steps: {
            reading: true,
            self_check: true,
            exercise: true,
            gabarito: true,
            deliverable: true,
            commit: true,
          },
          hoursSpent: 0,
          lastUpdated: '',
          deliverableSummary: '',
          notes: '',
          blockers: '',
        };
      }
      return migrated;
    }
  } catch {
    // ignore storage parse error
  }
  return {};
}

export default function App() {
  const [view, setView] = useState<ViewMode>('curriculum');
  const [phases, setPhases] = useState<PhaseInfo[]>([]);
  const [weeks, setWeeks] = useState<WeekMeta[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedWeek, setSelectedWeek] = useState(1);
  const [readerTab, setReaderTab] = useState<'concept' | 'notebook'>('concept');
  const [activeLabId, setActiveLabId] = useState('threshold-cost');

  const [weekLogs, setWeekLogs] = useState<Record<number, WeekLogState>>(loadInitialWeekLogs);

  const [logEntries, setLogEntries] = useState<LogbookEntry[]>(() => {
    try {
      const raw = localStorage.getItem(STORAGE_LOGBOOK_KEY);
      return raw ? (JSON.parse(raw) as LogbookEntry[]) : [];
    } catch {
      return [];
    }
  });

  useEffect(() => {
    fetch('/api/curriculum')
      .then(async (res) => {
        if (!res.ok) throw new Error('Não foi possível carregar a trilha de estudos.');
        return (await res.json()) as { phases: PhaseInfo[]; weeks: WeekMeta[] };
      })
      .then((data) => {
        setPhases(data.phases);
        setWeeks(data.weeks);
        setLoading(false);
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : 'Erro ao carregar dados.');
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_WEEK_LOGS_KEY, JSON.stringify(weekLogs));
    } catch {
      // ignore storage errors
    }
  }, [weekLogs]);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_LOGBOOK_KEY, JSON.stringify(logEntries));
    } catch {
      // ignore storage errors
    }
  }, [logEntries]);

  const completedWeeks = useMemo(() => {
    return Object.entries(weekLogs)
      .filter(([, state]) => countCompletedSteps(state) === WEEKLY_STEPS.length)
      .map(([w]) => Number(w))
      .sort((a, b) => a - b);
  }, [weekLogs]);

  const toggleCompleteWeek = (week: number) => {
    setWeekLogs((prev) => {
      const current = prev[week] ?? createEmptyWeekLog();
      const allDone = countCompletedSteps(current) === WEEKLY_STEPS.length;
      const nextDone = !allDone;
      return {
        ...prev,
        [week]: {
          ...current,
          lastUpdated: new Date().toISOString().slice(0, 10),
          steps: {
            reading: nextDone,
            self_check: nextDone,
            exercise: nextDone,
            gabarito: nextDone,
            deliverable: nextDone,
            commit: nextDone,
          },
        },
      };
    });
  };

  const toggleStep = (week: number, step: WeeklyStepId) => {
    setWeekLogs((prev) => {
      const current = prev[week] ?? createEmptyWeekLog();
      return {
        ...prev,
        [week]: {
          ...current,
          lastUpdated: new Date().toISOString().slice(0, 10),
          steps: {
            ...current.steps,
            [step]: !current.steps[step],
          },
        },
      };
    });
  };

  const updateWeekLog = (week: number, patch: Partial<WeekLogState>) => {
    setWeekLogs((prev) => {
      const current = prev[week] ?? createEmptyWeekLog();
      return {
        ...prev,
        [week]: {
          ...current,
          ...patch,
        },
      };
    });
  };

  const openWeek = (week: number, tab: 'concept' | 'notebook' = 'concept') => {
    setSelectedWeek(week);
    setReaderTab(tab);
    setView('reader');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const openLab = (labId: string) => {
    setActiveLabId(labId);
    setView('labs');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const nextWeek = weeks.find((w) => !completedWeeks.includes(w.week))?.week ?? 1;

  const addLogEntry = (entry: Omit<LogbookEntry, 'id'>) => {
    setLogEntries((prev) => [
      { ...entry, id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}` },
      ...prev,
    ]);
  };

  const deleteLogEntry = (id: string) => {
    setLogEntries((prev) => prev.filter((e) => e.id !== id));
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#F8FAFC] text-[#0F172A]">
      {/* Strict 3-Zone Top Bar Contract */}
      <header className="sticky top-0 z-30 bg-white/95 backdrop-blur border-b border-slate-200">
        <div className="mx-auto max-w-[1320px] px-6 py-3.5 flex items-center justify-between gap-8">
          {/* Zone 1: Single text element wordmark */}
          <button
            type="button"
            onClick={() => setView('curriculum')}
            className="text-base sm:text-lg font-bold tracking-tight text-[#14213D] whitespace-nowrap shrink-0 cursor-pointer"
          >
            AI Architect Foundations
          </button>

          {/* Zone 2: 5 concise single-line nav links */}
          <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-slate-600">
            <button
              type="button"
              onClick={() => setView('curriculum')}
              className={`hover:text-[#0F172A] transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                view === 'curriculum'
                  ? 'text-[#0F172A] underline underline-offset-8 decoration-2 decoration-[#FCA311]'
                  : ''
              }`}
            >
              Currículo
            </button>
            <button
              type="button"
              onClick={() => setView('reader')}
              className={`hover:text-[#0F172A] transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                view === 'reader'
                  ? 'text-[#0F172A] underline underline-offset-8 decoration-2 decoration-[#FCA311]'
                  : ''
              }`}
            >
              Leitura &amp; Notebooks
            </button>
            <button
              type="button"
              onClick={() => setView('labs')}
              className={`hover:text-[#0F172A] transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                view === 'labs'
                  ? 'text-[#0F172A] underline underline-offset-8 decoration-2 decoration-[#FCA311]'
                  : ''
              }`}
            >
              Labs Interativos
            </button>
            <button
              type="button"
              onClick={() => setView('logbook')}
              className={`hover:text-[#0F172A] transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                view === 'logbook'
                  ? 'text-[#0F172A] underline underline-offset-8 decoration-2 decoration-[#FCA311]'
                  : ''
              }`}
            >
              LogBook
            </button>
            <button
              type="button"
              onClick={() => setView('governance')}
              className={`hover:text-[#0F172A] transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                view === 'governance'
                  ? 'text-[#0F172A] underline underline-offset-8 decoration-2 decoration-[#FCA311]'
                  : ''
              }`}
            >
              Governança
            </button>
          </nav>

          {/* Zone 3: 1 primary action */}
          <div className="shrink-0">
            <button
              type="button"
              onClick={() => openWeek(nextWeek, 'concept')}
              className="px-4 py-2 text-xs font-medium text-white bg-[#14213D] rounded-lg hover:bg-[#1E3158] transition-colors whitespace-nowrap shrink-0 cursor-pointer tabular-nums"
            >
              Continuar Semana {String(nextWeek).padStart(2, '0')}
            </button>
          </div>
        </div>

        {/* Mobile navigation row */}
        <div className="md:hidden border-t border-slate-100 px-4 py-2 flex items-center gap-4 overflow-x-auto text-xs font-medium text-slate-600">
          <button
            type="button"
            onClick={() => setView('curriculum')}
            className={`whitespace-nowrap shrink-0 py-1 ${view === 'curriculum' ? 'text-[#0F172A] font-semibold' : ''}`}
          >
            Currículo
          </button>
          <button
            type="button"
            onClick={() => setView('reader')}
            className={`whitespace-nowrap shrink-0 py-1 ${view === 'reader' ? 'text-[#0F172A] font-semibold' : ''}`}
          >
            Leitura &amp; Notebooks
          </button>
          <button
            type="button"
            onClick={() => setView('labs')}
            className={`whitespace-nowrap shrink-0 py-1 ${view === 'labs' ? 'text-[#0F172A] font-semibold' : ''}`}
          >
            Labs Interativos
          </button>
          <button
            type="button"
            onClick={() => setView('logbook')}
            className={`whitespace-nowrap shrink-0 py-1 ${view === 'logbook' ? 'text-[#0F172A] font-semibold' : ''}`}
          >
            LogBook
          </button>
          <button
            type="button"
            onClick={() => setView('governance')}
            className={`whitespace-nowrap shrink-0 py-1 ${view === 'governance' ? 'text-[#0F172A] font-semibold' : ''}`}
          >
            Governança
          </button>
        </div>
      </header>

      <div className="flex-1">
        {loading && (
          <div className="mx-auto max-w-[1200px] px-6 py-16 text-sm text-slate-500">
            Carregando trilha de 32 semanas...
          </div>
        )}

        {error && !loading && (
          <div className="mx-auto max-w-[1200px] px-6 py-16">
            <div className="bg-white border border-red-200 rounded-xl p-6 text-sm text-red-700">
              {error}
            </div>
          </div>
        )}

        {!loading && !error && (
          <>
            {view === 'curriculum' && (
              <CurriculumView
                phases={phases}
                weeks={weeks}
                completedWeeks={completedWeeks}
                weekLogs={weekLogs}
                onToggleComplete={toggleCompleteWeek}
                onOpenWeek={openWeek}
                onOpenLab={openLab}
                onOpenLogbook={() => setView('logbook')}
              />
            )}

            {view === 'reader' && (
              <ConceptReaderView
                phases={phases}
                weeks={weeks}
                selectedWeek={selectedWeek}
                initialTab={readerTab}
                completedWeeks={completedWeeks}
                weekLogs={weekLogs}
                onSelectWeek={(w) => setSelectedWeek(w)}
                onToggleComplete={toggleCompleteWeek}
                onToggleStep={toggleStep}
                onOpenLab={openLab}
              />
            )}

            {view === 'labs' && (
              <InteractiveLabsView
                activeLabId={activeLabId}
                onSelectLab={setActiveLabId}
                onOpenWeek={(w) => openWeek(w, 'concept')}
              />
            )}

            {view === 'logbook' && (
              <LogbookView
                weeks={weeks}
                weekLogs={weekLogs}
                logEntries={logEntries}
                onToggleStep={toggleStep}
                onUpdateWeekLog={updateWeekLog}
                onAddLogEntry={addLogEntry}
                onDeleteLogEntry={deleteLogEntry}
                onOpenWeek={openWeek}
              />
            )}

            {view === 'governance' && <GovernanceView />}
          </>
        )}
      </div>

      <footer className="border-t border-slate-200 bg-white mt-12">
        <div className="mx-auto max-w-[1320px] px-6 py-6 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-500">
          <span>Enterprise AI Architect — Foundations · Trilha aberta de 32 semanas (Apache 2.0)</span>
          <a
            href="https://github.com/NetoRibeiro/ai-architect-foundations"
            target="_blank"
            rel="noreferrer"
            className="text-slate-600 hover:text-[#0F172A] underline underline-offset-2"
          >
            Repositório original no GitHub
          </a>
        </div>
      </footer>
    </div>
  );
}
