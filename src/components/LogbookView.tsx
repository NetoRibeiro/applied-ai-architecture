import { useState } from 'react';
import type { User } from 'firebase/auth';
import {
  BookOpen,
  Check,
  ChevronDown,
  ChevronUp,
  Cloud,
  Code2,
  Copy,
  Download,
  Lock,
  LogIn,
  Plus,
  Trash2,
  UploadCloud,
} from 'lucide-react';
import type { LogbookEntry, UserProfile, WeekLogState, WeekMeta, WeeklyStepId } from '../types';

export const WEEKLY_STEPS: Array<{ id: WeeklyStepId; shortLabel: string; fullLabel: string }> = [
  { id: 'reading', shortLabel: 'Leitura', fullLabel: '1. Leitura de conceito (sem código)' },
  { id: 'self_check', shortLabel: 'Autoaval.', fullLabel: '2. Perguntas de autoavaliação' },
  { id: 'exercise', shortLabel: 'Exercício', fullLabel: '3. Notebook de exercícios no Colab' },
  { id: 'gabarito', shortLabel: 'Gabarito', fullLabel: '4. Comparação com gabarito' },
  { id: 'deliverable', shortLabel: 'Entregável', fullLabel: '5. Entregável escrito em docs/' },
  { id: 'commit', shortLabel: 'Commit', fullLabel: '6. Commit e registro no GitHub' },
];

export function createEmptyWeekLog(): WeekLogState {
  return {
    steps: {
      reading: false,
      self_check: false,
      exercise: false,
      gabarito: false,
      deliverable: false,
      commit: false,
    },
    hoursSpent: 0,
    lastUpdated: '',
    deliverableSummary: '',
    notes: '',
    blockers: '',
  };
}

export function countCompletedSteps(state?: WeekLogState): number {
  if (!state) return 0;
  return WEEKLY_STEPS.filter((s) => state.steps[s.id]).length;
}

function getCycleLabel(week: number): string {
  if (week <= 4) return 'Ciclo 1 · Base Matemática & PyTorch';
  if (week <= 10) return 'Ciclo 2 · Anatomia dos LLMs';
  if (week <= 16) return 'Ciclo 3A · RAG & Search';
  if (week <= 22) return 'Ciclo 3B · PEFT & Alinhamento';
  return 'Ciclo 4 · Produção, Causalidade & Governança';
}

interface LogbookViewProps {
  weeks: WeekMeta[];
  weekLogs: Record<number, WeekLogState>;
  logEntries: LogbookEntry[];
  firebaseConfigured: boolean;
  user: User | null;
  profile: UserProfile | null;
  canPersistToCloud: boolean;
  hasLocalMigration: boolean;
  migrating: boolean;
  onMigrateLocal: () => Promise<void>;
  onSignIn: () => Promise<void>;
  onOpenAccess: () => void;
  onToggleStep: (week: number, step: WeeklyStepId) => void;
  onUpdateWeekLog: (week: number, patch: Partial<WeekLogState>) => void;
  onAddLogEntry: (entry: Omit<LogbookEntry, 'id'>) => void;
  onDeleteLogEntry: (id: string) => void;
  onOpenWeek: (week: number, tab?: 'concept' | 'notebook') => void;
}

type StatusFilter = 'all' | 'in_progress' | 'completed' | 'pending';

export default function LogbookView({
  weeks,
  weekLogs,
  logEntries,
  firebaseConfigured,
  user,
  profile,
  canPersistToCloud,
  hasLocalMigration,
  migrating,
  onMigrateLocal,
  onSignIn,
  onOpenAccess,
  onToggleStep,
  onUpdateWeekLog,
  onAddLogEntry,
  onDeleteLogEntry,
  onOpenWeek,
}: LogbookViewProps) {
  const [subTab, setSubTab] = useState<'steps' | 'journal'>('steps');
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');
  const [cycleFilter, setCycleFilter] = useState<string>('all');
  const [expandedWeek, setExpandedWeek] = useState<number | null>(1);
  const [copiedType, setCopiedType] = useState<'logbook' | 'checklist' | null>(null);

  // New journal entry form
  const [entryWeek, setEntryWeek] = useState(1);
  const [entryTitle, setEntryTitle] = useState('');
  const [entryHours, setEntryHours] = useState('2.0');
  const [entryNotes, setEntryNotes] = useState('');
  const [entryBlockers, setEntryBlockers] = useState('');
  const [entryCommand, setEntryCommand] = useState('');

  if (firebaseConfigured && !canPersistToCloud) {
    return (
      <div className="mx-auto max-w-[900px] px-6 py-12 space-y-6">
        <div className="bg-white border border-slate-200 rounded-xl p-8 space-y-5">
          <div className="flex items-center gap-3 text-[#14213D]">
            <Lock className="w-5 h-5" />
            <h1 className="text-xl font-bold tracking-tight">
              LogBook com Acesso Controlado
            </h1>
          </div>

          {!user ? (
            <>
              <p className="text-sm text-slate-600 leading-relaxed">
                O material didático e os labs interativos são abertos. Para registrar seus 192 passos, horas de estudo e histórico cronológico no banco de dados Firestore, faça login com sua conta Google.
              </p>
              <button
                type="button"
                onClick={() => void onSignIn()}
                className="inline-flex items-center gap-2 px-5 py-2.5 text-sm font-medium bg-[#14213D] text-white rounded-lg hover:bg-[#1E3158] cursor-pointer"
              >
                <LogIn className="w-4 h-4" />
                <span>Entrar com Google</span>
              </button>
            </>
          ) : profile?.status === 'pending' ? (
            <>
              <p className="text-sm text-slate-600 leading-relaxed">
                Sua conta <strong>{user.email}</strong> está aguardando aprovação do administrador para gravação no banco de dados. Assim que aprovada, seu LogBook será liberado automaticamente.
              </p>
              <button
                type="button"
                onClick={onOpenAccess}
                className="px-4 py-2 text-xs font-medium bg-[#14213D] text-white rounded-lg hover:bg-[#1E3158] cursor-pointer"
              >
                Ver Status da Conta
              </button>
            </>
          ) : (
            <>
              <p className="text-sm text-slate-600 leading-relaxed">
                O acesso ao banco de dados para <strong>{user.email}</strong> está suspenso ou ainda carregando permissões.
              </p>
              <button
                type="button"
                onClick={onOpenAccess}
                className="px-4 py-2 text-xs font-medium bg-[#14213D] text-white rounded-lg hover:bg-[#1E3158] cursor-pointer"
              >
                Gerenciar Conta &amp; Acesso
              </button>
            </>
          )}
        </div>
      </div>
    );
  }

  const totalSteps = weeks.length * WEEKLY_STEPS.length;
  const completedStepsCount = weeks.reduce(
    (sum, w) => sum + countCompletedSteps(weekLogs[w.week]),
    0,
  );
  const completedWeeksCount = weeks.filter(
    (w) => countCompletedSteps(weekLogs[w.week]) === WEEKLY_STEPS.length,
  ).length;
  const totalHoursFromWeeks = weeks.reduce(
    (sum, w) => sum + (weekLogs[w.week]?.hoursSpent ?? 0),
    0,
  );
  const totalHoursFromJournal = logEntries.reduce((sum, e) => sum + (e.hours || 0), 0);
  const totalHours = Math.max(totalHoursFromWeeks, totalHoursFromJournal);

  const filteredWeeks = weeks.filter((w) => {
    const doneSteps = countCompletedSteps(weekLogs[w.week]);
    if (statusFilter === 'completed' && doneSteps !== WEEKLY_STEPS.length) return false;
    if (statusFilter === 'in_progress' && (doneSteps === 0 || doneSteps === WEEKLY_STEPS.length))
      return false;
    if (statusFilter === 'pending' && doneSteps > 0) return false;

    if (cycleFilter === 'c1' && w.week > 4) return false;
    if (cycleFilter === 'c2' && (w.week < 5 || w.week > 10)) return false;
    if (cycleFilter === 'c3a' && (w.week < 11 || w.week > 16)) return false;
    if (cycleFilter === 'c3b' && (w.week < 17 || w.week > 22)) return false;
    if (cycleFilter === 'c4' && w.week < 23) return false;

    return true;
  });

  const buildFullLogbookMarkdown = () => {
    const lines: string[] = [
      '# LogBook de Estudos — Enterprise AI Architect Foundations',
      '',
      `Progresso total: ${completedStepsCount}/${totalSteps} passos (${completedWeeksCount}/32 semanas completas) · ${totalHours.toFixed(1)}h registradas`,
      '',
      '## Controle Passo a Passo por Semana',
      '',
      '| Semana | Tema | Leitura | Autoaval. | Exercício | Gabarito | Entregável | Commit | Horas |',
      '|---|---|---|---|---|---|---|---|---|',
    ];

    for (const w of weeks) {
      const state = weekLogs[w.week] ?? createEmptyWeekLog();
      const mark = (id: WeeklyStepId) => (state.steps[id] ? 'x' : ' ');
      lines.push(
        `| ${String(w.week).padStart(2, '0')} | ${w.title} | [${mark('reading')}] | [${mark('self_check')}] | [${mark('exercise')}] | [${mark('gabarito')}] | [${mark('deliverable')}] | [${mark('commit')}] | ${state.hoursSpent.toFixed(1)}h |`,
      );
    }

    lines.push('', '## Registro Cronológico de Sessões e Comandos', '');

    if (logEntries.length === 0) {
      lines.push('*Nenhuma sessão registrada manualmente ainda.*');
    } else {
      for (const e of logEntries) {
        lines.push(
          `### [${e.date}] — Semana ${String(e.week).padStart(2, '0')}: ${e.title} (${e.hours.toFixed(1)}h)`,
          '',
          e.notes,
        );
        if (e.blockers) {
          lines.push('', `**Travamentos:** ${e.blockers}`);
        }
        if (e.commandSnippet) {
          lines.push('', '```bash', e.commandSnippet, '```');
        }
        lines.push('', '---', '');
      }
    }

    return lines.join('\n');
  };

  const buildChecklistMarkdown = () => {
    return weeks
      .map((w) => {
        const done = countCompletedSteps(weekLogs[w.week]) === WEEKLY_STEPS.length;
        return `- [${done ? 'x' : ' '}] Semana ${w.week} — ${w.title}`;
      })
      .join('\n');
  };

  const handleCopy = async (kind: 'logbook' | 'checklist', content: string) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedType(kind);
      setTimeout(() => setCopiedType(null), 1600);
    } catch {
      // clipboard unavailable
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

  const handleSubmitEntry = (e: React.FormEvent) => {
    e.preventDefault();
    if (!entryTitle.trim() || !entryNotes.trim()) return;
    const parsedHours = Math.max(0, Number.parseFloat(entryHours) || 0);
    onAddLogEntry({
      date: new Date().toISOString().slice(0, 10),
      week: entryWeek,
      title: entryTitle.trim(),
      hours: parsedHours,
      notes: entryNotes.trim(),
      blockers: entryBlockers.trim(),
      commandSnippet: entryCommand.trim() || undefined,
    });
    const currentWeekHours = weekLogs[entryWeek]?.hoursSpent ?? 0;
    onUpdateWeekLog(entryWeek, {
      hoursSpent: Number((currentWeekHours + parsedHours).toFixed(1)),
      lastUpdated: new Date().toISOString().slice(0, 10),
    });
    setEntryTitle('');
    setEntryNotes('');
    setEntryBlockers('');
    setEntryCommand('');
  };

  return (
    <div className="mx-auto max-w-[1280px] px-6 py-8 space-y-8">
      {/* Header + Export Actions */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-[#0F172A] tracking-tight">
            LogBook de Progresso &amp; Passos
          </h1>
          <p className="text-sm text-slate-600 mt-1">
            Controle todos os 6 passos de cada uma das 32 semanas, registre horas, entregáveis e exporte seu{' '}
            <code className="text-xs font-mono bg-slate-100 px-1.5 py-0.5 rounded">LOGBOOK.md</code> para o GitHub.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 shrink-0">
          <button
            type="button"
            onClick={() => handleCopy('checklist', buildChecklistMarkdown())}
            className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium text-slate-700 bg-white border border-slate-200 hover:bg-slate-50 rounded-lg whitespace-nowrap shrink-0 cursor-pointer"
          >
            <Copy className="w-3.5 h-3.5" />
            <span>
              {copiedType === 'checklist' ? 'Checklist Copiado' : 'Copiar Checklist Plano'}
            </span>
          </button>
          <button
            type="button"
            onClick={() => handleCopy('logbook', buildFullLogbookMarkdown())}
            className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium text-slate-700 bg-white border border-slate-200 hover:bg-slate-50 rounded-lg whitespace-nowrap shrink-0 cursor-pointer"
          >
            <Copy className="w-3.5 h-3.5" />
            <span>{copiedType === 'logbook' ? 'LogBook Copiado' : 'Copiar LOGBOOK.md'}</span>
          </button>
          <button
            type="button"
            onClick={() => handleDownload('LOGBOOK.md', buildFullLogbookMarkdown())}
            className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-medium bg-[#14213D] text-white hover:bg-[#1E3158] rounded-lg whitespace-nowrap shrink-0 cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Baixar LOGBOOK.md</span>
          </button>
        </div>
      </div>

      <div className="bg-white border border-slate-200 rounded-xl px-5 py-3 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs text-slate-600">
        {firebaseConfigured ? (
          <>
            <div className="flex items-center gap-2">
              <Cloud className="w-4 h-4 text-[#0284C7] shrink-0" />
              <span>
                Sincronizado em tempo real na conta <strong>{user?.email}</strong> (Firestore)
              </span>
            </div>
            {hasLocalMigration && (
              <button
                type="button"
                disabled={migrating}
                onClick={() => void onMigrateLocal()}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-[#14213D] text-white rounded-md hover:bg-[#1E3158] cursor-pointer disabled:opacity-50 self-start sm:self-auto"
              >
                <UploadCloud className="w-3.5 h-3.5" />
                <span>
                  {migrating ? 'Importando...' : 'Importar progresso local do navegador'}
                </span>
              </button>
            )}
          </>
        ) : (
          <>
            <div className="flex items-center gap-2">
              <Lock className="w-4 h-4 text-slate-500 shrink-0" />
              <span>
                Modo local no navegador — adicione <code className="font-mono bg-slate-100 px-1 rounded">FIREBASE_CONFIG_JSON</code> no servidor para habilitar login Google e banco Firestore.
              </span>
            </div>
            <button
              type="button"
              onClick={onOpenAccess}
              className="px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-md cursor-pointer self-start sm:self-auto"
            >
              Ver Configuração
            </button>
          </>
        )}
      </div>

      {/* KPI Strip */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 tabular-nums">
        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <div className="text-xs text-slate-500">Passos Concluídos</div>
          <div className="text-2xl font-bold font-mono text-[#0F172A] mt-1">
            {completedStepsCount}{' '}
            <span className="text-sm font-normal text-slate-400">/ {totalSteps}</span>
          </div>
          <div className="mt-2 h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
            <div
              className="h-full bg-[#0284C7] transition-all"
              style={{ width: `${Math.round((completedStepsCount / Math.max(1, totalSteps)) * 100)}%` }}
            />
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <div className="text-xs text-slate-500">Semanas 100% Completas</div>
          <div className="text-2xl font-bold font-mono text-[#059669] mt-1">
            {completedWeeksCount}{' '}
            <span className="text-sm font-normal text-slate-400">/ 32</span>
          </div>
          <div className="text-xs text-slate-500 mt-1.5">Todos os 6 passos marcados</div>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <div className="text-xs text-slate-500">Horas Registradas</div>
          <div className="text-2xl font-bold font-mono text-[#0F172A] mt-1">
            {totalHours.toFixed(1)}h
          </div>
          <div className="text-xs text-slate-500 mt-1.5">Meta sugerida: ~14h / semana</div>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <div className="text-xs text-slate-500">Sessões no Diário</div>
          <div className="text-2xl font-bold font-mono text-[#0F172A] mt-1">
            {logEntries.length}
          </div>
          <div className="text-xs text-slate-500 mt-1.5">Entradas cronológicas salvas</div>
        </div>
      </div>

      {/* Mode Switcher */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-1 p-1 bg-slate-100 border border-slate-200 rounded-lg">
          <button
            type="button"
            onClick={() => setSubTab('steps')}
            className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              subTab === 'steps' ? 'bg-white text-[#0F172A] shadow-sm' : 'text-slate-600 hover:text-[#0F172A]'
            }`}
          >
            Controle Passo a Passo (32 Semanas)
          </button>
          <button
            type="button"
            onClick={() => setSubTab('journal')}
            className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              subTab === 'journal' ? 'bg-white text-[#0F172A] shadow-sm' : 'text-slate-600 hover:text-[#0F172A]'
            }`}
          >
            Diário Cronológico &amp; Comandos ({logEntries.length})
          </button>
        </div>

        {subTab === 'steps' && (
          <div className="flex flex-wrap items-center gap-3">
            <select
              value={cycleFilter}
              onChange={(e) => setCycleFilter(e.target.value)}
              aria-label="Filtrar por ciclo"
              className="px-3 py-1.5 text-xs bg-white border border-slate-200 rounded-lg text-[#0F172A]"
            >
              <option value="all">Todos os Ciclos (1–4)</option>
              <option value="c1">Ciclo 1 · Base Matemática (Sem. 1–4)</option>
              <option value="c2">Ciclo 2 · Anatomia LLMs (Sem. 5–10)</option>
              <option value="c3a">Ciclo 3A · RAG &amp; Search (Sem. 11–16)</option>
              <option value="c3b">Ciclo 3B · PEFT &amp; DPO (Sem. 17–22)</option>
              <option value="c4">Ciclo 4 · Produção &amp; Causal (Sem. 23–32)</option>
            </select>

            <div className="flex items-center gap-1 p-1 bg-slate-100 border border-slate-200 rounded-lg">
              {[
                { id: 'all' as const, label: 'Todas' },
                { id: 'in_progress' as const, label: 'Em andamento' },
                { id: 'pending' as const, label: 'Não iniciadas' },
                { id: 'completed' as const, label: 'Completas' },
              ].map((st) => (
                <button
                  key={st.id}
                  type="button"
                  onClick={() => setStatusFilter(st.id)}
                  className={`px-2.5 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                    statusFilter === st.id
                      ? 'bg-white text-[#0F172A] shadow-sm'
                      : 'text-slate-600 hover:text-[#0F172A]'
                  }`}
                >
                  {st.label}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Sub-view 1: Step-by-Step 32-Week Controller */}
      {subTab === 'steps' && (
        <div className="bg-white border border-slate-200 rounded-xl divide-y divide-slate-200 overflow-hidden">
          {filteredWeeks.length === 0 && (
            <div className="p-8 text-center text-sm text-slate-500">
              Nenhuma semana corresponde aos filtros selecionados.
            </div>
          )}

          {filteredWeeks.map((week) => {
            const state = weekLogs[week.week] ?? createEmptyWeekLog();
            const doneCount = countCompletedSteps(state);
            const isExpanded = expandedWeek === week.week;

            return (
              <div key={week.week} className="p-5 space-y-4">
                <div className="flex flex-col xl:flex-row xl:items-center justify-between gap-4">
                  {/* Week Info */}
                  <div className="space-y-1 min-w-0">
                    <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs text-slate-500 tabular-nums">
                      <span className="font-mono font-semibold text-[#0F172A]">
                        Semana {String(week.week).padStart(2, '0')}
                      </span>
                      <span aria-hidden="true">·</span>
                      <span>{getCycleLabel(week.week)}</span>
                      <span aria-hidden="true">·</span>
                      <span>
                        {doneCount}/6 passos
                      </span>
                      {state.hoursSpent > 0 && (
                        <>
                          <span aria-hidden="true">·</span>
                          <span>{state.hoursSpent.toFixed(1)}h</span>
                        </>
                      )}
                    </div>
                    <div className="text-base font-semibold text-[#0F172A]">{week.title}</div>
                  </div>

                  {/* 6 Step Checkboxes */}
                  <div className="flex flex-wrap items-center gap-1.5 shrink-0">
                    {WEEKLY_STEPS.map((step) => {
                      const active = state.steps[step.id];
                      return (
                        <button
                          key={step.id}
                          type="button"
                          onClick={() => onToggleStep(week.week, step.id)}
                          title={step.fullLabel}
                          className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-md border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                            active
                              ? 'bg-emerald-50 border-emerald-300 text-[#059669]'
                              : 'bg-slate-50 border-slate-200 text-slate-600 hover:bg-slate-100'
                          }`}
                        >
                          <Check
                            className={`w-3 h-3 ${active ? 'opacity-100' : 'opacity-25'}`}
                          />
                          <span>{step.shortLabel}</span>
                        </button>
                      );
                    })}

                    <button
                      type="button"
                      onClick={() => setExpandedWeek(isExpanded ? null : week.week)}
                      className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-200 hover:bg-slate-50 rounded-md whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      <span>Notas</span>
                      {isExpanded ? (
                        <ChevronUp className="w-3.5 h-3.5" />
                      ) : (
                        <ChevronDown className="w-3.5 h-3.5" />
                      )}
                    </button>
                  </div>
                </div>

                {/* Expandable Week Notes, Deliverable & Hours */}
                {isExpanded && (
                  <div className="bg-slate-50 border border-slate-200 rounded-lg p-5 space-y-4">
                    <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 pb-3">
                      <span className="text-xs font-medium text-slate-700">
                        Registro da Semana {String(week.week).padStart(2, '0')}
                        {state.lastUpdated ? ` · Atualizado em ${state.lastUpdated}` : ''}
                      </span>
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() => onOpenWeek(week.week, 'concept')}
                          className="inline-flex items-center gap-1 px-2.5 py-1 text-xs font-medium bg-white border border-slate-200 hover:bg-slate-100 rounded text-slate-700 cursor-pointer"
                        >
                          <BookOpen className="w-3.5 h-3.5" />
                          <span>Abrir Leitura</span>
                        </button>
                        <button
                          type="button"
                          onClick={() => onOpenWeek(week.week, 'notebook')}
                          className="inline-flex items-center gap-1 px-2.5 py-1 text-xs font-medium bg-white border border-slate-200 hover:bg-slate-100 rounded text-slate-700 cursor-pointer"
                        >
                          <Code2 className="w-3.5 h-3.5" />
                          <span>Abrir Notebook</span>
                        </button>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-12 gap-4">
                      <div className="md:col-span-3 space-y-1.5">
                        <label className="text-xs font-medium text-slate-700">
                          Horas Dedicadas
                        </label>
                        <input
                          type="number"
                          min="0"
                          step="0.5"
                          value={state.hoursSpent}
                          onChange={(e) =>
                            onUpdateWeekLog(week.week, {
                              hoursSpent: Math.max(0, Number.parseFloat(e.target.value) || 0),
                              lastUpdated: new Date().toISOString().slice(0, 10),
                            })
                          }
                          className="w-full px-3 py-2 text-sm font-mono bg-white border border-slate-200 rounded-lg"
                        />
                      </div>

                      <div className="md:col-span-9 space-y-1.5">
                        <label className="text-xs font-medium text-slate-700">
                          Entregável Produzido (arquivo em docs/ ou link do commit)
                        </label>
                        <input
                          type="text"
                          value={state.deliverableSummary}
                          onChange={(e) =>
                            onUpdateWeekLog(week.week, {
                              deliverableSummary: e.target.value,
                              lastUpdated: new Date().toISOString().slice(0, 10),
                            })
                          }
                          placeholder="Ex: docs/impacto-de-negocio.md — análise de threshold com custo FN=250 e FP=15"
                          className="w-full px-3 py-2 text-sm bg-white border border-slate-200 rounded-lg"
                        />
                      </div>

                      <div className="md:col-span-6 space-y-1.5">
                        <label className="text-xs font-medium text-slate-700">
                          Principais Aprendizados &amp; Decisões
                        </label>
                        <textarea
                          rows={3}
                          value={state.notes}
                          onChange={(e) =>
                            onUpdateWeekLog(week.week, {
                              notes: e.target.value,
                              lastUpdated: new Date().toISOString().slice(0, 10),
                            })
                          }
                          placeholder="Anote conceitos-chave, experimentos que funcionaram ou diferenças para o gabarito..."
                          className="w-full px-3 py-2 text-sm bg-white border border-slate-200 rounded-lg"
                        />
                      </div>

                      <div className="md:col-span-6 space-y-1.5">
                        <label className="text-xs font-medium text-slate-700">
                          Travamentos / Pontos de Revisão
                        </label>
                        <textarea
                          rows={3}
                          value={state.blockers}
                          onChange={(e) =>
                            onUpdateWeekLog(week.week, {
                              blockers: e.target.value,
                              lastUpdated: new Date().toISOString().slice(0, 10),
                            })
                          }
                          placeholder="O que travou nesta semana e precisa ser revisitado no checkpoint?"
                          className="w-full px-3 py-2 text-sm bg-white border border-slate-200 rounded-lg"
                        />
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Sub-view 2: Chronological Journal & Command Log */}
      {subTab === 'journal' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          <form
            onSubmit={handleSubmitEntry}
            className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-6 space-y-4"
          >
            <h2 className="text-base font-semibold text-[#0F172A] border-b border-slate-200 pb-3">
              Registrar Sessão de Estudo
            </h2>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div className="sm:col-span-2 space-y-1.5">
                <label className="text-xs font-medium text-slate-700">Semana</label>
                <select
                  value={entryWeek}
                  onChange={(e) => setEntryWeek(Number(e.target.value))}
                  className="w-full px-3 py-2 text-sm bg-white border border-slate-200 rounded-lg"
                >
                  {weeks.map((w) => (
                    <option key={w.week} value={w.week}>
                      Sem. {String(w.week).padStart(2, '0')} — {w.title}
                    </option>
                  ))}
                </select>
              </div>
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-slate-700">Horas</label>
                <input
                  type="number"
                  min="0"
                  step="0.5"
                  value={entryHours}
                  onChange={(e) => setEntryHours(e.target.value)}
                  className="w-full px-3 py-2 text-sm font-mono bg-white border border-slate-200 rounded-lg"
                />
              </div>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-medium text-slate-700">Título da Sessão</label>
              <input
                type="text"
                required
                value={entryTitle}
                onChange={(e) => setEntryTitle(e.target.value)}
                placeholder="Ex: Passo 1–4 do notebook de Self-Attention"
                className="w-full px-3 py-2 text-sm bg-white border border-slate-200 rounded-lg"
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-medium text-slate-700">O que foi executado</label>
              <textarea
                required
                rows={3}
                value={entryNotes}
                onChange={(e) => setEntryNotes(e.target.value)}
                placeholder="Resumo do estudo, conclusões da leitura ou resultados obtidos no Colab..."
                className="w-full px-3 py-2 text-sm bg-white border border-slate-200 rounded-lg"
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-medium text-slate-700">
                Comandos / Setup Utilizados (opcional)
              </label>
              <textarea
                rows={2}
                value={entryCommand}
                onChange={(e) => setEntryCommand(e.target.value)}
                placeholder="Ex: git commit -m 'Semana 07 concluída'"
                className="w-full px-3 py-2 text-xs font-mono bg-slate-50 border border-slate-200 rounded-lg"
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-medium text-slate-700">
                Travamentos / Dúvidas (opcional)
              </label>
              <input
                type="text"
                value={entryBlockers}
                onChange={(e) => setEntryBlockers(e.target.value)}
                placeholder="Ex: Dúvida na máscara causal com padding"
                className="w-full px-3 py-2 text-sm bg-white border border-slate-200 rounded-lg"
              />
            </div>

            <button
              type="submit"
              className="w-full inline-flex items-center justify-center gap-2 px-4 py-2.5 text-sm font-medium bg-[#14213D] text-white rounded-lg hover:bg-[#1E3158] cursor-pointer"
            >
              <Plus className="w-4 h-4" />
              <span>Salvar no LogBook</span>
            </button>
          </form>

          <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-6 space-y-4">
            <div className="border-b border-slate-200 pb-3 flex items-center justify-between">
              <h2 className="text-base font-semibold text-[#0F172A]">
                Histórico Cronológico ({logEntries.length})
              </h2>
              <span className="text-xs text-slate-500 tabular-nums">
                Sincronizado no banco Firestore
              </span>
            </div>

            {logEntries.length === 0 ? (
              <p className="text-sm text-slate-500 py-10 text-center">
                Nenhuma sessão registrada ainda. Adicione sua primeira sessão ao lado para começar seu histórico.
              </p>
            ) : (
              <div className="divide-y divide-slate-200">
                {logEntries.map((entry) => (
                  <div key={entry.id} className="py-4 space-y-2">
                    <div className="flex items-center justify-between gap-4">
                      <div className="text-xs text-slate-500 tabular-nums">
                        <span className="font-mono">{entry.date}</span>
                        <span className="mx-1.5" aria-hidden="true">·</span>
                        <span>Semana {String(entry.week).padStart(2, '0')}</span>
                        <span className="mx-1.5" aria-hidden="true">·</span>
                        <span>{entry.hours.toFixed(1)}h</span>
                      </div>
                      <button
                        type="button"
                        onClick={() => onDeleteLogEntry(entry.id)}
                        aria-label="Remover entrada"
                        className="text-slate-400 hover:text-red-600 cursor-pointer"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                    <h3 className="text-sm font-semibold text-[#0F172A]">{entry.title}</h3>
                    <p className="text-sm text-slate-600 whitespace-pre-line">{entry.notes}</p>
                    {entry.commandSnippet && (
                      <pre className="text-xs font-mono bg-[#0F172A] text-slate-100 p-3 rounded-lg overflow-x-auto">
                        <code>{entry.commandSnippet}</code>
                      </pre>
                    )}
                    {entry.blockers && (
                      <p className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded px-2.5 py-1.5">
                        Travamento: {entry.blockers}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
