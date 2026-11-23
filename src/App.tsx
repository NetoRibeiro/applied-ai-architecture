import { useEffect, useMemo, useRef, useState } from 'react';
import { onAuthStateChanged, type User } from 'firebase/auth';
import { collection, doc, onSnapshot, query, where } from 'firebase/firestore';
import AccessControlView from './components/AccessControlView';
import ConceptReaderView from './components/ConceptReaderView';
import CurriculumView from './components/CurriculumView';
import GovernanceView from './components/GovernanceView';
import InteractiveLabsView from './components/InteractiveLabsView';
import LogbookView, {
  WEEKLY_STEPS,
  countCompletedSteps,
  createEmptyWeekLog,
} from './components/LogbookView';
import {
  OperationType,
  auth,
  createLogEntry,
  db,
  ensureUserProfile,
  handleFirestoreError,
  isBootstrapAdminEmail,
  isFirebaseConfigured,
  migrateLocalProgressToCloud,
  removeLogEntry,
  saveGovernanceDraft,
  signInWithGoogle,
  signOutUser,
  upsertWeekLog,
} from './firebase';
import type {
  LogbookEntry,
  PhaseInfo,
  RegistrationMode,
  UserProfile,
  WeekLogState,
  WeekMeta,
  WeeklyStepId,
} from './types';

type ViewMode = 'curriculum' | 'reader' | 'labs' | 'logbook' | 'governance' | 'access';

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

  // Local guest state (also used as migration source if local progress exists)
  const [localWeekLogs, setLocalWeekLogs] = useState<Record<number, WeekLogState>>(loadInitialWeekLogs);
  const [localLogEntries, setLocalLogEntries] = useState<LogbookEntry[]>(() => {
    try {
      const raw = localStorage.getItem(STORAGE_LOGBOOK_KEY);
      return raw ? (JSON.parse(raw) as LogbookEntry[]) : [];
    } catch {
      return [];
    }
  });

  // Firebase Auth & RBAC state
  const [user, setUser] = useState<User | null>(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [authError, setAuthError] = useState<string | null>(null);
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [registrationMode, setRegistrationMode] = useState<RegistrationMode>('open_authenticated');
  const [allUsers, setAllUsers] = useState<UserProfile[]>([]);
  const [usersLoading, setUsersLoading] = useState(false);

  // Cloud Firestore state for approved users
  const [cloudWeekLogs, setCloudWeekLogs] = useState<Record<number, WeekLogState>>({});
  const [cloudWeekIds, setCloudWeekIds] = useState<Set<number>>(new Set());
  const [cloudLogEntries, setCloudLogEntries] = useState<LogbookEntry[]>([]);
  const [cloudAdrDraft, setCloudAdrDraft] = useState<string | undefined>(undefined);
  const [cloudRfcDraft, setCloudRfcDraft] = useState<string | undefined>(undefined);
  const [cloudLoaded, setCloudLoaded] = useState(false);
  const [migrating, setMigrating] = useState(false);

  const weekDebounceTimers = useRef<Record<number, number>>({});

  // Load curriculum catalog from Express backend
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

  // Persist guest local progress
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_WEEK_LOGS_KEY, JSON.stringify(localWeekLogs));
    } catch {
      // ignore storage errors
    }
  }, [localWeekLogs]);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_LOGBOOK_KEY, JSON.stringify(localLogEntries));
    } catch {
      // ignore storage errors
    }
  }, [localLogEntries]);

  // Cleanup debounce timers
  useEffect(() => {
    return () => {
      Object.values(weekDebounceTimers.current).forEach((timerId) => window.clearTimeout(timerId));
    };
  }, []);

  const firebaseReady = isFirebaseConfigured();

  // Auth listener
  useEffect(() => {
    if (!firebaseReady || !auth) {
      setAuthLoading(false);
      return;
    }

    const unsubscribe = onAuthStateChanged(auth, async (nextUser) => {
      setAuthError(null);
      if (!nextUser) {
        setUser(null);
        setProfile(null);
        setAllUsers([]);
        setCloudWeekLogs({});
        setCloudWeekIds(new Set());
        setCloudLogEntries([]);
        setCloudAdrDraft(undefined);
        setCloudRfcDraft(undefined);
        setCloudLoaded(false);
        setAuthLoading(false);
        return;
      }

      setUser(nextUser);
      try {
        await ensureUserProfile(nextUser);
      } catch (err) {
        setAuthError(err instanceof Error ? err.message : 'Erro ao inicializar perfil de acesso.');
      } finally {
        setAuthLoading(false);
      }
    });

    return () => unsubscribe();
  }, [firebaseReady]);

  // User profile & access policy listeners
  useEffect(() => {
    if (!firebaseReady || !db || !user || !user.emailVerified) return;

    const userDocPath = `users/${user.uid}`;
    const unsubProfile = onSnapshot(
      doc(db, 'users', user.uid),
      (snap) => {
        if (snap.exists()) {
          setProfile(snap.data() as UserProfile);
        }
      },
      (err) => {
        handleFirestoreError(err, OperationType.GET, userDocPath);
      },
    );

    const unsubConfig = onSnapshot(
      doc(db, 'config', 'access'),
      (snap) => {
        if (snap.exists()) {
          const mode = snap.data()?.registrationMode as RegistrationMode | undefined;
          if (mode === 'approval_required' || mode === 'open_authenticated') {
            setRegistrationMode(mode);
          }
        }
      },
      (err) => {
        handleFirestoreError(err, OperationType.GET, 'config/access');
      },
    );

    return () => {
      unsubProfile();
      unsubConfig();
    };
  }, [firebaseReady, user]);

  const isAdmin =
    Boolean(user?.emailVerified && isBootstrapAdminEmail(user.email)) ||
    profile?.role === 'admin';

  const canPersistToCloud = Boolean(
    firebaseReady && user && user.emailVerified && profile?.status === 'approved',
  );

  // Admin users directory listener
  useEffect(() => {
    if (!firebaseReady || !db || !user || !user.emailVerified || !isAdmin) {
      setAllUsers([]);
      return;
    }

    setUsersLoading(true);
    const unsubUsers = onSnapshot(
      collection(db, 'users'),
      (snap) => {
        const list = snap.docs.map((d) => d.data() as UserProfile);
        const orderRank: Record<string, number> = { pending: 0, approved: 1, revoked: 2 };
        list.sort((a, b) => {
          const diff = (orderRank[a.status] ?? 9) - (orderRank[b.status] ?? 9);
          if (diff !== 0) return diff;
          return a.email.localeCompare(b.email);
        });
        setAllUsers(list);
        setUsersLoading(false);
      },
      (err) => {
        setUsersLoading(false);
        handleFirestoreError(err, OperationType.LIST, 'users');
      },
    );

    return () => unsubUsers();
  }, [firebaseReady, user, isAdmin]);

  // Approved user cloud data listeners (weekLogs, logEntries, governance drafts)
  useEffect(() => {
    if (!firebaseReady || !db || !user || !canPersistToCloud) {
      setCloudWeekLogs({});
      setCloudWeekIds(new Set());
      setCloudLogEntries([]);
      setCloudAdrDraft(undefined);
      setCloudRfcDraft(undefined);
      setCloudLoaded(false);
      return;
    }

    let gotWeeks = false;
    let gotEntries = false;
    const markLoaded = () => {
      if (gotWeeks && gotEntries) {
        setCloudLoaded(true);
      }
    };

    const weekLogsPath = `users/${user.uid}/weekLogs`;
    const unsubWeeks = onSnapshot(
      query(collection(db, 'users', user.uid, 'weekLogs'), where('ownerUid', '==', user.uid)),
      (snap) => {
        const nextMap: Record<number, WeekLogState> = {};
        const nextIds = new Set<number>();
        for (const d of snap.docs) {
          const raw = d.data() as {
            week: number;
            steps: Record<WeeklyStepId, boolean>;
            hoursSpent: number;
            lastUpdated: string;
            deliverableSummary: string;
            notes: string;
            blockers: string;
          };
          if (typeof raw.week === 'number') {
            nextIds.add(raw.week);
            nextMap[raw.week] = {
              steps: raw.steps,
              hoursSpent: raw.hoursSpent ?? 0,
              lastUpdated: raw.lastUpdated ?? '',
              deliverableSummary: raw.deliverableSummary ?? '',
              notes: raw.notes ?? '',
              blockers: raw.blockers ?? '',
            };
          }
        }
        setCloudWeekLogs(nextMap);
        setCloudWeekIds(nextIds);
        gotWeeks = true;
        markLoaded();
      },
      (err) => {
        handleFirestoreError(err, OperationType.LIST, weekLogsPath);
      },
    );

    const entriesPath = `users/${user.uid}/logEntries`;
    const unsubEntries = onSnapshot(
      query(collection(db, 'users', user.uid, 'logEntries'), where('ownerUid', '==', user.uid)),
      (snap) => {
        const list: LogbookEntry[] = snap.docs.map((d) => {
          const raw = d.data() as Omit<LogbookEntry, 'id'>;
          return {
            id: d.id,
            date: raw.date,
            week: raw.week,
            title: raw.title,
            hours: raw.hours,
            notes: raw.notes,
            blockers: raw.blockers ?? '',
            commandSnippet: raw.commandSnippet,
          };
        });
        list.sort((a, b) => b.date.localeCompare(a.date));
        setCloudLogEntries(list);
        gotEntries = true;
        markLoaded();
      },
      (err) => {
        handleFirestoreError(err, OperationType.LIST, entriesPath);
      },
    );

    const draftsPath = `users/${user.uid}/drafts`;
    const unsubDrafts = onSnapshot(
      query(collection(db, 'users', user.uid, 'drafts'), where('ownerUid', '==', user.uid)),
      (snap) => {
        for (const d of snap.docs) {
          const raw = d.data() as { kind: 'adr' | 'rfc'; content: string };
          if (raw.kind === 'adr') setCloudAdrDraft(raw.content);
          if (raw.kind === 'rfc') setCloudRfcDraft(raw.content);
        }
      },
      (err) => {
        handleFirestoreError(err, OperationType.LIST, draftsPath);
      },
    );

    return () => {
      unsubWeeks();
      unsubEntries();
      unsubDrafts();
    };
  }, [firebaseReady, user, canPersistToCloud]);

  const weekLogs = canPersistToCloud ? cloudWeekLogs : localWeekLogs;
  const logEntries = canPersistToCloud ? cloudLogEntries : localLogEntries;

  const hasLocalMigration = useMemo(() => {
    if (!canPersistToCloud || !cloudLoaded) return false;
    const cloudEmpty = Object.keys(cloudWeekLogs).length === 0 && cloudLogEntries.length === 0;
    const localHasData = Object.keys(localWeekLogs).length > 0 || localLogEntries.length > 0;
    return cloudEmpty && localHasData;
  }, [canPersistToCloud, cloudLoaded, cloudWeekLogs, cloudLogEntries, localWeekLogs, localLogEntries]);

  const handleMigrateLocal = async () => {
    if (!user || !canPersistToCloud) return;
    setMigrating(true);
    try {
      await migrateLocalProgressToCloud(user.uid, localWeekLogs, localLogEntries);
    } finally {
      setMigrating(false);
    }
  };

  const completedWeeks = useMemo(() => {
    return Object.entries(weekLogs)
      .filter(([, state]) => countCompletedSteps(state) === WEEKLY_STEPS.length)
      .map(([w]) => Number(w))
      .sort((a, b) => a - b);
  }, [weekLogs]);

  const toggleCompleteWeek = (week: number) => {
    const current = weekLogs[week] ?? createEmptyWeekLog();
    const allDone = countCompletedSteps(current) === WEEKLY_STEPS.length;
    const nextDone = !allDone;
    const nextState: WeekLogState = {
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
    };

    if (canPersistToCloud && user) {
      setCloudWeekLogs((prev) => ({ ...prev, [week]: nextState }));
      void upsertWeekLog(user.uid, week, nextState, cloudWeekIds.has(week));
    } else {
      setLocalWeekLogs((prev) => ({ ...prev, [week]: nextState }));
    }
  };

  const toggleStep = (week: number, step: WeeklyStepId) => {
    const current = weekLogs[week] ?? createEmptyWeekLog();
    const nextState: WeekLogState = {
      ...current,
      lastUpdated: new Date().toISOString().slice(0, 10),
      steps: {
        ...current.steps,
        [step]: !current.steps[step],
      },
    };

    if (canPersistToCloud && user) {
      setCloudWeekLogs((prev) => ({ ...prev, [week]: nextState }));
      void upsertWeekLog(user.uid, week, nextState, cloudWeekIds.has(week));
    } else {
      setLocalWeekLogs((prev) => ({ ...prev, [week]: nextState }));
    }
  };

  const updateWeekLog = (week: number, patch: Partial<WeekLogState>) => {
    const current = weekLogs[week] ?? createEmptyWeekLog();
    const nextState: WeekLogState = {
      ...current,
      ...patch,
    };

    if (canPersistToCloud && user) {
      setCloudWeekLogs((prev) => ({ ...prev, [week]: nextState }));
      if (weekDebounceTimers.current[week]) {
        window.clearTimeout(weekDebounceTimers.current[week]);
      }
      weekDebounceTimers.current[week] = window.setTimeout(() => {
        void upsertWeekLog(user.uid, week, nextState, cloudWeekIds.has(week));
      }, 450);
    } else {
      setLocalWeekLogs((prev) => ({ ...prev, [week]: nextState }));
    }
  };

  const addLogEntry = (entry: Omit<LogbookEntry, 'id'>) => {
    if (canPersistToCloud && user) {
      void createLogEntry(user.uid, entry);
      return;
    }
    setLocalLogEntries((prev) => [
      { ...entry, id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}` },
      ...prev,
    ]);
  };

  const deleteLogEntry = (id: string) => {
    if (canPersistToCloud && user) {
      void removeLogEntry(user.uid, id);
      return;
    }
    setLocalLogEntries((prev) => prev.filter((e) => e.id !== id));
  };

  const handleSaveDraft = async (kind: 'adr' | 'rfc', content: string) => {
    if (!user || !canPersistToCloud) return;
    await saveGovernanceDraft(user.uid, kind, content);
  };

  const handleSignIn = async () => {
    setAuthError(null);
    try {
      await signInWithGoogle();
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : 'Falha ao entrar com Google.');
    }
  };

  const handleSignOut = async () => {
    setAuthError(null);
    await signOutUser();
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

  const accountButtonLabel = !firebaseReady
    ? 'Configurar Firebase'
    : !user
      ? 'Entrar com Google'
      : isAdmin
        ? 'Admin · Acesso'
        : profile?.status === 'pending'
          ? 'Conta · Pendente'
          : profile?.status === 'revoked'
            ? 'Conta · Suspensa'
            : 'Conta · Nuvem';

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
              onClick={() => {
                if (!firebaseReady || user) {
                  setView('access');
                } else {
                  void handleSignIn();
                }
              }}
              className="px-4 py-2 text-xs font-medium text-white bg-[#14213D] rounded-lg hover:bg-[#1E3158] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              {authLoading ? 'Carregando...' : accountButtonLabel}
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
          <button
            type="button"
            onClick={() => setView('access')}
            className={`whitespace-nowrap shrink-0 py-1 ${view === 'access' ? 'text-[#0F172A] font-semibold' : ''}`}
          >
            Acesso
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
                firebaseConfigured={firebaseReady}
                user={user}
                profile={profile}
                onSignIn={handleSignIn}
                onOpenAccess={() => setView('access')}
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
                firebaseConfigured={firebaseReady}
                user={user}
                profile={profile}
                canPersistToCloud={canPersistToCloud}
                hasLocalMigration={hasLocalMigration}
                migrating={migrating}
                onMigrateLocal={handleMigrateLocal}
                onSignIn={handleSignIn}
                onOpenAccess={() => setView('access')}
                onToggleStep={toggleStep}
                onUpdateWeekLog={updateWeekLog}
                onAddLogEntry={addLogEntry}
                onDeleteLogEntry={deleteLogEntry}
                onOpenWeek={openWeek}
              />
            )}

            {view === 'governance' && (
              <GovernanceView
                canPersistToCloud={canPersistToCloud}
                cloudAdrDraft={cloudAdrDraft}
                cloudRfcDraft={cloudRfcDraft}
                onSaveDraft={handleSaveDraft}
                onOpenAccess={() => setView('access')}
              />
            )}

            {view === 'access' && (
              <AccessControlView
                firebaseConfigured={firebaseReady}
                user={user}
                profile={profile}
                authLoading={authLoading}
                authError={authError}
                registrationMode={registrationMode}
                allUsers={allUsers}
                usersLoading={usersLoading}
                onSignIn={handleSignIn}
                onSignOut={handleSignOut}
                onOpenLogbook={() => setView('logbook')}
              />
            )}
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
