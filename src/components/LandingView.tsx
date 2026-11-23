import type { User } from 'firebase/auth';
import {
  ArrowRight,
  BookOpen,
  CheckCircle2,
  Cloud,
  Compass,
  FlaskConical,
  Laptop,
  LogIn,
  NotebookPen,
  ShieldCheck,
} from 'lucide-react';
import { WEEKLY_STEPS, countCompletedSteps } from './LogbookView';
import type { PhaseInfo, UserProfile, WeekLogState, WeekMeta, WeeklyStepId } from '../types';

const STEP_HINTS: Record<WeeklyStepId, string> = {
  reading: 'Compreenda os trade-offs arquiteturais antes de abrir código.',
  self_check: 'Valide o domínio conceitual respondendo às questões-chave.',
  exercise: 'Implemente e experimente os blocos práticos no notebook.',
  gabarito: 'Compare suas escolhas com a solução comentada da semana.',
  deliverable: 'Sintetize decisões, riscos e custos em um registro técnico.',
  commit: 'Feche a semana publicando o progresso e atualizando o LogBook.',
};

interface LandingViewProps {
  phases: PhaseInfo[];
  weeks: WeekMeta[];
  completedWeeks: number[];
  weekLogs: Record<number, WeekLogState>;
  logEntriesCount: number;
  firebaseConfigured: boolean;
  user: User | null;
  profile: UserProfile | null;
  onSignIn: () => Promise<void>;
  onOpenCurriculum: () => void;
  onOpenLogbook: () => void;
  onOpenLabs: () => void;
  onOpenAccess: () => void;
}

const STUDY_PILLARS = [
  {
    index: '01',
    title: 'Entenda o trade-off antes de executar código',
    description:
      'Comece pelo Leitor de Conceitos de cada semana e responda às perguntas de autoavaliação antes de abrir o gabarito. O foco é critério arquitetural: latência, custo por token, recall e risco em produção.',
    actionLabel: 'Explorar semanas no Currículo',
    target: 'curriculum' as const,
  },
  {
    index: '02',
    title: 'Teste hipóteses nos Labs Interativos',
    description:
      'Use os simuladores visuais para variar thresholds de decisão, chunking em RAG, memória de KV-cache e experimentos causais antes de escrever ou ajustar código.',
    actionLabel: 'Abrir Labs Interativos',
    target: 'labs' as const,
  },
  {
    index: '03',
    title: 'Feche cada semana com um entregável real',
    description:
      'Rode o notebook guiado no Google Colab ou localmente, compare sua abordagem com o gabarito e consolide o aprendizado em um commit, nota técnica ou diagrama.',
    actionLabel: 'Ver trilha completa',
    target: 'curriculum' as const,
  },
  {
    index: '04',
    title: 'Transforme o LogBook em portfólio técnico',
    description:
      'Registre horas investidas, comandos úteis e bloqueios superados. Esse histórico documenta sua evolução e serve de base direta para os ADRs e RFCs na aba Governança.',
    actionLabel: 'Abrir LogBook',
    target: 'logbook' as const,
  },
];

export default function LandingView({
  phases,
  weeks,
  completedWeeks,
  weekLogs,
  logEntriesCount,
  firebaseConfigured,
  user,
  profile,
  onSignIn,
  onOpenCurriculum,
  onOpenLogbook,
  onOpenLabs,
  onOpenAccess,
}: LandingViewProps) {
  const totalWeeks = weeks.length || 32;
  const totalStepsDone = weeks.reduce(
    (sum, w) => sum + countCompletedSteps(weekLogs[w.week]),
    0,
  );
  const totalPossibleSteps = totalWeeks * WEEKLY_STEPS.length;

  const isApprovedAccount = Boolean(user && user.emailVerified && profile?.status === 'approved');
  const isPendingAccount = Boolean(user && profile?.status === 'pending');

  const handlePrimaryAccountAction = () => {
    if (!firebaseConfigured) {
      onOpenAccess();
      return;
    }
    if (user) {
      onOpenCurriculum();
      return;
    }
    void onSignIn();
  };

  const handlePillarAction = (target: 'curriculum' | 'labs' | 'logbook') => {
    if (target === 'labs') {
      onOpenLabs();
    } else if (target === 'logbook') {
      onOpenLogbook();
    } else {
      onOpenCurriculum();
    }
  };

  return (
    <div className="mx-auto max-w-[1200px] px-6 py-10 sm:py-14 space-y-16">
      {/* Hero Split */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-stretch">
        <div className="lg:col-span-7 flex flex-col justify-between bg-white border border-slate-200 rounded-xl p-8 sm:p-10">
          <div className="space-y-5">
            <p className="text-xs font-medium text-slate-500 tabular-nums">
              Trilha aberta de Arquitetura de IA · {totalWeeks} semanas · {phases.length || 4} ciclos integradores
            </p>
            <h1 className="text-3xl sm:text-5xl font-bold tracking-tight text-[#0F172A] leading-[1.12] [text-wrap:balance]">
              Estude com método semanal e salve seu LogBook na nuvem
            </h1>
            <p className="text-base leading-relaxed text-slate-600 max-w-2xl">
              Ler teoria ou rodar notebooks soltos não constrói critério de arquitetura. Esta
              plataforma organiza sua jornada em ciclos práticos, simuladores de trade-off e um
              diário técnico semanal. Crie sua conta gratuita antes de começar para manter todo o
              andamento sincronizado.
            </p>
          </div>

          <div className="pt-8 space-y-4">
            <div className="flex flex-wrap items-center gap-4">
              {!user ? (
                <>
                  <button
                    type="button"
                    onClick={handlePrimaryAccountAction}
                    className="inline-flex items-center gap-2 px-5 py-3 text-sm font-semibold text-white bg-[#14213D] rounded-lg hover:bg-[#1E3158] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                  >
                    <LogIn className="w-4 h-4" />
                    {firebaseConfigured
                      ? 'Criar conta com Google e salvar progresso'
                      : 'Configurar conta em nuvem'}
                  </button>
                  <button
                    type="button"
                    onClick={onOpenCurriculum}
                    className="inline-flex items-center gap-1.5 px-3 py-2 text-sm font-medium text-slate-700 hover:text-[#0F172A] underline underline-offset-4 decoration-slate-300 hover:decoration-[#14213D] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                  >
                    Explorar Currículo primeiro
                    <ArrowRight className="w-4 h-4" />
                  </button>
                </>
              ) : (
                <>
                  <button
                    type="button"
                    onClick={onOpenCurriculum}
                    className="inline-flex items-center gap-2 px-5 py-3 text-sm font-semibold text-white bg-[#14213D] rounded-lg hover:bg-[#1E3158] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                  >
                    Ir para o Currículo
                    <ArrowRight className="w-4 h-4" />
                  </button>
                  <button
                    type="button"
                    onClick={onOpenLogbook}
                    className="inline-flex items-center gap-1.5 px-3 py-2 text-sm font-medium text-slate-700 hover:text-[#0F172A] underline underline-offset-4 decoration-slate-300 hover:decoration-[#14213D] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                  >
                    Abrir meu LogBook
                  </button>
                </>
              )}
            </div>

            <p className="text-xs text-slate-500 leading-relaxed">
              {!user
                ? 'Sem formulário longo: ao entrar com Google, seu perfil é criado automaticamente. Se você já iniciou em modo visitante, poderá migrar os registros locais para a nuvem com um clique.'
                : isPendingAccount
                  ? `Conectado como ${user.email}. Seu perfil aguarda aprovação do administrador para liberar a gravação em nuvem.`
                  : `Sessão ativa como ${user.email} · Seu checklist semanal, LogBook e rascunhos de governança estão vinculados à sua conta.`}
            </p>
          </div>
        </div>

        {/* Right Architectural Focal Panel: 6-Step Weekly Cadence + Account Status */}
        <aside className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-6 sm:p-8 flex flex-col justify-between gap-6">
          <div className="space-y-4">
            <div className="flex items-baseline justify-between gap-3 border-b border-slate-100 pb-3">
              <h2 className="text-base font-semibold text-[#0F172A]">
                Cadência semanal recomendada
              </h2>
              <span className="text-xs font-mono text-slate-500 tabular-nums shrink-0">
                6 etapas / semana
              </span>
            </div>

            <ol className="divide-y divide-slate-100">
              {WEEKLY_STEPS.map((step, idx) => (
                <li key={step.id} className="py-2.5 flex items-start gap-3">
                  <span className="font-mono text-xs font-semibold text-[#14213D] tabular-nums pt-0.5 shrink-0">
                    {String(idx + 1).padStart(2, '0')}.
                  </span>
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-[#0F172A]">
                      {step.fullLabel.replace(/^\d+\.\s*/, '')}
                    </p>
                    <p className="text-xs text-slate-500 leading-relaxed">
                      {STEP_HINTS[step.id]}
                    </p>
                  </div>
                </li>
              ))}
            </ol>
          </div>

          <div className="border-t border-slate-200 pt-4 grid grid-cols-3 gap-4">
            <div>
              <p className="text-xs text-slate-500">Semanas</p>
              <p className="text-lg font-bold text-[#0F172A] font-mono tabular-nums">
                {completedWeeks.length}/{totalWeeks}
              </p>
            </div>
            <div>
              <p className="text-xs text-slate-500">Etapas</p>
              <p className="text-lg font-bold text-[#0F172A] font-mono tabular-nums">
                {totalStepsDone}/{totalPossibleSteps}
              </p>
            </div>
            <div>
              <p className="text-xs text-slate-500">Salvamento</p>
              <p className="text-sm font-semibold text-[#14213D] pt-1 flex items-center gap-1.5">
                {isApprovedAccount ? (
                  <>
                    <Cloud className="w-3.5 h-3.5 text-emerald-700 shrink-0" />
                    <span>Em nuvem</span>
                  </>
                ) : (
                  <>
                    <Laptop className="w-3.5 h-3.5 text-slate-500 shrink-0" />
                    <span>Local</span>
                  </>
                )}
              </p>
            </div>
          </div>
        </aside>
      </section>

      {/* Section 1: How to get the most out of the Study Track */}
      <section className="space-y-6">
        <div className="max-w-2xl space-y-2">
          <h2 className="text-2xl font-bold tracking-tight text-[#0F172A] [text-wrap:balance]">
            Como tirar o melhor proveito da trilha
          </h2>
          <p className="text-sm leading-relaxed text-slate-600">
            Reserve de 4 a 6 horas por semana seguindo esta sequência para transformar leitura em
            decisão arquitetural verificável.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {STUDY_PILLARS.map((pillar) => (
            <article
              key={pillar.index}
              className="bg-white border border-slate-200 rounded-xl p-6 flex flex-col justify-between gap-5"
            >
              <div className="space-y-2.5">
                <p className="font-mono text-xs font-semibold text-[#B45309] tabular-nums">
                  {pillar.index}.
                </p>
                <h3 className="text-lg font-semibold text-[#0F172A] leading-snug">
                  {pillar.title}
                </h3>
                <p className="text-sm leading-relaxed text-slate-600">{pillar.description}</p>
              </div>
              <div>
                <button
                  type="button"
                  onClick={() => handlePillarAction(pillar.target)}
                  className="inline-flex items-center gap-1.5 text-xs font-semibold text-[#14213D] hover:text-[#1E3158] underline underline-offset-4 decoration-slate-300 hover:decoration-[#14213D] transition-colors cursor-pointer"
                >
                  {pillar.actionLabel}
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </article>
          ))}
        </div>
      </section>

      {/* Section 2: Real 4-Cycle Structure + Track Selection */}
      {phases.length > 0 && (
        <section className="space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
            <div className="max-w-2xl space-y-2">
              <h2 className="text-2xl font-bold tracking-tight text-[#0F172A] [text-wrap:balance]">
                Estrutura da jornada em {phases.length} ciclos
              </h2>
              <p className="text-sm leading-relaxed text-slate-600">
                Você pode seguir as 32 semanas em ordem ou usar os filtros do Currículo para focar
                em RAG, Fine-Tuning, Produção/FinOps ou antecipar Inferência Causal após o Ciclo 1.
              </p>
            </div>
            <button
              type="button"
              onClick={onOpenCurriculum}
              className="inline-flex items-center gap-1.5 text-sm font-semibold text-[#14213D] hover:text-[#1E3158] underline underline-offset-4 shrink-0 cursor-pointer"
            >
              Ver todas as semanas
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
            {phases.map((phase) => {
              const phaseWeeks = weeks.filter((w) => w.phase === phase.id);
              return (
                <div
                  key={phase.id}
                  className="bg-white border border-slate-200 rounded-xl p-5 flex flex-col justify-between gap-4"
                >
                  <div className="space-y-2">
                    <p className="text-xs font-mono text-slate-500 tabular-nums">
                      Ciclo {phase.id} · {phase.weeksRange}
                    </p>
                    <h3 className="text-base font-semibold text-[#0F172A] leading-snug">
                      {phase.title}
                    </h3>
                    <p className="text-xs leading-relaxed text-slate-600">{phase.subtitle}</p>
                  </div>
                  <p className="text-xs font-mono text-slate-500 tabular-nums border-t border-slate-100 pt-3">
                    {phaseWeeks.length} semanas neste ciclo
                  </p>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* Section 3: Why Create an Account + LogBook Cloud Sync */}
      <section className="bg-white border border-slate-200 rounded-xl p-8 sm:p-10 space-y-8">
        <div className="max-w-2xl space-y-2">
          <h2 className="text-2xl font-bold tracking-tight text-[#0F172A] [text-wrap:balance]">
            Crie sua conta para não perder seu LogBook e seu andamento
          </h2>
          <p className="text-sm leading-relaxed text-slate-600">
            Sem login, suas marcações ficam presas ao armazenamento temporário do navegador atual.
            Ao entrar com sua conta Google, todo o seu histórico fica salvo e acessível de qualquer
            dispositivo.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-[#F8FAFC] border border-slate-200 rounded-xl p-6 space-y-3">
            <div className="flex items-center gap-2 text-sm font-semibold text-slate-700">
              <Laptop className="w-4 h-4 text-slate-500 shrink-0" />
              <span>Sem conta (Modo visitante local)</span>
            </div>
            <ul className="space-y-2 text-xs text-slate-600 leading-relaxed">
              <li>Progresso gravado apenas no navegador atual (localStorage).</li>
              <li>Se limpar o cache, trocar de computador ou usar aba anônima, os dados somem.</li>
              <li>Rascunhos de governança (ADR/RFC) não acompanham sua troca de dispositivo.</li>
            </ul>
          </div>

          <div className="bg-[#F8FAFC] border border-[#14213D]/25 rounded-xl p-6 space-y-3">
            <div className="flex items-center gap-2 text-sm font-semibold text-[#14213D]">
              <ShieldCheck className="w-4 h-4 text-[#14213D] shrink-0" />
              <span>Com sua conta Google (LogBook em nuvem)</span>
            </div>
            <ul className="space-y-2 text-xs text-slate-700 leading-relaxed">
              <li className="flex items-start gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-700 shrink-0 mt-0.5" />
                <span>Checklist de 6 passos das 32 semanas sincronizado em tempo real.</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-700 shrink-0 mt-0.5" />
                <span>Diário do LogBook com horas, comandos e bloqueios salvo na sua conta.</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-700 shrink-0 mt-0.5" />
                <span>Importação em 1 clique do que você já começou em modo visitante.</span>
              </li>
            </ul>
          </div>
        </div>

        <div className="border-t border-slate-200 pt-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="space-y-1">
            <p className="text-sm font-semibold text-[#0F172A]">
              {user
                ? 'Conta conectada e pronta para registrar seus estudos'
                : 'Passo 1: entre com sua conta Google · Passo 2: abra o Currículo e marque a Semana 1'}
            </p>
            <p className="text-xs text-slate-500">
              {user
                ? `Você já possui ${completedWeeks.length} semana(s) concluída(s) e ${logEntriesCount} registro(s) no LogBook.`
                : 'O cadastro é instantâneo no primeiro login com Google, sem senha extra para lembrar.'}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3 shrink-0">
            {!user ? (
              <button
                type="button"
                onClick={handlePrimaryAccountAction}
                className="inline-flex items-center gap-2 px-5 py-2.5 text-sm font-semibold text-white bg-[#14213D] rounded-lg hover:bg-[#1E3158] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                <LogIn className="w-4 h-4" />
                {firebaseConfigured ? 'Criar conta e salvar LogBook' : 'Ver instruções de conta'}
              </button>
            ) : (
              <button
                type="button"
                onClick={onOpenCurriculum}
                className="inline-flex items-center gap-2 px-5 py-2.5 text-sm font-semibold text-white bg-[#14213D] rounded-lg hover:bg-[#1E3158] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                <BookOpen className="w-4 h-4" />
                Continuar no Currículo
              </button>
            )}
            <button
              type="button"
              onClick={onOpenLogbook}
              className="inline-flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium text-slate-700 bg-white border border-slate-300 rounded-lg hover:bg-slate-50 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              <NotebookPen className="w-4 h-4" />
              Abrir LogBook
            </button>
          </div>
        </div>
      </section>

      {/* Quiet quick-jump strip */}
      <section className="flex flex-wrap items-center justify-between gap-4 border-t border-slate-200 pt-6 text-xs text-slate-500">
        <div className="flex items-center gap-2">
          <Compass className="w-4 h-4 text-slate-400 shrink-0" />
          <span>Quer revisar primeiro? Navegue livremente pelas 32 semanas, leituras e simuladores.</span>
        </div>
        <div className="flex items-center gap-5">
          <button
            type="button"
            onClick={onOpenCurriculum}
            className="inline-flex items-center gap-1 font-medium text-slate-700 hover:text-[#0F172A] cursor-pointer"
          >
            <BookOpen className="w-3.5 h-3.5" />
            Currículo
          </button>
          <button
            type="button"
            onClick={onOpenLabs}
            className="inline-flex items-center gap-1 font-medium text-slate-700 hover:text-[#0F172A] cursor-pointer"
          >
            <FlaskConical className="w-3.5 h-3.5" />
            Labs Interativos
          </button>
        </div>
      </section>
    </div>
  );
}
