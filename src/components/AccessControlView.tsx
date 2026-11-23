import { useState } from 'react';
import type { User } from 'firebase/auth';
import {
  CheckCircle2,
  Clock,
  KeyRound,
  Lock,
  LogIn,
  LogOut,
  ShieldAlert,
  ShieldCheck,
  Trash2,
  UserCheck,
  UserX,
  Users,
} from 'lucide-react';
import {
  adminApproveAllPending,
  adminDeleteUser,
  adminUpdateUser,
  isBootstrapAdminEmail,
  updateAccessPolicy,
} from '../firebase';
import type { RegistrationMode, UserProfile, UserRole, UserStatus } from '../types';

interface AccessControlViewProps {
  firebaseConfigured: boolean;
  user: User | null;
  profile: UserProfile | null;
  authLoading: boolean;
  authError: string | null;
  registrationMode: RegistrationMode;
  allUsers: UserProfile[];
  usersLoading: boolean;
  onSignIn: () => Promise<void>;
  onSignOut: () => Promise<void>;
  onOpenLogbook: () => void;
}

const STATUS_LABELS: Record<UserStatus, string> = {
  approved: 'Aprovado',
  pending: 'Aguardando aprovação',
  revoked: 'Acesso suspenso',
};

const ROLE_LABELS: Record<UserRole, string> = {
  admin: 'Administrador',
  student: 'Estudante',
};

export default function AccessControlView({
  firebaseConfigured,
  user,
  profile,
  authLoading,
  authError,
  registrationMode,
  allUsers,
  usersLoading,
  onSignIn,
  onSignOut,
  onOpenLogbook,
}: AccessControlViewProps) {
  const [busyUid, setBusyUid] = useState<string | null>(null);
  const [policyBusy, setPolicyBusy] = useState(false);
  const [actionFeedback, setActionFeedback] = useState<string | null>(null);

  const isBootstrappedAdmin = isBootstrapAdminEmail(user?.email) && user?.emailVerified;
  const isAdmin = profile?.role === 'admin' || isBootstrappedAdmin;

  const approvedCount = allUsers.filter((u) => u.status === 'approved').length;
  const pendingCount = allUsers.filter((u) => u.status === 'pending').length;
  const revokedCount = allUsers.filter((u) => u.status === 'revoked').length;

  const handleTogglePolicy = async (mode: RegistrationMode) => {
    if (!user || !isAdmin) return;
    setPolicyBusy(true);
    setActionFeedback(null);
    try {
      await updateAccessPolicy(user.uid, mode);
      setActionFeedback(
        mode === 'approval_required'
          ? 'Política atualizada: novos usuários aguardam aprovação do administrador.'
          : 'Política atualizada: usuários autenticados via Google são aprovados automaticamente.',
      );
    } catch (err) {
      setActionFeedback(err instanceof Error ? err.message : 'Erro ao atualizar política.');
    } finally {
      setPolicyBusy(false);
    }
  };

  const handleUserStatus = async (target: UserProfile, status: UserStatus) => {
    if (!user || !isAdmin) return;
    setBusyUid(target.uid);
    setActionFeedback(null);
    try {
      await adminUpdateUser(target, { status }, user.uid, Boolean(isBootstrappedAdmin));
      setActionFeedback(`Status de ${target.email} atualizado para ${STATUS_LABELS[status]}.`);
    } catch (err) {
      setActionFeedback(err instanceof Error ? err.message : 'Não foi possível atualizar o usuário.');
    } finally {
      setBusyUid(null);
    }
  };

  const handleUserRole = async (target: UserProfile, role: UserRole) => {
    if (!user || !isBootstrappedAdmin) return;
    setBusyUid(target.uid);
    setActionFeedback(null);
    try {
      await adminUpdateUser(target, { role }, user.uid, true);
      setActionFeedback(`Papel de ${target.email} atualizado para ${ROLE_LABELS[role]}.`);
    } catch (err) {
      setActionFeedback(err instanceof Error ? err.message : 'Não foi possível alterar o papel.');
    } finally {
      setBusyUid(null);
    }
  };

  const handleDeleteUser = async (target: UserProfile) => {
    if (!user || !isAdmin) return;
    setBusyUid(target.uid);
    setActionFeedback(null);
    try {
      await adminDeleteUser(target.uid);
      setActionFeedback(`Registro de ${target.email} removido da lista de acesso.`);
    } catch (err) {
      setActionFeedback(err instanceof Error ? err.message : 'Não foi possível remover o registro.');
    } finally {
      setBusyUid(null);
    }
  };

  const handleApproveAllPending = async () => {
    if (!user || !isAdmin || pendingCount === 0) return;
    setPolicyBusy(true);
    setActionFeedback(null);
    try {
      const count = await adminApproveAllPending(allUsers);
      setActionFeedback(`${count} usuário(s) pendente(s) aprovado(s).`);
    } catch (err) {
      setActionFeedback(err instanceof Error ? err.message : 'Erro ao aprovar pendentes.');
    } finally {
      setPolicyBusy(false);
    }
  };

  return (
    <div className="mx-auto max-w-[1200px] px-6 py-8 space-y-8">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-[#0F172A] tracking-tight">
            Conta, Banco de Dados &amp; Controle de Acesso
          </h1>
          <p className="text-sm text-slate-600 mt-1">
            O conteúdo da trilha de 32 semanas é aberto para estudo. A persistência em nuvem do LogBook e dos documentos de Governança exige conta autenticada e autorizada.
          </p>
        </div>
      </div>

      {authError && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-sm text-red-800 flex items-center gap-3">
          <ShieldAlert className="w-4 h-4 shrink-0" />
          <span>{authError}</span>
        </div>
      )}

      {/* Unconfigured Firebase on VPS/EasyPanel */}
      {!firebaseConfigured && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-6 max-w-3xl space-y-3 text-sm text-amber-950">
          <div className="font-semibold">Configuração do Firebase pendente neste servidor</div>
          <p className="text-amber-900 leading-relaxed">
            A aplicação está rodando normalmente no modo local, mas como o arquivo{' '}
            <code className="font-mono text-xs bg-amber-100 px-1.5 py-0.5 rounded">firebase-applet-config.json</code>{' '}
            fica fora do repositório Git, é necessário informar as credenciais públicas do seu Firebase nas variáveis de ambiente do EasyPanel:
          </p>
          <pre className="text-xs font-mono bg-[#0F172A] text-slate-100 p-4 rounded-lg overflow-x-auto">
            <code>{`VITE_ADMIN_EMAIL=dsjuvencio@gmail.com
FIREBASE_CONFIG_JSON={"projectId":"...","appId":"...","apiKey":"...","authDomain":"...","firestoreDatabaseId":"(default)"}`}</code>
          </pre>
          <p className="text-xs text-amber-900">
            Após salvar as variáveis no EasyPanel e reiniciar o serviço, o login com Google e o Firestore serão ativados automaticamente sem precisar alterar o código.
          </p>
        </div>
      )}

      {/* Unauthenticated State */}
      {firebaseConfigured && !authLoading && !user && (
        <div className="bg-white border border-slate-200 rounded-xl p-8 max-w-2xl space-y-5">
          <div className="flex items-center gap-3 text-[#14213D]">
            <KeyRound className="w-5 h-5" />
            <h2 className="text-lg font-semibold">Entrar para salvar progresso na nuvem</h2>
          </div>
          <p className="text-sm text-slate-600 leading-relaxed">
            Faça login com sua conta Google para vincular seus checklists de 192 passos, diário cronológico de sessões e rascunhos de ADR/RFC ao banco Firestore com isolamento por usuário.
          </p>
          <button
            type="button"
            onClick={() => void onSignIn()}
            className="inline-flex items-center gap-2 px-5 py-2.5 text-sm font-medium bg-[#14213D] text-white rounded-lg hover:bg-[#1E3158] transition-colors cursor-pointer"
          >
            <LogIn className="w-4 h-4" />
            <span>Entrar com Google</span>
          </button>
        </div>
      )}

      {/* Authenticated Account Card */}
      {user && (
        <div className="bg-white border border-slate-200 rounded-xl p-6 flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="flex items-start gap-4">
            {user.photoURL ? (
              <img
                src={user.photoURL}
                alt={profile?.displayName || user.displayName || 'Usuário'}
                referrerPolicy="no-referrer"
                className="w-12 h-12 rounded-lg border border-slate-200 object-cover shrink-0"
              />
            ) : (
              <div className="w-12 h-12 rounded-lg bg-[#14213D] text-white flex items-center justify-center font-semibold text-base shrink-0">
                {(profile?.displayName || user.email || 'U').slice(0, 1).toUpperCase()}
              </div>
            )}

            <div className="space-y-1">
              <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
                <h2 className="text-base font-semibold text-[#0F172A]">
                  {profile?.displayName || user.displayName || 'Usuário'}
                </h2>
                <span className="text-xs text-slate-400" aria-hidden="true">·</span>
                <span className="text-xs font-mono text-slate-600">{user.email}</span>
              </div>

              <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
                <span className="inline-flex items-center gap-1.5 font-medium text-slate-700">
                  {isAdmin ? (
                    <ShieldCheck className="w-3.5 h-3.5 text-[#0284C7]" />
                  ) : (
                    <Users className="w-3.5 h-3.5 text-slate-500" />
                  )}
                  <span>{profile ? ROLE_LABELS[profile.role] : 'Carregando papel...'}</span>
                </span>
                <span className="text-slate-300" aria-hidden="true">·</span>
                {profile?.status === 'approved' && (
                  <span className="inline-flex items-center gap-1.5 font-medium text-[#059669]">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    <span>Acesso aprovado para leitura e escrita no Firestore</span>
                  </span>
                )}
                {profile?.status === 'pending' && (
                  <span className="inline-flex items-center gap-1.5 font-medium text-amber-700">
                    <Clock className="w-3.5 h-3.5" />
                    <span>Aguardando aprovação do administrador</span>
                  </span>
                )}
                {profile?.status === 'revoked' && (
                  <span className="inline-flex items-center gap-1.5 font-medium text-red-700">
                    <Lock className="w-3.5 h-3.5" />
                    <span>Acesso ao banco suspenso pelo administrador</span>
                  </span>
                )}
              </div>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2 shrink-0">
            {profile?.status === 'approved' && (
              <button
                type="button"
                onClick={onOpenLogbook}
                className="px-4 py-2 text-xs font-medium bg-[#14213D] text-white hover:bg-[#1E3158] rounded-lg cursor-pointer"
              >
                Abrir Meu LogBook
              </button>
            )}
            <button
              type="button"
              onClick={() => void onSignOut()}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium text-slate-700 bg-white border border-slate-200 hover:bg-slate-50 rounded-lg cursor-pointer"
            >
              <LogOut className="w-3.5 h-3.5" />
              <span>Sair</span>
            </button>
          </div>
        </div>
      )}

      {/* Pending / Revoked explanation for standard students */}
      {user && profile?.status === 'pending' && !isAdmin && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-6 space-y-2 text-sm text-amber-900">
          <div className="font-semibold">Solicitação registrada</div>
          <p>
            Sua conta <strong>{user.email}</strong> foi autenticada e registrada com status pendente. Assim que o administrador aprovar seu acesso, a sincronização do seu LogBook e rascunhos de Governança será liberada automaticamente nesta tela.
          </p>
        </div>
      )}

      {user && profile?.status === 'revoked' && !isAdmin && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-6 space-y-2 text-sm text-red-900">
          <div className="font-semibold">Acesso restrito</div>
          <p>
            O administrador suspendeu a permissão de gravação para <strong>{user.email}</strong>. Você ainda pode consultar o Currículo, Leituras e Labs Interativos abertos.
          </p>
        </div>
      )}

      {/* Admin Access Management Dashboard */}
      {isAdmin && (
        <div className="space-y-6">
          {/* Policy Card */}
          <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200 pb-4">
              <div>
                <h2 className="text-base font-semibold text-[#0F172A]">
                  Política de Entrada de Novos Usuários
                </h2>
                <p className="text-xs text-slate-600 mt-1">
                  Define se novos usuários que entrarem com Google recebem acesso imediato ou ficam pendentes até sua aprovação manual.
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2 self-start shrink-0">
                <div className="flex items-center gap-1 p-1 bg-slate-100 border border-slate-200 rounded-lg">
                  <button
                    type="button"
                    disabled={policyBusy}
                    onClick={() => void handleTogglePolicy('open_authenticated')}
                    className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap cursor-pointer ${
                      registrationMode === 'open_authenticated'
                        ? 'bg-white text-[#0F172A] shadow-sm'
                        : 'text-slate-600 hover:text-[#0F172A]'
                    }`}
                  >
                    Acesso Aberto (Login Google)
                  </button>
                  <button
                    type="button"
                    disabled={policyBusy}
                    onClick={() => void handleTogglePolicy('approval_required')}
                    className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap cursor-pointer ${
                      registrationMode === 'approval_required'
                        ? 'bg-white text-[#0F172A] shadow-sm'
                        : 'text-slate-600 hover:text-[#0F172A]'
                    }`}
                  >
                    Aprovação Manual
                  </button>
                </div>
                {pendingCount > 0 && (
                  <button
                    type="button"
                    disabled={policyBusy}
                    onClick={() => void handleApproveAllPending()}
                    className="px-3 py-2 text-xs font-medium bg-[#059669] text-white hover:bg-emerald-700 rounded-lg whitespace-nowrap cursor-pointer disabled:opacity-50"
                  >
                    Aprovar Pendentes ({pendingCount})
                  </button>
                )}
              </div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 tabular-nums">
              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                <div className="text-xs text-slate-500">Total Cadastrados</div>
                <div className="text-xl font-bold font-mono text-[#0F172A] mt-0.5">{allUsers.length}</div>
              </div>
              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                <div className="text-xs text-slate-500">Aprovados</div>
                <div className="text-xl font-bold font-mono text-[#059669] mt-0.5">{approvedCount}</div>
              </div>
              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                <div className="text-xs text-slate-500">Pendentes</div>
                <div className="text-xl font-bold font-mono text-amber-700 mt-0.5">{pendingCount}</div>
              </div>
              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                <div className="text-xs text-slate-500">Suspensos</div>
                <div className="text-xl font-bold font-mono text-red-700 mt-0.5">{revokedCount}</div>
              </div>
            </div>
          </div>

          {actionFeedback && (
            <div className="bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-xs text-slate-700">
              {actionFeedback}
            </div>
          )}

          {/* Registered Users Table */}
          <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-200 flex items-center justify-between">
              <h2 className="text-base font-semibold text-[#0F172A]">
                Usuários Autenticados ({allUsers.length})
              </h2>
              <span className="text-xs text-slate-500">
                Gestão em tempo real · Firestore Rules RBAC
              </span>
            </div>

            {usersLoading ? (
              <div className="p-8 text-sm text-slate-500">Carregando usuários...</div>
            ) : allUsers.length === 0 ? (
              <div className="p-8 text-sm text-slate-500">Nenhum usuário registrado ainda.</div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-sm">
                  <thead>
                    <tr className="border-b border-slate-200 bg-slate-50 text-xs font-medium text-slate-500">
                      <th className="py-3 px-5">Usuário</th>
                      <th className="py-3 px-4">Papel</th>
                      <th className="py-3 px-4">Status de Acesso</th>
                      <th className="py-3 px-5 text-right">Controles</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200">
                    {allUsers.map((u) => {
                      const isOwnerRow = isBootstrapAdminEmail(u.email);
                      const isBusy = busyUid === u.uid;
                      return (
                        <tr key={u.uid} className="hover:bg-slate-50/60">
                          <td className="py-3.5 px-5">
                            <div className="font-medium text-[#0F172A]">{u.displayName}</div>
                            <div className="text-xs font-mono text-slate-500">{u.email}</div>
                          </td>
                          <td className="py-3.5 px-4 text-xs font-medium text-slate-700">
                            {ROLE_LABELS[u.role]}
                          </td>
                          <td className="py-3.5 px-4 text-xs font-medium">
                            {u.status === 'approved' && (
                              <span className="text-[#059669]">{STATUS_LABELS.approved}</span>
                            )}
                            {u.status === 'pending' && (
                              <span className="text-amber-700">{STATUS_LABELS.pending}</span>
                            )}
                            {u.status === 'revoked' && (
                              <span className="text-red-700">{STATUS_LABELS.revoked}</span>
                            )}
                          </td>
                          <td className="py-3.5 px-5">
                            <div className="flex items-center justify-end gap-2">
                              {!isOwnerRow && u.status !== 'approved' && (
                                <button
                                  type="button"
                                  disabled={isBusy}
                                  onClick={() => void handleUserStatus(u, 'approved')}
                                  className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-[#059669] bg-emerald-50 border border-emerald-200 hover:bg-emerald-100 rounded-md cursor-pointer disabled:opacity-50"
                                >
                                  <UserCheck className="w-3.5 h-3.5" />
                                  <span>Aprovar</span>
                                </button>
                              )}
                              {!isOwnerRow && u.status !== 'revoked' && (
                                <button
                                  type="button"
                                  disabled={isBusy}
                                  onClick={() => void handleUserStatus(u, 'revoked')}
                                  className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-amber-800 bg-amber-50 border border-amber-200 hover:bg-amber-100 rounded-md cursor-pointer disabled:opacity-50"
                                >
                                  <UserX className="w-3.5 h-3.5" />
                                  <span>Suspender</span>
                                </button>
                              )}
                              {!isOwnerRow && isBootstrappedAdmin && (
                                <button
                                  type="button"
                                  disabled={isBusy}
                                  onClick={() =>
                                    void handleUserRole(u, u.role === 'admin' ? 'student' : 'admin')
                                  }
                                  className="px-2.5 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-200 hover:bg-slate-50 rounded-md cursor-pointer disabled:opacity-50"
                                >
                                  {u.role === 'admin' ? 'Remover Admin' : 'Tornar Admin'}
                                </button>
                              )}
                              {!isOwnerRow && u.uid !== user?.uid && (
                                <button
                                  type="button"
                                  disabled={isBusy}
                                  onClick={() => void handleDeleteUser(u)}
                                  aria-label={`Remover ${u.email}`}
                                  className="p-1.5 text-slate-400 hover:text-red-600 cursor-pointer disabled:opacity-50"
                                >
                                  <Trash2 className="w-4 h-4" />
                                </button>
                              )}
                              {isOwnerRow && (
                                <span className="text-xs text-slate-400">Admin Principal</span>
                              )}
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
