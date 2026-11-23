import { Component, type ErrorInfo, type ReactNode } from 'react';
import { AlertTriangle, RotateCcw } from 'lucide-react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  message: string;
}

export default class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    message: '',
  };

  public static getDerivedStateFromError(error: Error): State {
    let readable = error.message || 'Ocorreu um erro inesperado na aplicação.';
    try {
      const parsed = JSON.parse(readable) as { error?: string; path?: string; operationType?: string };
      if (parsed && typeof parsed === 'object' && parsed.error) {
        readable = `Falha de permissão ou rede (${parsed.operationType ?? 'operação'} em ${parsed.path ?? 'banco'}): ${parsed.error}`;
      }
    } catch {
      // plain string error message
    }
    return { hasError: true, message: readable };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Application ErrorBoundary:', error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-[#F8FAFC] flex items-center justify-center px-6">
          <div className="max-w-lg w-full bg-white border border-slate-200 rounded-xl p-6 space-y-4">
            <div className="flex items-center gap-3 text-red-700">
              <AlertTriangle className="w-5 h-5 shrink-0" />
              <h1 className="text-base font-semibold">Erro de Execução ou Acesso ao Banco</h1>
            </div>
            <p className="text-sm text-slate-600 leading-relaxed">{this.state.message}</p>
            <button
              type="button"
              onClick={() => window.location.reload()}
              className="inline-flex items-center gap-2 px-4 py-2 text-xs font-medium bg-[#14213D] text-white rounded-lg hover:bg-[#1E3158] cursor-pointer"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Recarregar Aplicação</span>
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
