export type EnvironmentType = 'CPU' | 'T4 GPU';

export type ValidationStatus =
  | 'End-to-End Validated'
  | 'Substitute Model Validated'
  | 'Syntax Verified';

export type WeeklyStepId =
  | 'reading'
  | 'self_check'
  | 'exercise'
  | 'gabarito'
  | 'deliverable'
  | 'commit';

export interface WeekLogState {
  steps: Record<WeeklyStepId, boolean>;
  hoursSpent: number;
  lastUpdated: string;
  deliverableSummary: string;
  notes: string;
  blockers: string;
}

export interface PhaseInfo {
  id: number;
  slug: string;
  title: string;
  subtitle: string;
  weeksRange: string;
}

export interface WeekMeta {
  week: number;
  phase: number;
  phaseTitle: string;
  title: string;
  environment: EnvironmentType;
  validationStatus: ValidationStatus;
  validationNote?: string;
  colabExerciseUrl: string;
  colabGabaritoUrl: string;
  conceptRelativePath: string;
  exerciseRelativePath: string;
  gabaritoRelativePath: string;
  hasInteractiveLab?: boolean;
  labId?: string;
  labTitle?: string;
}

export interface NotebookCell {
  id: string;
  cellType: 'markdown' | 'code';
  source: string;
  hasTodo: boolean;
}

export interface WeekDetail extends WeekMeta {
  conceptMarkdown: string;
  exerciseCells: NotebookCell[];
  gabaritoCells: NotebookCell[];
}

export interface SearchHit {
  week: number;
  title: string;
  phase: number;
  phaseTitle: string;
  snippet: string;
  matchCount: number;
}

export interface TemplatesPayload {
  adrTemplate: string;
  rfcTemplate: string;
  logbookTemplate: string;
}

export interface LogbookEntry {
  id: string;
  date: string;
  week: number;
  title: string;
  hours: number;
  notes: string;
  blockers: string;
  commandSnippet?: string;
}

export type UserRole = 'admin' | 'student';
export type UserStatus = 'pending' | 'approved' | 'revoked';
export type RegistrationMode = 'approval_required' | 'open_authenticated';

export interface UserProfile {
  uid: string;
  email: string;
  displayName: string;
  photoURL?: string;
  role: UserRole;
  status: UserStatus;
  createdAt?: unknown;
  updatedAt?: unknown;
  lastLoginAt?: unknown;
}

export interface AccessConfig {
  registrationMode: RegistrationMode;
  updatedBy: string;
  updatedAt?: unknown;
}

export type ExportScope = 'week' | 'phase' | 'full';
export type ExportFormat = 'pdf' | 'epub';

export interface ExportChapter {
  id: string;
  title: string;
  kind: 'intro' | 'reading' | 'appendix-divider' | 'gabarito';
  week?: number;
  phase?: number;
  html: string;
}

export interface ExportBundlePayload {
  scope: ExportScope;
  title: string;
  subtitle: string;
  scopeLabel: string;
  weeksRange: string;
  author: string;
  activeWeeks: number[];
  includeNotebooks: boolean;
  generatedAt: string;
  chapters: ExportChapter[];
}
