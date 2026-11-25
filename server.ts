import express from 'express';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  buildEpubBuffer,
  buildExportBundle,
  buildExportFilename,
} from './export-builder.js';
import type {
  EnvironmentType,
  ExportScope,
  NotebookCell,
  PhaseInfo,
  SearchHit,
  ValidationStatus,
  WeekDetail,
  WeekMeta,
} from './src/types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = __dirname;
const GITHUB_COLAB_BASE =
  'https://colab.research.google.com/github/NetoRibeiro/applied-ai-architecture/blob/main/';

const PHASES: PhaseInfo[] = [
  {
    id: 1,
    slug: '01-Phase-1-Math-and-ML-Core',
    title: 'Math Foundations e ML Core',
    subtitle: 'Pipelines, linear algebra, calculus, gradient descent e PyTorch tensors.',
    weeksRange: 'Semanas 1–4',
  },
  {
    id: 2,
    slug: '02-Phase-2-Deep-Learning',
    title: 'Deep Learning e Ecossistema Hugging Face',
    subtitle: 'Training loops, tokenization, self-attention, mini-GPT e encoders.',
    weeksRange: 'Semanas 5–10',
  },
  {
    id: 3,
    slug: '03-Phase-3-Advanced-RAG',
    title: 'Advanced RAG, Search e Reranking',
    subtitle: 'Embeddings, HNSW, vector DBs, hybrid search, NDCG e faithfulness.',
    weeksRange: 'Semanas 11–16',
  },
  {
    id: 4,
    slug: '04-Phase-4-Fine-Tuning',
    title: 'Parameter-Efficient Fine-Tuning (PEFT)',
    subtitle: 'Instruction datasets, LoRA do zero, QLoRA 4-bit, avaliação e DPO.',
    weeksRange: 'Semanas 17–22',
  },
  {
    id: 5,
    slug: '05-Phase-5-MLOps-and-Agents',
    title: 'MLOps, Deployment e Agent-First Products',
    subtitle: 'Model serving, batching, MLflow, observability, guardrails e structured outputs.',
    weeksRange: 'Semanas 23–26',
  },
  {
    id: 6,
    slug: '06-Phase-6-Causal-Inference',
    title: 'Causal Inference e Experimentation',
    subtitle: 'Cluster A/B tests, Difference-in-Differences, PSM, IV e Synthetic Control.',
    weeksRange: 'Semanas 27–30',
  },
  {
    id: 7,
    slug: '07-Phase-7-Governance-and-Scale',
    title: 'Governance e Engineering at Scale',
    subtitle: 'ADRs, RFCs, reprodutibilidade com um comando, OEC e capstone final.',
    weeksRange: 'Semanas 31–32',
  },
];

interface RawWeekDef {
  week: number;
  phase: number;
  title: string;
  environment: EnvironmentType;
  subFolder?: string;
  labId?: string;
  labTitle?: string;
}

const WEEK_DEFS: RawWeekDef[] = [
  { week: 1, phase: 1, title: 'Scikit-Learn Pipelines e o Accuracy Paradox', environment: 'CPU', subFolder: '01-sklearn-pipelines', labId: 'threshold-cost', labTitle: 'Accuracy Paradox & Cost-Sensitive Thresholding' },
  { week: 2, phase: 1, title: 'A Geometria dos Dados: Linear Algebra com NumPy', environment: 'CPU', subFolder: '02-linear-algebra-numpy', labId: 'vector-cosine', labTitle: 'Vector Geometry & Cosine Similarity' },
  { week: 3, phase: 1, title: 'O Motor do Aprendizado: Calculus e Gradient Descent', environment: 'CPU', subFolder: '03-calculus-gradient', labId: 'gradient-descent', labTitle: 'Gradient Descent & Learning Rate Stability' },
  { week: 4, phase: 1, title: 'A Ponte para Tensors: PyTorch e Autograd', environment: 'T4 GPU', subFolder: '04-pytorch-tensors' },
  { week: 5, phase: 2, title: 'O Training Loop: nn.Module, DataLoader e MLP', environment: 'CPU' },
  { week: 6, phase: 2, title: 'Tokenization, Embeddings e um Bigram Language Model', environment: 'CPU' },
  { week: 7, phase: 2, title: 'Self-Attention do Zero', environment: 'CPU' },
  { week: 8, phase: 2, title: 'Montando um Mini-GPT', environment: 'T4 GPU' },
  { week: 9, phase: 2, title: 'Hugging Face por Dentro: Tokenizers, Hidden States e Generation', environment: 'T4 GPU' },
  { week: 10, phase: 2, title: 'Projeto: Fine-Tuning de um Encoder com Training Loop Manual', environment: 'T4 GPU' },
  { week: 11, phase: 3, title: 'Embeddings e Semantic Search (Exact vs. HNSW)', environment: 'T4 GPU', labId: 'vector-cosine', labTitle: 'Vector Geometry & Cosine Similarity' },
  { week: 12, phase: 3, title: 'Vector Databases: Qdrant, pgvector e Metadata Filtering', environment: 'CPU' },
  { week: 13, phase: 3, title: 'Chunking e o Primeiro RAG Pipeline', environment: 'T4 GPU' },
  { week: 14, phase: 3, title: 'Hybrid Search e Multi-Stage Reranking', environment: 'T4 GPU' },
  { week: 15, phase: 3, title: 'Ranking Metrics: Recall@K, MRR e NDCG', environment: 'CPU', labId: 'ranking-metrics', labTitle: 'RAG Ranking Metrics (Recall@K, MRR, NDCG@K)' },
  { week: 16, phase: 3, title: 'RAG Evaluation: Faithfulness, Context e Capstone', environment: 'T4 GPU' },
  { week: 17, phase: 4, title: 'Instruction Dataset Engineering (JSONL)', environment: 'CPU' },
  { week: 18, phase: 4, title: 'LoRA do Zero: a Matemática do Low Rank', environment: 'CPU' },
  { week: 19, phase: 4, title: 'Fine-Tuning com PEFT/LoRA', environment: 'T4 GPU' },
  { week: 20, phase: 4, title: 'QLoRA: 4-bit Quantization', environment: 'T4 GPU' },
  { week: 21, phase: 4, title: 'Avaliando o Fine-Tuned Model', environment: 'T4 GPU' },
  { week: 22, phase: 4, title: 'Preference Alignment: DPO do Zero', environment: 'T4 GPU' },
  { week: 23, phase: 5, title: 'Model Serving: Latency, Throughput e Batching', environment: 'T4 GPU', labId: 'llm-serving', labTitle: 'LLM Serving, KV-Cache & Batching Planner' },
  { week: 24, phase: 5, title: 'Experiment Tracking e Model Registry com MLflow', environment: 'CPU' },
  { week: 25, phase: 5, title: 'Observability e Guardrails', environment: 'CPU' },
  { week: 26, phase: 5, title: 'Agent-First Analytics: Structured Outputs para Agents', environment: 'CPU' },
  { week: 27, phase: 6, title: 'A/B Tests, Statistical Power e Spillover em Redes', environment: 'CPU' },
  { week: 28, phase: 6, title: 'Difference-in-Differences (DiD)', environment: 'CPU' },
  { week: 29, phase: 6, title: 'Propensity Score Matching e Instrumental Variables', environment: 'CPU' },
  { week: 30, phase: 6, title: 'Synthetic Control', environment: 'CPU' },
  { week: 31, phase: 7, title: 'RFCs, ADRs e Reproducibility com um Comando', environment: 'CPU' },
  { week: 32, phase: 7, title: 'OEC: a Métrica Certa e o Capstone Final', environment: 'CPU' },
];

const SUBSTITUTE_WEEKS = new Set([9, 11, 13, 14, 16, 19, 21, 22, 23]);
const SYNTAX_ONLY_WEEKS = new Set([10, 20]);

function getValidationInfo(week: number): { status: ValidationStatus; note?: string } {
  if (SYNTAX_ONLY_WEEKS.has(week)) {
    return {
      status: 'Syntax Verified',
      note: week === 10
        ? 'Requer download do dataset AG News no Colab.'
        : 'Parte B (QLoRA com bitsandbytes) requer GPU CUDA.',
    };
  }
  if (SUBSTITUTE_WEEKS.has(week)) {
    return {
      status: 'Substitute Model Validated',
      note: 'Validado com modelo mínimo substituto; execute na T4 GPU para resultados completos.',
    };
  }
  if (week === 12) {
    return {
      status: 'End-to-End Validated',
      note: 'Executado de ponta a ponta (exceto célula opcional de pgvector).',
    };
  }
  return { status: 'End-to-End Validated' };
}

function buildWeekMeta(def: RawWeekDef): WeekMeta {
  const phase = PHASES.find((p) => p.id === def.phase)!;
  const pad = String(def.week).padStart(2, '0');
  const baseDir = def.subFolder ? `${phase.slug}/${def.subFolder}` : phase.slug;
  const conceptRelativePath = `${baseDir}/docs/Semana-${pad}-Conceitos.md`;
  const exerciseRelativePath = `${baseDir}/notebooks/semana_${pad}_exercicio.ipynb`;
  const gabaritoRelativePath = `${baseDir}/notebooks/semana_${pad}_gabarito.ipynb`;
  const val = getValidationInfo(def.week);

  return {
    week: def.week,
    phase: def.phase,
    phaseTitle: phase.title,
    title: def.title,
    environment: def.environment,
    validationStatus: val.status,
    validationNote: val.note,
    colabExerciseUrl: GITHUB_COLAB_BASE + exerciseRelativePath,
    colabGabaritoUrl: GITHUB_COLAB_BASE + gabaritoRelativePath,
    conceptRelativePath,
    exerciseRelativePath,
    gabaritoRelativePath,
    hasInteractiveLab: Boolean(def.labId),
    labId: def.labId,
    labTitle: def.labTitle,
  };
}

const WEEKS_META: WeekMeta[] = WEEK_DEFS.map(buildWeekMeta);

async function parseNotebook(filePath: string): Promise<NotebookCell[]> {
  const raw = await fs.readFile(filePath, 'utf-8');
  const parsed = JSON.parse(raw) as {
    cells?: Array<{
      id?: string;
      cell_type?: string;
      source?: string[] | string;
    }>;
  };
  const cells = parsed.cells ?? [];
  return cells.map((cell, index) => {
    const sourceText = Array.isArray(cell.source)
      ? cell.source.join('')
      : cell.source ?? '';
    const cellType = cell.cell_type === 'code' ? 'code' : 'markdown';
    return {
      id: cell.id ?? `cell-${index}`,
      cellType,
      source: sourceText,
      hasTodo: sourceText.includes('# [SEU CÓDIGO AQUI]'),
    };
  });
}

async function startServer() {
  const app = express();
  app.use(express.json());

  app.get('/api/health', (_req, res) => {
    res.json({ ok: true });
  });

  app.get('/api/public-config', async (_req, res) => {
    const cleanEnv = (val?: string) => (val ?? '').trim().replace(/^['"]|['"]$/g, '');
    const normalizeAuthDomain = (val: string) =>
      cleanEnv(val).replace(/^https?:\/\//i, '').replace(/\/+$/, '');

    const adminEmail = cleanEnv(process.env.VITE_ADMIN_EMAIL || process.env.ADMIN_EMAIL).toLowerCase();

    let firebase: Record<string, string> | null = null;

    if (process.env.FIREBASE_CONFIG_JSON) {
      try {
        const parsed = JSON.parse(cleanEnv(process.env.FIREBASE_CONFIG_JSON)) as Record<
          string,
          string
        >;
        if (parsed.apiKey && parsed.projectId && parsed.appId) {
          firebase = {
            ...parsed,
            apiKey: cleanEnv(parsed.apiKey),
            projectId: cleanEnv(parsed.projectId),
            appId: cleanEnv(parsed.appId),
            authDomain: normalizeAuthDomain(
              parsed.authDomain || `${cleanEnv(parsed.projectId)}.firebaseapp.com`,
            ),
            firestoreDatabaseId: cleanEnv(parsed.firestoreDatabaseId) || '(default)',
          };
        }
      } catch {
        // invalid FIREBASE_CONFIG_JSON env var
      }
    }

    if (
      !firebase &&
      process.env.FIREBASE_API_KEY &&
      process.env.FIREBASE_PROJECT_ID &&
      process.env.FIREBASE_APP_ID
    ) {
      const projectId = cleanEnv(process.env.FIREBASE_PROJECT_ID);
      firebase = {
        apiKey: cleanEnv(process.env.FIREBASE_API_KEY),
        authDomain: normalizeAuthDomain(
          process.env.FIREBASE_AUTH_DOMAIN || `${projectId}.firebaseapp.com`,
        ),
        projectId,
        storageBucket:
          cleanEnv(process.env.FIREBASE_STORAGE_BUCKET) || `${projectId}.firebasestorage.app`,
        messagingSenderId: cleanEnv(process.env.FIREBASE_MESSAGING_SENDER_ID),
        appId: cleanEnv(process.env.FIREBASE_APP_ID),
        firestoreDatabaseId: cleanEnv(process.env.FIREBASE_DATABASE_ID) || '(default)',
      };
    }

    if (!firebase) {
      try {
        const rawConfig = await fs.readFile(
          path.join(ROOT_DIR, 'firebase-applet-config.json'),
          'utf-8',
        );
        const parsed = JSON.parse(rawConfig) as Record<string, string>;
        if (parsed.apiKey && parsed.projectId && parsed.appId) {
          firebase = parsed;
        }
      } catch {
        // firebase-applet-config.json not present in container
      }
    }

    res.json({ adminEmail, firebase });
  });

  app.get('/api/curriculum', (_req, res) => {
    res.json({
      phases: PHASES,
      weeks: WEEKS_META,
    });
  });

  app.get('/api/week/:num', async (req, res) => {
    const num = Number.parseInt(req.params.num, 10);
    if (!Number.isInteger(num) || num < 1 || num > 32) {
      res.status(400).json({ error: 'Semana inválida. Escolha entre 1 e 32.' });
      return;
    }

    const meta = WEEKS_META.find((w) => w.week === num);
    if (!meta) {
      res.status(404).json({ error: 'Semana não encontrada.' });
      return;
    }

    try {
      const [conceptMarkdown, exerciseCells, gabaritoCells] = await Promise.all([
        fs.readFile(path.join(ROOT_DIR, meta.conceptRelativePath), 'utf-8'),
        parseNotebook(path.join(ROOT_DIR, meta.exerciseRelativePath)),
        parseNotebook(path.join(ROOT_DIR, meta.gabaritoRelativePath)),
      ]);

      const payload: WeekDetail = {
        ...meta,
        conceptMarkdown,
        exerciseCells,
        gabaritoCells,
      };
      res.json(payload);
    } catch (error) {
      console.error(`Failed loading week ${num}:`, error);
      res.status(500).json({ error: 'Não foi possível carregar os arquivos desta semana.' });
    }
  });

  app.get('/api/templates', async (_req, res) => {
    try {
      const [adrTemplate, rfcTemplate, logbookTemplate] = await Promise.all([
        fs.readFile(
          path.join(ROOT_DIR, '07-Phase-7-Governance-and-Scale/docs/templates/ADR-template.md'),
          'utf-8',
        ),
        fs.readFile(
          path.join(ROOT_DIR, '07-Phase-7-Governance-and-Scale/docs/templates/RFC-template.md'),
          'utf-8',
        ),
        fs.readFile(path.join(ROOT_DIR, 'LOGBOOK.md'), 'utf-8'),
      ]);
      res.json({ adrTemplate, rfcTemplate, logbookTemplate });
    } catch (error) {
      console.error('Failed loading templates:', error);
      res.status(500).json({ error: 'Erro ao carregar templates.' });
    }
  });

  app.get('/api/search', async (req, res) => {
    const query = String(req.query.q ?? '').trim();
    if (query.length < 2) {
      res.json({ hits: [] });
      return;
    }

    const qLower = query.toLowerCase();
    const hits: SearchHit[] = [];

    await Promise.all(
      WEEKS_META.map(async (meta) => {
        try {
          const text = await fs.readFile(path.join(ROOT_DIR, meta.conceptRelativePath), 'utf-8');
          const lower = text.toLowerCase();
          const titleMatch = meta.title.toLowerCase().includes(qLower);
          const idx = lower.indexOf(qLower);

          if (idx === -1 && !titleMatch) return;

          let matchCount = titleMatch ? 1 : 0;
          let pos = 0;
          while ((pos = lower.indexOf(qLower, pos)) !== -1) {
            matchCount += 1;
            pos += qLower.length;
          }

          const start = Math.max(0, (idx === -1 ? 0 : idx) - 70);
          const end = Math.min(text.length, (idx === -1 ? 140 : idx + qLower.length + 110));
          const rawSnippet = text
            .slice(start, end)
            .replace(/[#>*`\n]+/g, ' ')
            .replace(/\s+/g, ' ')
            .trim();

          hits.push({
            week: meta.week,
            title: meta.title,
            phase: meta.phase,
            phaseTitle: meta.phaseTitle,
            snippet: (start > 0 ? '…' : '') + rawSnippet + (end < text.length ? '…' : ''),
            matchCount,
          });
        } catch {
          // skip unreadable file
        }
      }),
    );

    hits.sort((a, b) => b.matchCount - a.matchCount || a.week - b.week);
    res.json({ hits: hits.slice(0, 12) });
  });

  function parseExportParams(query: Record<string, unknown>): {
    scope: ExportScope;
    phase: number;
    week: number;
    includeNotebooks: boolean;
  } | { error: string } {
    const rawScope = String(query.scope ?? 'full').trim();
    if (rawScope !== 'week' && rawScope !== 'phase' && rawScope !== 'full') {
      return { error: 'Escopo inválido. Use week, phase ou full.' };
    }
    const phase = Number.parseInt(String(query.phase ?? '1'), 10);
    if (rawScope === 'phase' && (!Number.isInteger(phase) || phase < 1 || phase > 7)) {
      return { error: 'Fase inválida. Escolha entre 1 e 7.' };
    }
    const week = Number.parseInt(String(query.week ?? '1'), 10);
    if (rawScope === 'week' && (!Number.isInteger(week) || week < 1 || week > 32)) {
      return { error: 'Semana inválida. Escolha entre 1 e 32.' };
    }
    const includeNotebooks = String(query.includeNotebooks ?? 'true') !== 'false';
    return {
      scope: rawScope,
      phase: Number.isInteger(phase) ? phase : 1,
      week: Number.isInteger(week) ? week : 1,
      includeNotebooks,
    };
  }

  app.get('/api/export/bundle', async (req, res) => {
    const parsed = parseExportParams(req.query as Record<string, unknown>);
    if ('error' in parsed) {
      res.status(400).json({ error: parsed.error });
      return;
    }

    try {
      const bundle = await buildExportBundle(
        ROOT_DIR,
        PHASES,
        WEEKS_META,
        parseNotebook,
        parsed,
      );
      res.json(bundle);
    } catch (error) {
      console.error('Failed building export bundle:', error);
      res.status(500).json({ error: 'Não foi possível gerar o conteúdo para exportação.' });
    }
  });

  app.get('/api/export/epub', async (req, res) => {
    const parsed = parseExportParams(req.query as Record<string, unknown>);
    if ('error' in parsed) {
      res.status(400).json({ error: parsed.error });
      return;
    }

    try {
      const bundle = await buildExportBundle(
        ROOT_DIR,
        PHASES,
        WEEKS_META,
        parseNotebook,
        parsed,
      );
      const epubBuffer = await buildEpubBuffer(bundle);
      const filename = buildExportFilename(
        parsed.scope,
        'epub',
        parsed.phase,
        parsed.week,
      );
      res.setHeader('Content-Type', 'application/epub+zip');
      res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
      res.send(epubBuffer);
    } catch (error) {
      console.error('Failed generating EPUB export:', error);
      res.status(500).json({ error: 'Não foi possível gerar o arquivo EPUB.' });
    }
  });

  const isProd = process.env.NODE_ENV === 'production';
  if (!isProd) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.join(ROOT_DIR, 'dist');
    app.use(express.static(distDir));
    app.use((req, res, next) => {
      if (req.method !== 'GET' || req.path.startsWith('/api/')) {
        next();
        return;
      }
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  const port = Number.parseInt(process.env.PORT ?? '3000', 10);
  app.listen(port, '0.0.0.0', () => {
    console.log(`AI Architect Foundations workbench listening on http://0.0.0.0:${port}`);
  });
}

startServer().catch((err) => {
  console.error('Fatal startup error:', err);
  process.exit(1);
});
