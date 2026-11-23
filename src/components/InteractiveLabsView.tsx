import { useEffect, useMemo, useState } from 'react';
import { Pause, Play, RotateCcw, StepForward } from 'lucide-react';

interface InteractiveLabsViewProps {
  activeLabId: string;
  onSelectLab: (labId: string) => void;
  onOpenWeek: (week: number) => void;
}

const LABS = [
  { id: 'threshold-cost', label: 'Semana 1 · Threshold & Custo', week: 1 },
  { id: 'vector-cosine', label: 'Semanas 2/11 · Vetores & Cosseno', week: 2 },
  { id: 'gradient-descent', label: 'Semana 3 · Gradient Descent', week: 3 },
  { id: 'ranking-metrics', label: 'Semana 15 · Recall@K, MRR & NDCG', week: 15 },
  { id: 'llm-serving', label: 'Semana 23 · LLM Serving & KV-Cache', week: 23 },
];

// Standard normal CDF approximation (Abramowitz & Stegun)
function normCdf(x: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989423 * Math.exp((-x * x) / 2);
  const p =
    d *
    t *
    (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return x > 0 ? 1 - p : p;
}

function logit(p: number): number {
  const clamped = Math.min(0.999, Math.max(0.001, p));
  return Math.log(clamped / (1 - clamped));
}

function ThresholdCostLab() {
  const [prevalencePct, setPrevalencePct] = useState(0.5);
  const [separation, setSeparation] = useState(2.3);
  const [threshold, setThreshold] = useState(0.5);
  const [costFP, setCostFP] = useState(15);
  const [costFN, setCostFN] = useState(250);

  const totalSamples = 10000;

  const evalAt = (thr: number) => {
    const zCut = logit(thr) / Math.max(0.5, separation * 0.6) + separation * 0.35;
    const posTotal = Math.round(totalSamples * (prevalencePct / 100));
    const negTotal = totalSamples - posTotal;
    const tpr = 1 - normCdf(zCut - separation);
    const fpr = 1 - normCdf(zCut);
    const tp = Math.round(posTotal * tpr);
    const fn = posTotal - tp;
    const fp = Math.round(negTotal * fpr);
    const tn = negTotal - fp;
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = posTotal > 0 ? tp / posTotal : 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    const accuracy = (tp + tn) / totalSamples;
    const totalCost = fp * costFP + fn * costFN;
    return { tp, fp, fn, tn, precision, recall, f1, accuracy, totalCost, posTotal, negTotal };
  };

  const current = evalAt(threshold);
  const naiveAccuracy = 1 - prevalencePct / 100;
  const naiveCost = Math.round(totalSamples * (prevalencePct / 100)) * costFN;

  const curvePoints = useMemo(() => {
    const pts: Array<{ thr: number; precision: number; recall: number; cost: number }> = [];
    for (let t = 0.05; t <= 0.951; t += 0.025) {
      const r = evalAt(Number(t.toFixed(3)));
      pts.push({
        thr: Number(t.toFixed(3)),
        precision: r.precision,
        recall: r.recall,
        cost: r.totalCost,
      });
    }
    return pts;
  }, [prevalencePct, separation, costFP, costFN]);

  const optimalPoint = useMemo(() => {
    return curvePoints.reduce((best, p) => (p.cost < best.cost ? p : best), curvePoints[0]);
  }, [curvePoints]);

  const maxCost = Math.max(...curvePoints.map((p) => p.cost), naiveCost, 1);

  const prPath = curvePoints
    .map((p, i) => {
      const x = 40 + p.recall * 250;
      const y = 210 - p.precision * 175;
      return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(' ');

  const costPath = curvePoints
    .map((p, i) => {
      const x = 40 + ((p.thr - 0.05) / 0.9) * 250;
      const y = 210 - (p.cost / maxCost) * 175;
      return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(' ');

  const curPrX = 40 + current.recall * 250;
  const curPrY = 210 - current.precision * 175;
  const curCostX = 40 + ((threshold - 0.05) / 0.9) * 250;
  const curCostY = 210 - (current.totalCost / maxCost) * 175;
  const optCostX = 40 + ((optimalPoint.thr - 0.05) / 0.9) * 250;
  const optCostY = 210 - (optimalPoint.cost / maxCost) * 175;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
      {/* Left Interactive Stage */}
      <div className="lg:col-span-8 space-y-6">
        <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-6">
          <div className="flex flex-wrap items-baseline justify-between gap-4 border-b border-slate-200 pb-4">
            <div>
              <h2 className="text-lg font-semibold text-[#0F172A]">
                Curva Precision-Recall &amp; Custo Total de Negócio
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Base simulada de {totalSamples.toLocaleString('pt-BR')} transações ({current.posTotal} fraudes reais)
              </p>
            </div>
            <div className="text-xs font-mono text-slate-600 tabular-nums">
              Baseline ingênuo (sempre 0): {(naiveAccuracy * 100).toFixed(2)}% accuracy · R$ {naiveCost.toLocaleString('pt-BR')}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Precision-Recall SVG */}
            <div className="bg-slate-50 border border-slate-200 rounded-lg p-4">
              <div className="text-xs font-medium text-slate-700 mb-2">
                Precision-Recall Curve (Threshold = {threshold.toFixed(2)})
              </div>
              <svg viewBox="0 0 320 245" className="w-full h-auto">
                <line x1="40" y1="35" x2="40" y2="210" stroke="#CBD5E1" strokeWidth="1" />
                <line x1="40" y1="210" x2="290" y2="210" stroke="#CBD5E1" strokeWidth="1" />
                <line x1="40" y1="122" x2="290" y2="122" stroke="#E2E8F0" strokeDasharray="3 3" />
                <line x1="165" y1="35" x2="165" y2="210" stroke="#E2E8F0" strokeDasharray="3 3" />
                <path d={prPath} fill="none" stroke="#0284C7" strokeWidth="2.5" />
                <circle cx={curPrX} cy={curPrY} r="5.5" fill="#D97706" stroke="#FFFFFF" strokeWidth="2" />
                <text x="165" y="234" textAnchor="middle" fontSize="11" fill="#475569">
                  Recall (Sensibilidade)
                </text>
                <text
                  x="14"
                  y="122"
                  textAnchor="middle"
                  fontSize="11"
                  fill="#475569"
                  transform="rotate(-90 14 122)"
                >
                  Precision
                </text>
                <text x="34" y="213" textAnchor="end" fontSize="10" fill="#64748B">0</text>
                <text x="34" y="40" textAnchor="end" fontSize="10" fill="#64748B">1.0</text>
                <text x="290" y="224" textAnchor="end" fontSize="10" fill="#64748B">1.0</text>
              </svg>
            </div>

            {/* Cost Curve SVG */}
            <div className="bg-slate-50 border border-slate-200 rounded-lg p-4">
              <div className="text-xs font-medium text-slate-700 mb-2">
                Custo Financeiro por Threshold (Ótimo t* = {optimalPoint.thr.toFixed(2)})
              </div>
              <svg viewBox="0 0 320 245" className="w-full h-auto">
                <line x1="40" y1="35" x2="40" y2="210" stroke="#CBD5E1" strokeWidth="1" />
                <line x1="40" y1="210" x2="290" y2="210" stroke="#CBD5E1" strokeWidth="1" />
                <line x1="40" y1="122" x2="290" y2="122" stroke="#E2E8F0" strokeDasharray="3 3" />
                <path d={costPath} fill="none" stroke="#14213D" strokeWidth="2.5" />
                <circle cx={optCostX} cy={optCostY} r="5" fill="#059669" stroke="#FFFFFF" strokeWidth="1.5" />
                <circle cx={curCostX} cy={curCostY} r="5.5" fill="#D97706" stroke="#FFFFFF" strokeWidth="2" />
                <text x="165" y="234" textAnchor="middle" fontSize="11" fill="#475569">
                  Decision Threshold (t)
                </text>
                <text
                  x="14"
                  y="122"
                  textAnchor="middle"
                  fontSize="11"
                  fill="#475569"
                  transform="rotate(-90 14 122)"
                >
                  Custo Total (R$)
                </text>
                <text x="40" y="224" textAnchor="start" fontSize="10" fill="#64748B">0.05</text>
                <text x="290" y="224" textAnchor="end" fontSize="10" fill="#64748B">0.95</text>
              </svg>
            </div>
          </div>

          {/* Confusion Matrix & Derived Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-12 gap-6 pt-2">
            <div className="md:col-span-6">
              <h3 className="text-xs font-semibold text-slate-700 mb-3">
                Confusion Matrix (t = {threshold.toFixed(2)})
              </h3>
              <div className="grid grid-cols-2 gap-2 text-center tabular-nums">
                <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
                  <div className="text-xs text-slate-500">True Negative (TN)</div>
                  <div className="text-xl font-bold font-mono text-[#0F172A] mt-1">
                    {current.tn.toLocaleString('pt-BR')}
                  </div>
                </div>
                <div className="bg-amber-50/70 border border-amber-200 rounded-lg p-3">
                  <div className="text-xs text-amber-800">False Positive (FP · Alarme)</div>
                  <div className="text-xl font-bold font-mono text-amber-900 mt-1">
                    {current.fp.toLocaleString('pt-BR')}
                  </div>
                </div>
                <div className="bg-red-50/70 border border-red-200 rounded-lg p-3">
                  <div className="text-xs text-red-800">False Negative (FN · Omissão)</div>
                  <div className="text-xl font-bold font-mono text-red-900 mt-1">
                    {current.fn.toLocaleString('pt-BR')}
                  </div>
                </div>
                <div className="bg-emerald-50/70 border border-emerald-200 rounded-lg p-3">
                  <div className="text-xs text-emerald-800">True Positive (TP · Detectada)</div>
                  <div className="text-xl font-bold font-mono text-emerald-900 mt-1">
                    {current.tp.toLocaleString('pt-BR')}
                  </div>
                </div>
              </div>
            </div>

            <div className="md:col-span-6 grid grid-cols-2 gap-3 tabular-nums">
              <div className="border border-slate-200 rounded-lg p-3.5">
                <div className="text-xs text-slate-500">Precision</div>
                <div className="text-xl font-bold font-mono text-[#0F172A] mt-1">
                  {(current.precision * 100).toFixed(1)}%
                </div>
              </div>
              <div className="border border-slate-200 rounded-lg p-3.5">
                <div className="text-xs text-slate-500">Recall</div>
                <div className="text-xl font-bold font-mono text-[#0F172A] mt-1">
                  {(current.recall * 100).toFixed(1)}%
                </div>
              </div>
              <div className="border border-slate-200 rounded-lg p-3.5">
                <div className="text-xs text-slate-500">F1-Score / Accuracy</div>
                <div className="text-lg font-bold font-mono text-[#0F172A] mt-1">
                  {current.f1.toFixed(3)} / {(current.accuracy * 100).toFixed(2)}%
                </div>
              </div>
              <div className="border border-slate-200 rounded-lg p-3.5 bg-slate-50">
                <div className="text-xs text-slate-500">Custo Total (FP·{costFP} + FN·{costFN})</div>
                <div className="text-xl font-bold font-mono text-[#14213D] mt-1">
                  R$ {current.totalCost.toLocaleString('pt-BR')}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Right Control & Concept Deck */}
      <div className="lg:col-span-4 bg-white border border-slate-200 rounded-xl p-6 space-y-6">
        <div className="flex items-center justify-between border-b border-slate-200 pb-4">
          <h3 className="text-sm font-semibold text-[#0F172A]">Parâmetros do Modelo</h3>
          <button
            type="button"
            onClick={() => setThreshold(optimalPoint.thr)}
            className="px-3 py-1.5 text-xs font-medium bg-[#059669] text-white rounded-md hover:bg-emerald-700 transition-colors cursor-pointer"
          >
            Aplicar Ótimo (t={optimalPoint.thr.toFixed(2)})
          </button>
        </div>

        <div className="space-y-5">
          <SliderControl
            label="Decision Threshold (t)"
            value={threshold}
            display={threshold.toFixed(2)}
            min={0.05}
            max={0.95}
            step={0.025}
            onChange={setThreshold}
          />
          <SliderControl
            label="Taxa de Fraude (Prevalence)"
            value={prevalencePct}
            display={`${prevalencePct.toFixed(1)}%`}
            min={0.2}
            max={10}
            step={0.2}
            onChange={setPrevalencePct}
          />
          <SliderControl
            label="Separação do Classificador (d')"
            value={separation}
            display={separation.toFixed(1)}
            min={1.0}
            max={3.5}
            step={0.1}
            onChange={setSeparation}
          />
          <SliderControl
            label="Custo False Positive (Alarme)"
            value={costFP}
            display={`R$ ${costFP}`}
            min={5}
            max={100}
            step={5}
            onChange={setCostFP}
          />
          <SliderControl
            label="Custo False Negative (Fraude)"
            value={costFN}
            display={`R$ ${costFN}`}
            min={50}
            max={1000}
            step={25}
            onChange={setCostFN}
          />
        </div>

        <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 text-xs leading-relaxed text-slate-600 space-y-2">
          <p className="font-semibold text-[#0F172A]">Por que t = 0,50 raramente é ótimo</p>
          <p>
            Quando o False Negative custa {Math.round(costFN / costFP)}x mais que o False Positive,
            o ponto ótimo se desloca para baixo. Reduzir o threshold captura mais fraudes sem
            retreinar o modelo.
          </p>
        </div>
      </div>
    </div>
  );
}

function VectorCosineLab() {
  const [ux, setUx] = useState(3.2);
  const [uy, setUy] = useState(1.8);
  const [vx, setVx] = useState(2.4);
  const [vy, setVy] = useState(3.0);

  const dot = ux * vx + uy * vy;
  const normU = Math.hypot(ux, uy);
  const normV = Math.hypot(vx, vy);
  const cosine = normU > 0 && normV > 0 ? dot / (normU * normV) : 0;
  const angleDeg = (Math.acos(Math.min(1, Math.max(-1, cosine))) * 180) / Math.PI;
  const l2 = Math.hypot(ux - vx, uy - vy);

  const toSvgX = (x: number) => 160 + x * 28;
  const toSvgY = (y: number) => 160 - y * 28;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
      <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-6 space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 pb-4">
          <div>
            <h2 className="text-lg font-semibold text-[#0F172A]">
              Espaço Vetorial 2D: Dot Product vs. Cosine Similarity
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Base matemática de embeddings e busca vetorial em RAG
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => { setUx(3.5); setUy(1.2); setVx(2.8); setVy(1.0); }}
              className="px-2.5 py-1 text-xs font-medium bg-slate-100 hover:bg-slate-200 rounded text-slate-700 cursor-pointer"
            >
              Similares
            </button>
            <button
              type="button"
              onClick={() => { setUx(3.5); setUy(0); setVx(0); setVy(3.5); }}
              className="px-2.5 py-1 text-xs font-medium bg-slate-100 hover:bg-slate-200 rounded text-slate-700 cursor-pointer"
            >
              Ortogonais
            </button>
            <button
              type="button"
              onClick={() => { setUx(3.0); setUy(1.5); setVx(-3.0); setVy(-1.5); }}
              className="px-2.5 py-1 text-xs font-medium bg-slate-100 hover:bg-slate-200 rounded text-slate-700 cursor-pointer"
            >
              Opostos
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
          <div className="md:col-span-7 bg-slate-50 border border-slate-200 rounded-lg p-3">
            <svg viewBox="0 0 320 320" className="w-full max-w-[340px] mx-auto h-auto">
              {[-4, -2, 2, 4].map((v) => (
                <g key={v}>
                  <line x1={toSvgX(v)} y1="20" x2={toSvgX(v)} y2="300" stroke="#E2E8F0" strokeWidth="1" />
                  <line x1="20" y1={toSvgY(v)} x2="300" y2={toSvgY(v)} stroke="#E2E8F0" strokeWidth="1" />
                </g>
              ))}
              <line x1="20" y1="160" x2="300" y2="160" stroke="#94A3B8" strokeWidth="1.5" />
              <line x1="160" y1="20" x2="160" y2="300" stroke="#94A3B8" strokeWidth="1.5" />

              <line
                x1={toSvgX(ux)}
                y1={toSvgY(uy)}
                x2={toSvgX(vx)}
                y2={toSvgY(vy)}
                stroke="#94A3B8"
                strokeDasharray="4 4"
                strokeWidth="1.5"
              />

              <line
                x1="160"
                y1="160"
                x2={toSvgX(ux)}
                y2={toSvgY(uy)}
                stroke="#0284C7"
                strokeWidth="3"
                strokeLinecap="round"
              />
              <circle cx={toSvgX(ux)} cy={toSvgY(uy)} r="5" fill="#0284C7" />
              <text x={toSvgX(ux) + 8} y={toSvgY(uy) - 6} fontSize="12" fontWeight="600" fill="#0284C7">
                u ({ux.toFixed(1)}, {uy.toFixed(1)})
              </text>

              <line
                x1="160"
                y1="160"
                x2={toSvgX(vx)}
                y2={toSvgY(vy)}
                stroke="#D97706"
                strokeWidth="3"
                strokeLinecap="round"
              />
              <circle cx={toSvgX(vx)} cy={toSvgY(vy)} r="5" fill="#D97706" />
              <text x={toSvgX(vx) + 8} y={toSvgY(vy) - 6} fontSize="12" fontWeight="600" fill="#D97706">
                v ({vx.toFixed(1)}, {vy.toFixed(1)})
              </text>
            </svg>
          </div>

          <div className="md:col-span-5 space-y-3 tabular-nums">
            <div className="border border-slate-200 rounded-lg p-3.5">
              <div className="text-xs text-slate-500">Cosine Similarity (u·v / ||u||||v||)</div>
              <div className="text-2xl font-bold font-mono text-[#0F172A] mt-1">
                {cosine.toFixed(4)}
              </div>
              <div className="text-xs text-slate-500 mt-1">Ângulo θ = {angleDeg.toFixed(1)}°</div>
            </div>
            <div className="border border-slate-200 rounded-lg p-3.5">
              <div className="text-xs text-slate-500">Dot Product (u · v)</div>
              <div className="text-xl font-bold font-mono text-[#0F172A] mt-1">
                {dot.toFixed(3)}
              </div>
            </div>
            <div className="border border-slate-200 rounded-lg p-3.5">
              <div className="text-xs text-slate-500">Distância Euclidiana (L2)</div>
              <div className="text-xl font-bold font-mono text-[#0F172A] mt-1">
                {l2.toFixed(3)}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-6 space-y-5">
        <h3 className="text-sm font-semibold text-[#0F172A] border-b border-slate-200 pb-3">
          Coordenadas dos Embeddings
        </h3>
        <SliderControl label="Vetor Query u_x" value={ux} display={ux.toFixed(1)} min={-4.5} max={4.5} step={0.1} onChange={setUx} />
        <SliderControl label="Vetor Query u_y" value={uy} display={uy.toFixed(1)} min={-4.5} max={4.5} step={0.1} onChange={setUy} />
        <SliderControl label="Vetor Documento v_x" value={vx} display={vx.toFixed(1)} min={-4.5} max={4.5} step={0.1} onChange={setVx} />
        <SliderControl label="Vetor Documento v_y" value={vy} display={vy.toFixed(1)} min={-4.5} max={4.5} step={0.1} onChange={setVy} />

        <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 text-xs leading-relaxed text-slate-600">
          Dobre a magnitude de <strong>v</strong> mantendo a direção: o <strong>Dot Product</strong> dobra, mas a <strong>Cosine Similarity</strong> permanece idêntica porque normaliza pelo comprimento dos vetores.
        </div>
      </div>
    </div>
  );
}

function GradientDescentLab() {
  const [lr, setLr] = useState(0.12);
  const [w0, setW0] = useState(-2.8);
  const [history, setHistory] = useState<number[]>([-2.8]);
  const [isPlaying, setIsPlaying] = useState(false);

  // L(w) = 2*(w - 1.5)^2 + 0.5 => L'(w) = 4*(w - 1.5)
  const lossFn = (w: number) => 2 * Math.pow(w - 1.5, 2) + 0.5;
  const gradFn = (w: number) => 4 * (w - 1.5);

  const stepOnce = () => {
    setHistory((prev) => {
      const curr = prev[prev.length - 1];
      const next = curr - lr * gradFn(curr);
      if (!Number.isFinite(next) || Math.abs(next) > 25) {
        setIsPlaying(false);
        return prev;
      }
      return [...prev.slice(-29), next];
    });
  };

  const reset = (start = w0) => {
    setIsPlaying(false);
    setHistory([start]);
  };

  useEffect(() => {
    if (!isPlaying) return;
    const id = setInterval(stepOnce, 320);
    return () => clearInterval(id);
  }, [isPlaying, lr]);

  const currentW = history[history.length - 1];
  const currentLoss = lossFn(currentW);
  const currentGrad = gradFn(currentW);

  let regime = 'CONVERGÊNCIA SUAVE';
  let regimeClass = 'text-[#059669] bg-emerald-50 border-emerald-200';
  if (lr >= 0.5) {
    regime = 'DIVERGÊNCIA (η ≥ 0.50)';
    regimeClass = 'text-red-700 bg-red-50 border-red-200';
  } else if (lr > 0.25) {
    regime = 'OSCILAÇÃO AMORTECIDA';
    regimeClass = 'text-amber-800 bg-amber-50 border-amber-200';
  }

  const mapX = (w: number) => 40 + ((Math.max(-4, Math.min(5, w)) + 4) / 9) * 320;
  const mapY = (l: number) => 210 - (Math.min(45, l) / 45) * 175;

  const curveD = Array.from({ length: 73 }, (_, i) => {
    const w = -4 + (i / 72) * 9;
    return `${i === 0 ? 'M' : 'L'} ${mapX(w).toFixed(1)} ${mapY(lossFn(w)).toFixed(1)}`;
  }).join(' ');

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
      <div className="lg:col-span-8 bg-white border border-slate-200 rounded-xl p-6 space-y-5">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 pb-4">
          <div>
            <h2 className="text-lg font-semibold text-[#0F172A]">
              Loss Surface L(w) = 2(w − 1.5)² + 0.5
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Passo {history.length - 1} · Mínimo global em w* = 1.50 (Loss = 0.50)
            </p>
          </div>
          <span className={`px-3 py-1 text-xs font-mono font-medium border rounded-md ${regimeClass}`}>
            {regime}
          </span>
        </div>

        <div className="bg-slate-50 border border-slate-200 rounded-lg p-4">
          <svg viewBox="0 0 400 245" className="w-full h-auto">
            <line x1="40" y1="210" x2="360" y2="210" stroke="#CBD5E1" strokeWidth="1" />
            <line x1="40" y1="35" x2="40" y2="210" stroke="#CBD5E1" strokeWidth="1" />
            <path d={curveD} fill="none" stroke="#14213D" strokeWidth="2.5" />
            <circle cx={mapX(1.5)} cy={mapY(0.5)} r="4" fill="#059669" />
            {history.map((w, idx) => {
              const x = mapX(w);
              const y = mapY(lossFn(w));
              const prevW = idx > 0 ? history[idx - 1] : null;
              const isLast = idx === history.length - 1;
              return (
                <g key={idx}>
                  {prevW !== null && (
                    <line
                      x1={mapX(prevW)}
                      y1={mapY(lossFn(prevW))}
                      x2={x}
                      y2={y}
                      stroke="#0284C7"
                      strokeWidth="1.5"
                      strokeDasharray="3 3"
                    />
                  )}
                  <circle
                    cx={x}
                    cy={y}
                    r={isLast ? 6 : 3}
                    fill={isLast ? '#D97706' : '#0284C7'}
                    stroke={isLast ? '#FFFFFF' : 'none'}
                    strokeWidth="2"
                  />
                </g>
              );
            })}
            <text x="200" y="234" textAnchor="middle" fontSize="11" fill="#475569">
              Parâmetro w
            </text>
          </svg>
        </div>

        <div className="grid grid-cols-3 gap-4 tabular-nums">
          <div className="border border-slate-200 rounded-lg p-3.5">
            <div className="text-xs text-slate-500">Peso Atual (w)</div>
            <div className="text-xl font-bold font-mono text-[#0F172A] mt-1">
              {currentW.toFixed(4)}
            </div>
          </div>
          <div className="border border-slate-200 rounded-lg p-3.5">
            <div className="text-xs text-slate-500">Loss L(w)</div>
            <div className="text-xl font-bold font-mono text-[#0F172A] mt-1">
              {currentLoss.toFixed(4)}
            </div>
          </div>
          <div className="border border-slate-200 rounded-lg p-3.5">
            <div className="text-xs text-slate-500">Gradiente L&apos;(w)</div>
            <div className="text-xl font-bold font-mono text-[#0F172A] mt-1">
              {currentGrad.toFixed(4)}
            </div>
          </div>
        </div>
      </div>

      <div className="lg:col-span-4 bg-white border border-slate-200 rounded-xl p-6 space-y-6">
        <div className="flex items-center gap-2 border-b border-slate-200 pb-4">
          <button
            type="button"
            onClick={() => setIsPlaying((p) => !p)}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium bg-[#14213D] text-white rounded-lg hover:bg-[#1E3158] cursor-pointer"
          >
            {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            <span>{isPlaying ? 'Pausar' : 'Simular'}</span>
          </button>
          <button
            type="button"
            onClick={stepOnce}
            className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium bg-slate-100 hover:bg-slate-200 text-[#0F172A] rounded-lg cursor-pointer"
          >
            <StepForward className="w-3.5 h-3.5" />
            <span>1 Passo</span>
          </button>
          <button
            type="button"
            onClick={() => reset()}
            className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium bg-slate-100 hover:bg-slate-200 text-[#0F172A] rounded-lg cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset</span>
          </button>
        </div>

        <SliderControl
          label="Learning Rate (η)"
          value={lr}
          display={lr.toFixed(2)}
          min={0.02}
          max={0.56}
          step={0.02}
          onChange={setLr}
        />
        <SliderControl
          label="Peso Inicial (w₀)"
          value={w0}
          display={w0.toFixed(1)}
          min={-3.5}
          max={4.5}
          step={0.1}
          onChange={(val) => {
            setW0(val);
            reset(val);
          }}
        />

        <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 text-xs leading-relaxed text-slate-600">
          Regra de atualização: <code className="font-mono">w ← w − η · L&apos;(w)</code>. Teste{' '}
          <code className="font-mono">η = 0.52</code> para observar o gradiente explodindo na prática.
        </div>
      </div>
    </div>
  );
}

function RankingMetricsLab() {
  const [relevances, setRelevances] = useState<number[]>([2, 0, 1, 2, 0, 1]);
  const [totalRelevantInCorpus, setTotalRelevantInCorpus] = useState(5);

  const k = relevances.length;
  const binaryHits = relevances.filter((r) => r > 0).length;
  const recallAtK = Math.min(1, binaryHits / Math.max(1, totalRelevantInCorpus));
  const precisionAtK = binaryHits / k;
  const firstHitIdx = relevances.findIndex((r) => r > 0);
  const mrr = firstHitIdx === -1 ? 0 : 1 / (firstHitIdx + 1);

  const dcg = relevances.reduce((sum, rel, i) => {
    const gain = Math.pow(2, rel) - 1;
    const discount = Math.log2(i + 2);
    return sum + gain / discount;
  }, 0);

  const idealSorted = [...relevances].sort((a, b) => b - a);
  const idcg = idealSorted.reduce((sum, rel, i) => {
    const gain = Math.pow(2, rel) - 1;
    const discount = Math.log2(i + 2);
    return sum + gain / discount;
  }, 0);

  const ndcg = idcg > 0 ? dcg / idcg : 0;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
      <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-6 space-y-6">
        <div className="border-b border-slate-200 pb-4">
          <h2 className="text-lg font-semibold text-[#0F172A]">
            Avaliação de Retrieval Top-{k} (Recall@K, MRR, NDCG@K)
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Ajuste a relevância de cada chunk retornado e veja o impacto nas métricas de ranking
          </p>
        </div>

        <div className="space-y-3">
          {relevances.map((rel, idx) => {
            const gain = Math.pow(2, rel) - 1;
            const discounted = gain / Math.log2(idx + 2);
            return (
              <div
                key={idx}
                className="flex flex-wrap items-center justify-between gap-4 p-3.5 border border-slate-200 rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <span className="font-mono text-xs font-semibold text-slate-500 tabular-nums">
                    Rank #{idx + 1}
                  </span>
                  <span className="text-sm font-medium text-[#0F172A]">
                    Chunk Recuperado {idx + 1}
                  </span>
                </div>

                <div className="flex items-center gap-3">
                  <span className="text-xs font-mono text-slate-500 tabular-nums">
                    DCG contrib: {discounted.toFixed(3)}
                  </span>
                  <div className="flex items-center gap-1 p-0.5 bg-slate-100 rounded-md">
                    {[
                      { val: 0, label: '0 · Irrel.' },
                      { val: 1, label: '1 · Parcial' },
                      { val: 2, label: '2 · Exato' },
                    ].map((opt) => (
                      <button
                        key={opt.val}
                        type="button"
                        onClick={() =>
                          setRelevances((prev) => prev.map((v, i) => (i === idx ? opt.val : v)))
                        }
                        className={`px-2.5 py-1 text-xs font-medium rounded transition-colors cursor-pointer ${
                          rel === opt.val
                            ? 'bg-white text-[#0F172A] shadow-sm'
                            : 'text-slate-600 hover:text-[#0F172A]'
                        }`}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-6 space-y-5">
        <h3 className="text-sm font-semibold text-[#0F172A] border-b border-slate-200 pb-3">
          Métricas Calculadas
        </h3>

        <div className="grid grid-cols-2 gap-3 tabular-nums">
          <div className="border border-slate-200 rounded-lg p-3.5">
            <div className="text-xs text-slate-500">NDCG@{k}</div>
            <div className="text-2xl font-bold font-mono text-[#0284C7] mt-1">
              {ndcg.toFixed(4)}
            </div>
          </div>
          <div className="border border-slate-200 rounded-lg p-3.5">
            <div className="text-xs text-slate-500">MRR</div>
            <div className="text-2xl font-bold font-mono text-[#0F172A] mt-1">
              {mrr.toFixed(4)}
            </div>
          </div>
          <div className="border border-slate-200 rounded-lg p-3.5">
            <div className="text-xs text-slate-500">Recall@{k}</div>
            <div className="text-xl font-bold font-mono text-[#0F172A] mt-1">
              {(recallAtK * 100).toFixed(1)}%
            </div>
          </div>
          <div className="border border-slate-200 rounded-lg p-3.5">
            <div className="text-xs text-slate-500">Precision@{k}</div>
            <div className="text-xl font-bold font-mono text-[#0F172A] mt-1">
              {(precisionAtK * 100).toFixed(1)}%
            </div>
          </div>
        </div>

        <SliderControl
          label="Docs Relevantes no Corpus"
          value={totalRelevantInCorpus}
          display={`${totalRelevantInCorpus} docs`}
          min={1}
          max={12}
          step={1}
          onChange={setTotalRelevantInCorpus}
        />

        <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 text-xs leading-relaxed text-slate-600 space-y-1.5">
          <p className="font-mono text-[#0F172A]">
            DCG@{k} = {dcg.toFixed(3)} · IDCG@{k} = {idcg.toFixed(3)}
          </p>
          <p>
            A ordem importa no NDCG e no MRR: um chunk exato (2) na posição #1 vale muito mais que na posição #4 devido ao desconto logarítmico <code className="font-mono">log₂(rank + 1)</code>.
          </p>
        </div>
      </div>
    </div>
  );
}

const MODELS = [
  { id: 'qwen-0.5b', name: 'Qwen2.5-0.5B-Instruct', paramsB: 0.5, layers: 24, kvHeads: 2, headDim: 64 },
  { id: 'qwen-1.5b', name: 'Qwen2.5-1.5B-Instruct', paramsB: 1.5, layers: 28, kvHeads: 2, headDim: 128 },
  { id: 'llama-8b', name: 'Llama-3-8B-Instruct', paramsB: 8.0, layers: 32, kvHeads: 8, headDim: 128 },
  { id: 'qwen-14b', name: 'Qwen2.5-14B-Instruct', paramsB: 14.0, layers: 48, kvHeads: 8, headDim: 128 },
];

function LlmServingLab() {
  const [modelId, setModelId] = useState('qwen-1.5b');
  const [bytesPerParam, setBytesPerParam] = useState<2 | 1 | 0.5>(2);
  const [batchSize, setBatchSize] = useState(8);
  const [seqLen, setSeqLen] = useState(2048);

  const model = MODELS.find((m) => m.id === modelId) ?? MODELS[1];
  const weightsGb = model.paramsB * bytesPerParam;
  // KV cache in bytes: 2 (K and V) * layers * kvHeads * headDim * seqLen * batchSize * 2 (fp16 kv)
  const kvCacheGb =
    (2 * model.layers * model.kvHeads * model.headDim * seqLen * batchSize * 2) / 1e9;
  const overheadGb = 0.8 + batchSize * 0.04;
  const totalGb = weightsGb + kvCacheGb + overheadGb;

  const fitsT4 = totalGb <= 15.0;
  const fitsA100 = totalGb <= 78.0;
  const estTokPerSec = Math.round((batchSize * 38) / Math.max(0.5, model.paramsB * 0.65));

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
      <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-6 space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 pb-4">
          <div>
            <h2 className="text-lg font-semibold text-[#0F172A]">
              Capacity Planning: VRAM, KV-Cache &amp; Continuous Batching
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Estime memória e throughput antes de subir um endpoint vLLM ou TGI
            </p>
          </div>
          <span
            className={`px-3 py-1 text-xs font-mono font-medium border rounded-md ${
              fitsT4
                ? 'bg-emerald-50 text-[#059669] border-emerald-200'
                : fitsA100
                  ? 'bg-amber-50 text-amber-800 border-amber-200'
                  : 'bg-red-50 text-red-700 border-red-200'
            }`}
          >
            {fitsT4 ? 'CABEM EM T4 (16 GB)' : fitsA100 ? 'REQUER A100 (80 GB)' : 'EXCEDE 80 GB VRAM'}
          </span>
        </div>

        {/* Stacked VRAM Bar */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs font-mono tabular-nums text-slate-600">
            <span>Uso Total Estimado: {totalGb.toFixed(2)} GB</span>
            <span>Limite T4: 15.0 GB úteis</span>
          </div>
          <div className="h-8 w-full bg-slate-100 border border-slate-200 rounded-lg overflow-hidden flex">
            <div
              style={{ width: `${Math.min(100, (weightsGb / Math.max(totalGb, 16)) * 100)}%` }}
              className="bg-[#14213D] h-full transition-all"
              title={`Pesos: ${weightsGb.toFixed(2)} GB`}
            />
            <div
              style={{ width: `${Math.min(100, (kvCacheGb / Math.max(totalGb, 16)) * 100)}%` }}
              className="bg-[#0284C7] h-full transition-all"
              title={`KV-Cache: ${kvCacheGb.toFixed(2)} GB`}
            />
            <div
              style={{ width: `${Math.min(100, (overheadGb / Math.max(totalGb, 16)) * 100)}%` }}
              className="bg-[#FCA311] h-full transition-all"
              title={`Ativações/CUDA: ${overheadGb.toFixed(2)} GB`}
            />
          </div>
          <div className="flex flex-wrap gap-5 text-xs text-slate-600 tabular-nums pt-1">
            <span>Pesos ({weightsGb.toFixed(2)} GB)</span>
            <span aria-hidden="true">·</span>
            <span className="text-[#0284C7] font-medium">KV-Cache ({kvCacheGb.toFixed(2)} GB)</span>
            <span aria-hidden="true">·</span>
            <span className="text-amber-700">CUDA Overhead ({overheadGb.toFixed(2)} GB)</span>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-4 tabular-nums">
          <div className="border border-slate-200 rounded-lg p-3.5">
            <div className="text-xs text-slate-500">VRAM Total</div>
            <div className="text-xl font-bold font-mono text-[#0F172A] mt-1">
              {totalGb.toFixed(2)} GB
            </div>
          </div>
          <div className="border border-slate-200 rounded-lg p-3.5">
            <div className="text-xs text-slate-500">KV-Cache Share</div>
            <div className="text-xl font-bold font-mono text-[#0284C7] mt-1">
              {((kvCacheGb / totalGb) * 100).toFixed(1)}%
            </div>
          </div>
          <div className="border border-slate-200 rounded-lg p-3.5">
            <div className="text-xs text-slate-500">Throughput Estimado</div>
            <div className="text-xl font-bold font-mono text-[#0F172A] mt-1">
              ~{estTokPerSec} tok/s
            </div>
          </div>
        </div>
      </div>

      <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-6 space-y-5">
        <h3 className="text-sm font-semibold text-[#0F172A] border-b border-slate-200 pb-3">
          Configuração do Deployment
        </h3>

        <div className="space-y-1.5">
          <label className="text-xs font-medium text-slate-700">Modelo Open-Weight</label>
          <select
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            className="w-full px-3 py-2 text-sm bg-white border border-slate-200 rounded-lg text-[#0F172A]"
          >
            {MODELS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} ({m.paramsB}B)
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-1.5">
          <span className="text-xs font-medium text-slate-700 block">Quantização dos Pesos</span>
          <div className="grid grid-cols-3 gap-1 p-1 bg-slate-100 rounded-lg">
            {[
              { bytes: 2 as const, label: 'FP16' },
              { bytes: 1 as const, label: 'INT8' },
              { bytes: 0.5 as const, label: '4-bit' },
            ].map((q) => (
              <button
                key={q.label}
                type="button"
                onClick={() => setBytesPerParam(q.bytes)}
                className={`py-1.5 text-xs font-medium rounded-md transition-colors cursor-pointer ${
                  bytesPerParam === q.bytes
                    ? 'bg-white text-[#0F172A] shadow-sm'
                    : 'text-slate-600 hover:text-[#0F172A]'
                }`}
              >
                {q.label}
              </button>
            ))}
          </div>
        </div>

        <SliderControl
          label="Batch Size (Requisições Simultâneas)"
          value={batchSize}
          display={`${batchSize} reqs`}
          min={1}
          max={64}
          step={1}
          onChange={setBatchSize}
        />
        <SliderControl
          label="Context Length (Tokens)"
          value={seqLen}
          display={`${seqLen} tok`}
          min={512}
          max={16384}
          step={512}
          onChange={setSeqLen}
        />
      </div>
    </div>
  );
}

function SliderControl({
  label,
  value,
  display,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  display: string;
  min: number;
  max: number;
  step: number;
  onChange: (val: number) => void;
}) {
  return (
    <label className="block space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className="font-medium text-slate-700">{label}</span>
        <span className="font-mono font-semibold text-[#0F172A] tabular-nums">{display}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-[#14213D] cursor-pointer"
      />
    </label>
  );
}

export default function InteractiveLabsView({
  activeLabId,
  onSelectLab,
  onOpenWeek,
}: InteractiveLabsViewProps) {
  const currentLab = LABS.find((l) => l.id === activeLabId) ?? LABS[0];

  return (
    <div className="mx-auto max-w-[1280px] px-6 py-8 space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-[#0F172A] tracking-tight">
            Laboratórios Interativos
          </h1>
          <p className="text-sm text-slate-600 mt-1">
            Simuladores interativos que replicam a matemática e a lógica dos notebooks diretamente no navegador.
          </p>
        </div>
        <button
          type="button"
          onClick={() => onOpenWeek(currentLab.week)}
          className="px-4 py-2 text-xs font-medium bg-white border border-slate-200 hover:bg-slate-50 text-[#0F172A] rounded-lg whitespace-nowrap shrink-0 cursor-pointer"
        >
          Ver Leitura da Semana {currentLab.week}
        </button>
      </div>

      {/* Lab Selector Segmented Bar */}
      <div className="flex flex-wrap items-center gap-1 p-1 bg-slate-100 border border-slate-200 rounded-lg">
        {LABS.map((lab) => {
          const active = lab.id === currentLab.id;
          return (
            <button
              key={lab.id}
              type="button"
              onClick={() => onSelectLab(lab.id)}
              className={`px-3 py-2 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                active
                  ? 'bg-white text-[#0F172A] shadow-sm'
                  : 'text-slate-600 hover:text-[#0F172A]'
              }`}
            >
              {lab.label}
            </button>
          );
        })}
      </div>

      {currentLab.id === 'threshold-cost' && <ThresholdCostLab />}
      {currentLab.id === 'vector-cosine' && <VectorCosineLab />}
      {currentLab.id === 'gradient-descent' && <GradientDescentLab />}
      {currentLab.id === 'ranking-metrics' && <RankingMetricsLab />}
      {currentLab.id === 'llm-serving' && <LlmServingLab />}
    </div>
  );
}
