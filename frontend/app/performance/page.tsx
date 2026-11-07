"use client";

import { useState } from "react";
import useSWR from "swr";
import { apiURL, fetcher, GRADE_COLORS, ACCENT } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";
import { DisclaimerBanner } from "@/components/shared/disclaimer-banner";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

export default function PerformancePage() {
  const { data: summary, isLoading } = useSWR(
    apiURL("/api/results/summary"),
    fetcher
  );
  const { data: comparison } = useSWR(
    apiURL("/api/results/model-comparison"),
    fetcher
  );
  const { data: cm } = useSWR(
    apiURL("/api/results/confusion-matrix"),
    fetcher
  );
  const { data: curves } = useSWR(apiURL("/api/results/curves"), fetcher);
  const { data: cvResults } = useSWR(apiURL("/api/results/cv"), fetcher);

  const [sortCol, setSortCol] = useState("macro_f1_mean");
  const [sortAsc, setSortAsc] = useState(false);

  return (
    <div className="max-w-6xl mx-auto px-6 py-12">
      <h1 className="text-[28px] font-semibold tracking-[-0.02em] mb-2">
        Model Performance
      </h1>
      <p className="text-[16px] text-[#6B6B6B] mb-8 max-w-[65ch]">
        Evaluation on the held-out test set (n=252), touched exactly once after
        all model selection was finalized.
      </p>

      {/* Headline metrics */}
      {isLoading ? (
        <div className="grid grid-cols-4 gap-4 mb-10">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-20 rounded-md" />
          ))}
        </div>
      ) : summary ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10">
          <MetricCard
            label="Macro-F1"
            value={summary.metrics.macro_f1}
            ci={summary.bootstrap_ci.macro_f1_ci}
          />
          <MetricCard
            label="Balanced Accuracy"
            value={summary.metrics.balanced_accuracy}
            ci={summary.bootstrap_ci.balanced_accuracy_ci}
          />
          <MetricCard
            label="Weighted-F1"
            value={summary.metrics.weighted_f1}
          />
          <MetricCard
            label="QWK"
            value={summary.metrics.quadratic_weighted_kappa}
            ci={summary.bootstrap_ci.qwk_ci}
          />
        </div>
      ) : null}

      {/* Honesty callout */}
      <div className="border border-[#FCD34D]/40 bg-[#FFFBEB] rounded-md px-5 py-4 mb-10">
        <p className="text-[14px] text-[#92400E] leading-relaxed">
          <strong>Constraint note:</strong> Grade 1 has only 4 test samples
          (22 total in the dataset). Per-class metrics for minority grades have
          wide confidence intervals and should be interpreted with caution. The
          5-class model struggles to distinguish adjacent grades, which is
          expected with this level of class imbalance.
        </p>
      </div>

      {/* Model Comparison */}
      {comparison && (
        <section className="mb-12">
          <h2 className="text-[18px] font-semibold mb-4">Model Comparison</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {([
              ["binary", "Binary Baseline", false],
              ["two_stage", "Two-stage", true],
              ["five_class", "Multi-class (5)", false],
            ] as const).map(([key, title, recommended]) => {
              const data = comparison[key];
              if (!data) return (
                <div key={key} className="border border-black/[0.08] rounded-md px-5 py-4 opacity-50">
                  <h3 className="text-[14px] font-medium mb-2">{title}</h3>
                  <p className="text-[13px] text-[#6B6B6B]">Not available</p>
                </div>
              );
              const m = data.metrics || {};
              const ci = data.bootstrap_ci || {};
              return (
                <div key={key}
                  className={`border rounded-md px-5 py-4 ${
                    recommended ? "border-l-2 border-l-[#3B5BDB] border-black/[0.08]" : "border-black/[0.08]"
                  }`}>
                  <h3 className="text-[14px] font-medium mb-3">
                    {title}
                    {recommended && <span className="text-[11px] text-[#3B5BDB] ml-2">recommended</span>}
                  </h3>
                  {m.auc_roc != null && (
                    <div className="mb-2">
                      <p className="text-[12px] text-[#6B6B6B]">AUC-ROC</p>
                      <p className="text-[20px] font-semibold">{m.auc_roc.toFixed(3)}
                        {ci.auc && <span className="text-[11px] text-[#94A3B8] ml-1">
                          [{ci.auc[0].toFixed(3)}, {ci.auc[1].toFixed(3)}]
                        </span>}
                      </p>
                    </div>
                  )}
                  <div className="mb-2">
                    <p className="text-[12px] text-[#6B6B6B]">Macro-F1</p>
                    <p className="text-[20px] font-semibold">{m.macro_f1?.toFixed(3) ?? "--"}
                      {ci.macro_f1_ci && <span className="text-[11px] text-[#94A3B8] ml-1">
                        [{ci.macro_f1_ci[0].toFixed(3)}, {ci.macro_f1_ci[1].toFixed(3)}]
                      </span>}
                    </p>
                  </div>
                  {m.balanced_accuracy != null && (
                    <div>
                      <p className="text-[12px] text-[#6B6B6B]">Balanced Acc</p>
                      <p className="text-[16px] font-medium">{m.balanced_accuracy.toFixed(3)}</p>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
          {comparison.binary_validation && (
            <p className="text-[13px] text-[#6B6B6B] mt-3">
              Binary baseline validation: <strong className={
                comparison.binary_validation.status === "PASSED" ? "text-[#2F5D50]" : "text-[#991B1B]"
              }>{comparison.binary_validation.status}</strong> (AUC={comparison.binary_validation.auc.toFixed(3)},
              benchmark=0.74, van Dijk et al. 2021)
            </p>
          )}
        </section>
      )}

      {/* Per-class metrics table */}
      {summary?.per_class && (
        <section className="mb-12">
          <h2 className="text-[18px] font-semibold mb-4">
            Per-class Metrics (Test Set)
          </h2>
          <div className="border border-black/[0.08] rounded-md overflow-x-auto">
            <table className="w-full text-[14px]">
              <thead>
                <tr className="border-b border-black/[0.06]">
                  {["Class", "Precision", "Recall", "F1", "Support"].map(
                    (h) => (
                      <th
                        key={h}
                        className="px-4 py-3 text-left text-[12px] font-medium uppercase tracking-wider text-[#6B6B6B]"
                      >
                        {h}
                      </th>
                    )
                  )}
                </tr>
              </thead>
              <tbody>
                {Object.entries(summary.per_class as Record<string, Record<string, unknown>>).map(
                  ([label, m]) => (
                    <tr
                      key={label}
                      className="border-b border-black/[0.03]"
                    >
                      <td className="px-4 py-2.5 font-medium">{label}</td>
                      <td className="px-4 py-2.5 text-right">
                        {fmt(m.precision as number)}
                      </td>
                      <td className="px-4 py-2.5 text-right">
                        {fmt(m.recall as number)}
                      </td>
                      <td className="px-4 py-2.5 text-right">
                        {fmt(m.f1 as number)}
                        {Array.isArray(m.f1_ci) && (
                          <span className="text-[11px] text-[#94A3B8] ml-1">
                            [{fmt((m.f1_ci as number[])[0])},{" "}
                            {fmt((m.f1_ci as number[])[1])}]
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-2.5 text-right">
                        {String(m.support)}
                      </td>
                    </tr>
                  )
                )}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* Confusion matrix */}
      {cm && (
        <section className="mb-12">
          <h2 className="text-[18px] font-semibold mb-4">Confusion Matrix</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <ConfusionMatrix
              matrix={cm.matrix}
              labels={cm.labels}
              title="Counts"
              normalized={false}
            />
            <ConfusionMatrix
              matrix={cm.normalized}
              labels={cm.labels}
              title="Normalized"
              normalized={true}
            />
          </div>
        </section>
      )}

      {/* ROC and PR curves */}
      {curves && (
        <section className="mb-12">
          <h2 className="text-[18px] font-semibold mb-4">
            ROC and Precision-Recall Curves
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <CurveChart
              data={curves.roc}
              xKey="fpr"
              yKey="tpr"
              xLabel="False Positive Rate"
              yLabel="True Positive Rate"
              title="ROC (one-vs-rest)"
              showDiagonal
              metricKey="auc"
              metricLabel="AUC"
            />
            <CurveChart
              data={curves.pr}
              xKey="recall"
              yKey="precision"
              xLabel="Recall"
              yLabel="Precision"
              title="Precision-Recall"
              metricKey="ap"
              metricLabel="AP"
            />
          </div>
        </section>
      )}

      {/* CV comparison table */}
      {cvResults && (
        <section className="mb-12">
          <h2 className="text-[18px] font-semibold mb-4">
            Cross-validation Comparison ({cvResults.n_configurations}{" "}
            configurations)
          </h2>
          <div className="border border-black/[0.08] rounded-md overflow-x-auto">
            <table className="w-full text-[14px]">
              <thead>
                <tr className="border-b border-black/[0.06]">
                  {[
                    { key: "formulation", label: "Form." },
                    { key: "model", label: "Model" },
                    { key: "imbalance_strategy", label: "Imbalance" },
                    { key: "macro_f1_mean", label: "Macro-F1" },
                    { key: "balanced_acc_mean", label: "Bal. Acc." },
                  ].map((col) => (
                    <th
                      key={col.key}
                      onClick={() => {
                        if (sortCol === col.key) setSortAsc(!sortAsc);
                        else {
                          setSortCol(col.key);
                          setSortAsc(false);
                        }
                      }}
                      className="px-4 py-3 text-left text-[12px] font-medium uppercase tracking-wider text-[#6B6B6B] cursor-pointer hover:text-[#1A1A1A]"
                    >
                      {col.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[...cvResults.results]
                  .sort((a: Record<string, unknown>, b: Record<string, unknown>) => {
                    const va = (a[sortCol] as number) ?? 0;
                    const vb = (b[sortCol] as number) ?? 0;
                    return sortAsc ? va - vb : vb - va;
                  })
                  .map((r: Record<string, unknown>, i: number) => {
                    const isBest =
                      i === 0 && sortCol === "macro_f1_mean" && !sortAsc;
                    return (
                      <tr
                        key={i}
                        className={`border-b border-black/[0.03] ${
                          isBest ? "border-l-2 border-l-[#3B5BDB]" : ""
                        }`}
                      >
                        <td className="px-4 py-2.5">{String(r.formulation)}</td>
                        <td className="px-4 py-2.5 font-medium">
                          {String(r.model)}
                        </td>
                        <td className="px-4 py-2.5">
                          {String(r.imbalance_strategy)}
                        </td>
                        <td className="px-4 py-2.5 text-right">
                          {fmt(r.macro_f1_mean as number)} &plusmn;{" "}
                          {fmt(r.macro_f1_std as number)}
                        </td>
                        <td className="px-4 py-2.5 text-right">
                          {fmt(r.balanced_acc_mean as number)} &plusmn;{" "}
                          {fmt(r.balanced_acc_std as number)}
                        </td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <DisclaimerBanner />
    </div>
  );
}

function fmt(n: number | undefined | null): string {
  if (n == null) return "--";
  return n.toFixed(2);
}

function MetricCard({
  label,
  value,
  ci,
}: {
  label: string;
  value: number;
  ci?: [number, number];
}) {
  return (
    <div className="border border-black/[0.08] rounded-md px-5 py-4">
      <p className="text-[13px] text-[#6B6B6B] mb-1">{label}</p>
      <p className="text-[24px] font-semibold tracking-[-0.02em]">
        {value.toFixed(3)}
      </p>
      {ci && (
        <p className="text-[12px] text-[#94A3B8]">
          95% CI: [{ci[0].toFixed(3)}, {ci[1].toFixed(3)}]
        </p>
      )}
    </div>
  );
}

function ConfusionMatrix({
  matrix,
  labels,
  title,
  normalized,
}: {
  matrix: number[][];
  labels: string[];
  title: string;
  normalized: boolean;
}) {
  const maxVal = Math.max(...matrix.flat());

  return (
    <div className="border border-black/[0.08] rounded-md p-5">
      <h3 className="text-[14px] font-medium mb-3">{title}</h3>
      <div className="overflow-x-auto">
        <table className="text-[13px]">
          <thead>
            <tr>
              <th className="px-2 py-1 text-[11px] text-[#6B6B6B]"></th>
              {labels.map((l) => (
                <th
                  key={l}
                  className="px-2 py-1 text-[11px] text-[#6B6B6B] font-medium"
                >
                  {l}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => (
              <tr key={i}>
                <td className="px-2 py-1 text-[11px] text-[#6B6B6B] font-medium">
                  {labels[i]}
                </td>
                {row.map((val, j) => {
                  const intensity = normalized
                    ? val
                    : maxVal > 0
                      ? val / maxVal
                      : 0;
                  return (
                    <td
                      key={j}
                      className="px-3 py-2 text-center font-medium"
                      style={{
                        backgroundColor: `rgba(59, 91, 219, ${intensity * 0.6})`,
                        color: intensity > 0.5 ? "#fff" : "#1A1A1A",
                      }}
                      title={`True: ${labels[i]}, Pred: ${labels[j]}`}
                    >
                      {normalized ? val.toFixed(2) : val}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-[11px] text-[#6B6B6B] mt-2">
        Rows: true grade. Columns: predicted grade.
      </p>
    </div>
  );
}

function CurveChart({
  data,
  xKey,
  yKey,
  xLabel,
  yLabel,
  title,
  showDiagonal,
  metricKey,
  metricLabel,
}: {
  data: Record<string, { [k: string]: number[] | number }>;
  xKey: string;
  yKey: string;
  xLabel: string;
  yLabel: string;
  title: string;
  showDiagonal?: boolean;
  metricKey: string;
  metricLabel: string;
}) {
  // Build chart data - combine all class curves
  const allPoints: Record<string, unknown>[] = [];
  const classNames = Object.keys(data);

  // For each class, create line data
  return (
    <div className="border border-black/[0.08] rounded-md p-5">
      <h3 className="text-[14px] font-medium mb-3">{title}</h3>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart>
          <CartesianGrid strokeDasharray="3 3" stroke="#E5E5E5" />
          <XAxis
            dataKey="x"
            type="number"
            domain={[0, 1]}
            tick={{ fontSize: 11 }}
            label={{ value: xLabel, position: "bottom", offset: 0, fontSize: 12 }}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fontSize: 11 }}
            label={{
              value: yLabel,
              angle: -90,
              position: "insideLeft",
              offset: 10,
              fontSize: 12,
            }}
          />
          <Tooltip
            contentStyle={{
              fontSize: 12,
              borderRadius: 4,
              border: "1px solid rgba(0,0,0,0.1)",
            }}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {showDiagonal && (
            <Line
              data={[
                { x: 0, y: 0 },
                { x: 1, y: 1 },
              ]}
              dataKey="y"
              stroke="#D1D5DB"
              strokeDasharray="4 4"
              dot={false}
              name="Random"
              strokeWidth={1}
            />
          )}
          {classNames.map((cls) => {
            const curveData = data[cls];
            const xArr = curveData[xKey] as number[];
            const yArr = curveData[yKey] as number[];
            const metric = curveData[metricKey] as number;
            const points = xArr.map((xv: number, i: number) => ({
              x: xv,
              y: yArr[i],
            }));
            // Downsample for performance
            const step = Math.max(1, Math.floor(points.length / 100));
            const downsampled = points.filter(
              (_: unknown, i: number) => i % step === 0
            );
            return (
              <Line
                key={cls}
                data={downsampled}
                dataKey="y"
                stroke={GRADE_COLORS[cls] || ACCENT}
                strokeWidth={1.5}
                dot={false}
                name={`${cls} (${metricLabel}=${metric.toFixed(2)})`}
              />
            );
          })}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
