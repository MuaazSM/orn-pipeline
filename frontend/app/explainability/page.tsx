"use client";

import { useState } from "react";
import useSWR from "swr";
import { apiURL, fetcher, ACCENT, GRADE_COLORS } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";
import { DisclaimerBanner } from "@/components/shared/disclaimer-banner";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  Legend,
} from "recharts";

export default function ExplainabilityPage() {
  const { data: globalShap } = useSWR(
    apiURL("/api/explain/global"),
    fetcher
  );
  const { data: perClassShap } = useSWR(
    apiURL("/api/explain/per-class"),
    fetcher
  );

  const [pdpFeature, setPdpFeature] = useState<string>("");
  const { data: pdpData } = useSWR(
    pdpFeature
      ? apiURL(`/api/explain/partial-dependence?feature=${pdpFeature}`)
      : null,
    fetcher
  );

  // Set default PDP feature when global data loads
  if (globalShap?.features?.length > 0 && !pdpFeature) {
    setPdpFeature(globalShap.features[0]);
  }

  return (
    <div className="max-w-6xl mx-auto px-6 py-12">
      <h1 className="text-[28px] font-semibold tracking-[-0.02em] mb-2">
        Explainability
      </h1>
      <p className="text-[16px] text-[#6B6B6B] mb-8 max-w-[65ch]">
        How the model makes predictions, analyzed with SHAP (SHapley Additive
        exPlanations). SHAP values quantify each feature&rsquo;s contribution to
        shifting the prediction away from the baseline.
      </p>

      {/* Plain-language explanation */}
      <div className="border border-black/[0.08] bg-[#F9FAFB] rounded-md px-5 py-4 mb-10 max-w-[65ch]">
        <h3 className="text-[14px] font-medium mb-2">
          What SHAP values mean
        </h3>
        <p className="text-[14px] text-[#4B5563] leading-relaxed">
          For each prediction, SHAP assigns a value to every feature indicating
          how much that feature pushed the prediction toward or away from a
          particular grade. A positive SHAP value for a feature means it
          increased the model&rsquo;s predicted probability of that grade. The
          magnitude indicates the strength of the effect. Global importance is
          the average of these absolute values across all test patients.
        </p>
      </div>

      {/* Global feature importance */}
      <section className="mb-12">
        <h2 className="text-[18px] font-semibold mb-4">
          Global Feature Importance (Top 15)
        </h2>
        {globalShap ? (
          <div className="border border-black/[0.08] rounded-md p-6">
            <ResponsiveContainer width="100%" height={420}>
              <BarChart
                data={globalShap.features
                  .slice(0, 15)
                  .map((f: string, i: number) => ({
                    feature: cleanFeatureName(f),
                    importance: globalShap.importance[i],
                  }))
                  .reverse()}
                layout="vertical"
                margin={{ left: 140 }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="#E5E5E5"
                  horizontal={false}
                />
                <XAxis
                  type="number"
                  tick={{ fontSize: 12 }}
                  label={{
                    value: "Mean |SHAP value|",
                    position: "bottom",
                    offset: 0,
                    fontSize: 12,
                  }}
                />
                <YAxis
                  type="category"
                  dataKey="feature"
                  tick={{ fontSize: 12 }}
                  width={130}
                />
                <Tooltip
                  contentStyle={{
                    fontSize: 12,
                    borderRadius: 4,
                    border: "1px solid rgba(0,0,0,0.1)",
                  }}
                  formatter={(v) => Number(v).toFixed(4)}
                />
                <Bar dataKey="importance" fill={ACCENT} radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <Skeleton className="h-[420px] w-full rounded-md" />
        )}
      </section>

      {/* Per-class importance heatmap */}
      {perClassShap && (
        <section className="mb-12">
          <h2 className="text-[18px] font-semibold mb-4">
            Per-class Feature Importance
          </h2>
          <p className="text-[14px] text-[#6B6B6B] mb-4 max-w-[65ch]">
            How feature importance varies across predicted grades. Columns are
            grades, rows are features. Darker cells indicate stronger influence.
          </p>
          <div className="border border-black/[0.08] rounded-md p-5 overflow-x-auto">
            <HeatmapTable data={perClassShap} topN={15} globalShap={globalShap} />
          </div>
        </section>
      )}

      {/* Partial dependence */}
      <section className="mb-12">
        <h2 className="text-[18px] font-semibold mb-4">
          Partial Dependence Plot
        </h2>
        <p className="text-[14px] text-[#6B6B6B] mb-4 max-w-[65ch]">
          Shows the marginal effect of a single feature on predicted
          probabilities, averaged over all patients.
        </p>
        <div className="flex items-center gap-3 mb-4">
          <label className="text-[14px] text-[#6B6B6B]">Feature:</label>
          <Select value={pdpFeature} onValueChange={(v) => v && setPdpFeature(v)}>
            <SelectTrigger className="w-[220px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {(globalShap?.features || []).slice(0, 10).map((f: string) => (
                <SelectItem key={f} value={f}>
                  {cleanFeatureName(f)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {pdpData ? (
          <div className="border border-black/[0.08] rounded-md p-6">
            <ResponsiveContainer width="100%" height={320}>
              <LineChart
                data={pdpData.values.map((v: number, i: number) => {
                  const row: Record<string, number> = { value: v };
                  pdpData.predictions[i].forEach(
                    (p: number, cls: number) => {
                      row[`Grade ${cls}`] = p;
                    }
                  );
                  return row;
                })}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E5E5" />
                <XAxis
                  dataKey="value"
                  tick={{ fontSize: 12 }}
                  label={{
                    value: cleanFeatureName(pdpFeature),
                    position: "bottom",
                    offset: 0,
                    fontSize: 12,
                  }}
                />
                <YAxis
                  tick={{ fontSize: 12 }}
                  label={{
                    value: "Mean predicted probability",
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
                <Legend wrapperStyle={{ fontSize: 12 }} />
                {Array.from(
                  { length: pdpData.predictions[0]?.length ?? 0 },
                  (_, i) => (
                    <Line
                      key={i}
                      dataKey={`Grade ${i}`}
                      stroke={
                        GRADE_COLORS[`Grade ${i}`] || ACCENT
                      }
                      strokeWidth={2}
                      dot={false}
                    />
                  )
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <Skeleton className="h-[320px] w-full rounded-md" />
        )}
      </section>

      <DisclaimerBanner />
    </div>
  );
}

function cleanFeatureName(name: string): string {
  return name
    .replace(/^(continuous__|onehot__|ordinal__|binary__)/, "")
    .replace(/_/g, " ");
}

function HeatmapTable({
  data,
  topN,
  globalShap,
}: {
  data: Record<string, Record<string, number>>;
  topN: number;
  globalShap: { features: string[]; importance: number[] } | null;
}) {
  const grades = Object.keys(data);
  // Get top features from global importance
  const topFeatures = globalShap
    ? globalShap.features.slice(0, topN)
    : Object.keys(Object.values(data)[0] || {}).slice(0, topN);

  // Find max for normalization
  let maxVal = 0;
  for (const grade of grades) {
    for (const feat of topFeatures) {
      const v = data[grade]?.[feat] ?? 0;
      if (v > maxVal) maxVal = v;
    }
  }

  return (
    <table className="text-[13px]">
      <thead>
        <tr>
          <th className="px-3 py-2 text-left text-[11px] font-medium uppercase tracking-wider text-[#6B6B6B]">
            Feature
          </th>
          {grades.map((g) => (
            <th
              key={g}
              className="px-3 py-2 text-center text-[11px] font-medium uppercase tracking-wider text-[#6B6B6B]"
            >
              {g}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {topFeatures.map((feat) => (
          <tr key={feat} className="border-t border-black/[0.04]">
            <td className="px-3 py-2 font-medium whitespace-nowrap">
              {cleanFeatureName(feat)}
            </td>
            {grades.map((g) => {
              const val = data[g]?.[feat] ?? 0;
              const intensity = maxVal > 0 ? val / maxVal : 0;
              return (
                <td
                  key={g}
                  className="px-3 py-2 text-center"
                  style={{
                    backgroundColor: `rgba(59, 91, 219, ${intensity * 0.5})`,
                    color: intensity > 0.6 ? "#fff" : "#1A1A1A",
                  }}
                >
                  {val.toFixed(3)}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
