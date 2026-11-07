"use client";

import { useState } from "react";
import useSWR from "swr";
import { apiURL, fetcher, GRADE_COLORS_ARRAY } from "@/lib/api";
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
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
} from "recharts";

export default function DataPage() {
  const [gradeFilter, setGradeFilter] = useState<string>("all");
  const [histFeature, setHistFeature] = useState<string>("mean");

  const gradeParam = gradeFilter !== "all" ? `&grade=${gradeFilter}` : "";
  const { data: patients, isLoading: patientsLoading } = useSWR(
    apiURL(`/api/data/sample-patients?n=50${gradeParam}`),
    fetcher
  );
  const { data: dvhData } = useSWR(
    apiURL(`/api/data/dvh-curves?n=30${gradeFilter !== "all" ? `&grade=${gradeFilter}` : ""}`),
    fetcher
  );
  const { data: histData } = useSWR(
    apiURL(`/api/data/feature-histogram?feature=${histFeature}`),
    fetcher
  );
  const { data: eda } = useSWR(apiURL("/api/results/eda"), fetcher);

  const featureOptions = eda?.column_groups?.dvh
    ? [...(eda.column_groups.demographic || []), ...(eda.column_groups.dvh || []).slice(0, 10)]
    : ["mean", "Age", "pack_years", "V30", "V60", "D50"];

  return (
    <div className="max-w-6xl mx-auto px-6 py-12">
      <h1 className="text-[28px] font-semibold tracking-[-0.02em] mb-2">
        Data Explorer
      </h1>
      <p className="text-[16px] text-[#6B6B6B] mb-8 max-w-[65ch]">
        Browse test set patients, dose-volume histograms, and feature
        distributions. Filters apply across all panels.
      </p>

      {/* Grade filter */}
      <div className="flex items-center gap-3 mb-8">
        <label className="text-[14px] text-[#6B6B6B]">Filter by grade:</label>
        <Select value={gradeFilter} onValueChange={(v) => v && setGradeFilter(v)}>
          <SelectTrigger className="w-[140px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All grades</SelectItem>
            <SelectItem value="0">Grade 0</SelectItem>
            <SelectItem value="1">Grade 1</SelectItem>
            <SelectItem value="2">Grade 2</SelectItem>
            <SelectItem value="3">Grade 3</SelectItem>
            <SelectItem value="4">Grade 4</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Patient table */}
      <section className="mb-12">
        <h2 className="text-[18px] font-semibold mb-4">Patient Records</h2>
        <div className="border border-black/[0.08] rounded-md overflow-x-auto">
          {patientsLoading ? (
            <div className="p-6 space-y-3">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-8 w-full" />
              ))}
            </div>
          ) : patients?.patients?.length > 0 ? (
            <table className="w-full text-[14px]">
              <thead>
                <tr className="border-b border-black/[0.06]">
                  {["ORN Grade", "Age", "Gender", "Subsite", "T_stage", "mean", "V30", "V60"].map(
                    (col) => (
                      <th
                        key={col}
                        className="px-4 py-3 text-left text-[12px] font-medium uppercase tracking-wider text-[#6B6B6B]"
                      >
                        {col}
                      </th>
                    )
                  )}
                </tr>
              </thead>
              <tbody>
                {patients.patients.slice(0, 20).map((p: Record<string, unknown>, i: number) => (
                  <tr
                    key={i}
                    className="border-b border-black/[0.03] hover:bg-black/[0.02]"
                  >
                    <td className="px-4 py-2.5 font-medium">
                      <span
                        className="inline-block w-2 h-2 rounded-full mr-2"
                        style={{
                          backgroundColor:
                            GRADE_COLORS_ARRAY[Number(p["ORN Grade"]) || 0],
                        }}
                      />
                      {String(p["ORN Grade"] ?? "")}
                    </td>
                    <td className="px-4 py-2.5 text-right">{String(p["Age"] ?? "")}</td>
                    <td className="px-4 py-2.5">{String(p["Gender"] ?? "")}</td>
                    <td className="px-4 py-2.5">{String(p["Subsite"] ?? "")}</td>
                    <td className="px-4 py-2.5">{String(p["T_stage"] ?? "")}</td>
                    <td className="px-4 py-2.5 text-right">
                      {p["mean"] != null ? Number(p["mean"]).toFixed(1) : ""}
                    </td>
                    <td className="px-4 py-2.5 text-right">
                      {p["V30"] != null ? Number(p["V30"]).toFixed(1) : ""}
                    </td>
                    <td className="px-4 py-2.5 text-right">
                      {p["V60"] != null ? Number(p["V60"]).toFixed(1) : ""}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="p-6 text-[14px] text-[#6B6B6B]">
              No patients match the current filter.
            </p>
          )}
        </div>
        {patients && (
          <p className="text-[13px] text-[#6B6B6B] mt-2">
            Showing {Math.min(20, patients.patients?.length ?? 0)} of{" "}
            {patients.n} test set patients
          </p>
        )}
      </section>

      {/* DVH curves */}
      <section className="mb-12">
        <h2 className="text-[18px] font-semibold mb-4">DVH Curves</h2>
        <p className="text-[14px] text-[#6B6B6B] mb-4">
          Volume (%) receiving at least a given dose, color-coded by ORN grade.
        </p>
        {dvhData?.curves?.length > 0 ? (
          <div className="border border-black/[0.08] rounded-md p-6">
            <ResponsiveContainer width="100%" height={360}>
              <LineChart>
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E5E5" />
                <XAxis
                  dataKey="dose"
                  type="number"
                  domain={[0, 70]}
                  label={{ value: "Dose (Gy)", position: "bottom", offset: 0 }}
                  tick={{ fontSize: 12 }}
                />
                <YAxis
                  label={{
                    value: "Volume (%)",
                    angle: -90,
                    position: "insideLeft",
                    offset: 10,
                  }}
                  tick={{ fontSize: 12 }}
                />
                <Tooltip
                  contentStyle={{
                    fontSize: 12,
                    borderRadius: 4,
                    border: "1px solid rgba(0,0,0,0.1)",
                  }}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                {dvhData.curves.slice(0, 15).map((curve: { grade: number; values: number[] }, idx: number) => {
                  const lineData = dvhData.dose_points.map(
                    (d: number, j: number) => ({
                      dose: d,
                      volume: curve.values[j],
                    })
                  );
                  return (
                    <Line
                      key={idx}
                      data={lineData}
                      dataKey="volume"
                      stroke={GRADE_COLORS_ARRAY[curve.grade] || "#94A3B8"}
                      strokeWidth={1.5}
                      dot={false}
                      name={`Grade ${curve.grade}`}
                      opacity={0.7}
                    />
                  );
                })}
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <Skeleton className="h-[360px] w-full rounded-md" />
        )}
      </section>

      {/* Feature histogram */}
      <section className="mb-12">
        <h2 className="text-[18px] font-semibold mb-4">
          Feature Distribution
        </h2>
        <div className="flex items-center gap-3 mb-4">
          <label className="text-[14px] text-[#6B6B6B]">Feature:</label>
          <Select value={histFeature} onValueChange={(v) => v && setHistFeature(v)}>
            <SelectTrigger className="w-[180px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {featureOptions.map((f: string) => (
                <SelectItem key={f} value={f}>
                  {f}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {histData?.grades ? (
          <div className="border border-black/[0.08] rounded-md p-6">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={buildHistogramData(histData)}
                barCategoryGap="0%"
                barGap={0}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E5E5" />
                <XAxis dataKey="bin" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip
                  contentStyle={{
                    fontSize: 12,
                    borderRadius: 4,
                    border: "1px solid rgba(0,0,0,0.1)",
                  }}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                {Object.keys(histData.grades).map((grade) => (
                  <Bar
                    key={grade}
                    dataKey={`grade_${grade}`}
                    name={`Grade ${grade}`}
                    fill={GRADE_COLORS_ARRAY[Number(grade)]}
                    opacity={0.7}
                    stackId="a"
                  />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <Skeleton className="h-[300px] w-full rounded-md" />
        )}
      </section>

      <DisclaimerBanner />
    </div>
  );
}

function buildHistogramData(histData: {
  grades: Record<string, { counts?: number[]; bin_edges?: number[]; values?: Record<string, number> }>;
}) {
  const grades = histData.grades;
  const firstGrade = Object.values(grades)[0];

  if (firstGrade?.counts && firstGrade?.bin_edges) {
    const edges = firstGrade.bin_edges;
    return edges.slice(0, -1).map((edge: number, i: number) => {
      const row: Record<string, string | number> = {
        bin: `${edge.toFixed(1)}`,
      };
      for (const [grade, data] of Object.entries(grades)) {
        row[`grade_${grade}`] = data.counts?.[i] ?? 0;
      }
      return row;
    });
  }

  // Categorical
  const allValues = new Set<string>();
  for (const data of Object.values(grades)) {
    if (data.values) Object.keys(data.values).forEach((v) => allValues.add(v));
  }
  return Array.from(allValues).map((val) => {
    const row: Record<string, string | number> = { bin: val };
    for (const [grade, data] of Object.entries(grades)) {
      row[`grade_${grade}`] = data.values?.[val] ?? 0;
    }
    return row;
  });
}
