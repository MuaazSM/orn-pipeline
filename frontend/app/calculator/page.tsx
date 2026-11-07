"use client";

import { useState } from "react";
import useSWR from "swr";
import { apiURL, fetcher, GRADE_COLORS_ARRAY, ACCENT } from "@/lib/api";
import { DisclaimerBanner } from "@/components/shared/disclaimer-banner";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

type ModelMode = "two-stage" | "binary" | "multiclass";

interface TwoStageResult {
  stage1_probability: number;
  stage1_flagged: boolean;
  stage2_probability_mild: number | null;
  stage2_probability_severe: number | null;
  final_class: string;
  final_probabilities: Record<string, number>;
  stage1_shap?: Record<string, Record<string, number>> | null;
  stage2_shap?: Record<string, Record<string, number>> | null;
}

interface BinaryResult {
  predicted_grade: number;
  probabilities: Record<string, number>;
  shap_values?: Record<string, Record<string, number>> | null;
  confidence_note: string;
}

export default function CalculatorPage() {
  const [modelMode, setModelMode] = useState<ModelMode>("two-stage");
  const [twoStageResult, setTwoStageResult] = useState<TwoStageResult | null>(null);
  const [binaryResult, setBinaryResult] = useState<BinaryResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedPatient, setSelectedPatient] = useState<string>("");

  const { data: samplePatients } = useSWR(
    apiURL("/api/data/sample-patients?n=30"),
    fetcher
  );

  const runPredict = async (features: Record<string, unknown>) => {
    setLoading(true);
    setError(null);
    setTwoStageResult(null);
    setBinaryResult(null);
    try {
      const endpoint = modelMode === "two-stage"
        ? "/api/predict/two-stage"
        : "/api/predict";
      const res = await fetch(apiURL(endpoint), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      if (modelMode === "two-stage") {
        setTwoStageResult(data);
      } else {
        setBinaryResult(data);
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const handleSamplePredict = async (patient: Record<string, unknown>) => {
    const features = { ...patient };
    delete features["ORN Grade"];
    delete features["ORN_binary"];
    delete features["predicted_index"];
    delete features["ID"];
    await runPredict(features);
  };

  const handleManualPredict = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const features: Record<string, unknown> = {};
    formData.forEach((val, key) => {
      const num = Number(val);
      features[key] = isNaN(num) || val === "" ? val : num;
    });
    await runPredict(features);
  };

  return (
    <div className="max-w-5xl mx-auto px-6 py-12">
      <h1 className="text-[28px] font-semibold tracking-[-0.02em] mb-2">
        Patient Risk Calculator
      </h1>
      <p className="text-[16px] text-[#6B6B6B] mb-6 max-w-[65ch]">
        Predict ORN grade for a patient profile. View probability distributions
        and feature-level explanations.
      </p>

      <DisclaimerBanner prominent />

      {/* Model selector */}
      <div className="mt-6 mb-8 flex items-center gap-3">
        <span className="text-[14px] text-[#6B6B6B]">Model:</span>
        <div className="flex border border-black/[0.08] rounded-md overflow-hidden">
          {([
            ["two-stage", "Two-stage (recommended)"],
            ["binary", "Binary only"],
            ["multiclass", "Multi-class (experimental)"],
          ] as const).map(([mode, label]) => (
            <button
              key={mode}
              onClick={() => { setModelMode(mode); setTwoStageResult(null); setBinaryResult(null); }}
              className={`px-4 py-2 text-[13px] transition-colors ${
                modelMode === mode
                  ? "bg-[#2F5D50] text-white"
                  : "text-[#6B6B6B] hover:bg-black/[0.03]"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input panel */}
        <div>
          <Tabs defaultValue="sample">
            <TabsList className="mb-4">
              <TabsTrigger value="sample">Sample patient</TabsTrigger>
              <TabsTrigger value="manual">Manual entry</TabsTrigger>
            </TabsList>

            <TabsContent value="sample">
              <div className="space-y-4">
                <div>
                  <Label className="text-[14px] mb-2 block">
                    Select a test set patient
                  </Label>
                  <Select
                    value={selectedPatient}
                    onValueChange={(val: string | null) => {
                      if (val == null) return;
                      setSelectedPatient(val);
                      const patient = samplePatients?.patients?.[Number(val)];
                      if (patient) handleSamplePredict(patient);
                    }}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Choose a patient..." />
                    </SelectTrigger>
                    <SelectContent>
                      {samplePatients?.patients?.map(
                        (p: Record<string, unknown>, i: number) => (
                          <SelectItem key={i} value={String(i)}>
                            Patient {i + 1} — Grade{" "}
                            {String(p["ORN Grade"])}, {String(p["Subsite"])},{" "}
                            {String(p["Gender"])}
                          </SelectItem>
                        )
                      )}
                    </SelectContent>
                  </Select>
                </div>

                {selectedPatient !== "" &&
                  samplePatients?.patients?.[Number(selectedPatient)] && (
                    <div className="border border-black/[0.08] rounded-md p-4">
                      <h3 className="text-[14px] font-medium mb-2">
                        Patient Details
                      </h3>
                      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[13px]">
                        {Object.entries(
                          samplePatients.patients[Number(selectedPatient)]
                        )
                          .filter(
                            ([k]) =>
                              !["ID", "predicted_index", "ORN_binary"].includes(
                                k
                              )
                          )
                          .slice(0, 16)
                          .map(([k, v]) => (
                            <div key={k} className="flex justify-between">
                              <span className="text-[#6B6B6B]">{k}:</span>
                              <span className="font-medium">
                                {v != null
                                  ? typeof v === "number"
                                    ? Number(v).toFixed(2)
                                    : String(v)
                                  : "N/A"}
                              </span>
                            </div>
                          ))}
                      </div>
                      <p className="text-[12px] text-[#6B6B6B] mt-2">
                        Actual outcome: Grade{" "}
                        {String(
                          samplePatients.patients[Number(selectedPatient)][
                            "ORN Grade"
                          ]
                        )}
                      </p>
                    </div>
                  )}
              </div>
            </TabsContent>

            <TabsContent value="manual">
              <form onSubmit={handleManualPredict} className="space-y-4">
                <fieldset className="space-y-3">
                  <legend className="text-[14px] font-medium mb-2">
                    Demographics
                  </legend>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <Label htmlFor="Age" className="text-[13px]">
                        Age
                      </Label>
                      <Input
                        id="Age"
                        name="Age"
                        type="number"
                        defaultValue={60}
                      />
                    </div>
                    <div>
                      <Label htmlFor="Gender" className="text-[13px]">
                        Gender
                      </Label>
                      <select
                        id="Gender"
                        name="Gender"
                        className="w-full h-9 rounded-md border border-input px-3 text-[14px]"
                      >
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                      </select>
                    </div>
                    <div>
                      <Label htmlFor="Smoking_status" className="text-[13px]">
                        Smoking
                      </Label>
                      <select
                        id="Smoking_status"
                        name="Smoking_status"
                        className="w-full h-9 rounded-md border border-input px-3 text-[14px]"
                      >
                        <option value="Never">Never</option>
                        <option value="Former">Former</option>
                        <option value="Current">Current</option>
                      </select>
                    </div>
                    <div>
                      <Label htmlFor="pack_years" className="text-[13px]">
                        Pack-years
                      </Label>
                      <Input
                        id="pack_years"
                        name="pack_years"
                        type="number"
                        defaultValue={20}
                      />
                    </div>
                  </div>
                </fieldset>

                <fieldset className="space-y-3">
                  <legend className="text-[14px] font-medium mb-2">
                    Clinical
                  </legend>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <Label htmlFor="Subsite" className="text-[13px]">
                        Subsite
                      </Label>
                      <select
                        id="Subsite"
                        name="Subsite"
                        className="w-full h-9 rounded-md border border-input px-3 text-[14px]"
                      >
                        <option value="OPC.O">OPC.O</option>
                        <option value="Oral">Oral</option>
                        <option value="Hypo.Larynx.Naso.unkn">
                          Hypo/Larynx/Naso
                        </option>
                      </select>
                    </div>
                    <div>
                      <Label htmlFor="T_stage" className="text-[13px]">
                        T Stage
                      </Label>
                      <select
                        id="T_stage"
                        name="T_stage"
                        className="w-full h-9 rounded-md border border-input px-3 text-[14px]"
                      >
                        {["T1", "T2", "T3", "T4"].map((t) => (
                          <option key={t} value={t}>
                            {t}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <Label htmlFor="N_stage" className="text-[13px]">
                        N Stage
                      </Label>
                      <select
                        id="N_stage"
                        name="N_stage"
                        className="w-full h-9 rounded-md border border-input px-3 text-[14px]"
                      >
                        {["N0", "N1", "N2", "N3"].map((n) => (
                          <option key={n} value={n}>
                            {n}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <Label
                        htmlFor="Dental_extraction"
                        className="text-[13px]"
                      >
                        Pre-RT dental extraction
                      </Label>
                      <select
                        id="Dental_extraction"
                        name="Dental_extraction"
                        className="w-full h-9 rounded-md border border-input px-3 text-[14px]"
                      >
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                      </select>
                    </div>
                  </div>
                </fieldset>

                <fieldset className="space-y-3">
                  <legend className="text-[14px] font-medium mb-2">
                    Dosimetric (key DVH parameters)
                  </legend>
                  <div className="grid grid-cols-3 gap-3">
                    {[
                      ["mean", "Mean dose (Gy)", "45"],
                      ["V30", "V30 (%)", "80"],
                      ["V60", "V60 (%)", "30"],
                      ["D50", "D50 (Gy)", "40"],
                      ["Volume", "Mandible vol (cc)", "50"],
                    ].map(([name, label, def]) => (
                      <div key={name}>
                        <Label htmlFor={name} className="text-[13px]">
                          {label}
                        </Label>
                        <Input
                          id={name}
                          name={name}
                          type="number"
                          step="0.1"
                          defaultValue={def}
                        />
                      </div>
                    ))}
                  </div>
                </fieldset>

                <Button
                  type="submit"
                  disabled={loading}
                  className="bg-[#2F5D50] hover:bg-[#264A40] text-white"
                >
                  {loading ? "Predicting..." : "Predict ORN Grade"}
                </Button>
              </form>
            </TabsContent>
          </Tabs>
        </div>

        {/* Results panel */}
        <div>
          {error && (
            <div className="border border-[#EF4444]/20 bg-[#FEF2F2] rounded-md px-4 py-3 mb-4">
              <p className="text-[14px] text-[#991B1B]">
                Prediction failed. The backend may not be running, or the input
                may be incomplete.
              </p>
            </div>
          )}

          {loading && (
            <div className="space-y-4">
              <Skeleton className="h-24 w-full rounded-md" />
              <Skeleton className="h-64 w-full rounded-md" />
            </div>
          )}

          {/* Two-stage results */}
          {twoStageResult && !loading && (
            <div className="space-y-6">
              {/* Stage 1 */}
              <div className="border border-black/[0.08] rounded-md px-6 py-5">
                <p className="text-[12px] uppercase tracking-wider text-[#6B6B6B] mb-2">Stage 1: ORN Detection</p>
                <div className="flex items-baseline gap-3 mb-1">
                  <span className="text-[14px] text-[#6B6B6B]">P(ORN) =</span>
                  <span className="text-[28px] font-semibold tracking-[-0.02em]"
                    style={{ color: twoStageResult.stage1_flagged ? "#EF4444" : "#94A3B8" }}>
                    {(twoStageResult.stage1_probability * 100).toFixed(1)}%
                  </span>
                  <span className={`text-[14px] font-medium ${twoStageResult.stage1_flagged ? "text-[#EF4444]" : "text-[#94A3B8]"}`}>
                    {twoStageResult.stage1_flagged ? "→ flagged" : "→ below threshold"}
                  </span>
                </div>
              </div>

              {/* Stage 2 */}
              {twoStageResult.stage1_flagged ? (
                <div className="border border-black/[0.08] rounded-md px-6 py-5">
                  <p className="text-[12px] uppercase tracking-wider text-[#6B6B6B] mb-2">Stage 2: Severity</p>
                  <div className="flex items-baseline gap-3 mb-2">
                    <span className="text-[14px] text-[#6B6B6B]">P(Severe | ORN) =</span>
                    <span className="text-[24px] font-semibold">
                      {twoStageResult.stage2_probability_severe != null
                        ? `${(twoStageResult.stage2_probability_severe * 100).toFixed(1)}%`
                        : "--"}
                    </span>
                  </div>
                  <p className="text-[14px] text-[#6B6B6B]">
                    P(Mild | ORN) = {twoStageResult.stage2_probability_mild != null
                      ? `${(twoStageResult.stage2_probability_mild * 100).toFixed(1)}%`
                      : "--"}
                  </p>
                </div>
              ) : (
                <div className="border border-black/[0.06] rounded-md px-6 py-4 bg-[#F9FAFB]">
                  <p className="text-[13px] text-[#6B6B6B]">Stage 2 not run (below ORN threshold)</p>
                </div>
              )}

              {/* Final prediction */}
              <div className="border-2 border-black/[0.12] rounded-md px-6 py-5">
                <p className="text-[12px] uppercase tracking-wider text-[#6B6B6B] mb-2">Final Prediction</p>
                <p className="text-[24px] font-semibold tracking-[-0.02em]"
                  style={{
                    color: twoStageResult.final_class === "No ORN" ? "#94A3B8"
                      : twoStageResult.final_class === "Mild ORN" ? "#FB923C" : "#991B1B"
                  }}>
                  {twoStageResult.final_class}
                </p>
                <div className="mt-3 flex gap-4 text-[13px] text-[#6B6B6B]">
                  {Object.entries(twoStageResult.final_probabilities).map(([cls, p]) => (
                    <span key={cls}>{cls}: {(p * 100).toFixed(1)}%</span>
                  ))}
                </div>
              </div>

              {/* SHAP for stage 1 */}
              {twoStageResult.stage1_shap && (
                <div className="border border-black/[0.08] rounded-md px-6 py-5">
                  <h3 className="text-[14px] font-medium mb-1">Stage 1 Feature Contributions</h3>
                  <p className="text-[13px] text-[#6B6B6B] mb-3">Features driving ORN probability</p>
                  <SHAPWaterfall shapValues={twoStageResult.stage1_shap["1"] || {}} />
                </div>
              )}

              {/* SHAP for stage 2 if applicable */}
              {twoStageResult.stage1_flagged && twoStageResult.stage2_shap && (
                <div className="border border-black/[0.08] rounded-md px-6 py-5">
                  <h3 className="text-[14px] font-medium mb-1">Stage 2 Feature Contributions</h3>
                  <p className="text-[13px] text-[#6B6B6B] mb-3">Features driving severity classification</p>
                  <SHAPWaterfall shapValues={twoStageResult.stage2_shap["1"] || {}} />
                </div>
              )}
            </div>
          )}

          {/* Binary / multiclass results */}
          {binaryResult && !loading && (
            <div className="space-y-6">
              <div className="border border-black/[0.08] rounded-md px-6 py-5">
                <p className="text-[13px] text-[#6B6B6B] mb-1">
                  {modelMode === "binary" ? "ORN Detection" : "Predicted Grade"}
                </p>
                <div className="flex items-baseline gap-3">
                  <span className="text-[40px] font-semibold tracking-[-0.02em]"
                    style={{ color: GRADE_COLORS_ARRAY[binaryResult.predicted_grade] || "#1A1A1A" }}>
                    {modelMode === "binary"
                      ? (binaryResult.predicted_grade === 1 ? "ORN" : "No ORN")
                      : `Grade ${binaryResult.predicted_grade}`}
                  </span>
                </div>
                {binaryResult.confidence_note && (
                  <p className="text-[13px] text-[#92400E] mt-2 bg-[#FFFBEB] px-3 py-2 rounded">
                    {binaryResult.confidence_note}
                  </p>
                )}
              </div>

              <div className="border border-black/[0.08] rounded-md px-6 py-5">
                <h3 className="text-[14px] font-medium mb-3">Probabilities</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart
                    data={Object.entries(binaryResult.probabilities).map(
                      ([k, prob]) => ({
                        label: modelMode === "binary"
                          ? (k === "0" ? "No ORN" : "ORN")
                          : `Grade ${k}`,
                        probability: prob,
                      })
                    )}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#E5E5E5" vertical={false} />
                    <XAxis dataKey="label" tick={{ fontSize: 12 }} />
                    <YAxis domain={[0, 1]} tick={{ fontSize: 12 }}
                      tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
                    <Tooltip formatter={(v) => `${(Number(v) * 100).toFixed(1)}%`}
                      contentStyle={{ fontSize: 12, borderRadius: 4, border: "1px solid rgba(0,0,0,0.1)" }} />
                    <Bar dataKey="probability" radius={[3, 3, 0, 0]}>
                      {Object.keys(binaryResult.probabilities).map((_, i) => (
                        <Cell key={i} fill={GRADE_COLORS_ARRAY[i] || ACCENT} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {binaryResult.shap_values && (
                <div className="border border-black/[0.08] rounded-md px-6 py-5">
                  <h3 className="text-[14px] font-medium mb-3">Feature Contributions (SHAP)</h3>
                  <SHAPWaterfall shapValues={
                    binaryResult.shap_values[String(binaryResult.predicted_grade)]
                    || binaryResult.shap_values["1"]
                    || {}
                  } />
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="mt-8">
        <DisclaimerBanner />
      </div>
    </div>
  );
}

function SHAPWaterfall({
  shapValues,
}: {
  shapValues: Record<string, number>;
}) {
  const sorted = Object.entries(shapValues)
    .map(([feat, val]) => ({
      feature: feat
        .replace(/^(continuous__|onehot__|ordinal__|binary__)/, "")
        .replace(/_/g, " "),
      value: val,
    }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 10);

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart
        data={sorted.reverse()}
        layout="vertical"
        margin={{ left: 120 }}
      >
        <CartesianGrid
          strokeDasharray="3 3"
          stroke="#E5E5E5"
          horizontal={false}
        />
        <XAxis type="number" tick={{ fontSize: 11 }} />
        <YAxis
          type="category"
          dataKey="feature"
          tick={{ fontSize: 11 }}
          width={110}
        />
        <Tooltip
          contentStyle={{
            fontSize: 12,
            borderRadius: 4,
            border: "1px solid rgba(0,0,0,0.1)",
          }}
          formatter={(v) => Number(v).toFixed(4)}
        />
        <Bar dataKey="value" radius={[0, 3, 3, 0]}>
          {sorted.map((entry, i) => (
            <Cell
              key={i}
              fill={entry.value >= 0 ? "#EF4444" : "#3B82F6"}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
