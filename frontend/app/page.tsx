"use client";

import Link from "next/link";
import useSWR from "swr";
import { apiURL, fetcher } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";
import { DisclaimerBanner } from "@/components/shared/disclaimer-banner";

const navCards = [
  {
    href: "/data",
    title: "Data Explorer",
    desc: "Browse patient records, DVH curves, and feature distributions stratified by grade.",
  },
  {
    href: "/performance",
    title: "Model Performance",
    desc: "Confusion matrices, ROC/PR curves, and per-class metrics with bootstrap confidence intervals.",
  },
  {
    href: "/explainability",
    title: "Explainability",
    desc: "SHAP feature importance, per-class drivers, and partial dependence plots.",
  },
  {
    href: "/calculator",
    title: "Risk Calculator",
    desc: "Predict ORN grade for a patient profile with probability estimates and SHAP explanations.",
  },
  {
    href: "/about",
    title: "Methods",
    desc: "Background, methodology, limitations, and citations.",
  },
];

export default function HomePage() {
  const { data: summary, isLoading } = useSWR(
    apiURL("/api/results/summary"),
    fetcher
  );
  const { data: eda } = useSWR(apiURL("/api/results/eda"), fetcher);

  const macroF1 = summary?.metrics?.macro_f1;
  const macroF1CI = summary?.bootstrap_ci?.macro_f1_ci;
  const nPatients = eda?.n_patients ?? 1259;
  const ornRate = eda?.orn_incidence_rate;

  return (
    <div className="max-w-3xl mx-auto px-6 py-16">
      {/* Paper-style header */}
      <header className="mb-12">
        <h1 className="text-[36px] font-semibold leading-[1.15] tracking-[-0.02em] text-[#1A1A1A] mb-3">
          Osteoradionecrosis Grade Prediction
        </h1>
        <p className="text-[18px] text-[#6B6B6B] leading-relaxed mb-4">
          A machine learning pipeline for predicting ORN severity in head and
          neck cancer patients treated with radiation therapy
        </p>
        <p className="text-[14px] text-[#6B6B6B]">
          MD Anderson Cancer Center retrospective cohort, 2005&ndash;2015
        </p>
      </header>

      {/* Abstract */}
      <section className="mb-14">
        <p className="text-[17px] leading-[1.7] text-[#1A1A1A] max-w-[65ch]">
          This dashboard presents a gradient boosting model trained on clinical
          and dosimetric data from {nPatients.toLocaleString()} head and neck
          cancer patients to predict osteoradionecrosis (ORN) grade (0&ndash;4).
          The dataset is severely class-imbalanced, with Grade 1 comprising only
          22 patients. Model selection used stratified cross-validation with
          macro-F1 as the primary metric. All results include bootstrap 95%
          confidence intervals to quantify uncertainty.
        </p>
      </section>

      {/* Key stats */}
      <section className="mb-14 flex items-baseline gap-0 border-y border-black/[0.06] py-8">
        <StatBlock
          value={nPatients.toLocaleString()}
          label="patients"
          loading={!eda}
        />
        <div className="w-px h-12 bg-black/[0.08] mx-8 self-center" />
        <StatBlock
          value={ornRate ? `${(ornRate * 100).toFixed(1)}%` : "13.7%"}
          label="ORN incidence"
          loading={!eda}
        />
        <div className="w-px h-12 bg-black/[0.08] mx-8 self-center" />
        <StatBlock
          value={macroF1 ? macroF1.toFixed(2) : "--"}
          label="macro-F1 (test)"
          ci={macroF1CI}
          loading={isLoading}
        />
      </section>

      {/* Methods at a glance */}
      <section className="mb-14">
        <h2 className="text-[20px] font-semibold tracking-[-0.01em] mb-4 text-[#1A1A1A]">
          Methods at a glance
        </h2>
        <p className="text-[16px] leading-[1.7] text-[#1A1A1A] max-w-[65ch]">
          Features include patient demographics, tumor staging, treatment
          parameters, and 40 dose-volume histogram (DVH) metrics. Models
          evaluated: Random Forest, XGBoost, LightGBM, ordinal logistic
          regression, and multinomial logistic regression, each tested with
          class weighting and SMOTE oversampling (applied inside CV folds). The
          held-out test set was evaluated exactly once after model selection was
          finalized.
        </p>
      </section>

      {/* Navigation cards */}
      <section className="mb-14 space-y-3">
        {navCards.map((card) => (
          <Link
            key={card.href}
            href={card.href}
            className="block border border-black/[0.08] rounded-md px-6 py-5 hover:border-black/[0.16] transition-colors"
          >
            <h3 className="text-[16px] font-medium text-[#1A1A1A] mb-1">
              {card.title}
            </h3>
            <p className="text-[14px] text-[#6B6B6B] leading-relaxed">
              {card.desc}
            </p>
          </Link>
        ))}
      </section>

      <DisclaimerBanner />
    </div>
  );
}

function StatBlock({
  value,
  label,
  ci,
  loading,
}: {
  value: string;
  label: string;
  ci?: [number, number];
  loading?: boolean;
}) {
  if (loading) {
    return (
      <div className="flex-1">
        <Skeleton className="h-10 w-24 mb-1" />
        <Skeleton className="h-4 w-16" />
      </div>
    );
  }

  return (
    <div className="flex-1">
      <p className="text-[32px] font-semibold tracking-[-0.02em] text-[#1A1A1A] leading-none mb-1">
        {value}
      </p>
      <p className="text-[14px] text-[#6B6B6B]">
        {label}
        {ci && (
          <span className="text-[12px] ml-1 text-[#94A3B8]">
            [{ci[0].toFixed(2)}, {ci[1].toFixed(2)}]
          </span>
        )}
      </p>
    </div>
  );
}
