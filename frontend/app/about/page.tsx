import { DisclaimerBanner } from "@/components/shared/disclaimer-banner";

export default function AboutPage() {
  return (
    <div className="max-w-[65ch] mx-auto px-6 py-12">
      <h1 className="text-[28px] font-semibold tracking-[-0.02em] mb-8">
        Methods and Background
      </h1>

      <section className="mb-10">
        <h2 className="text-[20px] font-semibold mb-3">Background</h2>
        <p className="text-[17px] leading-[1.7] text-[#1A1A1A] mb-4">
          Osteoradionecrosis (ORN) of the mandible is a serious complication of
          head and neck radiation therapy. It ranges from asymptomatic bone
          exposure (Grade 1) to pathological fracture requiring surgical
          intervention (Grade 4). Predicting ORN severity before it occurs could
          help guide treatment planning and patient counseling.
        </p>
        <p className="text-[17px] leading-[1.7] text-[#1A1A1A]">
          This project uses a retrospective cohort of 1,259 patients treated at
          MD Anderson Cancer Center between 2005 and 2015, with detailed
          dosimetric data (dose-volume histogram parameters for the mandible)
          and clinical variables.
        </p>
      </section>

      <section className="mb-10">
        <h2 className="text-[20px] font-semibold mb-3">Methods</h2>
        <p className="text-[17px] leading-[1.7] text-[#1A1A1A] mb-4">
          We trained gradient boosting models (Random Forest, XGBoost, LightGBM)
          alongside logistic regression and ordinal regression baselines. The
          dataset was split 60/20/20 (train/validation/test) with stratification
          on the 5-class ORN grade.
        </p>
        <p className="text-[17px] leading-[1.7] text-[#1A1A1A] mb-4">
          Class imbalance was addressed through class weighting and SMOTE
          oversampling, applied strictly inside cross-validation folds to prevent
          data leakage. Model selection used 5-fold stratified cross-validation
          with macro-F1 as the primary metric. The held-out test set was
          evaluated exactly once.
        </p>
        <p className="text-[17px] leading-[1.7] text-[#1A1A1A] mb-4">
          Features include 40 DVH parameters (V5&ndash;V70, D2&ndash;D99, mean,
          min, and max dose), patient demographics (age, gender, smoking status),
          tumor characteristics (subsite, T-stage, N-stage), and treatment
          details (chemotherapy, postoperative RT, pre-RT dental extraction).
        </p>
        <p className="text-[17px] leading-[1.7] text-[#1A1A1A]">
          Explainability uses SHAP (SHapley Additive exPlanations) with
          TreeExplainer for the selected tree-based model. All metrics include
          1,000-iteration bootstrap 95% confidence intervals.
        </p>
      </section>

      <section className="mb-10">
        <h2 className="text-[20px] font-semibold mb-3">Why two stages?</h2>
        <p className="text-[17px] leading-[1.7] text-[#1A1A1A] mb-4">
          Binary ORN prediction (any grade vs. none) is well-established in the
          literature. Van Dijk et al. (2021) achieved AUC 0.77 on this dataset
          with just two features (D30 and dental extraction status). Reber et al.
          (2023) benchmarked multiple ML methods on the same cohort, confirming
          logistic regression as competitive (AUC 0.70). Our pipeline reproduces
          these results as a validation step.
        </p>
        <p className="text-[17px] leading-[1.7] text-[#1A1A1A] mb-4">
          The 5-class grade prediction problem, however, is not solvable with
          this sample size. Grade 1 has only 22 patients total (4 in the test
          set). The model collapses all minority grades into Grade 0, producing
          F1 = 0.00 for Grades 1-3. This is a data limitation, not a modeling
          failure.
        </p>
        <p className="text-[17px] leading-[1.7] text-[#1A1A1A]">
          The two-stage hierarchical approach is the practical middle ground: a
          binary gate (ORN vs. no ORN) with established performance, followed by
          severity stratification (mild vs. severe) only among predicted
          positives. The second stage operates on a more balanced subset
          (approximately 1:2 mild-to-severe ratio among ORN-positive patients),
          making the discrimination task tractable.
        </p>
      </section>

      <section className="mb-10">
        <h2 className="text-[20px] font-semibold mb-3">Limitations</h2>
        <ul className="space-y-3 text-[17px] leading-[1.7] text-[#1A1A1A]">
          <li className="pl-4 border-l-2 border-black/[0.08]">
            <strong>Single institution.</strong> All patients are from MD
            Anderson. There is no external validation cohort.
          </li>
          <li className="pl-4 border-l-2 border-black/[0.08]">
            <strong>Severe class imbalance.</strong> Grade 1 has only 22
            patients. The model cannot reliably distinguish fine-grained severity
            levels with this sample size. Per-class metrics for minority grades
            have wide confidence intervals.
          </li>
          <li className="pl-4 border-l-2 border-black/[0.08]">
            <strong>Retrospective design.</strong> Associations identified may
            not hold in a prospective setting. Treatment decisions were not
            randomized.
          </li>
          <li className="pl-4 border-l-2 border-black/[0.08]">
            <strong>DVH limitations.</strong> Dose-volume histograms are summary
            statistics of complex 3D dose distributions. Spatial dose patterns
            are not captured.
          </li>
          <li className="pl-4 border-l-2 border-black/[0.08]">
            <strong>No neural nets.</strong> With 1,259 tabular rows, gradient
            boosting outperforms deep learning (consistent with published
            benchmarks on this exact dataset). Neural approaches were
            deliberately excluded.
          </li>
        </ul>
      </section>

      <section className="mb-10">
        <h2 className="text-[20px] font-semibold mb-3">Citations</h2>
        <p className="text-[17px] leading-[1.7] text-[#1A1A1A] mb-3">
          van Dijk LV, et al. &ldquo;Normal tissue complication probability
          (NTCP) prediction model for osteoradionecrosis of the mandible in
          patients with head and neck cancer after radiation therapy.&rdquo;{" "}
          <em>Radiotherapy and Oncology</em>. 2021.
        </p>
        <p className="text-[17px] leading-[1.7] text-[#1A1A1A] mb-3">
          Reber S, et al. &ldquo;Machine learning and deep learning for
          prediction of osteoradionecrosis of the mandible.&rdquo;{" "}
          <em>Scientific Reports</em>. 2023. Benchmark comparison of ML vs. DL
          on the same cohort.
        </p>
        <p className="text-[17px] leading-[1.7] text-[#1A1A1A]">
          Dataset available at figshare (anonymized, 1,259 patients). Original
          collection: MD Anderson Cancer Center, 2005&ndash;2015.
        </p>
      </section>

      <section className="mb-10">
        <h2 className="text-[20px] font-semibold mb-3">Disclaimer</h2>
        <p className="text-[17px] leading-[1.7] text-[#1A1A1A]">
          This is a research prototype developed for academic purposes. It has
          not been validated for clinical use and must not be used to inform
          patient care decisions. Predicted grades are model estimates with
          substantial uncertainty, particularly for minority classes. Any
          clinical application would require prospective validation, regulatory
          review, and integration with clinical judgment.
        </p>
      </section>

      <DisclaimerBanner />
    </div>
  );
}
