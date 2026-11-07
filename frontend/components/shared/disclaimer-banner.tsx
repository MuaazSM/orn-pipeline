export function DisclaimerBanner({ prominent = false }: { prominent?: boolean }) {
  if (prominent) {
    return (
      <div className="border border-[#EF4444]/20 bg-[#FEF2F2] rounded-md px-5 py-4">
        <p className="text-[14px] text-[#991B1B] font-medium leading-relaxed">
          Research tool only. Predictions are not validated for clinical
          decision-making. Do not use to inform patient care.
        </p>
      </div>
    );
  }

  return (
    <div className="border-t border-black/[0.06] mt-16 pt-4 pb-2">
      <p className="text-[13px] text-[#6B6B6B]">
        Research prototype. Not validated for clinical use.
      </p>
    </div>
  );
}
