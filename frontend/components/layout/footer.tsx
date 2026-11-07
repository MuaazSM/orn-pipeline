export function Footer() {
  return (
    <footer className="border-t border-black/[0.06] py-6 px-6">
      <div className="max-w-6xl mx-auto flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
        <p className="text-[13px] text-[#6B6B6B]">
          Research prototype. Not validated for clinical use.
        </p>
        <p className="text-[13px] text-[#6B6B6B]">
          Data: van Dijk et al. 2021 &middot; MD Anderson 2005&ndash;2015
        </p>
      </div>
    </footer>
  );
}
