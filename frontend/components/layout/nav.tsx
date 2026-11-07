"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { Menu, X } from "lucide-react";

const links = [
  { href: "/", label: "Overview" },
  { href: "/data", label: "Data" },
  { href: "/performance", label: "Performance" },
  { href: "/explainability", label: "Explainability" },
  { href: "/calculator", label: "Calculator" },
  { href: "/about", label: "About" },
];

export function Nav() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <nav className="sticky top-0 z-50 bg-[#FAFAF7]/95 backdrop-blur-sm border-b border-black/[0.06]">
      <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
        <Link
          href="/"
          className="text-[15px] font-semibold tracking-tight text-[#1A1A1A]"
        >
          ORN Prediction
        </Link>

        {/* Desktop nav */}
        <div className="hidden md:flex items-center gap-8">
          {links.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={`text-[14px] transition-colors ${
                pathname === link.href
                  ? "text-[#1A1A1A] font-medium"
                  : "text-[#6B6B6B] hover:text-[#1A1A1A]"
              }`}
            >
              {link.label}
            </Link>
          ))}
        </div>

        {/* Mobile hamburger */}
        <button
          className="md:hidden p-1"
          onClick={() => setMobileOpen(!mobileOpen)}
          aria-label="Toggle menu"
        >
          {mobileOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
      </div>

      {/* Mobile menu */}
      {mobileOpen && (
        <div className="md:hidden border-t border-black/[0.06] bg-[#FAFAF7] px-6 py-4 space-y-3">
          {links.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              onClick={() => setMobileOpen(false)}
              className={`block text-[15px] ${
                pathname === link.href
                  ? "text-[#1A1A1A] font-medium"
                  : "text-[#6B6B6B]"
              }`}
            >
              {link.label}
            </Link>
          ))}
        </div>
      )}
    </nav>
  );
}
