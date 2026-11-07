import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Nav } from "@/components/layout/nav";
import { Footer } from "@/components/layout/footer";
import { Toaster } from "sonner";

const inter = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "ORN Grade Prediction — Research Dashboard",
  description:
    "Osteoradionecrosis grade prediction in head and neck cancer patients. Research prototype.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col font-sans">
        <Nav />
        <main className="flex-1">{children}</main>
        <Footer />
        <Toaster position="bottom-right" />
      </body>
    </html>
  );
}
