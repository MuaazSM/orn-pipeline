const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function fetchAPI<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${res.statusText}`);
  }
  return res.json();
}

export function apiURL(path: string): string {
  return `${API_BASE}${path}`;
}

export const fetcher = (url: string) =>
  fetch(url).then((r) => {
    if (!r.ok) throw new Error(`${r.status}`);
    return r.json();
  });

// Grade color palette (sequential severity)
export const GRADE_COLORS: Record<string, string> = {
  "Grade 0": "#94A3B8",
  "Grade 1": "#FCD34D",
  "Grade 2": "#FB923C",
  "Grade 3": "#EF4444",
  "Grade 4": "#991B1B",
  // 3-class
  None: "#94A3B8",
  Mild: "#FB923C",
  Severe: "#991B1B",
};

export const GRADE_COLORS_ARRAY = [
  "#94A3B8",
  "#FCD34D",
  "#FB923C",
  "#EF4444",
  "#991B1B",
];

export const ACCENT = "#3B5BDB";
