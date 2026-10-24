const usd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
const usd2 = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });

export const money = (v: number | null | undefined): string => (v == null ? "-" : usd.format(v));
export const money2 = (v: number | null | undefined): string => (v == null ? "-" : usd2.format(v));
export const pct = (v: number | null | undefined, digits = 1): string =>
  v == null ? "-" : `${(v * 100).toFixed(digits)}%`;
export const ms = (seconds: number | null | undefined): string =>
  seconds == null ? "-" : `${(seconds * 1e3).toFixed(seconds < 0.01 ? 2 : 1)} ms`;
export const count = (v: number | null | undefined): string =>
  v == null ? "-" : Math.round(v).toLocaleString("en-US");

export function value(v: unknown): string {
  if (v == null) return "-";
  if (typeof v === "number") return Number.isInteger(v) ? v.toLocaleString("en-US") : v.toFixed(3);
  if (typeof v === "boolean") return v ? "yes" : "no";
  return String(v);
}

export function time(iso: string | null | undefined): string {
  if (!iso) return "-";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toISOString().replace("T", " ").slice(0, 19);
}
