import type { Costs, Outcome } from "./api";

export interface CostControl {
  key: Exclude<keyof Costs, "review_capacity_per_day">;
  label: string;
  min: number;
  max: number;
  step: number;
  unit: "usd" | "rate";
}

export const COST_CONTROLS: readonly CostControl[] = [
  { key: "chargeback_fee", label: "Chargeback fee", min: 0, max: 100, step: 1, unit: "usd" },
  { key: "margin_rate", label: "Margin rate", min: 0, max: 1, step: 0.01, unit: "rate" },
  { key: "churn_penalty", label: "Churn penalty (false block)", min: 0, max: 100, step: 1, unit: "usd" },
  { key: "analyst_cost", label: "Analyst cost per review", min: 0, max: 50, step: 0.5, unit: "usd" },
  { key: "review_friction", label: "Review friction (legit)", min: 0, max: 20, step: 0.5, unit: "usd" },
];

/** Slider range that always contains the current value (configs may exceed the defaults). */
export function rangeFor(c: CostControl, v: number): { min: number; max: number } {
  return { min: Math.min(c.min, v), max: Math.max(c.max, c.unit === "rate" ? v : Math.ceil(v * 2)) };
}

export const OUTCOME_ROWS: readonly { key: keyof Outcome; label: string; money: boolean; better: "up" | "down" }[] = [
  { key: "net_savings_vs_approve_all", label: "Net savings vs approve-all", money: true, better: "up" },
  { key: "total_cost", label: "Total cost", money: true, better: "down" },
  { key: "fraud_amount_stopped", label: "Fraud $ stopped", money: true, better: "up" },
  { key: "fraud_amount_missed", label: "Fraud $ missed", money: true, better: "down" },
  { key: "legit_blocked", label: "Legit transactions blocked", money: false, better: "down" },
  { key: "n_review", label: "Reviews", money: false, better: "down" },
  { key: "n_block", label: "Blocks", money: false, better: "down" },
  { key: "n_approve", label: "Approvals", money: false, better: "up" },
];

/** Body for POST /policy/simulate: every field explicit (null capacity = unlimited). */
export function simulateBody(c: Costs): Costs {
  return { ...c, review_capacity_per_day: c.review_capacity_per_day == null ? null : Math.max(0, Math.round(c.review_capacity_per_day)) };
}
