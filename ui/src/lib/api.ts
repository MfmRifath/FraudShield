// Typed client for the FraudShield API. Everything goes through /api (proxied to the service).
export const API_BASE: string = import.meta.env.VITE_API_BASE ?? "/api";

export type Decision = "approve" | "review" | "block";

export interface ReasonCode {
  code: string;
  message: string;
  feature: string | null;
  value: unknown;
  typical: unknown;
}

export interface DecisionEvent {
  txn_id: string;
  uid: string;
  event_time: string;
  amount: number;
  decision: Decision;
  probability: number | null;
  model_version: string | null;
  variant: string;
  degraded: boolean;
  latency_ms: number;
  rules_fired: string[];
  reason_codes: ReasonCode[];
}

export interface Case {
  case_id: string;
  created_at: string;
  txn: Record<string, unknown>;
  probability: number | null;
  decision: Decision;
  reasons: ReasonCode[];
  expected_loss: number;
  model_version: string | null;
  status: "open" | "labeled";
  label: "fraud" | "legit" | null;
  label_source: "analyst" | "chargeback" | null;
  analyst_id: string | null;
  notes: string | null;
  labeled_at: string | null;
}

export interface ActivityEntry {
  txn_id: string;
  event_time: string;
  amount: number | null;
  decision: Decision | null;
  probability: number | null;
  product: string | null;
  device_info: string | null;
  p_email_domain: string | null;
  addr: string | null;
  degraded: boolean | null;
  model_version: string | null;
}

export interface CaseContext {
  case_id: string;
  uid: string;
  timeline: ActivityEntry[];
  linked: Record<string, string[]>;
  store_available: boolean;
}

export interface AuditEntry {
  at: string;
  case_id: string;
  action: string;
  actor: string;
  detail: Record<string, unknown>;
}

export interface ModelInfo {
  champion: string | null;
  challenger: string | null;
  canary_pct: number;
  dataset: string;
}

export interface DriftSummary {
  as_of: string;
  drift_share: number | null;
  score_psi?: number | null;
  triggers: string[];
  drifted_columns: { column: string; drifted: boolean }[];
  has_html: boolean;
  [k: string]: unknown;
}

export interface MonitoringSummary {
  models: { champion: string | null; challenger: string | null };
  canary_pct: number;
  decisions: Record<string, number>;
  fallbacks: Record<string, number>;
  latency_s: Record<string, number | null>;
  review_queue_depth: number;
  drift: DriftSummary | null;
}

export interface Costs {
  chargeback_fee: number;
  margin_rate: number;
  churn_penalty: number;
  analyst_cost: number;
  review_friction: number;
  review_capacity_per_day: number | null;
}

export interface Outcome {
  total_cost: number;
  fraud_amount: number;
  fraud_amount_stopped: number;
  fraud_amount_missed: number;
  legit_blocked: number;
  n_approve: number;
  n_review: number;
  n_block: number;
  approve_all_cost: number;
  net_savings_vs_approve_all: number;
  [k: string]: number;
}

export interface PolicyWindow {
  rows: number;
  days: number;
  frauds: number;
}

export interface PolicyCurrent {
  costs: Costs;
  outcome: Outcome;
  window: PolicyWindow;
}

export interface PolicySimulation {
  window: PolicyWindow;
  current: { costs: Costs; outcome: Outcome };
  simulated: { costs: Costs; outcome: Outcome };
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
  ) {
    super(`${status}: ${detail}`);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...init?.headers },
  });
  if (!r.ok) {
    let detail = r.statusText;
    try {
      const body = (await r.json()) as { detail?: unknown };
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      // non-JSON error body: keep the status text
    }
    throw new ApiError(r.status, detail);
  }
  return (await r.json()) as T;
}

export const api = {
  model: () => request<ModelInfo>("/model"),
  cases: (status: "open" | "labeled" | null, limit = 200) =>
    request<Case[]>(`/cases?limit=${limit}${status ? `&status=${status}` : ""}`),
  case: (id: string) => request<Case>(`/cases/${encodeURIComponent(id)}`),
  context: (id: string) => request<CaseContext>(`/cases/${encodeURIComponent(id)}/context`),
  audit: (id: string) => request<AuditEntry[]>(`/cases/${encodeURIComponent(id)}/audit`),
  label: (id: string, label: "fraud" | "legit", analyst_id: string, notes: string | null) =>
    request<Case>(`/cases/${encodeURIComponent(id)}/label`, {
      method: "POST",
      body: JSON.stringify({ label, analyst_id, notes }),
    }),
  monitoring: () => request<MonitoringSummary>("/monitoring/summary"),
  policyCurrent: () => request<PolicyCurrent>("/policy/current"),
  simulate: (costs: Costs) =>
    request<PolicySimulation>("/policy/simulate", { method: "POST", body: JSON.stringify(costs) }),
};

export const driftReportUrl = `${API_BASE}/monitoring/drift/latest.html`;
export const streamUrl = `${API_BASE}/events/stream`;
