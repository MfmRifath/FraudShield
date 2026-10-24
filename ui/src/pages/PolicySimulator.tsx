import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { QueryError } from "../components/QueryState";
import { Tile } from "../components/Tile";
import { api, type Costs } from "../lib/api";
import { count, money } from "../lib/format";
import { COST_CONTROLS, OUTCOME_ROWS, rangeFor, simulateBody } from "../lib/policy";

function useDebounced<T>(v: T, delayMs: number): T {
  const [d, setD] = useState(v);
  useEffect(() => {
    const t = window.setTimeout(() => setD(v), delayMs);
    return () => window.clearTimeout(t);
  }, [v, delayMs]);
  return d;
}

export function PolicySimulator({ debounceMs = 250 }: { debounceMs?: number }) {
  const current = useQuery({ queryKey: ["policy", "current"], queryFn: api.policyCurrent });
  const [costs, setCosts] = useState<Costs | null>(null);
  useEffect(() => {
    if (current.data && costs === null) setCosts(current.data.costs);
  }, [current.data, costs]);
  const body = useDebounced(costs, debounceMs);
  const sim = useQuery({
    queryKey: ["policy", "simulate", body],
    queryFn: () => api.simulate(simulateBody(body!)),
    enabled: body !== null,
    placeholderData: keepPreviousData,
  });

  if (current.error) return <QueryError error={current.error} />;
  if (!current.data || !costs) return <p className="muted">Loading…</p>;
  const w = current.data.window;
  const base = current.data.outcome;
  const out = sim.data?.simulated.outcome;
  const unlimited = costs.review_capacity_per_day == null;
  const set = (patch: Partial<Costs>) => setCosts({ ...costs, ...patch });
  const delta = out ? out.net_savings_vs_approve_all - base.net_savings_vs_approve_all : null;

  return (
    <div className="stack">
      <h1>Policy simulator</h1>
      <p className="muted" style={{ margin: 0 }}>
        Recomputes approve / review / block and the dollar outcome on the champion's held-out window
        ({count(w.rows)} transactions, {w.days} days, {count(w.frauds)} frauds). Nothing is changed in production.
      </p>
      <div className="grid cols-4">
        <Tile label="Net savings, current policy" value={money(base.net_savings_vs_approve_all)} />
        <Tile label="Net savings, simulated" value={<span data-testid="sim-savings">{out ? money(out.net_savings_vs_approve_all) : "…"}</span>} />
        <Tile label="Difference" value={delta == null ? "…" : <span className={delta >= 0 ? "good" : "error"}>{delta >= 0 ? "+" : ""}{money(delta)}</span>} />
        <Tile label="Reviews, simulated" value={out ? count(out.n_review) : "…"} hint={`current ${count(base.n_review)}`} />
      </div>
      <div className="grid cols-2">
        <section className="card stack">
          <h2>Costs</h2>
          {COST_CONTROLS.map((c) => {
            const v = costs[c.key];
            const r = rangeFor(c, v);
            const id = `cost-${c.key}`;
            return (
              <div key={c.key} className="slider">
                <label htmlFor={id}>{c.label}</label>
                <input id={id} type="range" min={r.min} max={r.max} step={c.step} value={v}
                  onChange={(e) => set({ [c.key]: Number(e.target.value) })} />
                <span className="mono">{c.unit === "usd" ? `$${v}` : `${(v * 100).toFixed(0)}%`}</span>
              </div>
            );
          })}
          <div className="slider">
            <label htmlFor="cost-capacity">Review capacity / day</label>
            <input id="cost-capacity" type="range" min={0} max={Math.max(2000, costs.review_capacity_per_day ?? 0)} step={10}
              disabled={unlimited} value={costs.review_capacity_per_day ?? 0}
              onChange={(e) => set({ review_capacity_per_day: Number(e.target.value) })} />
            <span className="mono">{unlimited ? "unlimited" : costs.review_capacity_per_day}</span>
          </div>
          <label className="row small">
            <input type="checkbox" checked={unlimited}
              onChange={(e) => set({ review_capacity_per_day: e.target.checked ? null : (current.data.costs.review_capacity_per_day ?? 500) })} />
            Unlimited review capacity
          </label>
          <div><button onClick={() => setCosts(current.data.costs)}>Reset to current config</button></div>
        </section>
        <section className="card table-wrap">
          <h2>Outcome on the test window</h2>
          {sim.error ? <QueryError error={sim.error} /> : null}
          <table aria-label="policy outcome">
            <thead><tr><th>Metric</th><th className="num">Current</th><th className="num">Simulated</th><th className="num">Δ</th></tr></thead>
            <tbody>
              {OUTCOME_ROWS.map((r) => {
                const a = base[r.key] ?? 0;
                const b = out?.[r.key];
                const d = b == null ? null : b - a;
                const good = d == null || d === 0 ? "" : (d > 0) === (r.better === "up") ? "good" : "error";
                const fmt = r.money ? money : count;
                return (
                  <tr key={r.key}>
                    <td>{r.label}</td>
                    <td className="num">{fmt(a)}</td>
                    <td className="num">{b == null ? "…" : fmt(b)}</td>
                    <td className={`num ${good}`}>{d == null ? "" : `${d > 0 ? "+" : ""}${fmt(d)}`}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>
      </div>
    </div>
  );
}
