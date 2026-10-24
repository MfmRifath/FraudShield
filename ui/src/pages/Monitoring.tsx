import { useQuery } from "@tanstack/react-query";

import { QueryError } from "../components/QueryState";
import { Tile } from "../components/Tile";
import { api, driftReportUrl } from "../lib/api";
import { count, ms, pct } from "../lib/format";

const DECISIONS = ["approve", "review", "block"] as const;

export function Monitoring() {
  const model = useQuery({ queryKey: ["model"], queryFn: api.model, refetchInterval: 10_000 });
  const s = useQuery({ queryKey: ["monitoring"], queryFn: api.monitoring, refetchInterval: 5000 });
  if (s.error) return <QueryError error={s.error} />;
  if (!s.data) return <p className="muted">Loading…</p>;
  const m = s.data;
  const total = DECISIONS.reduce((a, d) => a + (m.decisions[d] ?? 0), 0);
  const drift = m.drift;

  return (
    <div className="stack">
      <h1>Model &amp; monitoring</h1>
      <div className="grid cols-4">
        <Tile label="Champion" value={<span className="small">{m.models.champion ?? "none (rules only)"}</span>}
          hint={model.data ? `dataset ${model.data.dataset}` : undefined} />
        <Tile label="Challenger" value={<span className="small">{m.models.challenger ?? "none"}</span>}
          hint={`canary ${m.canary_pct}% of cards`} />
        <Tile label="Review queue" value={count(m.review_queue_depth)} hint="open cases" />
        <Tile label="Decisions since start" value={count(total)} />
      </div>
      <div className="grid cols-2">
        <section className="card">
          <h2>Decision mix</h2>
          {total === 0 ? <p className="muted">No decisions yet.</p> : (
            <>
              <div className="bar-track" role="img"
                aria-label={DECISIONS.map((d) => `${d} ${pct((m.decisions[d] ?? 0) / total)}`).join(", ")}>
                {DECISIONS.map((d) => (
                  <div key={d} className={`bar-seg dot ${d}`} style={{ width: `${((m.decisions[d] ?? 0) / total) * 100}%`, borderRadius: 0 }}
                    title={`${d}: ${count(m.decisions[d])}`} />
                ))}
              </div>
              <table style={{ marginTop: 10 }}>
                <thead><tr><th>Decision</th><th className="num">Count</th><th className="num">Share</th></tr></thead>
                <tbody>
                  {DECISIONS.map((d) => (
                    <tr key={d}>
                      <td><span className="badge"><span className={`dot ${d}`} aria-hidden />{d}</span></td>
                      <td className="num">{count(m.decisions[d] ?? 0)}</td>
                      <td className="num">{pct((m.decisions[d] ?? 0) / total)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </section>
        <section className="card">
          <h2>Scoring latency (since start)</h2>
          <div className="grid cols-4">
            {(["p50", "p95", "p99"] as const).map((q) => (
              <div key={q} className="tile"><div className="label">{q}</div><div className="value">{ms(m.latency_s[q])}</div></div>
            ))}
          </div>
          <h2 style={{ marginTop: 14 }}>Fallbacks</h2>
          {Object.keys(m.fallbacks).length === 0 ? <p className="muted">None.</p> : (
            <table><tbody>
              {Object.entries(m.fallbacks).map(([k, v]) => <tr key={k}><td>{k}</td><td className="num">{count(v)}</td></tr>)}
            </tbody></table>
          )}
        </section>
      </div>
      <section className="card">
        <h2>Latest drift check</h2>
        {!drift ? <p className="muted">No drift report yet (the daily drift flow writes reports/drift/latest.json).</p> : (
          <div className="stack">
            <div className="grid cols-4">
              <Tile label="As of" value={<span className="small">{drift.as_of}</span>} />
              <Tile label="Drifted feature share" value={pct(drift.drift_share)} hint="trigger > 30%" />
              <Tile label="Score PSI" value={drift.score_psi == null ? "-" : drift.score_psi.toFixed(3)} hint="trigger > 0.2" />
              <Tile label="Retrain triggered" value={drift.should_retrain ? "yes" : "no"} />
            </div>
            {drift.triggers.length ? (
              <ul className="error">{drift.triggers.map((t) => <li key={t}>{t}</li>)}</ul>
            ) : <p className="good">No trigger fired.</p>}
            {drift.drifted_columns.length ? (
              <p className="small">Drifted: <span className="mono">{drift.drifted_columns.map((c) => c.column).join(", ")}</span></p>
            ) : null}
            {drift.has_html ? (
              <>
                <a href={driftReportUrl} target="_blank" rel="noreferrer">Open the Evidently report in a new tab</a>
                <iframe className="report" title="Evidently drift report" src={driftReportUrl} sandbox="allow-scripts" />
              </>
            ) : <p className="muted">The HTML report is not available.</p>}
          </div>
        )}
      </section>
    </div>
  );
}
