import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useParams } from "react-router";

import { DecisionBadge } from "../components/DecisionBadge";
import { QueryError } from "../components/QueryState";
import { api } from "../lib/api";
import { money2, pct, time, value } from "../lib/format";

const FIELD_ORDER = ["txn_id", "uid", "event_time", "amount", "product", "card_network", "card_kind",
  "p_email_domain", "r_email_domain", "device_type", "device_info", "addr"];

export function CaseDetail({ analystId }: { analystId: string }) {
  const { caseId = "" } = useParams();
  const qc = useQueryClient();
  const c = useQuery({ queryKey: ["case", caseId], queryFn: () => api.case(caseId) });
  const ctx = useQuery({ queryKey: ["context", caseId], queryFn: () => api.context(caseId) });
  const audit = useQuery({ queryKey: ["audit", caseId], queryFn: () => api.audit(caseId) });
  const [notes, setNotes] = useState("");
  const label = useMutation({
    mutationFn: (l: "fraud" | "legit") => api.label(caseId, l, analystId, notes.trim() || null),
    onSuccess: (updated) => {
      qc.setQueryData(["case", caseId], updated);
      void qc.invalidateQueries({ queryKey: ["audit", caseId] });
      void qc.invalidateQueries({ queryKey: ["cases"] });
    },
  });

  if (c.error) return <QueryError error={c.error} />;
  if (!c.data) return <p className="muted">Loading…</p>;
  const k = c.data;
  const extras = Object.keys(k.txn).filter((f) => !FIELD_ORDER.includes(f) && !f.startsWith("label"));

  return (
    <div className="stack">
      <div className="row">
        <Link to="/queue">← Queue</Link>
        <h1 style={{ margin: 0 }}>Case <span className="mono">{k.case_id}</span></h1>
        <DecisionBadge decision={k.decision} />
        <span className="badge">{k.status === "open" ? "open" : `labelled ${k.label} by ${k.analyst_id}`}</span>
      </div>
      <div className="grid cols-4">
        <div className="card tile"><div className="label">Expected loss</div><div className="value">{money2(k.expected_loss)}</div></div>
        <div className="card tile"><div className="label">P(fraud)</div><div className="value">{pct(k.probability)}</div></div>
        <div className="card tile"><div className="label">Amount</div><div className="value">{money2(Number(k.txn.amount))}</div></div>
        <div className="card tile"><div className="label">Model</div><div className="value small">{k.model_version ?? "rules"}</div></div>
      </div>
      <div className="grid cols-2">
        <section className="card">
          <h2>Why it was flagged</h2>
          {k.reasons.length === 0 ? <p className="muted">No reason codes (rules-only decision).</p> : (
            <table>
              <thead><tr><th>Reason</th><th>Feature</th><th className="num">Value</th><th className="num">Typical</th></tr></thead>
              <tbody>
                {k.reasons.slice(0, 3).map((r) => (
                  <tr key={r.code}>
                    <td style={{ whiteSpace: "normal" }}>{r.message}</td>
                    <td className="mono">{r.feature ?? "-"}</td>
                    <td className="num">{value(r.value)}</td>
                    <td className="num">{value(r.typical)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
        <section className="card">
          <h2>Label</h2>
          <textarea placeholder="Notes (optional)" value={notes} onChange={(e) => setNotes(e.target.value)}
            aria-label="notes" />
          <div className="row" style={{ marginTop: 8 }}>
            <button className="danger" disabled={label.isPending} onClick={() => label.mutate("fraud")}>Fraud</button>
            <button className="primary" disabled={label.isPending} onClick={() => label.mutate("legit")}>Legit</button>
            <span className="small muted">as {analystId}</span>
          </div>
          {label.error ? <QueryError error={label.error} /> : null}
          {k.notes ? <p className="small">Notes: {k.notes}</p> : null}
        </section>
      </div>
      <div className="grid cols-2">
        <section className="card">
          <h2>Transaction</h2>
          <dl className="fields">
            {[...FIELD_ORDER, ...extras].filter((f) => f in k.txn).map((f) => (
              <div key={f} style={{ display: "contents" }}>
                <dt className="mono">{f}</dt>
                <dd>{f === "event_time" ? time(String(k.txn[f])) : value(k.txn[f])}</dd>
              </div>
            ))}
          </dl>
        </section>
        <section className="card">
          <h2>Linked entities</h2>
          {ctx.error ? <QueryError error={ctx.error} /> : !ctx.data ? <p className="muted">Loading…</p> : (
            <>
              {!ctx.data.store_available ? <p className="error small">Feature store unavailable; links may be missing.</p> : null}
              {Object.keys(ctx.data.linked).length === 0 ? <p className="muted">No shared device or email.</p> : null}
              {Object.entries(ctx.data.linked).map(([col, uids]) => (
                <div key={col} style={{ marginBottom: 8 }}>
                  <div className="small muted">
                    Other cards on this {col === "device_info" ? "device" : "email domain"} ({value(k.txn[col])}): {uids.length}
                  </div>
                  <div className="mono">{uids.length ? uids.slice(0, 20).join(", ") + (uids.length > 20 ? " …" : "") : "none"}</div>
                </div>
              ))}
            </>
          )}
        </section>
      </div>
      <section className="card table-wrap">
        <h2>Card timeline (last {ctx.data?.timeline.length ?? 0} events, newest first)</h2>
        <table>
          <thead><tr><th>Event time</th><th>Txn</th><th className="num">Amount</th><th>Decision</th>
            <th className="num">P(fraud)</th><th>Device</th><th>Email</th><th>Product</th></tr></thead>
          <tbody>
            {(ctx.data?.timeline ?? []).map((t) => (
              <tr key={t.txn_id} className={t.txn_id === k.case_id ? "selected" : ""}>
                <td className="mono">{time(t.event_time)}</td>
                <td className="mono">{t.txn_id}</td>
                <td className="num">{money2(t.amount)}</td>
                <td><DecisionBadge decision={t.decision} /></td>
                <td className="num">{pct(t.probability)}</td>
                <td className="mono">{value(t.device_info)}</td>
                <td className="mono">{value(t.p_email_domain)}</td>
                <td>{value(t.product)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      <section className="card table-wrap">
        <h2>Audit trail</h2>
        <table>
          <thead><tr><th>At</th><th>Action</th><th>Actor</th><th>Detail</th></tr></thead>
          <tbody>
            {(audit.data ?? []).map((a, i) => (
              <tr key={i}>
                <td className="mono">{time(a.at)}</td><td>{a.action}</td><td>{a.actor}</td>
                <td className="mono">{Object.keys(a.detail).length ? JSON.stringify(a.detail) : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
