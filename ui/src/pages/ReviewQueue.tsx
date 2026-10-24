import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router";

import { DecisionBadge } from "../components/DecisionBadge";
import { QueryError } from "../components/QueryState";
import { api } from "../lib/api";
import { money2, pct, time } from "../lib/format";
import { isTyping, keyAction, sortQueue } from "../lib/queue";

export function ReviewQueue({ analystId }: { analystId: string }) {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const q = useQuery({ queryKey: ["cases", "open"], queryFn: () => api.cases("open"), refetchInterval: 5000 });
  const queue = useMemo(() => sortQueue(q.data ?? []), [q.data]);
  const [selected, setSelected] = useState(0);
  const [last, setLast] = useState<string | null>(null);
  const idx = Math.min(selected, Math.max(queue.length - 1, 0));
  const current = queue[idx];

  const label = useMutation({
    mutationFn: (v: { id: string; label: "fraud" | "legit" }) => api.label(v.id, v.label, analystId, null),
    onSuccess: (c) => {
      setLast(`${c.case_id} labelled ${c.label}`);
      qc.setQueryData(["cases", "open"], (old: typeof q.data) => old?.filter((x) => x.case_id !== c.case_id));
      void qc.invalidateQueries({ queryKey: ["cases"] });
    },
  });

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e.target) || e.metaKey || e.ctrlKey || e.altKey) return;
      const a = keyAction(e.key);
      if (!a) return;
      e.preventDefault();
      if (a.kind === "move") setSelected(Math.max(0, Math.min(queue.length - 1, idx + a.by)));
      else if (!current) return;
      else if (a.kind === "open") navigate(`/cases/${encodeURIComponent(current.case_id)}`);
      else if (!label.isPending) label.mutate({ id: current.case_id, label: a.label });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [queue, idx, current, label, navigate]);

  return (
    <div className="stack">
      <div className="row">
        <h1 style={{ margin: 0 }}>Review queue</h1>
        <span className="muted">{queue.length} open, ordered by expected loss</span>
      </div>
      <p className="small muted" style={{ margin: 0 }}>
        <kbd>F</kbd> fraud · <kbd>L</kbd> legit · <kbd>E</kbd> examine case · <kbd>J</kbd>/<kbd>K</kbd> move
      </p>
      {last ? <p className="good small" role="status" style={{ margin: 0 }}>{last}</p> : null}
      {label.error ? <QueryError error={label.error} /> : null}
      {q.error ? <QueryError error={q.error} /> : null}
      <div className="card table-wrap">
        <table aria-label="review queue">
          <thead>
            <tr>
              <th className="num">Expected loss</th><th>Case</th><th>Card (uid)</th><th className="num">Amount</th>
              <th className="num">P(fraud)</th><th>Decision</th><th>Top reason</th><th>Opened</th>
            </tr>
          </thead>
          <tbody>
            {q.isPending ? (
              <tr><td colSpan={8} className="muted">Loading…</td></tr>
            ) : queue.length === 0 ? (
              <tr><td colSpan={8} className="muted">Queue is empty.</td></tr>
            ) : (
              queue.map((c, i) => (
                <tr
                  key={c.case_id}
                  className={`clickable${i === idx ? " selected" : ""}`}
                  aria-selected={i === idx}
                  onClick={() => setSelected(i)}
                  onDoubleClick={() => navigate(`/cases/${encodeURIComponent(c.case_id)}`)}
                >
                  <td className="num">{money2(c.expected_loss)}</td>
                  <td className="mono">{c.case_id}</td>
                  <td className="mono">{String(c.txn.uid ?? "-")}</td>
                  <td className="num">{money2(Number(c.txn.amount))}</td>
                  <td className="num">{pct(c.probability)}</td>
                  <td><DecisionBadge decision={c.decision} /></td>
                  <td>{c.reasons[0]?.message ?? "-"}</td>
                  <td className="mono">{time(c.created_at)}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
