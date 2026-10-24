import { useState } from "react";
import { Line, LineChart, ResponsiveContainer, Tooltip, YAxis } from "recharts";

import { DecisionBadge } from "../components/DecisionBadge";
import { Tile } from "../components/Tile";
import { money2, pct, time } from "../lib/format";
import { useDecisionStream } from "../lib/useDecisionStream";

export function LiveFeed() {
  const [paused, setPaused] = useState(false);
  const { events, throughput, connected } = useDecisionStream(100, paused);
  const series = throughput.map((n, i) => ({ t: i - throughput.length + 1, n }));
  const lastSec = throughput.at(-1) ?? 0;
  const avg = throughput.reduce((a, b) => a + b, 0) / throughput.length;
  const share = (d: string) => (events.length ? events.filter((e) => e.decision === d).length / events.length : null);

  return (
    <div className="stack">
      <div className="row">
        <h1 style={{ margin: 0 }}>Live feed</h1>
        <span className="badge" role="status">
          <span className={`dot ${connected ? "approve" : "block"}`} aria-hidden />
          {connected ? "connected" : "disconnected"}
        </span>
        <button onClick={() => setPaused((p) => !p)}>{paused ? "Resume table" : "Pause table"}</button>
      </div>
      <div className="grid cols-4">
        <Tile label="Throughput (last second)" value={`${lastSec} tx/s`} hint={`${avg.toFixed(1)} tx/s, 60 s average`} />
        <Tile label="Approve (last 100)" value={pct(share("approve"))} />
        <Tile label="Review (last 100)" value={pct(share("review"))} />
        <Tile label="Block (last 100)" value={pct(share("block"))} />
      </div>
      <div className="card">
        <h2>Transactions per second, last 60 s</h2>
        <div style={{ height: 90 }} aria-label="throughput sparkline">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={series} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
              <YAxis width={32} allowDecimals={false} tick={{ fill: "var(--muted)", fontSize: 11 }}
                axisLine={false} tickLine={false} />
              <Tooltip
                formatter={(v) => [`${String(v)} tx/s`, "throughput"]}
                labelFormatter={(_, p) => `${-Number(p[0]?.payload?.t ?? 0)} s ago`}
                contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--ink)" }}
              />
              <Line type="monotone" dataKey="n" stroke="var(--series-1)" strokeWidth={2} dot={false}
                isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div className="card table-wrap">
        <table>
          <thead>
            <tr>
              <th>Event time</th><th>Txn</th><th>Card (uid)</th><th className="num">Amount</th>
              <th>Decision</th><th className="num">P(fraud)</th><th>Top reason</th><th>Model</th>
              <th className="num">Latency</th>
            </tr>
          </thead>
          <tbody>
            {events.length === 0 ? (
              <tr><td colSpan={9} className="muted">Waiting for decisions… (run <code>make replay</code>)</td></tr>
            ) : (
              events.map((e) => (
                <tr key={e.txn_id}>
                  <td className="mono">{time(e.event_time)}</td>
                  <td className="mono">{e.txn_id}</td>
                  <td className="mono">{e.uid}</td>
                  <td className="num">{money2(e.amount)}</td>
                  <td><DecisionBadge decision={e.decision} /></td>
                  <td className="num">{pct(e.probability)}</td>
                  <td>{e.reason_codes[0]?.message ?? (e.rules_fired[0] ? `rule: ${e.rules_fired[0]}` : "-")}</td>
                  <td className="small">{e.model_version ?? "rules"}{e.degraded ? " (degraded)" : ""}</td>
                  <td className="num">{e.latency_ms.toFixed(1)} ms</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
