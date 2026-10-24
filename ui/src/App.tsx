import { lazy, Suspense } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router";

import { useAnalystId } from "./lib/analyst";
import { ReviewQueue } from "./pages/ReviewQueue";

// the queue is the landing page; the rest (and Recharts) load on first visit
const CaseDetail = lazy(() => import("./pages/CaseDetail").then((m) => ({ default: m.CaseDetail })));
const LiveFeed = lazy(() => import("./pages/LiveFeed").then((m) => ({ default: m.LiveFeed })));
const Monitoring = lazy(() => import("./pages/Monitoring").then((m) => ({ default: m.Monitoring })));
const PolicySimulator = lazy(() => import("./pages/PolicySimulator").then((m) => ({ default: m.PolicySimulator })));

const PAGES = [
  ["/live", "Live feed"],
  ["/queue", "Review queue"],
  ["/monitoring", "Model & monitoring"],
  ["/policy", "Policy simulator"],
] as const;

export function App() {
  const [analystId, setAnalystId] = useAnalystId();
  return (
    <div className="shell">
      <header className="topbar">
        <span className="brand">FraudShield</span>
        <nav className="nav" aria-label="Pages">
          {PAGES.map(([to, label]) => (
            <NavLink key={to} to={to}>
              {label}
            </NavLink>
          ))}
        </nav>
        <label className="analyst">
          Analyst
          <input value={analystId} onChange={(e) => setAnalystId(e.target.value.trim() || "analyst")} />
        </label>
      </header>
      <main>
        <Suspense fallback={<p className="muted">Loading…</p>}>
        <Routes>
          <Route path="/" element={<Navigate to="/queue" replace />} />
          <Route path="/live" element={<LiveFeed />} />
          <Route path="/queue" element={<ReviewQueue analystId={analystId} />} />
          <Route path="/cases/:caseId" element={<CaseDetail analystId={analystId} />} />
          <Route path="/monitoring" element={<Monitoring />} />
          <Route path="/policy" element={<PolicySimulator />} />
          <Route path="*" element={<p>Page not found.</p>} />
        </Routes>
        </Suspense>
      </main>
    </div>
  );
}
