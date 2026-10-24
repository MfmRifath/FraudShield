import type { Decision } from "../lib/api";

/** Decision as a status colour dot + its name: never colour alone. */
export function DecisionBadge({ decision }: { decision: Decision | null | undefined }) {
  if (!decision) return <span className="muted">-</span>;
  return (
    <span className="badge">
      <span className={`dot ${decision}`} aria-hidden />
      {decision}
    </span>
  );
}
