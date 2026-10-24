import type { ReactNode } from "react";

export function Tile({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="card tile">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {hint ? <div className="small muted">{hint}</div> : null}
    </div>
  );
}
