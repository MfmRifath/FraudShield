import { ApiError } from "../lib/api";

export function QueryError({ error }: { error: unknown }) {
  const msg = error instanceof ApiError ? error.detail : error instanceof Error ? error.message : String(error);
  return (
    <p className="error" role="alert">
      {msg}
    </p>
  );
}
