import { useCallback, useState } from "react";

const KEY = "fraudshield.analyst_id";

function read(): string {
  try {
    return window.localStorage.getItem(KEY) || "analyst";
  } catch {
    return "analyst";
  }
}

/** The analyst id attached to labels; remembered per browser (a convenience, not auth). */
export function useAnalystId(): [string, (v: string) => void] {
  const [id, setId] = useState(read);
  const set = useCallback((v: string) => {
    setId(v);
    try {
      window.localStorage.setItem(KEY, v);
    } catch {
      // storage unavailable (private mode): keep it for this session only
    }
  }, []);
  return [id, set];
}
