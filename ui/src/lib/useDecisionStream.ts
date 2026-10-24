import { useEffect, useRef, useState } from "react";

import { type DecisionEvent, streamUrl } from "./api";
import { pruneArrivals, throughputSeries } from "./throughput";

export interface StreamState {
  events: DecisionEvent[];
  throughput: number[];
  connected: boolean;
}

/** Subscribe to the API's SSE decision stream; keep the newest `keep` events. */
export function useDecisionStream(keep = 100, paused = false): StreamState {
  const [events, setEvents] = useState<DecisionEvent[]>([]);
  const [throughput, setThroughput] = useState<number[]>(() => throughputSeries([], Date.now()));
  const [connected, setConnected] = useState(false);
  const arrivals = useRef<number[]>([]);
  const pausedRef = useRef(paused);
  pausedRef.current = paused;

  useEffect(() => {
    const es = new EventSource(streamUrl);
    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false); // EventSource reconnects by itself
    es.addEventListener("decision", (msg) => {
      arrivals.current.push(Date.now());
      if (pausedRef.current) return;
      const ev = JSON.parse((msg as MessageEvent<string>).data) as DecisionEvent;
      setEvents((prev) => [ev, ...prev].slice(0, keep));
    });
    const tick = window.setInterval(() => {
      const now = Date.now();
      arrivals.current = pruneArrivals(arrivals.current, now);
      setThroughput(throughputSeries(arrivals.current, now));
    }, 1000);
    return () => {
      es.close();
      window.clearInterval(tick);
    };
  }, [keep]);

  return { events, throughput, connected };
}
