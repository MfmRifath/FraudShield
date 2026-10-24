/** Events per second over the last `seconds` seconds, oldest bucket first (sparkline input). */
export function throughputSeries(arrivalsMs: readonly number[], nowMs: number, seconds = 60): number[] {
  const out = new Array<number>(seconds).fill(0);
  const end = Math.floor(nowMs / 1000);
  for (const t of arrivalsMs) {
    const age = end - Math.floor(t / 1000);
    if (age >= 0 && age < seconds) out[seconds - 1 - age]! += 1;
  }
  return out;
}

/** Keep only arrivals inside the window (bounded memory for a long-lived feed). */
export function pruneArrivals(arrivalsMs: number[], nowMs: number, seconds = 60): number[] {
  const cutoff = (Math.floor(nowMs / 1000) - seconds + 1) * 1000;
  const i = arrivalsMs.findIndex((t) => t >= cutoff);
  return i === -1 ? [] : i === 0 ? arrivalsMs : arrivalsMs.slice(i);
}
