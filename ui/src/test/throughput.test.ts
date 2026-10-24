import { describe, expect, it } from "vitest";

import { pruneArrivals, throughputSeries } from "../lib/throughput";

describe("throughput", () => {
  const now = 100_500; // t = 100.5 s
  it("buckets arrivals per second, oldest first, ignoring old and future events", () => {
    const s = throughputSeries([100_100, 100_400, 99_000, 99_999, 40_000, 101_000], now, 3);
    expect(s).toEqual([0, 2, 2]); // 98 s, 99 s, 100 s
  });
  it("prunes arrivals that fell out of the window", () => {
    expect(pruneArrivals([97_000, 98_100, 99_000, 100_200], now, 3)).toEqual([98_100, 99_000, 100_200]);
    expect(pruneArrivals([1, 2], now, 3)).toEqual([]);
  });
});
