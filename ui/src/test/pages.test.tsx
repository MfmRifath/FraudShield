import { act, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { MonitoringSummary } from "../lib/api";
import { CaseDetail } from "../pages/CaseDetail";
import { LiveFeed } from "../pages/LiveFeed";
import { Monitoring } from "../pages/Monitoring";
import { makeCase, mockApi, renderPage } from "./helpers";

describe("CaseDetail", () => {
  it("shows reasons with value vs typical, timeline, linked cards, audit, and labels with notes", async () => {
    const k = makeCase("t1", 42);
    const calls = mockApi({
      "GET /cases/t1": () => k,
      "GET /cases/t1/context": () => ({
        case_id: "t1", uid: "u-t1", store_available: true, linked: { device_info: ["u-x", "u-y"] },
        timeline: [{ txn_id: "t1", event_time: "2021-03-01T12:00:00Z", amount: 100, decision: "review",
          probability: 0.3, product: "web", device_info: "d1", p_email_domain: "a.com", addr: null,
          degraded: false, model_version: "m1" }],
      }),
      "GET /cases/t1/audit": () => [{ at: "2021-03-01T12:00:00Z", case_id: "t1", action: "open", actor: "system", detail: {} }],
      "POST /cases/t1/label": (b) => ({ ...k, status: "labeled", label: "fraud", analyst_id: "ana", notes: (b as { notes: string }).notes }),
    });
    renderPage(<CaseDetail analystId="ana" />, "/cases/t1", "/cases/:caseId");
    expect(await screen.findByText("Amount unusually high")).toBeTruthy();
    expect(screen.getByText("20")).toBeTruthy(); // typical value next to the observed 100
    expect(await screen.findByText("u-x, u-y")).toBeTruthy();
    expect(screen.getByText("system")).toBeTruthy();
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("notes"), "ring device");
    await user.click(screen.getByRole("button", { name: "Fraud" }));
    await waitFor(() => expect(calls.some((c) => c.method === "POST")).toBe(true));
    expect(calls.find((c) => c.method === "POST")?.body).toEqual({ label: "fraud", analyst_id: "ana", notes: "ring device" });
    expect(await screen.findByText("labelled fraud by ana")).toBeTruthy();
  });
});

describe("Monitoring", () => {
  it("renders versions, decision mix, latency and the drift report", async () => {
    const summary: MonitoringSummary = {
      models: { champion: "v3", challenger: "v4" }, canary_pct: 10,
      decisions: { approve: 90, review: 8, block: 2 }, fallbacks: { feature_timeout: 3 },
      latency_s: { p50: 0.002, p95: 0.004, p99: 0.009 }, review_queue_depth: 8,
      drift: { as_of: "2021-03-01", drift_share: 0.6, score_psi: 0.12, triggers: ["feature drift: 60%"],
        drifted_columns: [{ column: "amount", drifted: true }], has_html: true, should_retrain: true },
    };
    mockApi({ "GET /monitoring/summary": () => summary,
      "GET /model": () => ({ champion: "v3", challenger: "v4", canary_pct: 10, dataset: "sparkov" }) });
    renderPage(<Monitoring />);
    expect(await screen.findByText("v3")).toBeTruthy();
    expect(screen.getByText("90.0%")).toBeTruthy();
    expect(screen.getByText("4.00 ms")).toBeTruthy();
    expect(screen.getByText("feature_timeout")).toBeTruthy();
    expect(screen.getByText("feature drift: 60%")).toBeTruthy();
    expect(screen.getByTitle("Evidently drift report")).toBeTruthy();
  });
});

class FakeEventSource {
  static last: FakeEventSource | null = null;
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  private listeners: Record<string, ((e: MessageEvent) => void)[]> = {};
  constructor(readonly url: string) {
    FakeEventSource.last = this;
  }
  addEventListener(type: string, fn: (e: MessageEvent) => void) {
    (this.listeners[type] ??= []).push(fn);
  }
  emit(type: string, data: unknown) {
    for (const fn of this.listeners[type] ?? []) fn(new MessageEvent(type, { data: JSON.stringify(data) }));
  }
  close() {}
}

describe("LiveFeed", () => {
  it("subscribes to the decision stream and shows events newest first", async () => {
    vi.stubGlobal("EventSource", FakeEventSource);
    renderPage(<LiveFeed />);
    const es = FakeEventSource.last!;
    expect(es.url).toBe("/api/events/stream");
    act(() => es.onopen?.());
    expect(await screen.findByText("connected")).toBeTruthy();
    const ev = (id: string, decision: string) => ({
      txn_id: id, uid: "u1", event_time: "2021-03-01T12:00:00Z", amount: 12.5, decision, probability: 0.02,
      model_version: "m1", variant: "champion", degraded: false, latency_ms: 3.2, rules_fired: [],
      reason_codes: [{ code: "c", message: "New device", feature: "f", value: 1, typical: 0 }],
    });
    act(() => {
      es.emit("decision", ev("a", "approve"));
      es.emit("decision", ev("b", "block"));
    });
    const rows = await screen.findAllByText(/^[ab]$/);
    expect(rows.map((r) => r.textContent)).toEqual(["b", "a"]);
    expect(screen.getAllByText("New device")).toHaveLength(2);
  });
});
