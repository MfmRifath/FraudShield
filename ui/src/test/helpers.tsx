import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import type { ReactElement } from "react";
import { MemoryRouter, Route, Routes } from "react-router";
import { vi } from "vitest";

import type { Case } from "../lib/api";

export interface Call {
  method: string;
  path: string;
  body: unknown;
}

/** Stub fetch with a router of `METHOD /path` -> handler; records every call. */
export function mockApi(routes: Record<string, (body: unknown, path: string) => unknown>): Call[] {
  const calls: Call[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      const path = url.replace(/^\/api/, "");
      const method = init?.method ?? "GET";
      const body = init?.body ? JSON.parse(String(init.body)) : undefined;
      calls.push({ method, path, body });
      const key = Object.keys(routes).find((k) => {
        const [m, p] = k.split(" ");
        return m === method && (p === path || (p!.endsWith("*") && path.startsWith(p!.slice(0, -1))));
      });
      if (!key) return new Response(JSON.stringify({ detail: `no mock for ${method} ${path}` }), { status: 404 });
      return new Response(JSON.stringify(routes[key]!(body, path)), { status: 200 });
    }),
  );
  return calls;
}

/** Render `ui` at `route`; other case pages render a placeholder (to observe navigation). */
export function renderPage(ui: ReactElement, path = "/", route = "*") {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path={route} element={ui} />
          {route === "/cases/:caseId" ? null : <Route path="/cases/:caseId" element={<p>case page</p>} />}
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

export function makeCase(id: string, expected_loss: number, created_at = "2021-03-01T12:00:00Z"): Case {
  return {
    case_id: id, created_at, txn: { txn_id: id, uid: `u-${id}`, amount: 100 }, probability: 0.3,
    decision: "review", reasons: [{ code: "amount", message: "Amount unusually high", feature: "amount", value: 100, typical: 20 }],
    expected_loss, model_version: "m1", status: "open", label: null, label_source: null, analyst_id: null,
    notes: null, labeled_at: null,
  };
}
