import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type { Costs, Outcome } from "../lib/api";
import { PolicySimulator } from "../pages/PolicySimulator";
import { mockApi, renderPage } from "./helpers";

const COSTS: Costs = { chargeback_fee: 25, margin_rate: 0.03, churn_penalty: 10, analyst_cost: 4, review_friction: 2,
  review_capacity_per_day: 200 };

/** A toy outcome that depends on the inputs, so the test can tell which costs were simulated. */
function outcome(c: Costs): Outcome {
  const reviews = c.review_capacity_per_day ?? 999;
  return { total_cost: 1000 + c.analyst_cost * reviews, fraud_amount: 5000, fraud_amount_stopped: 4000,
    fraud_amount_missed: 1000, legit_blocked: 3, n_approve: 900, n_review: reviews, n_block: 10,
    approve_all_cost: 6000, net_savings_vs_approve_all: 5000 - c.analyst_cost * reviews };
}

describe("PolicySimulator", () => {
  function setup() {
    const calls = mockApi({
      "GET /policy/current": () => ({ costs: COSTS, outcome: outcome(COSTS), window: { rows: 1000, days: 26, frauds: 40 } }),
      "POST /policy/simulate": (b) => ({ window: { rows: 1000, days: 26, frauds: 40 },
        current: { costs: COSTS, outcome: outcome(COSTS) }, simulated: { costs: b, outcome: outcome(b as Costs) } }),
    });
    renderPage(<PolicySimulator debounceMs={0} />, "/policy");
    return calls;
  }

  it("starts from the configured costs and shows the current outcome", async () => {
    const calls = setup();
    await waitFor(() => expect(screen.getByTestId("sim-savings").textContent).toBe("$4,200"));
    expect(calls.find((c) => c.method === "POST")?.body).toEqual(COSTS);
    expect(screen.getByText(/1,000 transactions, 26 days, 40 frauds/)).toBeTruthy();
  });

  it("recomputes savings when a slider moves", async () => {
    const calls = setup();
    await waitFor(() => expect(screen.getByTestId("sim-savings").textContent).toBe("$4,200"));
    fireEvent.change(screen.getByLabelText("Analyst cost per review"), { target: { value: "10" } });
    await waitFor(() => expect(screen.getByTestId("sim-savings").textContent).toBe("$3,000"));
    expect(calls.at(-1)?.body).toEqual({ ...COSTS, analyst_cost: 10 });
    expect(screen.getAllByText("-$1,200")).toHaveLength(2); // difference tile + net-savings row
    expect(screen.getByText("+$1,200")).toBeTruthy(); // total cost went up
  });

  it("sends null capacity for unlimited reviews", async () => {
    const calls = setup();
    await waitFor(() => expect(screen.getByTestId("sim-savings").textContent).toBe("$4,200"));
    await userEvent.setup().click(screen.getByLabelText("Unlimited review capacity"));
    await waitFor(() => expect(calls.at(-1)?.body).toEqual({ ...COSTS, review_capacity_per_day: null }));
    expect((screen.getByLabelText("Review capacity / day") as HTMLInputElement).disabled).toBe(true);
  });

  it("explains a model without a policy window", async () => {
    mockApi({});
    renderPage(<PolicySimulator debounceMs={0} />, "/policy");
    expect(await screen.findByRole("alert")).toBeTruthy();
  });
});
