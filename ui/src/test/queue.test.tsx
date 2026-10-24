import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { keyAction, sortQueue } from "../lib/queue";
import { ReviewQueue } from "../pages/ReviewQueue";
import { makeCase, mockApi, renderPage } from "./helpers";

describe("sortQueue", () => {
  it("orders by expected loss desc, then oldest, then id", () => {
    const cases = [
      makeCase("a", 10), makeCase("b", 50, "2021-03-01T12:05:00Z"), makeCase("c", 50, "2021-03-01T12:01:00Z"),
      makeCase("d", 50, "2021-03-01T12:01:00Z"),
    ];
    expect(sortQueue(cases).map((c) => c.case_id)).toEqual(["c", "d", "b", "a"]);
    expect(cases.map((c) => c.case_id)).toEqual(["a", "b", "c", "d"]); // input untouched
  });

  it("maps F/L/E and movement keys, case-insensitively", () => {
    expect(keyAction("f")).toEqual({ kind: "label", label: "fraud" });
    expect(keyAction("L")).toEqual({ kind: "label", label: "legit" });
    expect(keyAction("e")).toEqual({ kind: "open" });
    expect(keyAction("j")).toEqual({ kind: "move", by: 1 });
    expect(keyAction("ArrowUp")).toEqual({ kind: "move", by: -1 });
    expect(keyAction("x")).toBeNull();
  });
});

describe("ReviewQueue page", () => {
  const cases = [makeCase("low", 5), makeCase("high", 90), makeCase("mid", 40)];

  function setup() {
    const labelled = new Set<string>();
    const calls = mockApi({
      "GET /cases*": () => cases.filter((c) => !labelled.has(c.case_id)),
      "POST /cases/*": (body, path) => {
        const id = path.split("/")[2]!;
        labelled.add(id);
        return { ...makeCase(id, 0), status: "labeled", ...(body as object) };
      },
    });
    const user = userEvent.setup();
    renderPage(<ReviewQueue analystId="ana" />, "/queue");
    return { calls, user };
  }

  const rowIds = () =>
    within(screen.getByRole("table", { name: "review queue" }))
      .getAllByRole("row")
      .slice(1)
      .map((r) => within(r).getAllByRole("cell")[1]!.textContent);

  it("shows open cases sorted by expected loss", async () => {
    setup();
    await waitFor(() => expect(rowIds()).toEqual(["high", "mid", "low"]));
  });

  it("F labels the selected case fraud, then L labels the next one legit", async () => {
    const { calls, user } = setup();
    await waitFor(() => expect(rowIds()).toHaveLength(3));
    await user.keyboard("f");
    await waitFor(() => expect(rowIds()).toEqual(["mid", "low"]));
    await user.keyboard("l");
    await waitFor(() => expect(calls.filter((c) => c.method === "POST")).toHaveLength(2));
    const posts = calls.filter((c) => c.method === "POST");
    expect(posts.map((c) => [c.path, c.body])).toEqual([
      ["/cases/high/label", { label: "fraud", analyst_id: "ana", notes: null }],
      ["/cases/mid/label", { label: "legit", analyst_id: "ana", notes: null }],
    ]);
  });

  it("J moves the selection and E opens that case", async () => {
    const { user } = setup();
    await waitFor(() => expect(rowIds()).toHaveLength(3));
    await user.keyboard("j");
    expect(screen.getAllByRole("row")[2]?.getAttribute("aria-selected")).toBe("true");
    await user.keyboard("e");
    expect(await screen.findByText("case page")).toBeTruthy();
  });

  it("ignores shortcuts while typing in an input", async () => {
    const { calls, user } = setup();
    await waitFor(() => expect(rowIds()).toHaveLength(3));
    const input = document.createElement("input");
    document.body.appendChild(input);
    await user.click(input);
    await user.keyboard("f");
    expect(calls.some((c) => c.method === "POST")).toBe(false);
    input.remove();
  });
});
