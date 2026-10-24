import type { Case } from "./api";

/** Review order: highest expected loss first, then oldest, then id (same as the API). */
export function sortQueue(cases: readonly Case[]): Case[] {
  return [...cases].sort(
    (a, b) =>
      b.expected_loss - a.expected_loss ||
      a.created_at.localeCompare(b.created_at) ||
      a.case_id.localeCompare(b.case_id),
  );
}

export type QueueAction = { kind: "label"; label: "fraud" | "legit" } | { kind: "open" } | { kind: "move"; by: number };

/** Keyboard map of the review queue: F fraud, L legit, E examine (open case), J/K or arrows move. */
export function keyAction(key: string): QueueAction | null {
  switch (key.toLowerCase()) {
    case "f":
      return { kind: "label", label: "fraud" };
    case "l":
      return { kind: "label", label: "legit" };
    case "e":
    case "enter":
      return { kind: "open" };
    case "j":
    case "arrowdown":
      return { kind: "move", by: 1 };
    case "k":
    case "arrowup":
      return { kind: "move", by: -1 };
    default:
      return null;
  }
}

/** Ignore shortcuts while the analyst is typing (notes, analyst id). */
export function isTyping(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
}
