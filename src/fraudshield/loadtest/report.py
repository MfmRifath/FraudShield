"""Load-test report: Locust CSVs + /metrics samples -> ``results.json``, ``report.md``, charts.

Two latency views per scenario:
* **client** (``/score`` row): what Locust measured, request sent -> response read, over the
  compose network (includes HTTP parsing, uvicorn, JSON);
* **service** (``service`` row): the ``latency_ms`` the API returns for each request
  (validation -> response built), recorded by the locustfile as its own Locust series.
Fallbacks and the decision mix come from the API's Prometheus counters (deltas over the run).
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fraudshield.loadtest.metrics import Snapshot

CLIENT_ROW = "/score"
SERVICE_ROW = "service"
POLICY_FALLBACKS = {"review_capacity"}  # a policy outcome, not a degradation
PCTS = ("50%", "95%", "99%")


def _num(v: str | None) -> float | None:
    if v in (None, "", "N/A"):
        return None
    f = float(v)
    return None if math.isnan(f) else f


def read_stats(path: Path, name: str) -> dict[str, float | None] | None:
    """One row of a Locust ``*_stats.csv`` by request name."""
    if not path.is_file():
        return None
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["Name"] == name:
                return {
                    "requests": _num(row["Request Count"]),
                    "failures": _num(row["Failure Count"]),
                    "rps": _num(row["Requests/s"]),
                    "mean_ms": _num(row["Average Response Time"]),
                    "max_ms": _num(row["Max Response Time"]),
                    **{f"p{p[:-1]}_ms": _num(row[p]) for p in PCTS},
                }
    return None


def read_history(path: Path, name: str) -> list[dict[str, float | None]]:
    """Per-second rows of ``*_stats_history.csv`` (percentiles over Locust's sliding window)."""
    if not path.is_file():
        return []
    out = []
    with path.open() as f:
        rows = [r for r in csv.DictReader(f) if r["Name"] == name]
    if not rows:
        return []
    t0 = float(rows[0]["Timestamp"])
    for r in rows:
        out.append({"t": float(r["Timestamp"]) - t0, "rps": _num(r["Requests/s"]),
                    "failures_s": _num(r["Failures/s"]),
                    **{f"p{p[:-1]}_ms": _num(r[p]) for p in PCTS}})  # fmt: skip
    return out


def snapshots(server: dict[str, Any]) -> list[Snapshot]:
    return [
        Snapshot(
            s["t"],
            s["decisions"],
            s["fallbacks"],
            {math.inf if k == "inf" else float(k): v for k, v in s["latency_buckets"].items()},
        )
        for s in server["samples"]
    ]


def degraded_series(samples: list[Snapshot]) -> list[dict[str, float]]:
    """Per-interval share of requests that took a fallback path (policy outcomes excluded)."""
    out = []
    for a, b in itertools.pairwise(samples):
        d = b.minus(a)
        n = d.scored
        fb = sum(v for k, v in d.fallbacks.items() if k not in POLICY_FALLBACKS)
        if n > 0:
            out.append({"t": b.t, "scored": n, "fallback_share": fb / n})
    return out


RECOVERED_BELOW = 0.05


def recovery_seconds(
    series: list[dict[str, float]], restart_at: float, below: float = RECOVERED_BELOW
) -> float | None:
    """Seconds after Redis restarted until the fallback share stays below ``below`` (3 samples).

    5 %, not 0: a loaded single-worker API misses the 10 ms feature budget on a few requests in
    steady state too (see the normal scenario's fallback share)."""
    after = [p for p in series if p["t"] >= restart_at]
    for i in range(len(after) - 2):
        if all(p["fallback_share"] < below for p in after[i : i + 3]):
            return after[i]["t"] - restart_at
    return None


def model_info(bundle_dir: Path) -> dict[str, Any]:
    """Trees and leaves of the served full model, read from LightGBM's text format."""
    spec = json.loads((bundle_dir / "bundle.json").read_text())
    text = (bundle_dir / spec["full"]["booster_file"]).read_text()
    leaves = [
        int(line.split("=", 1)[1]) for line in text.splitlines() if line.startswith("num_leaves=")
    ]
    meta = spec["metadata"]
    return {
        "model_version": meta.get("model_version"),
        "dataset": meta.get("dataset"),
        "created_at": meta.get("created_at"),
        "git_sha": meta.get("git_sha"),
        "trees_full": len(leaves),
        "mean_leaves_per_tree": round(sum(leaves) / len(leaves), 1) if leaves else None,
        "features_full": len(spec["full"]["feature_names"]),
    }


def scenario_result(raw: Path, name: str) -> dict[str, Any]:
    server = json.loads((raw / f"{name}_server.json").read_text())
    samples = snapshots(server)
    total = samples[-1].minus(samples[0]) if len(samples) >= 2 else None
    series = degraded_series(samples)
    outage = server.get("outage")
    out: dict[str, Any] = {
        "name": name,
        "title": server["scenario"]["title"],
        "client": read_stats(raw / f"{name}_stats.csv", CLIENT_ROW),
        "service": read_stats(raw / f"{name}_stats.csv", SERVICE_ROW),
        "history": read_history(raw / f"{name}_stats_history.csv", CLIENT_ROW),
        "degraded_series": series,
        "load_start": server.get("load_start"),
        "outage": outage,
        "container_stats": server.get("container_stats", {}),
    }
    if total is not None:
        fb = {k: v for k, v in total.fallbacks.items() if v > 0}
        out["server"] = {
            "scored": total.scored,
            "decisions": {k: v for k, v in total.decisions.items() if v > 0},
            "fallbacks": fb,
            "fallback_share": sum(v for k, v in fb.items() if k not in POLICY_FALLBACKS)
            / max(total.scored, 1),
            "bucketed_latency_s": total.latency_quantiles(),
        }
    if outage:
        out["recovery_s"] = recovery_seconds(series, outage[1])
    return out


def build_results(out_dir: Path) -> dict[str, Any]:
    raw = out_dir / "raw"
    meta = json.loads((raw / "run.json").read_text())
    bundle = Path(meta["config"]["bundle_dir"])
    return {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "config": meta["config"],
        "environment": meta["environment"],
        "model": model_info(bundle) if (bundle / "bundle.json").is_file() else None,
        "scenarios": [scenario_result(raw, n) for n in meta["scenarios"]],
    }


def _ms(v: float | None) -> str:
    return "-" if v is None else f"{v:.1f}"


def _synthetic(m: dict[str, Any]) -> bool:
    return m.get("dataset") == "synthetic" or str(m.get("model_version", "")).startswith(
        "synthetic"
    )


def _model_label(m: dict[str, Any]) -> str:
    if _synthetic(m) or m.get("dataset") is None:
        return f"`{m.get('model_version')}`"
    return f"the trained `{m['dataset']}` model (trained {m.get('created_at')}, git `{m.get('git_sha')}`)"


def _model_note(m: dict[str, Any]) -> str:
    if _synthetic(m):
        return "A *synthetic* bundle of production-like size: it measures the serving path, not model quality."
    return "Payloads are that dataset's held-out test window, as served."


def to_markdown(r: dict[str, Any]) -> str:
    c, e, m = r["config"], r["environment"], r["model"] or {}
    lines = [
        "# Load test",
        "",
        f"Generated {r['generated_at']} by `make loadtest` (`python -m fraudshield.loadtest`). "
        "Rerun it to refresh every number on this page.",
        "",
        f"- **Load:** {c['users']} Locust users x {c['rps_per_user']} req/s (Poisson arrivals) = "
        f"**{c['users'] * c['rps_per_user']:.0f} req/s target**, {c['duration_s']} s per scenario "
        f"after a {c['warmup_s']} s warm-up. {round(c['known_share'] * 100)} % of requests come "
        "from returning cards, the rest from never-seen cards; the API stamps server time.",
        f"- **Model:** {_model_label(m)}, {m.get('trees_full')} trees "
        f"(mean {m.get('mean_leaves_per_tree')} leaves/tree), {m.get('features_full')} features. "
        + _model_note(m),
        "- **System under test:** the compose `core` stack (API: 1 uvicorn worker, 768 MiB limit; "
        "Redis feature store; Kafka publishing; processor and shadow scorer consuming).",
        f"- **Machine:** {e.get('host_system')} {e.get('host_machine')} host, Docker "
        f"{e.get('docker_os')} ({e.get('docker_arch')}), {e.get('docker_cpus')} CPUs, "
        f"{e.get('docker_mem_gib')} GiB. Locust runs on the same machine.",
        "",
        "| Scenario | Requests | Achieved req/s | Failed | Client p50 / p95 / p99 (ms) "
        "| Service p50 / p95 / p99 (ms) | Fallback share |",
        "|---|---:|---:|---:|---|---|---:|",
    ]
    for s in r["scenarios"]:
        cl, sv, srv = s["client"] or {}, s["service"] or {}, s.get("server", {})
        lines.append(
            f"| {s['title']} | {cl.get('requests', 0):,.0f} | {cl.get('rps') or 0:.1f} "
            f"| {cl.get('failures', 0):,.0f} "
            f"| {_ms(cl.get('p50_ms'))} / {_ms(cl.get('p95_ms'))} / {_ms(cl.get('p99_ms'))} "
            f"| {_ms(sv.get('p50_ms'))} / {_ms(sv.get('p95_ms'))} / {_ms(sv.get('p99_ms'))} "
            f"| {srv.get('fallback_share', 0) * 100:.1f} % |"
        )
    lines += [
        "",
        "*Client* = Locust, request sent to response read, over the compose network. *Service* = "
        "the `latency_ms` the API reports (validation to response). *Failed* = non-200 responses "
        "or client errors. *Fallback share* = requests answered on a fallback path "
        "(`review_capacity` excluded: it is a policy outcome).",
        "",
        "![Latency over time](latency.png)",
        "",
    ]
    for s in r["scenarios"]:
        srv = s.get("server", {})
        lines += [f"## {s['title']}", ""]
        if s.get("outage"):
            k, st = s["outage"]
            rec = s.get("recovery_s")
            lines += [
                f"Redis was killed {k - (s['load_start'] or 0):.0f} s into the load and started "
                f"again {st - k:.0f} s later. "
                + (
                    f"The fallback share was back under {RECOVERED_BELOW:.0%} {rec:.0f} s after the restart."
                    if rec is not None
                    else f"The fallback share did not drop back under {RECOVERED_BELOW:.0%} before the run ended."
                ),
                "",
                "![Fallback share during the Redis outage](redis_killed.png)",
                "",
            ]
        if srv:
            dec = ", ".join(f"{k} {v:,.0f}" for k, v in srv["decisions"].items())
            fb = ", ".join(f"`{k}` {v:,.0f}" for k, v in srv["fallbacks"].items()) or "none"
            q = srv["bucketed_latency_s"]
            lines += [
                f"- Decisions: {dec}.",
                f"- Fallbacks by reason: {fb}.",
                "- Server histogram (Prometheus buckets, coarse): p50 / p95 / p99 = "
                + " / ".join(
                    _ms(None if q.get(p) is None else q[p] * 1e3) for p in ("p50", "p95", "p99")
                )
                + " ms.",
            ]
        if s.get("container_stats"):
            cs = s["container_stats"]
            lines.append(
                f"- API container mid-run: {cs.get('MemUsage')} memory, {cs.get('CPUPerc')} CPU."
            )
        lines.append("")
    lines += [
        "## Caveats",
        "",
        "- One laptop runs the load generator, the API, Redis, Redpanda and the stream workers, so "
        "they compete for CPU; a dedicated host would show lower tails.",
        "- Locust's per-second percentiles (charts) use its 10 s sliding window; the table uses all "
        "requests of the scenario.",
        "- The decision mix says nothing about the policy: each returning card transacts every few "
        "seconds (the burst pattern the synthetic model learnt as fraud), and the 200/day review "
        "capacity is used up within seconds, after which reviews become approve/block "
        "(`review_capacity`).",
        "- The real-model rerun (IEEE-CIS champion, test-window payloads) is in PLAN.md "
        '"Deferred real-data runs".',
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- charts (light theme PNGs)
INK, INK2, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#f0efec"
SERIES = {"p50_ms": ("#2a78d6", "p50"), "p95_ms": ("#eb6834", "p95"), "p99_ms": ("#1baf7a", "p99")}


def _style(ax: Any) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)


def plot(r: dict[str, Any], out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    scen = r["scenarios"]
    fig, axes = plt.subplots(
        1, len(scen), figsize=(4.2 * len(scen), 3.2), sharey=True, squeeze=False
    )
    for ax, s in zip(axes[0], scen, strict=True):
        h = s["history"]
        _style(ax)
        for key, (color, label) in SERIES.items():
            pts = [(p["t"], p[key]) for p in h if p[key] is not None]
            if pts:
                ax.plot(*zip(*pts, strict=True), color=color, linewidth=2, label=label)
        if s.get("outage") and h:
            ls = s["load_start"] or 0.0
            # history time starts when Locust starts reporting, roughly at load start
            ax.axvspan(s["outage"][0] - ls, s["outage"][1] - ls, color=BAND, zorder=0)
            ax.text((s["outage"][0] + s["outage"][1]) / 2 - ls, 0.97, "Redis down", color=INK2,
                    ha="center", va="top", fontsize=8, transform=ax.get_xaxis_transform())  # fmt: skip
        ax.set_title(s["title"], fontsize=9, color=INK, loc="left")
        ax.set_xlabel("seconds", fontsize=8, color=INK2)
    axes[0][0].set_ylabel("client latency (ms)", fontsize=8, color=INK2)
    axes[0][0].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    p = out_dir / "latency.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)

    killed = next((s for s in scen if s.get("outage")), None)
    if killed and killed["degraded_series"]:
        fig, ax = plt.subplots(figsize=(6.4, 2.8))
        _style(ax)
        ser = killed["degraded_series"]
        ax.plot([p["t"] for p in ser], [p["fallback_share"] * 100 for p in ser], color="#2a78d6",
                linewidth=2)  # fmt: skip
        k, st = killed["outage"]
        ax.axvspan(k, st, color=BAND, zorder=0)
        ax.text((k + st) / 2, 0.97, "Redis down", color=INK2, ha="center", va="top", fontsize=8,
                transform=ax.get_xaxis_transform())  # fmt: skip
        ax.set_ylim(-2, 102)
        ax.set_ylabel("requests on a fallback path (%)", fontsize=8, color=INK2)
        ax.set_xlabel("seconds since scenario start", fontsize=8, color=INK2)
        ax.set_title("Redis killed: degraded (request-only) scoring, no errors", fontsize=9,
                     color=INK, loc="left")  # fmt: skip
        fig.tight_layout()
        p = out_dir / "redis_killed.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    return paths


def write_report(out_dir: Path, charts: bool = True) -> dict[str, Any]:
    r = build_results(out_dir)
    (out_dir / "results.json").write_text(json.dumps(r, indent=1) + "\n")
    (out_dir / "report.md").write_text(to_markdown(r))
    if charts:
        plot(r, out_dir)
    return r
