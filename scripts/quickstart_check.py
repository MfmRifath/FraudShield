"""Follow the README's demo quickstart in a fresh clone and time it (Phase 9 DoD: < 10 minutes).

    make quickstart-check

1. clone the repository (HEAD) into a temporary directory;
2. cold-build the demo image there (``--no-cache``), timed;
3. run the commands from the README's ``quickstart`` block verbatim, except that the clone URL
   points at this checkout (there may be no public remote yet), timed until the container is
   healthy;
4. check what the README promises: UI at the demo port, API ready, API docs, decisions arriving;
5. tear everything down and write ``reports/demo/quickstart.{json,md}``.

A separate compose project, image tag and port keep it apart from a running ``make demo``.
Base images already in the local Docker cache are not re-downloaded; the report lists their sizes
so the download time on a fresh machine can be estimated.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fraudshield.loadtest.runner import image_size_bytes

START, END = "<!-- quickstart:start -->", "<!-- quickstart:end -->"
BASE_IMAGES = ("node:24.21.0-alpine", "ghcr.io/astral-sh/uv:0.12.23-python3.12-trixie-slim",
               "python:3.12.15-slim-trixie")  # fmt: skip
PROJECT = "fraudshield-quickstart"


def quickstart_commands(readme: str) -> list[str]:
    """Shell lines of the README's quickstart code block (comments and blanks dropped)."""
    a, b = readme.index(START), readme.index(END)
    block = re.search(r"```(?:bash|sh)\n(.*?)```", readme[a:b], re.S)
    if block is None:
        raise ValueError("no ```bash block between the quickstart markers")
    lines = [ln.split(" #")[0].strip() for ln in block.group(1).splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def localise(cmds: list[str], repo: Path) -> list[str]:
    """Point the README's `git clone <url>` at this checkout."""
    out = []
    for c in cmds:
        m = re.search(r"git clone (\S+)", c)
        if (
            m
        ):  # keep the directory the README's URL would create (.../fraudshield.git -> fraudshield)
            dest = m.group(1).rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
            c = c.replace(m.group(0), f"git clone --quiet {repo.as_uri()} {dest}", 1)
        out.append(c)
    return out


def _sh(cmd: str, cwd: Path, env: dict[str, str]) -> float:
    """Run ``cmd``; on failure show the end of its output (the build log says why)."""
    t = time.monotonic()
    p = subprocess.run(["bash", "-euo", "pipefail", "-c", cmd], cwd=cwd, env=env,  # noqa: S603, S607
                       capture_output=True, text=True, check=False)  # fmt: skip
    if p.returncode != 0:
        tail = "\n".join((p.stdout + p.stderr).splitlines()[-40:])
        raise SystemExit(f"step failed ({p.returncode}): {cmd}\n{tail}")
    return time.monotonic() - t


def _get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:  # noqa: S310 - local demo
            return r.status, r.read()
    except OSError:
        return 0, b""


def _scored(base: str) -> float:
    status, body = _get(f"{base}/api/monitoring/summary")
    return float(sum(json.loads(body)["decisions"].values())) if status == 200 else -1.0


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path.cwd().resolve()
    cmds = quickstart_commands((repo / "README.md").read_text())
    env = os.environ | {"COMPOSE_PROJECT_NAME": PROJECT, "FS_TAG": "quickstart",
                        "FS_DEMO_PORT": str(args.port)}  # fmt: skip
    base = f"http://localhost:{args.port}"
    work = Path(tempfile.mkdtemp(prefix="fs-quickstart-"))
    timings: dict[str, float] = {}
    m = re.search(r"git clone (?:--\S+ )*\S+ (\S+)", localise(cmds, repo)[0])
    clone = work / (m.group(1) if m else "fraudshield")  # the directory the README's clone creates
    try:
        # the README's own clone step comes first; build cold in that clone, then run the rest
        clone_cmd, rest = localise(cmds, repo)[0], localise(cmds, repo)[1:]
        first, _, tail = clone_cmd.partition("&&")
        timings["clone_s"] = _sh(first.strip(), work, env)
        tail_cmds = [
            c.strip() for c in tail.split("&&") if c.strip() and not c.strip().startswith("cd ")
        ]
        if args.cold:
            timings["cold_build_s"] = _sh(
                "docker compose --progress plain -f compose.demo.yml build --no-cache", clone, env
            )
        timings["start_until_healthy_s"] = _sh(" && ".join(tail_cmds + rest), clone, env)
        first_scored = _scored(base)
        time.sleep(10)
        ui_status, ui = _get(f"{base}/queue")
        checks = {
            "ui_served": ui_status == 200 and b'id="root"' in ui,
            "api_ready": _get(f"{base}/api/ready")[0] == 200,
            "api_docs": _get(f"{base}/api/docs")[0] == 200
            and _get(f"{base}/api/openapi.json")[0] == 200,
            "decisions_arriving": _scored(base) > first_scored >= 0,
        }
    finally:
        if clone.exists():
            subprocess.run(["docker", "compose", "-f", "compose.demo.yml", "down", "-v"],  # noqa: S607
                           cwd=clone, env=env, check=False, capture_output=True)  # fmt: skip
        shutil.rmtree(work, ignore_errors=True)
    total = sum(timings.values())
    ver = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"],  # noqa: S607
                         capture_output=True, text=True, check=False).stdout.strip()  # fmt: skip
    return {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "readme_commands": cmds,
        "timings_s": {k: round(v, 1) for k, v in timings.items()},
        "total_s": round(total, 1),
        "under_10_min": total < 600,
        "cold_build": args.cold,
        "checks": checks,
        "base_images_mib": {
            i: None if (b := image_size_bytes(i)) is None else round(b / 2**20) for i in BASE_IMAGES
        },
        "machine": f"{platform.system()} {platform.machine()}, Docker {ver}",
    }


def to_markdown(r: dict[str, Any]) -> str:
    t = r["timings_s"]
    ok = r["under_10_min"] and all(r["checks"].values())
    cmds = "\n".join(r["readme_commands"])
    checks = "\n".join(f"| {k} | {'pass' if v else 'FAIL'} |" for k, v in r["checks"].items())
    bases = ", ".join(
        f"`{k}` {'size unknown (pulled by BuildKit, not in the image store)' if v is None else f'{v} MiB'}"
        for k, v in r["base_images_mib"].items()
    )
    return f"""# README quickstart check

Generated {r["generated_at"]} by `make quickstart-check` (`scripts/quickstart_check.py`) on {r["machine"]}.
**{"Passed" if ok else "FAILED"}: {r["total_s"] / 60:.1f} minutes from clone to a working demo** (goal: under 10).

The script ran the README's quickstart commands in a fresh clone, building the image
{"from scratch (`--no-cache`)" if r["cold_build"] else "with the local build cache"}:

```bash
{cmds}
```

| Step | Seconds |
|---|---:|
| clone | {t.get("clone_s", 0):.0f} |
| build the demo image (UI, Python deps, demo model) | {t.get("cold_build_s", 0):.0f} |
| start until healthy | {t.get("start_until_healthy_s", 0):.0f} |
| **total** | **{r["total_s"]:.0f}** |

| Check | Result |
|---|---|
{checks}

Base images were already in the local Docker cache. On a fresh machine add their download:
{bases} (uncompressed sizes; downloads are compressed and smaller).
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, default=8098)
    p.add_argument("--no-cold", dest="cold", action="store_false", help="reuse the build cache")
    p.add_argument("--out", type=Path, default=Path("reports/demo"))
    args = p.parse_args()
    r = run(args)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "quickstart.json").write_text(json.dumps(r, indent=1) + "\n")
    (args.out / "quickstart.md").write_text(to_markdown(r))
    print(to_markdown(r))
    return 0 if r["under_10_min"] and all(r["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
