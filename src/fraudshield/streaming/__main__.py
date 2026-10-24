"""``python -m fraudshield.streaming <worker>``: run one stream worker (used by docker compose)."""

from __future__ import annotations

import logging
import sys

from fraudshield.config import get_settings

WORKERS = ("processor", "gateway", "label-emitter", "parquet-sink", "shadow")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or args[0] not in WORKERS:
        print(f"usage: python -m fraudshield.streaming {{{','.join(WORKERS)}}}", file=sys.stderr)
        return 2
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    name = args[0]
    if name == "processor":
        from fraudshield.streaming.processor import run
    elif name == "gateway":
        from fraudshield.streaming.gateway import run
    elif name == "label-emitter":
        from fraudshield.streaming.label_emitter import run
    elif name == "parquet-sink":
        from fraudshield.streaming.parquet_sink import run
    else:
        from fraudshield.streaming.shadow import run
    run(settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
