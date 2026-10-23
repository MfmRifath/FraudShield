# ADR-0003: Configuration with pydantic-settings and YAML

- Status: Accepted
- Date: 2026-10-23
- Phase: 0

## Context

The spec asks for "config via pydantic-settings + YAML in configs/". It leaves open how files are layered, how env vars override, and how typos are handled.

## Decision

- One typed root `fraudshield.config.Settings` (pydantic-settings). Each phase adds a nested section (`features`, `training`, `serving`, ...) as a frozen pydantic model.
- **Precedence:** init kwargs > env vars (`FRAUDSHIELD_` prefix, `__` nesting) > YAML file > code defaults.
- **One YAML file per profile**, selected with `FRAUDSHIELD_CONFIG` (default `configs/base.yaml`). No implicit multi-file deep merge: profile files are complete, which keeps them explicit and diffable.
- **Domain configs stay separate** (`configs/costs.yaml`, `configs/rules.yaml`, ...). They are loaded by their own typed models because they change independently (the policy simulator, rule edits).
- `extra="forbid"` everywhere, so an unknown key is an error rather than being silently ignored.
- `.env` is **not** auto-loaded by the app. Docker Compose reads it; locally use `direnv` or export the variables. This keeps tests hermetic.
- Settings are frozen; `get_settings()` caches one instance per process.

## Alternatives considered

- Hydra/OmegaConf: powerful composition, but a second config system with weaker static typing.
- Deep-merged base + overlay files: convenient, but it hides the effective config. We can revisit if profiles drift.

## Consequences

- A profile change means editing a complete file. `tests/unit/test_config.py::test_profile_validates` checks every profile listed in `PROFILES`; each new profile must be added there.
