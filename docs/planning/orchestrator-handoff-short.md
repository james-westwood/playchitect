# Orchestrator handoff (short) — playchitect

Paste into a fresh opencode session with the **orchestrator** agent active. Working directory: `~/Programming/personal/playchitect`.

---

You are the orchestrator for **playchitect**. Read, in this order:

1. **Plan:** `docs/planning/ml-playlist-generator-plan.md` — purpose, approach, ML design
2. **Tasks:** `prd.json` — take the lowest-priority incomplete task without `"status": "blocked"`
3. **Operating rules:** `docs/planning/orchestrator-handoff.md` — repo overrides, gates, discipline

**Do now:** TASK-17, TASK-18 (Phase 1 remainder). Then STOP and report to James before starting Phase 2 (TASK-19+).

**Rules that matter most:** local only — no GitHub issues, PRs, or pushes; run `uv run pytest tests/ -o addopts='' -q` and `uv run pre-commit run --all-files` before every commit; no AI attribution in commits; eval data (golden sets, held-out labels) is never training data; TASK-H1 gates the enhancement track only, never the mainline.
