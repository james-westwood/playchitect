# Orchestrator handoff prompt — playchitect ML playlist generator

Paste into a fresh opencode session with the **orchestrator** agent active (Tab to select). Working directory: `~/Programming/personal/playchitect`.

---

You are coordinating work on **playchitect**, not hcm_model. Your agent body's repo-specific details are overridden as follows:

- **Task source: `prd.json`, local only.** Do NOT create GitHub issues, do NOT open PRs, do NOT push. The prd.json is the single source of truth — pick the lowest-priority incomplete task without `"status": "blocked"`. This is a solo local project; GitHub ceremony is overhead with no reviewer value.
- **Local checkout:** `~/Programming/personal/playchitect`. Base branch: `main`. Work on local-only branches `feature/<task-id>-<slug>` and merge locally, or commit straight to `main` — James's call if you are unsure, default to local branches merged with `--no-ff`.
- **Tooling:** `uv` (NOT poetry), Python 3.13, `pytest`, `ruff`, `ty`. Quality checks from prd.json run before every commit: `uv run pytest tests/ -o addopts='' -q` and `uv run pre-commit run --all-files`. Ignore `tests/benchmarks/` unless a task targets it.
- **No MkDocs here** — docs are plain markdown under `docs/`. The docs gate per task becomes: docstrings and type hints on new public functions, plus any plan/readme doc the task touches. pytest + ruff + ty green.
- **Commits:** no AI attribution — no Co-Authored-By trailers, no "Generated with" lines, no tool mentions. Conventional commits (`feat:`, `fix:`, `test:`, `docs:`).
- **Marking progress:** when a task is done (code + tests green + committed), set `"completed": true` in prd.json in the same commit or a chore commit immediately after.

## Mission

Execute `docs/planning/ml-playlist-generator-plan.md` via the prd.json tasks. Read the plan doc first — it contains verified findings about the current broken default path, the seed artist list, and the data assets. Do not re-derive them; trust the document.

## How to run it

1. Drive each task through your **Tests → Implement → Docs** workflow, delegating to test-writer, coder, docs-writer with complete self-contained prompts (they do not see this conversation). One task at a time, in priority order.
2. Phase boundaries are human gates: when all Phase 1 tasks (TASK-15..18) are complete, report a phase summary and wait for James before starting Phase 2.
3. **HARD GATE:** Phase 2 ends at TASK-24 producing `docs/planning/match-rate-report.md`, followed by TASK-H1 which is **owner: human**. Present the match-rate numbers (surviving pairs per source, resolver precision, the >= 2000-pair threshold read) and STOP. Do not invent Phase 3 tasks until James completes TASK-H1 with a direction.
4. **Eval discipline (non-negotiable):** nothing ships unless it beats the BPM-window-only baseline with bootstrap CIs; James's own sets and held-out personal labels are eval-only, never training data. If a delegated agent proposes training on eval data, reject the work.
5. **Scope discipline:** no GUI, packaging, or Flatpak work. The blocked TASK-01..14 stay blocked until Phase 6 rescoping. If a task grows beyond its description, split it — one task, one commit series.
6. If a task is ambiguous or its acceptance criteria conflict with the codebase, STOP and ask James — do not guess.

## Human gates

James reviews at: every phase transition, the TASK-H1 hard gate, and any ambiguity. When blocked, report status in your standard format and wait.

## First action

Read `docs/planning/ml-playlist-generator-plan.md`, then take **TASK-15** (split_cluster_recluster) through Triage: read the current `split_cluster` in `playchitect/core/clustering.py`, confirm the task description matches reality, and delegate Phase Tests to test-writer. Report your status.
