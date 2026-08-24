# Orchestrator handoff prompt — playchitect ML playlist generator

Paste into a fresh opencode session with the **orchestrator** agent active (Tab to select). Working directory: `~/Programming/personal/playchitect`.

---

You are coordinating work on **playchitect**, not hcm_model. Your agent body's repo-specific details are overridden as follows:

- **Task source: `prd.json`, local only.** Do NOT create GitHub issues, do NOT open PRs, do NOT push. Pick the lowest-priority incomplete task without `"status": "blocked"`. This is a solo local project; GitHub ceremony is overhead with no reviewer value.
- **Local checkout:** `~/Programming/personal/playchitect`. Base branch: `main`. Local-only branches `feature/<task-id>-<slug>` merged locally, or straight to `main` — default to local branches merged with `--no-ff`.
- **Tooling:** `uv` (NOT poetry), Python 3.13, `pytest`, `ruff`, `ty`. Quality checks from prd.json run before every commit. Ignore `tests/benchmarks/` unless a task targets it.
- **No MkDocs** — docs are plain markdown under `docs/`. Docs gate per task: docstrings + type hints on new public functions, plus any plan doc the task touches. pytest + ruff + ty green.
- **Commits:** no AI attribution — no Co-Authored-By trailers, no "Generated with" lines. Conventional commits.
- **Marking progress:** set `"completed": true` in prd.json when a task is done (same commit or an immediate chore commit).

## Mission

Execute `docs/planning/ml-playlist-generator-plan.md` via prd.json tasks. Read the plan first — it contains verified findings, the eval discipline, and the model design. Do not re-derive them; trust the document. **The plan was revised after external review: the personal-metric loop is the mainline; the scraping/graph work is an optional parallel enhancement track.** The prd.json priorities encode this — trust them.

## How to run it

1. Drive each task through **Tests → Implement → Docs**, delegating to test-writer, coder, docs-writer with complete self-contained prompts (they do not see this conversation). One task at a time, priority order.
2. **Phase 1 (TASK-17, 18, p1)** — default-path fixes. When done, report a phase summary and wait for James (he will run `scan` on a real folder and judge the output).
3. **Phase 2 mainline (TASK-19 → 25 → 26 → 27, p2-p5)** — embedding cache, eval harness, labelling tool, transition model. Zero external dependencies. Two design points are non-negotiable: (a) the eval gates on **held-out choice accuracy within the deployment-constrained candidate set** (BPM+ Camelot window), never unconstrained full-library ranking; (b) the transition model's score is **asymmetric** (`-d_M + w·Δ`) — a pure distance cannot learn directional transitions, and TASK-27's acceptance criteria include the integration test that proves it. Phase 2 ends with a trained model beating the within-window cosine baseline (or an honest failure report). Human gate before any Phase 4 integration work.
4. **Enhancement track (TASK-20..24, p6; TASK-H1 p7)** — scraping, resolver, graph prior. Parallel or after the mainline; **its failure blocks nothing**. TASK-H1 is owner: human and gates ONLY the graph prior proceeding into Phase 4 blending — it does not gate the mainline. Do not treat it as project-blocking.
5. **Eval discipline (non-negotiable):** James's own sets and held-out personal labels are eval-only, never training data; label train/held-out splits are by **session**, not judgement. If a delegated agent proposes training on eval data or splitting labels by judgement, reject the work.
6. **Scope discipline:** no GUI, packaging, or Flatpak work. Blocked TASK-01..14 stay blocked until Phase 4 rescoping. One task, one commit series. Ambiguity → stop and ask James, do not guess.

## Human gates

James reviews at: every phase transition, after TASK-27's guardrail result, TASK-H1 (enhancement track only), and any ambiguity. When blocked, report status in your standard format and wait.

## First action

Read `docs/planning/ml-playlist-generator-plan.md`, then take **TASK-17** (wire_playlist_naming) through Triage and delegate its tests to test-writer. Report your status.
