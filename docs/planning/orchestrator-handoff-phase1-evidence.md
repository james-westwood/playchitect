# Orchestrator handoff — Phase 1 evidence run and gate

**Date:** 2026-08-25
**Supplements:** `orchestrator-handoff.md` (the standing rules — read that first)
**Covers:** the Phase 1 "Done when" evidence run against the real library, and the two defects it exposed

This document records the evidence behind the Phase 1 human gate so a fresh session does not
have to re-run a two-hour scan to know what was found.

---

## Standing rules (unchanged)

All of `orchestrator-handoff.md` still applies. The short version:

- **Task source is `prd.json`, local only.** No GitHub issues, no PRs. Pick the lowest
  priority number among incomplete, non-blocked tasks.
- **Quality checks before every commit:** `uv run pytest tests/ -o addopts='' -q` and
  `uv run pre-commit run --all-files`.
- **Conventional commits, no AI attribution.**
- **Phase boundaries are human gates.** James reviews at every phase transition, after
  TASK-27's guardrail result, at TASK-H1, and at any ambiguity.
- **Eval data is never training data.** James's own sets and held-out personal labels are
  eval-only; label splits are by session, never by judgement.
- **TASK-H1 gates the enhancement track only**, never the mainline.

One deviation from the original handoff: **pushing is now allowed.** `main` was reconciled with
`origin/main` on 2026-08-25 (see "The fork and the merge" below) and pushed. The no-push rule was
a "don't create GitHub ceremony" rule, not a "never back up" rule — treat pushing `main` as
allowed and PRs/issues as still off.

## Where the project stands

- **Phase 1 complete on paper:** TASK-15, 16, 17, 18 all merged to `main`.
- **Suite green:** 1379 passed, 2 skipped (re-verified 2026-08-25).
- **Phase 2 not started.** TASK-19 (embedding cache ETL, `PCA(n_components=64, whiten=True)`)
  is the next mainline task and is **held at the human gate**.
- `docs/planning/stitch-screens/` is intentionally untracked.

## The evidence run

The plan's Phase 1 exit criterion is: *"default `scan` on `dark 4` produces distinguishable,
character-named playlists with per-cluster stats; tests green."*

The `dark 4` folder no longer exists, so the run used
`/mnt/1tb_ssd/Media/Music/Bandcamp-lycragimp` instead — **405 tracks**, a larger and more
genre-homogeneous library (hypnotic / industrial techno) than the plan's 190-track baseline.
This makes it a harder separation problem than the one the plan was written against.

Command: default `scan --dry-run`, no `--use-embeddings`, no `--genre`, no `--fast`.
Log: `/tmp/opencode/phase1-evidence-scan2.log` (283 lines). Started 11:51, finished 12:13 —
22 minutes rather than the estimated 2–2.5h, because the intensity cache
(`~/.cache/playchitect/playchitect.db`, content-hash keyed) was already warm.

### Results

```
Found 405 audio files
Extracted BPM from 405/405 tracks
405 tracks analysed, 0 failed
Clustering 405 tracks on 8 features
PCA weights (ci=0.230, n=405): onset_strength=0.166, rms_energy=0.151, bass_harmonics=0.133
Using K=2 clusters (weight source: pca)
EWKM per-cluster weights applied
Cluster 1: 52 tracks, BPM: 128.6 ± 29.5, top feature: onset_strength (0.29)
Cluster 0: 346 tracks, BPM: 128.8 ± 19.8, top feature: kick_energy (0.20)
WARNING - Clusters did not separate: largest cluster holds 86.9% of 398 tracks.
          Consider --use-embeddings or --genre hints for better separation.
  (split 2 clusters → 17 playlists to meet target size)
Weight source: pca | Top features: onset_strength=0.29, brightness=0.24, rms_energy=0.21
```

17 playlists, 398 tracks total, BPM means spanning 68.1 → 149.6.

## Gate check against the plan's Phase 1 criteria

| Criterion | Task | Result |
|---|---|---|
| Character names present | TASK-17 | **Pass.** All 17 playlists named — *Powerful Luminous Wave*, *Punchy Luminous Wave*, *Percussive Wave*, *Vocal Wave*, *Ethereal Driving / Rapid / Steady Wave*, *Quick Wave*. Names propagate to M3U filenames. |
| Per-cluster stats differ, not copied from parent | TASK-15 | **Pass.** Parent cluster 0 was 346 tracks @ 128.8 ± 19.8 BPM; its sub-clusters report their own, much tighter stats (136.7 ± 8.9, 104.2 ± 6.2, 73.3 ± 6.7, 68.1 ± 4.7). Durations are per-cluster too, 18.8 → 348.2 min. This is genuine recursive re-clustering, not the old shuffle-and-dice. |
| "Weight source" is not BPM-only | TASK-16 | **Pass.** `Weight source: pca`, clustering on all 8 features with EWKM per-cluster weights. The old `uniform` / BPM-only default is gone. |
| Degenerate-K warning fires | TASK-18 | **Pass.** Fired on real data at 86.9% dominance, surfaced in both the log and stdout. |
| Tests green | — | **Pass.** 1379 passed, 2 skipped. |
| Playlists are *distinguishable* | — | **Passed** by James, 2026-08-25, with the sameness accepted as designed stopgap behaviour. See below. |

### The judgement call

Mechanically, all four Phase 1 tasks do what they were specified to do. Whether the *output*
clears the plan's bar is a DJ's call, not a test's. The honest read:

**Against.** The degenerate warning fired because it should have — top-level K collapsed to 2
with 86.9% in one cluster, and 16 of the 17 playlists are splits of that single blob. Eight of
the 17 playlists (**274 tracks, 69% of the library**) have BPM means inside 133–139. Only six
distinct character names cover all 17 playlists, so **12 of 17 needed a BPM-range suffix to be
told apart**: *Ethereal Rapid Wave [136-145]* vs *[136-146]* vs *[136-143]* vs *[136-136]* is
not a distinction anyone can use while mixing. On this library the default path is still
substantially BPM buckets — better labelled, and no longer randomised, but still buckets.

**For.** This is a single-genre techno folder, the hardest possible case for 8-feature
separation, and precisely the case the plan says embeddings (TASK-19) exist to solve. The real
wins are visible: the low-BPM material separated cleanly (*Quick Wave [68-72]* 14 tracks,
*[73-80]* 6 tracks, *Vocal Wave* @ 104 BPM, 23 tracks), and the system now *tells you* it
failed to separate instead of silently handing over diced noise. Phase 1 was scoped as a cheap
stopgap — "clustering is scaffolding, do not perfect it".

**Orchestrator's read:** Phase 1 did what it was scoped to do. The residual indistinguishability
is exactly the problem Phase 2 is designed to solve, so this reads as a pass with a known
limitation rather than a failure — but the gate is James's.

### Gate outcome — PASSED, 2026-08-25

James passed the gate. Two orchestrators (Claude Opus and Kimi K3) independently reached the
same recommendation and he took it. **Phase 1 does not close until TASK-28 lands** — silent
track loss is an integrity bug rather than polish, and it sits inside the Phase 1 epic at
priority 1. TASK-29 stays deferred at priority 6.

The reasoning on the sameness, for the record: it is the designed stopgap behaviour on a
genuinely homogeneous single-genre folder, and the tool now says so out loud rather than
silently shipping diced noise. The remedy is Phase 2's embedding space, not more K-selection
tuning. Sending Phase 1 back for it would contradict the plan's own "clustering is scaffolding,
do not perfect it".

## Defects the evidence run exposed

Both are now specified as prd.json tasks. Neither was in Phase 1's scope; both were found by
running against a real library rather than a fixture.

### TASK-28 — silent track loss in cluster dedup (priority 1)

405 tracks went into clustering and **398 came out**. Seven tracks vanished with no log line
and no user-visible message; the cluster stat lines and the 17 playlists both sum to 398.

`PlaylistClusterer._deduplicate_clusters` (`playchitect/core/clustering.py:824`, called from
`cluster_by_features` at line 446) recomputes each track's home cluster as the nearest **raw
K-means** centroid. But by the time it runs, the labels have been rewritten by **EWKM
refinement** (`ewkm_refine`, ~line 422), which reassigns tracks using per-cluster feature
weights. A track EWKM moved therefore sits in cluster A while its nearest raw K-means centroid
is cluster B — and the rebuild loop is remove-only:

```python
kept_tracks = [t for t in r.tracks if track_to_cluster.get(t) == r.cluster_id]
```

so it is dropped from A and never added to B. It leaves the output entirely.

Same function, second defect: the rebuilt `ClusterResult` recomputes `track_count` from
`kept_tracks` but copies `bpm_mean`, `bpm_std`, `total_duration` and `feature_means` verbatim
from the pre-dedup result — so any cluster that lost members reports **stale statistics**.

Prioritised at **1** (ahead of TASK-19) on the grounds that it is silent data loss on the
mainline default path, it is cheap and TDD-shaped, and it undermines confidence in any future
evidence run. Flip it to a later priority if you would rather Phase 2 start first.

### TASK-29 — same track as both opener and closer (priority 6)

Two clusters in this run — holding 7 and 14 tracks — reported an identical Opener and Closer.
`TrackSelector.select` (`playchitect/core/track_selector.py:86`) ranks openers and closers
independently over the same track list with no mutual exclusion (sorts at ~line 151), so a
track topping both rankings is returned at the head of both lists and
`selected_first` / `selected_last` return the same path.

Cosmetic rather than corrupting, so prioritised at **6** — backlog, do it when convenient.

## The fork and the merge (2026-08-25)

The repo had been forked in two for two months and nobody had noticed. Recorded here because it
explains why the plan says what it says.

`main` and `origin/main` diverged at `0268829` (**2026-06-15**) and never rejoined:

- **`origin/main`** carried the **seed-playlist feature, built and merged 2026-06-15/16** via
  PRs #220-224 (issue #219): `core/features.py`, `core/seed_playlist.py`, the CLI `playlist`
  command, the LibraryView make-playlist button, ~1,070 lines of tests.
- **local `main`** carried the ML replan, first commit **2026-08-20** — branched off that same
  June fork point.

So `ml-playlist-generator-plan.md` was authored against a **two-month-stale checkout**. It
recorded TASK-01..14 as unstarted, and its founding premise ("~18k LOC, 1,334 tests, and is not
usable") was formed without the seed-playlist feature in view. **James deferred revisiting that
premise to Phase 4.** The practical consequence is that Phase 4's "re-scope TASK-01..14 to the
transition model" has a head start: adapt `seed_playlist.py`, do not rebuild it.

The merge itself was small. `playchitect/cli/commands.py` auto-merged cleanly — `scan` with
`--fast` and the `playlist` command occupy disjoint regions. Only `prd.json` and `STATUS.md`
conflicted. Task states were set from what actually landed, not from what either file claimed:
TASK-01..10 and TASK-13 completed; **TASK-11/12 still blocked — implemented but unmerged on
`origin/feature/219-task11-12-gui-wiring`** (`main_window.py` +72, `test_main_window_seed.py`
+750), to be merged during the Phase 4 GUI pass; TASK-14 blocked, never started.

Merged tree: **1419 passed, 2 skipped**, pre-commit all green.

### The ralph loop is dormant — no guard needed

It was flagged that a live ralph loop might autonomously pick up the human-gated TASK-19 once
the merge marked earlier tasks complete. Checked, and it will not:

- No cron entry, no systemd user unit, no timer for this repo, no running process.
- RalphZilla's `ralph.log` was last written **2026-06-06**; its newest summary is 2026-04-29.
- `ralphzilla/prd.json` has `"project": "ralphzilla"` — it drives itself, not playchitect. The
  stray `ralphzilla/playchitect/` directory holds only `__pycache__`.

If a loop is ever pointed at this repo again, note that it must skip tasks carrying
`"status": "blocked"` — that is the only thing keeping the parked TASK-11/12/14 out of the
pickup order.

## What a fresh session should do next

1. **TASK-28** is the next pickup (priority 1) and closes Phase 1. Take it through
   Triage → Tests → Implement → Docs, delegating with complete self-contained prompts.
2. Re-run the evidence scan after TASK-28 to confirm **405 in / 405 out**. It is cheap now the
   intensity cache is warm — about 20 minutes for 405 tracks.
3. Then **TASK-19**, and the Phase 2 mainline in priority order: 19 → 25 → 26 → 27. Phase 2
   ends at a human gate on TASK-27's guardrail result.
4. Phase 1's gate is passed; **the next human gate is the Phase 1 → Phase 2 transition once
   TASK-28 is green.** Do not run past it.

## Gotchas

- **For long scans, invoke `.venv/bin/playchitect` directly.** A `uv run`-wrapped scan hung
  once — futex wait, no child process.
- **Never run two scans against the same cache DB concurrently.**
- **On a fresh checkout run `uv sync --extra dev`**, or `pytest` resolves to conda's install.
- The intensity cache is keyed by content hash, so renames and moves are free but re-encodes
  are not.
