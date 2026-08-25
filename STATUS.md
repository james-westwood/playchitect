# Playchitect
**Status:** active
**Priority:** medium
**Progress:** ~55%
**Last updated:** 2026-08-25
**Repo:** `~/Programming/personal/playchitect`

## What it does
Smart DJ playlist manager. Clusters a music library on BPM plus 7 audio intensity features so
playlists cohere by character, not just tempo. Also generates seed-based playlists: give it one
track and a target duration and it gathers the most sonically similar library tracks. CLI and
GTK4 GUI; M3U and CUE export.

## Why it matters
Personal DJ tooling for building sets that sound right, not just look right on paper. Replaces
the rigid BPM-bucket playlist scripts.

## Completed
- Core audio analysis (scanner, metadata, intensity features)
- K-means clustering with genre-aware weighting, EWKM, PCA weights
- CLI: `scan`, `info`, `cluster`, `export`, `playlist`
- GTK4 GUI (Library, Playlists, Set Builder, Export views) + LibraryView make-playlist button
- Energy arc sequencing, harmonic mixing, Rekordbox import, Mixxx sync
- Seed-based playlist engine (`features.py`, `seed_playlist.py`, CLI `playlist`) — PRs #220-224
- ML plan Phase 1: recursive re-clustering, multi-dimensional scan default, character naming,
  degenerate-K warning (TASK-15..18)
- 1,379 tests, CI (ruff, ty, pytest, Fedora 41)

## Next actions
- [ ] TASK-28: fix silent track loss in cluster dedup (405 tracks in, 398 out) — closes Phase 1
- [ ] TASK-19: embedding cache ETL, start of the Phase 2 personal-metric mainline
- [ ] TASK-25/26/27: eval harness, choice-labelling tool, transition model

## Parked
- TASK-11/12: MainWindow seed generation wiring — implemented on the unmerged branch
  `origin/feature/219-task11-12-gui-wiring`; merge during the Phase 4 GUI pass
- TASK-14: CLI reference docs for the `playlist` command — never started
- Milestone 6: Packaging (Flatpak + PyPI)

## Notes
Two lines of work were reconciled on 2026-08-25. The seed-playlist feature shipped 2026-06-16
(issue #219). The ML replan (`docs/planning/ml-playlist-generator-plan.md`) was authored
2026-08-20 from a checkout predating those merges, so it recorded that work as unstarted;
revisiting the plan's premise is deferred to Phase 4. Phase 1 of the ML plan passed its human
gate on 2026-08-25 — evidence and gate reasoning in
`docs/planning/orchestrator-handoff-phase1-evidence.md`.
