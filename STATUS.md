# Playchitect
**Status:** active
**Priority:** high
**Progress:** ~60%
**Last updated:** 2026-06-16
**Repo:** `~/Programming/personal/playchitect`

## What it does
Smart DJ Playlist Manager with intelligent multi-dimensional K-means clustering
(BPM + 7 intensity features). Generates playlists that feel coherent — not just
mathematically similar. Supports seed-based playlist generation: given a single
track and target duration, finds the most sonically similar library tracks.

## Why it matters
Personal DJ tooling for creating sets that sound right, not just look right on paper.

## Completed
- Core audio analysis (scanner, metadata, intensity features)
- K-means clustering with genre-aware weighting
- CLI: `scan`, `info`, `cluster`, `export`
- GTK4 GUI (Library, Playlists, Set Builder, Export views)
- Energy arc sequencing, harmonic mixing, Rekordbox import, Mixxx sync
- Seed-based playlist engine (`build_feature_vector`, `seed_playlist`, CLI `playlist` command)
- 938+ tests, CI (ruff, ty, pytest, Fedora 41)

## In progress
- [ ] GUI: "Make playlist like this…" button + dialog in LibraryView (TASK-09/10)
- [ ] GUI: MainWindow seed generation wiring (TASK-11/12)
- [ ] Docs: fix STATUS.md + ROADMAP.md (TASK-13)
- [ ] Docs: CLI reference for `playlist` command (TASK-14)
- [ ] Milestone 6: Packaging (Flatpak + PyPI)

## Notes
Milestones 1–5 complete. PRs #221, #222, #223 merged (seed-playlist feature).
Tracking issue: #219.
