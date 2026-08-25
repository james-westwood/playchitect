# Playchitect
**Status:** active
**Priority:** medium
**Progress:** ~45%
**Last updated:** 2026-08-25
**Repo:** `~/Programming/personal/playchitect`

## What it does
Smart DJ playlist manager. Clusters a music library on BPM plus audio intensity features so
playlists cohere by character, not just tempo. CLI and GTK4 GUI; M3U and CUE export.

## Why it matters
Replaces rigid BPM-bucket playlist scripts for real DJ sets, and is the test case for the
ralph dev loop.

## Next actions
- [ ] Human gate: judge Phase 1 output quality — see `docs/planning/orchestrator-handoff-phase1-evidence.md`
- [ ] TASK-28: fix silent track loss in cluster dedup (405 in, 398 out)
- [ ] TASK-19: embedding cache ETL, start of the Phase 2 personal-metric mainline

## Notes
<!-- Milestones 1-5 complete. Now executing docs/planning/ml-playlist-generator-plan.md:
     Phase 1 (default-path fixes, TASK-15..18) complete and awaiting the human gate.
     Phase 2 is the personal-metric loop: embedding cache, eval harness, labelling tool,
     transition model. Seed-playlist work TASK-01..14 is on hold pending Phase 4 rescoping. -->
