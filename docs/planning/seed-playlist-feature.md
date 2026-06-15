# Seed-based length-targeted playlist generation — feature plan

*Tracking issue: #219*

## Context

Playchitect can already partition a whole library into K clusters and trim clusters to a
target duration, but it has no way to do the thing users most often want: **"give me ~90
minutes of music like *this* track."**

- `core/clustering.py` partitions the *whole* library into K clusters; `--target-duration`
  only chooses *how many* clusters, not the length of any single playlist.
- `core/playlist_builder.py::build_duration_constrained_playlists` trims clusters to a length
  by centroid distance, but it is wired into the GUI Playlists view only
  (`gui/views/playlists_view.py:782`) and works per auto-discovered cluster.
- There is **no "given a seed track, gather the most similar tracks and fill to a target
  length" operation** anywhere.

This feature adds that primitive in `core`, exposed through both the **CLI** and the **GUI**,
reusing the existing feature/scaling/weighting/duration/sequencing machinery and returning a
`ClusterResult` so export and GUI rendering work unchanged.

**Confirmed decisions:** seed = a single track; the seed track is **included** in the result;
both CLI and GUI surfaces.

## How to read this breakdown

Issues are grouped into **epics** and ordered by dependency. Each issue is sized for a junior
developer and states, explicitly: **Deliverable**, **Reuse** (existing code to call instead of
reinventing), **TDD tests** (written first, red → green), **Smoke / human testing**, and the
**CLI / GUI / Docs** updates it needs (or "N/A").

### Definition of Done (every issue)
- Tests written first, all green: `uv run pytest <target> -o addopts="" -q`.
- `uv run pre-commit run --all-files` clean (ruff, ty, unit, smoke hooks).
- New/changed code >85% covered; complete type hints; no magic numbers (use named constants).
- Docstrings updated for any touched public function.
- Branch off `main` (`feature/219-<slug>`), PR opened, then `./scripts/review_pr.sh` for review.

---

## EPIC A — Core seed-similarity engine

> Goal: a pure, surface-agnostic engine turning *(seed track, target length)* into an ordered,
> length-correct `ClusterResult`. Build bottom-up so each piece is tested before the public
> function ties them together.

### Shared 8-D feature-vector helper
- **Deliverable:** one helper, `build_feature_vector(metadata, features) -> np.ndarray | None`,
  returning the same **8-D** vector clustering uses (BPM + the 7 intensity features, in
  `weighting.FEATURE_NAMES` order). Place it where both `clustering.py` and the new module can
  import it (e.g. a small `core/features.py`, or extend `weighting.py`). Refactor
  `playlist_builder._features_to_vector` (currently 7-D, no BPM) and the clustering vector
  assembly to call it — single source of truth.
- **Reuse:** `weighting.FEATURE_NAMES`; `IntensityFeatures` fields (`rms_energy, brightness,
  sub_bass_energy, kick_energy, bass_harmonics, percussiveness, onset_strength`);
  `TrackMetadata.bpm`.
- **TDD tests** (`tests/unit/test_features.py`): vector length 8; ordering matches
  `FEATURE_NAMES`; BPM in the right slot; returns `None` when BPM/features missing; values
  copied, not aliased.
- **CLI:** N/A. **GUI:** N/A. **Docs:** docstring.
- **Human testing:** none beyond review; existing clustering tests must stay green (proves the
  refactor changed nothing).

### Seed similarity ranking (scale + weight + distance)
- **Deliverable:** internal `rank_by_similarity(seed_vec, candidate_vecs, *, genre,
  weight_overrides) -> list[tuple[Path, float]]` (closest first). Fit `StandardScaler` on the
  candidate matrix, transform seed + candidates, apply the weight vector, rank by weighted
  Euclidean distance to the seed.
- **Reuse:** `StandardScaler` pattern from `clustering.py`; `weighting.select_weights(...)` +
  `weight_config.apply_weight_overrides(...)`; the ranking shape of
  `playlist_builder._rank_tracks_by_distance`.
- **TDD tests** (`tests/unit/test_seed_playlist.py::TestRanking`): identical-to-seed track
  ranks ≈ 0 and first; deterministic ordering on synthetic features; changing weights changes
  ordering; single-candidate and empty-candidate edge cases.
- **CLI / GUI:** N/A. **Docs:** docstrings. **Human testing:** none (deterministic).

### Duration-constrained assembly with seed included
- **Deliverable:** internal `fill_to_duration(ranked, metadata_dict, target_secs, tolerance,
  seed_path) -> list[Path]`. Greedily add nearest tracks until cumulative duration first
  reaches `target_secs` within `±tolerance`; the seed is **always included**; skip
  zero-/None-duration tracks.
- **Reuse:** `playlist_builder._get_track_duration` and the cumulative-fill loop in
  `build_duration_constrained_playlists`.
- **TDD tests** (`...::TestFill`): total duration within `[target*(1-tol), target*(1+tol)]`
  when the library is large enough; seed always present; target longer than the whole library
  returns everything without erroring; zero-duration tracks skipped; tolerance boundary
  respected.
- **CLI / GUI:** N/A. **Docs:** docstrings. **Human testing:** none (deterministic).

### Public `generate_playlist_from_seed` + sequencing + `ClusterResult`
- **Deliverable:** `core/seed_playlist.py` exposing
  `generate_playlist_from_seed(seed_path, candidate_features, metadata_dict,
  target_duration_mins, *, genre=None, weight_overrides=None, tolerance=0.1,
  sequence_mode="ramp") -> ClusterResult`. Compose the three helpers, order the chosen tracks
  with the energy sequencer, and populate a `ClusterResult` (tracks, `total_duration`,
  `centroid` = seed's scaled vector, auto name like `Like: <seed title>`).
- **Reuse:** `sequencer` / `arc_sequencer` (modes ramp/peak/valley/wave/flat, matching the
  Playlists view); `ClusterResult` dataclass.
- **TDD tests** (`...::TestGenerate`): returns a `ClusterResult` with seed included and
  duration within tolerance; `sequence_mode` reorders (ramp vs flat differ); invalid inputs
  raise `ValueError` (seed absent from candidates, empty candidates, `target<=0`); the result
  is consumable by `core/export.py` (write an M3U in the test).
- **CLI / GUI:** N/A. **Docs:** module + function docstrings with an example.
- **Human testing:** quick REPL run on a tiny real folder (≤20 tracks); record the command in
  the PR.

---

## EPIC B — CLI surface

> Goal: `playchitect playlist --seed <track> --duration <mins>` end to end.

### `playlist` command skeleton, options & validation
- **Deliverable:** new Click command in `cli/commands.py`: `--seed PATH` (required),
  `--music-dir PATH` (required), `--duration FLOAT` minutes (required, >0), `--output PATH`
  (optional), `--sequence [ramp|peak|valley|wave|flat]` (default `ramp`), `--genre TEXT`
  (optional). Validate seed exists, music-dir exists, duration > 0; clear errors with non-zero
  exit.
- **Reuse:** option/validation style of the existing `scan` command (`commands.py:34`).
- **TDD tests** (`tests/integration/test_cli_playlist.py`, `CliRunner`): `--help` lists the
  command; missing/invalid `--seed`, missing `--music-dir`, `--duration 0` each fail helpfully
  with non-zero exit.
- **CLI:** the work itself. **GUI:** N/A. **Docs:** stub into `docs/guide/cli-reference.md`.
- **Human testing:** `uv run playchitect playlist --help` reads correctly.

### CLI library loading, generation & export wiring
- **Deliverable:** command body — discover the library, gather metadata + intensity features
  (using the cache), call `generate_playlist_from_seed`, export the `ClusterResult`, print a
  short summary (track count, total minutes, output path). Default output name if `--output`
  omitted; support `.cue` when the output extension is `.cue`.
- **Reuse:** `audio_scanner`, `metadata_extractor`, `intensity_analyzer` (+ JSON cache) — same
  loading path `scan` uses; `core/export.py` for M3U/CUE.
- **TDD tests** (`tests/integration/test_cli_playlist.py`): with a small fixture library + a
  seed, exit 0, M3U written that contains the seed and totals ≈ target within tolerance;
  unreadable/0-track library handled gracefully.
- **CLI:** complete behaviour. **GUI:** N/A.
- **Docs:** full `playlist` section in `docs/guide/cli-reference.md` per `UPDATING_DOCS.md`
  (synopsis, every flag, worked example); update the command docstring.
- **Human testing:** `uv run playchitect playlist --seed "<real track>" --music-dir ~/Music
  --duration 90 --output /tmp/like-this.m3u`; open the M3U and confirm ~90 min of audibly
  similar tracks including the seed. Record in PR.

---

## EPIC C — GUI surface

> Goal: from the Library view, pick a track → "Make playlist like this…" → choose a length →
> see the result. Mirror existing patterns; reuse the Playlists view to display results.

### Library view "Make playlist like this…" action + dialog
- **Deliverable:** in `gui/views/library_view.py`, add a header button **and** right-click
  context entry "Make playlist like this…", enabled only when exactly one track is selected.
  Clicking opens a small modal with target length (30/60/90/120/Custom minutes) and sequence
  mode (ramp/peak/valley/wave/flat); it returns the chosen values via a callback. No generation
  logic here (stub the handler).
- **Reuse:** existing libadwaita dialog/widget patterns in `gui/` (preferences window,
  Playlists header controls).
- **GUI smoke tests** (`tests/gui/test_library_seed_action.py`, mock harness): action exists;
  disabled with no selection, enabled with one; dialog constructs and exposes the selected
  length + sequence values.
- **CLI:** N/A. **GUI:** the work itself. **Docs:** note the new control (full page in Epic D).
- **Human testing:** launch `uv run playchitect-gui`, select a track, open the dialog, confirm
  controls behave (no generation yet).

### GUI background generation & result display
- **Deliverable:** wire the dialog callback to run `generate_playlist_from_seed` on a
  background thread, then hand the resulting `ClusterResult` to the Playlists view and switch to
  it. Spinner/disabled state while running; error toast on failure. Features/metadata come from
  the already-loaded library state.
- **Reuse:** the background-thread + main-thread-callback pattern in
  `gui/views/playlists_view.py` (~line 782); the Playlists view's `ClusterResult` rendering.
- **GUI smoke tests:** completion callback populates the Playlists view with the result;
  failure path shows an error and re-enables the control (worker run synchronously / mocked).
- **CLI:** N/A. **GUI:** the work itself. **Docs:** covered in Epic D.
- **Human testing:** in the running GUI, select a track → "Make playlist like this…" → 90 min →
  confirm a coherent set appears in Playlists and can be exported. Record in PR.

---

## EPIC D — Documentation & status reconciliation

> Goal: make the project's own docs trustworthy and document the new feature.

### Reconcile project status docs
- **Deliverable:** fix `STATUS.md` (description = Smart DJ Playlist Manager, not "game design
  tool"; status `active`; realistic progress; accurate next actions). Update
  `docs/planning/ROADMAP.md` so the Milestone 7 GUI views are marked built/merged, and add this
  seed-playlist feature.
- **TDD / CLI / GUI:** N/A. **Human testing:** James confirms both match reality.

### User & developer documentation for the feature
- **Deliverable:** a user-guide page/section covering both surfaces (CLI `playlist` command and
  the GUI "Make playlist like this…" flow), added to VitePress per `UPDATING_DOCS.md` and the
  `docs/.vitepress/config.ts` sidebar. Finalize `cli-reference.md`. Confirm touched docstrings.
- **TDD:** N/A. **CLI / GUI:** documentation of both.
- **Human testing:** the docs site builds and the new page renders in the nav.

---

## Suggested delivery order & PR slicing
1. **Epic A** as one PR (the four issues are tightly coupled — ship the engine together, commit
   per issue for reviewable history). Land green before anything depends on it.
2. **Epic B** as one PR (depends on A).
3. **Epic C** as one PR — split into its two issues if review gets large (depends on A).
4. **Epic D** as one PR (can start in parallel; the status-doc fix needs no code).

Each PR: TDD first, smoke tests, the human-test command/result pasted into the PR body, then
`./scripts/review_pr.sh` before squash-merge to `main`.
