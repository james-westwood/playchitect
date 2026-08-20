# Plan: ML-Powered Playlist Generator

**Status:** Approved direction, pre-Phase-1
**Author:** James + Kimi K3 (opencode session 2026-08-20), synthesised with a Claude Opus review
**Supersedes:** the clustering-centric roadmap in ROADMAP.md; prd.json TASK-01..14 (seed-playlist) is ON HOLD pending Phase 2

---

## Why this plan exists

Playchitect has ~18k LOC, 1,334 passing tests, 76% coverage — and is not usable. Verified 2026-08-20 against the real library (`dark 4`, 190 tracks, `--dry-run`):

- Default `scan` does **BPM-only** clustering ("Weight source: uniform"). The entire intensity/embedding/mood/sequencing stack is opt-in via flags.
- K=2 on 190 tracks → one 184-track cluster → `split_cluster()` **randomly shuffles and dices** it into 7 indistinguishable 25-track playlists, copying parent stats. The headline feature degrades to BPM buckets plus a randomiser.
- Nobody ever evaluated playlist *quality*; the dev loop optimised for tasks closed.

Root cause: no measurable definition of "good". This plan installs one before any further modelling.

## Guiding principles

1. **Eval before modelling.** Nothing ships without beating the BPM-window baseline.
2. **Two data sources, kept separate.** Population adjacency (other DJs' sets) = training prior. Personal labels (James's A/B/C/D judgements) = personalisation + eval. Never train on eval data.
3. **Compose existing components.** intensity_analyzer, embedding_extractor, sequencer, compatibility (Camelot), cache_db, mixxx_sync all exist. This plan wires them into one pipeline; it does not greenfield them.
4. **Clustering is scaffolding.** The endgame is seed → learned metric → beam-search sequence → Mixxx crate. Fix the default path cheaply; do not perfect it.
5. **GUI and packaging last.** No GUI polish or Flatpak work until the generated playlists are worth playing.

## The library and data assets (verified)

- Library: `/mnt/1tb_ssd/Media/Music` — **1,726 tracks**
- Mixxx DB: `~/.mixxx/mixxxdb.sqlite` (2.4MB, only 4-5 sets — too small to train on)
- James's own sets: 4-5 on Mixcloud with CUE sheets (ordering known) — **golden eval set**, never train
- External: 1001tracklists.com, mixesdb.com — millions of sets by other artists

### Seed artist list (for scraping queries)

Union of library roster + r/Techno canon, three groups:

- **Axis 1 — UK industrial/Birmingham:** Surgeon, Regis, Blawan, Perc, Makaton, Black Merlin, Ancient Methods, Clouds, Ansome, Vatican Shadow, SNTS (Perc Trax, Downwards, Sandwell District)
- **Axis 2 — Hypnotic/Spanish + aligned:** Oscar Mulero, Reeko, Exium, Lewis Fautzi, Kwartz, Unbalance, Kessell, CRVEL, Kmyle, Keikari, Jurango, DVS1, Planetary Assault Systems, Function, Takaaki Itoh (PoleGroup, Semantica, Warm Up, Axis, HUSH)
- **r/Techno canon:** Robert Hood, Jeff Mills, Ben Klock, Marcel Dettmann, Shed, Rødhåd, Sleeparchive, Terrence Dixon, Luke Slater, Paula Temple, Blawan, Joey Beltram, Basic Channel / Moritz von Oswald
- **EXCLUDED — Axis 3 (UK bass/dub/dubstep, ~40% of library):** Peverelist, Kowton, Asusu, Simo Cell, Stenny, Forest Drive West, Toma Kami, Mala, Coki, Skream, Benga, Jack Sparrow, Kromestar, etc. Not techno per James 2026-08-20. Tag separately in the data model (`uk-bass/dub/dubstep`); exclude from techno adjacency mining. May get its own mining pass later.
- **EXCLUDED — business techno / mainstage:** Charlotte de Witte, Amelie Lens, Drumcode flagship names. They poison the adjacency graph with mainstage co-occurrences.

---

## Phase 1 — Fix the default path (stopgap usability)

Cheap, TDD-able, makes the tool useful tonight. Clustering stays scaffolding.

1. **Fix `split_cluster`** (core/clustering.py:840): replace random shuffle-and-dice with recursive re-clustering — re-run K-means inside an over-sized cluster with higher K; if features don't separate, split by energy-arc (RMS) ordering, never randomly. Sub-clusters must report their *own* stats, not parent stats.
2. **Multi-dimensional clustering becomes the default** in `scan`; BPM-only moves behind `--fast`. Intensity analysis runs by default (cached in cache_db).
3. **Wire the naming package** (core/naming/ — vibe_profiler, grammar_engine) into CLI playlist output so playlists get character names, not `Playlist N [130-133bpm]`.
4. **Fail loudly** when K selection collapses (e.g. K=2 on 190 tracks with dominant cluster): log a warning that features didn't separate and suggest `--use-embeddings`.

**Done when:** default `scan` on `dark 4` produces distinguishable, character-named playlists with per-cluster stats; tests green.

## Phase 2 — Embedding cache + match-rate spike (HARD GATE)

The deliverable is a **number** that gates the architecture. STOP and report to James before Phase 3.

1. **Embedding cache ETL:** essentia `discogs-effnet` embeddings for all 1,726 tracks; store in Parquet keyed by **file content hash** (not path — path-keyed caches rot on library reorg); fit PCA to 64 dims and persist the transform. Reuse `core/embedding_extractor.py` (swap musicnn → discogs-effnet) and `core/cache_db.py` patterns. Pull BPM/key from Mixxx DB where present rather than recomputing.
2. **Seed artist extractor:** derive the axis-1/axis-2 artist list from library metadata (filename + tags), merge with the r/Techno canon list above → `data/seed_artists.txt`.
3. **Scrapers:** 1001tracklists + MixesDB, seeded by the artist list. Rate-limit politely, cache every fetched page to disk. ToS-grey: personal research only.
4. **Fuzzy resolver:** match tracklist entries (artist/title strings) to library tracks: normalisation (case, punctuation, `(Original Mix)` stripping), label cat-number match, duration agreement. Report precision on a hand-checked sample.
5. **Own sets:** parse James's 4-5 Mixcloud tracklists + CUE sheets → golden eval pairs (held out, never trained on).

**Gate — report to James and STOP:**
- Tracks embedded / failed, PCA variance retained
- Sets scraped per source; tracklist entries parsed
- **Pairs surviving resolution** (both tracks in library), per source and total
- Resolver precision estimate
- Recommendation: is the surviving-pair count enough for a population prior (rule of thumb: ≥2k pairs), or does the design pivot (e.g. metadata co-occurrence graph as prior instead of learned metric pretraining)?

## Phase 3 — Eval harness (before any modelling)

1. `playchitect eval` command: given track A, rank library; report where the true next track lands. Metrics: Recall@10, Recall@50, MRR. **Split by set, not by pair** (pair-splitting leaks).
2. Bootstrap CIs on all metrics (small eval sets → wide error bars; the guardrail must not misfire on noise).
3. Baselines: random, BPM-window-only, raw embedding cosine. **Guardrail: any learned metric that cannot beat BPM-window-only is worth nothing.**
4. Eval data = golden sets + held-out personal labels. Log every eval run (git-style history) so metric-vs-label-count curves are plottable later.

## Phase 4 — Choice-labelling tool

Terminal script, no GUI:

1. Present: last 20s of track A crossfaded into first 20s of candidates B, C, D (render the blend with ffmpeg — DJs judge the overlap, not the concatenation). Keys 1/2/3 + **s(kip)**. Skip is informative: ambiguous triplets get excluded.
2. **Active sampling**: candidates from the same metric-neighbourhood under the current model (random pairs are ~90% obvious rejects and teach nothing). This is what makes ~600 labels enough instead of 6,000.
3. Append-only JSONL log: timestamp, session ID, tracks, choice. (Late-night judgement drift is checkable only if logged.)
4. Budget: ~30s/judgement; 200 choices for first signal, 600-800 for a usable metric. 15-minute sessions.

## Phase 5 — Metric learning

1. **Diagonal Mahalanobis** over the 64-d PCA space, triplet/listwise margin loss (~30 lines of torch, seconds on CPU). Move to low-rank only if diagonal plateaus.
2. Two-stage: pretrain on population adjacency (generic techno DJ practice), fine-tune on James's labels (personal deviations). Pragmatic fallback at this data scale: `score = α·population_prior + (1−α)·personal_metric`, α tuned on held-out personal labels.
3. Retrain as labels accumulate; plot eval metric vs label count; stop labelling when the curve flattens.
4. Watch-items: population data encodes *other DJs'* context (filter scraped sets to the seed-list gene pool to control for it); "sounds similar" ≠ "mixes well" (deliberate contrast) — attack only after the eval is stable.

## Phase 6 — Sequencing + integration

1. Beam search over the learned metric: hard constraints (per-transition BPM drift, Camelot compatibility via existing `core/compatibility.py`), soft energy-arc objective (wrap existing `core/sequencer.py` strategies — do not rewrite).
2. Output: Mixxx crate + M3U/CUE via existing exporters.
3. Re-scope the on-hold prd.json seed-playlist tasks (TASK-01..14) to use the learned metric.
4. Only then: GUI wiring for seed → playlist in the GTK app.

## Explicitly deferred

- GUI polish (GUI-10/11/12-style tasks), Flatpak/PyPI packaging (old Milestone 6), Five Rhythms extras, Rekordbox import extras.
- UK-bass axis mining (own pass, later).
- RalphZilla auto-loop: this plan runs human-gated via the opencode orchestrator, not AFK auto-merge.
