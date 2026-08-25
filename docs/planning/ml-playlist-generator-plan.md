# Plan: ML-Powered Playlist Generator

**Status:** Phase 1 passed its human gate 2026-08-25 (TASK-15..18 done); closes on TASK-28. Phase 2 next.
**Author:** James + Kimi K3 (opencode session 2026-08-20), synthesised with a Claude Opus review
**Revised 2026-08-20** after external review: mainline reordered to the zero-external-dependency personal-metric loop; scraping demoted to a gated enhancement track. Technical fixes incorporated: choice-accuracy primary eval, deployment-constrained candidate sets, asymmetric transition score, rank-fusion blending, PCA whitening, judgement pre-rendering, response-time logging.
**Supersedes:** the clustering-centric roadmap in ROADMAP.md; prd.json TASK-01..14 (seed-playlist) is parked pending Phase 4
**Correction (2026-08-25):** this plan was authored from a checkout that predated the seed-playlist merges of 2026-06-16 (PRs #220-224), so it recorded TASK-01..14 as unstarted when TASK-01..10 and TASK-13 had in fact shipped. The premise below is stated without that work in view. James deferred revisiting it to Phase 4 — see `orchestrator-handoff-phase1-evidence.md`.

---

## Why this plan exists

Playchitect has ~18k LOC, 1,334 passing tests, 76% coverage — and is not usable. Verified 2026-08-20 against the real library (`dark 4`, 190 tracks, `--dry-run`):

- Default `scan` does **BPM-only** clustering ("Weight source: uniform"). The entire intensity/embedding/mood/sequencing stack is opt-in via flags.
- K=2 on 190 tracks → one 184-track cluster → `split_cluster()` **randomly shuffles and dices** it into 7 indistinguishable 25-track playlists, copying parent stats. The headline feature degrades to BPM buckets plus a randomiser.
- Nobody ever evaluated playlist *quality*; the dev loop optimised for tasks closed.

Root cause: no measurable definition of "good". This plan installs one before any further modelling.

## Guiding principles

1. **Eval before modelling.** Nothing ships without beating the baselines with bootstrap CIs that exclude them.
2. **The personal metric is the endgame — so the mainline has zero external dependencies.** Embedding ETL → eval harness → labelling tool → personal transition model is entirely under local control. The scraping/graph-prior work is an *enhancement track*: valuable if it lands, blocking nothing if it dies on anti-scraping or untracklisted sets.
3. **Two label sources, kept separate.** Population adjacency (other DJs' sets, enhancement track) = prior only. Personal labels (James's A/B/C/D judgements) = training + eval (split by session). James's own sets = golden sanity set, never trained on.
4. **Compose existing components.** intensity_analyzer, embedding_extractor, sequencer, compatibility (Camelot), cache_db all exist. Wire them into one pipeline; do not greenfield them.
5. **Clustering is scaffolding.** The endgame is seed → transition model → beam-search sequence → Mixxx crate. Fix the default path cheaply; do not perfect it.
6. **GUI and packaging last.**

## The library and data assets (verified)

- Library: `/mnt/1tb_ssd/Media/Music` — **1,726 tracks**
- Mixxx DB: `~/.mixxx/mixxxdb.sqlite` (2.4MB, only 4-5 sets — too small to train on)
- James's own sets: 4-5 on Mixcloud with CUE sheets — **golden sanity set**, eval-only, never trained on
- External (enhancement track only): 1001tracklists.com, mixesdb.com

---

## Phase 1 — Fix the default path (stopgap usability)

Cheap, TDD-able, no ML risk. Clustering stays scaffolding.

1. **Fix `split_cluster`** (core/clustering.py:840): recursive re-clustering instead of random shuffle-and-dice; sub-clusters report their own stats. *(TASK-15 — done)*
2. **Multi-dimensional clustering default** in `scan`; BPM-only behind `--fast`. *(TASK-16 — done)*
3. **Wire the naming package** into CLI/export names. *(TASK-17)*
4. **Fail loudly** on degenerate K (dominant cluster >= 70% with K <= 2 on >= 50 tracks → warning suggesting `--use-embeddings`). *(TASK-18)*

**Done when:** default `scan` on `dark 4` produces distinguishable, character-named playlists with per-cluster stats; tests green. **Human gate before Phase 2.**

## Phase 2 — Personal metric mainline (no external dependencies)

### 2.1 Embedding cache (TASK-19)

essentia `discogs-effnet` embeddings for all 1,726 tracks; Parquet keyed by **content hash** (sha256 of first 1MB + size), not path. Fit `PCA(n_components=64, whiten=True)` and persist — whitening so the Phase 2.4 diagonal weights are interpretable as pure feature importance rather than relearning the variance profile. Pull BPM/key from Mixxx DB where present rather than recomputing.

### 2.2 Eval harness (TASK-25)

`playchitect eval`. Two metrics, two candidate sets:

- **Primary: held-out choice accuracy.** For each held-out A/B/C/D judgement, the model must rank James's chosen candidate first; chance is 33%. ~150 held-out labels gives real statistical power — this is why the primary metric lives on the labels, not the golden sets.
- **Secondary (sanity): next-track retrieval** on the golden sets (Recall@10/50, MRR), split by set never by pair. With only ~100 golden pairs this has wide CIs forever — it is a directional check, not a gate.
- **Candidate sets:** (a) *deployment-constrained* — the BPM-drift + Camelot-feasible window around the anchor, the same constraint beam search will apply; (b) unconstrained full library. **The gate uses (a).** Rationale: unconstrained ranking punishes the metric for surfacing tracks it will never be asked about, and hands the BPM baseline free recall from a constraint the system applies anyway.
- **Baselines:** random-within-window, raw-cosine-within-window, BPM-proximity-within-window. All metrics with bootstrap CIs (1000 resamples). Harness must run cleanly in the zero-label state (baselines only) and grow as labels accumulate.

### 2.3 Choice-labelling tool (TASK-26)

Terminal script, no GUI:

1. Anchor A: last 20s crossfaded (ffmpeg) into first 20s of candidates B, C, D. Keys 1/2/3 + s(kip). DJs judge the blend, not the concatenation.
2. **Pre-render the next judgement's three blends while the current one plays.** Without this, a 15-minute session yields ~8 judgements instead of ~25 — the naive ~30s/judgement budget ignores render time and three ~40s crossfades.
3. **Active sampling, explicit cold start:** candidates from A's ~30-nearest neighbourhood under the *current* model; **round one uses raw discogs-effnet cosine** from TASK-19 (no learned metric exists yet). Random pairs are ~90% obvious rejects and teach nothing.
4. Append-only `data/labels.jsonl`: `{ts, session_id, anchor, candidates, choice (null = skip), response_ms}`. Response time is logged from day one — fast choices are higher-confidence labels and can be loss-weighted later at zero cost now. Split by **session** (not judgement) into train/held-out.
5. Realistic budget: ~15-25 judgements per 15-minute session; 200 labels ≈ 8-13 sessions.

### 2.4 Personal transition model (TASK-27)

`score(A→B) = −d_M(A,B) + wᵀ·Δ(A,B)`

- `d_M`: diagonal Mahalanobis over the whitened 64-d embeddings (64 params), margin loss on chosen-beats-losers constraints (3 per judgement), torch, CPU-seconds.
- `Δ`: signed deltas (BPM, RMS energy, brightness, percussiveness, sub_bass) — **mandatory, not optional**. A pure distance is symmetric; transitions are directional (A→B ≠ B→A, energy usually rises, deliberate contrast exists). Without the delta term the model class structurally cannot represent what the labels encode.
- **Guardrail:** must beat the raw-cosine-within-window baseline on held-out choice accuracy with a bootstrap CI excluding the baseline. Report failure honestly if not.

**Done when:** a trained model artifact beats the within-window baselines on held-out labels, or the harness reports exactly where it falls short. Human gate before Phase 4 integration.

## Phase 3 — Enhancement track: co-occurrence graph prior (parallel, gated, optional)

Runs in parallel with or after Phase 2. **Its failure blocks nothing** — the mainline stands alone. If it lands, it slots into the Phase 4 blend.

1. Seed artist extractor (TASK-20), scrapers (TASK-21), fuzzy resolver (TASK-22), own-sets golden harvest (TASK-23), match-rate + coverage report (TASK-24) → human gate TASK-H1.
2. **Scraping reality check:** MixesDB is MediaWiki — use its **API**, not page scraping. 1001tracklists sits behind Cloudflare with aggressive anti-scraping, and underground hypnotic/industrial sets are exactly the ones full of `ID - ID` entries — expect real engineering pain and low resolve rates there. Both fine: this track is optional.
3. **Graph embeddings:** factorise the **directed** transition matrix (transitions are directional; check empirically whether direction matters vs undirected), artist/label backoff for unseen tracks. The graph records *actual mixes*, including deliberate contrast — complementing the audio metric, which is blind to it.

## Phase 4 — Sequencing + integration

1. Beam search over the transition score: hard constraints (per-transition BPM drift, Camelot via existing `core/compatibility.py`), soft energy arc (wrap existing `core/sequencer.py` — do not rewrite).
2. **Blending (only if the graph prior landed):** never linearly blend raw scores — the signals live on unrelated, query-dependent scales. Per-query rank-transform, then combine ranks (reciprocal-rank fusion is the boring, robust choice at this data scale); tune fusion weights on held-out personal labels.
3. Output: Mixxx crate + M3U/CUE via existing exporters.
4. Re-scope the parked prd.json TASK-01..14 to the transition model — **adapt the shipped `core/seed_playlist.py` and `core/features.py` rather than building anew**, and merge the outstanding GUI wiring from `origin/feature/219-task11-12-gui-wiring` (TASK-11/12) plus the missing `playlist` CLI reference (TASK-14). **Revisit this plan's premise here**, per the correction note at the top. GUI last.

## Explicitly deferred

- GUI polish, Flatpak/PyPI packaging, Five Rhythms extras, Rekordbox extras.
- UK-bass axis mining (own pass, later — tagged `uk-bass/dub/dubstep`, excluded from the techno seeds).
- RalphZilla AFK auto-loop: this plan runs human-gated via the opencode orchestrator.
