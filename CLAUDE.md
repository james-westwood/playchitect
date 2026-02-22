# Claude Session Context - Playchitect Project

## Project Overview

**Name**: Playchitect
**Purpose**: Smart DJ Playlist Manager with Intelligent BPM Clustering
**Repository**: https://github.com/james-westwood/playchitect
**Location**: `/home/james/audio/playchitect/`
**Status**: Milestone 1 Complete (2026-02-19)


# Gemini — As a code developer

Claude is the senior developer of the Playchitect project. You are responsible for writing most of the code, which will be sent to Gemini for review via the `./scripts/review_pr.sh` script. However, sometimes, you can spawn a Gemini CLI instance on to carry out some tasks for you and create code, using git worktrees. Look up the details in the "Parallel Development with Git Worktree + Background Gemini" section below.

## What is Playchitect?

Playchitect transforms DJ playlist creation from rigid BPM-based grouping to intelligent multi-dimensional clustering. Instead of grouping tracks solely by tempo (e.g., "all 125 BPM tracks together"), it uses K-means analysis on BPM + 4 audio intensity features to create playlists that feel coherent—not just mathematically similar.

**Key Innovation**: A 125 BPM ambient intro sounds nothing like a 125 BPM hard techno track. Playchitect understands this by analyzing:
- BPM (tempo)
- Spectral brightness (treble content)
- High-frequency energy (>8kHz)
- RMS energy (loudness)
- Percussiveness (kick drum strength)

## Technology Stack

**Backend**:
- Python 3.13+ with native type hints
- librosa (audio intensity analysis)
- mutagen (metadata extraction)
- scikit-learn (K-means clustering)
- numpy, scipy

**Frontend** (Milestone 3):
- GTK4 + libadwaita (native GNOME)
- PyGObject bindings (system package — cannot be pip-installed)
- GNOME Sushi integration (spacebar preview)

**Development**:
- Package management: uv
- Testing: pytest with 218 tests (unit + integration)
- Pre-commit hooks: ruff, ty, pytest-unit, cli-smoke-test
- CI/CD: GitHub Actions (.github/workflows/ci.yml)

## Project Structure

```
playchitect/
├── playchitect/                    # Main package
│   ├── core/                       # Core business logic
│   │   ├── audio_scanner.py        # File discovery (92% coverage)
│   │   ├── metadata_extractor.py   # BPM extraction (61% coverage)
│   │   ├── intensity_analyzer.py   # TODO: Milestone 2
│   │   ├── clustering.py           # TODO: Milestone 2
│   │   ├── track_selector.py       # TODO: Milestone 2
│   │   ├── playlist_generator.py   # TODO: Milestone 2
│   │   └── export.py               # TODO: Milestone 4
│   ├── cli/                        # CLI interface
│   │   └── commands.py             # scan, info commands
│   ├── gui/                        # GTK4 interface (Milestone 3)
│   └── utils/                      # Config, logging
├── tests/
│   ├── unit/                       # 27 tests passing
│   ├── integration/                # TODO
│   └── gui/                        # TODO: Milestone 3
├── docs/
│   └── MILESTONE1_COMPLETE.md      # Milestone 1 summary
├── pyproject.toml                  # Project metadata
├── uv.lock                         # Locked dependencies
└── CLAUDE.md                       # This file
```

## Development Workflow

### Setup
```bash
cd /home/james/audio/playchitect
# Venv uses system Python 3.13 + system-site-packages for GTK4/gi access
uv venv --python /usr/bin/python3 --system-site-packages  # Already done
uv pip install -e ".[dev]"  # Already done
```

### Common Commands
```bash
# Run tests
uv run pytest -v
uv run pytest --cov=playchitect --cov-report=html

# Run CLI
uv run playchitect --help
uv run playchitect info /path/to/music
uv run playchitect scan /path/to/music --output /path/to/playlists

# Run pre-commit checks
uv run pre-commit run --all-files

# Format code
uv run ruff format playchitect/ tests/

# Type check
uv run ty check
```

## Feature Branching Policy

**Rule: Never commit directly to `main`. Every change goes through a feature branch + PR + Gemini review.**

### Branch Naming Convention

| Type | Pattern | Example |
|------|---------|---------|
| Feature | `feature/<issue-number>-<slug>` | `feature/1-intensity-analyzer` |
| Bug fix | `fix/<issue-number>-<slug>` | `fix/7-rms-overflow` |
| Docs | `docs/<slug>` | `docs/update-readme` |
| Chore | `chore/<slug>` | `chore/chore/bump-librosa` |

### Before Writing Any Code (Claude's checklist)

1. Create a GitHub issue (`gh issue create`) to track the work
2. Discuss the plan with James before implementing — enter plan mode,
   write the plan, and wait for approval via ExitPlanMode
3. Only then create a feature branch and start coding
4. Never commit or edit files directly on `main`

### Claude's Git Workflow

```bash
# 1. Start from main
git checkout main && git pull

# 2. Create feature branch (use issue number)
git checkout -b feature/1-intensity-analyzer

# 3. Implement with TDD — write tests first
uv run pytest -v  # watch them fail, then pass

# 4. Commit (pre-commit hooks run automatically)
git commit -m "feat(analysis): implement librosa intensity analyzer

- RMS energy with frame-level normalization
- RMS-weighted spectral centroid (brightness)
- 3-way bass split: sub-bass / kick / harmonics
- HPSS percussiveness ratio
- JSON caching with MD5 hash validation

Closes #1"

# 5. Push and open PR
git push -u origin feature/1-intensity-analyzer
gh pr create --title "feat(analysis): implement librosa intensity analyzer" \
  --body "Closes #1" --assignee james-westwood
```

### PR Checklist (Claude must verify before raising PR)

- [ ] All tests pass: `uv run pytest -v`
- [ ] Coverage >85% on modified modules
- [ ] Pre-commit hooks pass: `uv run pre-commit run --all-files`
- [ ] Type hints complete and ty clean
- [ ] No magic numbers — use named constants
- [ ] No direct commits to `main`

### Gemini Review Workflow

After Claude opens a PR, **James must run the Gemini reviewer**:

```bash
# From repo root, on the feature branch (or just run after gh pr checkout <num>):
./scripts/review_pr.sh

# Or to compare against a different base:
./scripts/review_pr.sh develop
```

Gemini will output either:
- **APPROVE** → James merges the PR via `gh pr merge --squash`
- **REQUEST CHANGES** → Claude reads the feedback, fixes blocking issues, pushes to the same branch, and asks James to run the review again

**Gemini's instructions live in `GEMINI.md`** at the repo root. Do not modify GEMINI.md without discussing with James first.

### Merge Strategy

- **Squash merge** to keep `main` history clean
- PR title becomes the squash commit message (must follow conventional commits)
- Delete the feature branch after merge

---

## Parallel Development with Git Worktree + Background Gemini

For larger features, Claude and Gemini can work in **true parallel** using `git worktree`.
This avoids the bottleneck of waiting for Gemini to finish before Claude can continue.

### How It Works

The feature is split into two parts:
- **Claude** takes the higher-level implementation (main feature branch)
- **Gemini** takes a smaller, well-scoped sub-task (sub-branch in a separate worktree)

Both run simultaneously — Claude in the main directory, Gemini in `/tmp/playchitect-gemini`.

### Setup

```bash
# 1. Create the main feature branch (Claude's branch)
git checkout main && git pull
git checkout -b feature/11-cue-sheet

# 2. Create Gemini's sub-branch off the feature branch
git checkout -b feature/11-cue-timing
git push -u origin feature/11-cue-timing
git checkout feature/11-cue-sheet

# 3. Add a worktree for Gemini's sub-branch
git worktree add /tmp/playchitect-gemini feature/11-cue-timing
```

### Launching Gemini in the Background

```bash
gemini --yolo -p "$(cat /tmp/gemini_prompt.txt)" > /tmp/gemini_task.log 2>&1 &
echo "Gemini PID: $!"
```

Key points for the Gemini prompt:
- Specify `Working directory: /tmp/playchitect-gemini`
- Name the branch explicitly and say it is already checked out
- Tell Gemini to open its PR against the **feature branch** (not `main`)
- Tell Gemini **not** to commit to main or the parent feature branch

### Checking on Gemini

```bash
# Check if it's still running
ps aux | grep gemini | grep -v grep

# Tail the log
tail -30 /tmp/gemini_task.log

# Check if Gemini has committed anything
git -C /tmp/playchitect-gemini log --oneline -5
```

### Merging the Sub-Branch

```bash
# Run Gemini review from the sub-branch, with the feature branch as base
git checkout feature/11-cue-timing
./scripts/review_pr.sh feature/11-cue-sheet

# On APPROVE, merge locally (GitHub merge may conflict if both branches
# touched the same file — resolve, commit, then the PR closes automatically)
git checkout feature/11-cue-sheet
git merge origin/feature/11-cue-timing
# resolve any conflicts, then:
git push origin feature/11-cue-sheet
```

### Cleaning Up

```bash
git worktree remove /tmp/playchitect-gemini
git branch -D feature/11-cue-timing
git push origin --delete feature/11-cue-timing
```

### Lessons Learned (Issue #11)

- **Works well when tasks are truly independent.** The split between timing utilities (Gemini) and the generator/CLI (Claude) meant zero code overlap during development.
- **Gemini can hit 429 rate-limit errors** and stall without producing output. Monitor the log; if the line count stops growing, kill (`kill <PID>`) and relaunch.
- **If the worktree disappears**, check `git worktree list`. Untracked files from the lost worktree may be left in the main working directory — check `git status`, read them, and commit what's useful directly on the sub-branch.
- **Conflict on shared files is expected** when both branches independently create the same file (e.g. both writing `cue_timing.py`). Resolve with `git checkout --theirs <file>` to prefer the sub-branch version, or merge manually. The PR will close automatically once the conflict is resolved and pushed.
- **Sub-branch PR base must be the feature branch**, not `main`. Use `gh pr create --base feature/11-cue-sheet`.
- **Run `./scripts/review_pr.sh <base-branch>` from the sub-branch**, not the feature branch.

## Milestone Status

### ✅ Milestone 1: Foundation & Core Refactoring (Complete — 2026-02-19)
- GitHub repo, audio_scanner (92% cov), metadata_extractor (61% cov), CLI, pre-commit, uv

### ✅ Milestone 2: Intelligent Analysis Engine (Complete — 2026-02-19)
- `intensity_analyzer.py` — librosa spectral analysis, 7-feature vector, JSON caching
- `clustering.py` — K-means on 8D space, elbow method, genre-aware multi-clustering, EWKM
- `track_selector.py` — smart first/last track scoring with user overrides
- `weighting.py` — genre-specific PCA + EWKM feature weighting
- `embedding_extractor.py` — MusiCNN Block PCA semantic embedding integration (#5)

### ✅ Milestone 3: GTK4 GUI (Complete — 2026-02-20)
- Main application window (`playchitect/gui/app.py`, `windows/main_window.py`)
- Track list widget with `Gtk.ColumnView` sorting/filtering
- Cluster visualization panel with split-pane layout
- GNOME Sushi / xdg-open spacebar track preview
- GUI smoke tests (32 tests, conftest.py mock harness)

### ✅ Milestone 4: Export & Integration (Complete — 2026-02-20)
- CUE sheet generator with frame-accurate timing
- Desktop file, AppStream metainfo, 9 PNG icon sizes + .ico
- `playchitect-install-desktop` entry point

### ✅ Milestone 5: Testing & QA (Complete — 2026-02-21)
- GitHub Actions CI/CD (lint, unit, integration, Fedora 41 container)
- GUI smoke tests (Milestone 3 above)
- Performance benchmark suite: `tests/benchmarks/` with synthetic_library fixture,
  component-level benchmarks (AudioScanner, MetadataExtractor, IntensityAnalyzer,
  PlaylistClusterer), regression alerts via `--benchmark-compare`
- `scripts/review_pr.sh` Gemini code review workflow (default: gemini-2.5-pro)

### 🚧 Milestone 6: Packaging & Distribution (Next)
- [ ] **#16** — Flatpak manifest for Flathub submission ⚠️ see Flathub note below
- [ ] **#17** — PyPI package publishing

### 📋 Post-Milestone 6: Enhancement Backlog (no milestone assigned)
- **#19** — Wire intensity cache dir through central config
- **#21** — Boost `metadata_extractor` test coverage to >85%
- **#22** — Parallel batch analysis with `ProcessPoolExecutor`
- **#23** — Silhouette score for auto-K cluster selection
- **#26/#27** — User-configurable weight overrides (YAML config + CLI flags)
- **#36–43** — Key/Harmonic Mixing, Energy Flow, Timbre/Texture, Structural features + GUI
- **#51** — "5 Rhythms" intensity sequencing mode

## Origin Story

Playchitect originated from `/home/james/audio-management/scripts/create_random_playlists.py` (314 lines), which used rigid BPM range grouping. The new approach uses intelligent clustering to understand track character beyond just tempo.

**Original Script Location**: `/home/james/audio-management/scripts/create_random_playlists.py`

## Key Decisions

1. **License**: GPL-3.0 (copyleft, requires derivatives to be open source)
2. **Package Manager**: uv (fast, reproducible builds)
3. **Python Version**: 3.13+ (native type hints, modern features)
4. **GUI Framework**: GTK4 + libadwaita (native GNOME)
5. **Testing Strategy**: TDD with pytest, >85% coverage target
6. **Type Checking**: Strict ty with native type hints (Rust-based, replaces mypy)
7. **Code Style**: Ruff with 100-char line length (ruff format)

## Important Notes

- **Working Directory**: `/home/james/audio/playchitect/`
- **Virtual Environment**: `.venv/` (managed by uv)
- **Pre-commit**: Always run before commits (installed via `uv run pre-commit install`)
- **Tests**: Must pass before merging to main
- **Coverage Target**: >85% for core modules
- **No AI attribution**: Never mention Claude, AI, or any AI tool in commit messages, PR titles, PR bodies, issue comments, or any other Git/GitHub content. Write all such content as the developer.

## ⚠️ Flathub AI Policy (relevant to Milestone 6)

Flathub's [submission requirements](https://docs.flathub.org/docs/for-app-authors/requirements) include:

> "Submissions or changes where most of the code is written by or using AI without any meaningful human input, review, justification or moderation of the code are not allowed."
> "Submission pull requests must not be generated, opened, or automated using AI tools or agents."

**Implications for Playchitect:**
- The Flathub submission PR (#16) **must be opened manually by James**, not by Claude or Gemini
- The review request in that PR must not be delegated to an AI tool
- The key qualifier is "without any meaningful human input" — James has reviewed and approved all PRs, provided requirements, and made architectural decisions. Whether this constitutes sufficient human oversight is a judgement call that Flathub makes on a case-by-case basis
- PyPI (#17) has no equivalent policy and is straightforward
- **Alternative distribution**: COPR (Fedora), OBS (openSUSE), or a direct `.flatpak` download are all viable if Flathub declines

## Related Documentation

- **General Audio System**: `/home/james/audio-management/claude.md`
- **Original Script**: `/home/james/audio-management/scripts/create_random_playlists.py`
- **Project Plan**: Full implementation plan stored in session transcript

## Contact

**Developer**: James Westwood
**Machine**: Muddlehead (primary audio workstation)
**Platform**: Fedora 42 with GNOME

---

**Last Updated**: 2026-02-21
**Current Milestone**: Milestone 6 — Packaging & Distribution (Milestones 1–5 complete)
