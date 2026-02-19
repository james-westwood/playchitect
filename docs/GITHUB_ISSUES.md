# GitHub Issues Created

## Summary

Created **17 comprehensive issues** across 6 milestones for the Playchitect project.

**Project Board**: https://github.com/users/james-westwood/projects/1
**Repository**: https://github.com/james-westwood/playchitect

---

## Milestone 2: Intelligent Analysis Engine (6 issues)

### Critical Priority

**#1 - Implement librosa intensity analyzer with RMS-weighted features**
- 8-dimensional feature vector (BPM + 7 intensity features)
- Bass energy split: sub-bass (20-60Hz), kick (60-120Hz), harmonics (120-250Hz)
- RMS weighting: louder frames count more
- JSON caching system
- **Labels**: `type-feature`, `area-analysis`, `priority-critical`, `area-core`
- **URL**: https://github.com/james-westwood/playchitect/issues/1

**#2 - Integrate intensity features into K-means clustering**
- Update clustering to use 8-dimensional features
- Backwards compatible with BPM-only mode
- Feature importance reporting
- **Labels**: `type-feature`, `area-core`, `priority-critical`
- **URL**: https://github.com/james-westwood/playchitect/issues/2

### High Priority

**#3 - Implement genre-specific PCA feature weighting**
- Learn optimal weights per genre using PCA
- User override capability via config file
- Default equal weights fallback
- **Labels**: `type-feature`, `area-core`, `priority-high`
- **URL**: https://github.com/james-westwood/playchitect/issues/3

**#4 - Implement smart first/last track selector**
- Intro/outro detection
- Score tracks for playlist position suitability
- Top-N recommendations with confidence scores
- **Labels**: `type-feature`, `area-core`, `priority-high`
- **URL**: https://github.com/james-westwood/playchitect/issues/4

### Medium Priority

**#5 - Integrate Essentia/MusiCNN for music embeddings**
- 128-dimension embedding vectors
- Genre auto-detection
- 136-dim feature vectors total (8 + 128)
- **Labels**: `type-feature`, `area-analysis`, `priority-medium`
- **URL**: https://github.com/james-westwood/playchitect/issues/5

**#6 - Implement genre-aware multi-clustering for mixed-genre sets**
- Three modes: single-genre, per-genre, mixed-genre
- Manual genre override support
- **Labels**: `type-feature`, `area-core`, `priority-medium`
- **URL**: https://github.com/james-westwood/playchitect/issues/6

---

## Milestone 3: GTK4 GUI Development (4 issues)

### Critical Priority

**#7 - Create GTK4 main application window with libadwaita**
- Adw.ApplicationWindow with header bar
- Menu structure and keyboard shortcuts
- Adaptive layout and dark mode
- **Labels**: `type-feature`, `area-gui`, `priority-critical`
- **URL**: https://github.com/james-westwood/playchitect/issues/7

### High Priority

**#8 - Implement track list widget with sorting and filtering**
- Gtk.ColumnView with multiple columns
- Search/filter, multi-selection
- Handle 10k+ tracks efficiently
- **Labels**: `type-feature`, `area-gui`, `priority-high`
- **URL**: https://github.com/james-westwood/playchitect/issues/8

**#9 - Create cluster visualization panel**
- Visual cluster cards with stats
- BPM and intensity distribution
- Interactive cluster selection
- **Labels**: `type-feature`, `area-gui`, `priority-high`
- **URL**: https://github.com/james-westwood/playchitect/issues/9

### Medium Priority

**#10 - Integrate GNOME Sushi for spacebar track preview**
- Spacebar shortcut for quick preview
- Fallback to xdg-open
- **Labels**: `type-feature`, `area-gui`, `priority-medium`
- **URL**: https://github.com/james-westwood/playchitect/issues/10

---

## Milestone 4: Export & Integration (2 issues)

**#11 - Implement CUE sheet generator with proper timing**
- Accurate track timing calculation
- Metadata inclusion
- **Labels**: `type-feature`, `area-export`, `priority-high`
- **URL**: https://github.com/james-westwood/playchitect/issues/11

**#12 - Create desktop file and icon for system integration**
- Application icon (SVG + PNG)
- .desktop file with MIME associations
- **Labels**: `type-feature`, `area-gui`, `priority-medium`
- **URL**: https://github.com/james-westwood/playchitect/issues/12

---

## Milestone 5: Testing & Quality Assurance (3 issues)

**#13 - Set up GitHub Actions CI/CD pipeline**
- Automated testing on push/PR
- Coverage reporting
- Multi-version testing (Python 3.11, 3.12)
- **Labels**: `type-feature`, `area-testing`, `priority-high`
- **URL**: https://github.com/james-westwood/playchitect/issues/13

**#14 - Create GUI smoke tests for layout and accessibility**
- GNOME HIG compliance checks
- Accessibility validation
- Pre-commit hook integration
- **Labels**: `type-test`, `area-gui`, `area-testing`, `priority-high`
- **URL**: https://github.com/james-westwood/playchitect/issues/14

**#15 - Performance benchmarking for 10k+ track libraries**
- Synthetic test library generation
- Performance targets and alerts
- **Labels**: `type-test`, `area-core`, `priority-medium`
- **URL**: https://github.com/james-westwood/playchitect/issues/15

---

## Milestone 6: Packaging & Distribution (2 issues)

**#16 - Create Flatpak manifest for Flathub submission**
- Complete Flatpak manifest
- Sandbox configuration
- Flathub submission and review
- **Labels**: `type-feature`, `priority-critical`
- **URL**: https://github.com/james-westwood/playchitect/issues/16

**#17 - Publish to PyPI with entry points**
- PyPI package preparation
- Entry points for CLI and GUI
- v1.0.0 release
- **Labels**: `type-feature`, `priority-high`
- **URL**: https://github.com/james-westwood/playchitect/issues/17

---

## Issue Statistics

**Total Issues**: 17
- **Critical**: 3 (18%)
- **High**: 6 (35%)
- **Medium**: 5 (29%)
- **Low**: 0

**By Type**:
- **Feature**: 16 (94%)
- **Test**: 3 (18%)
- **Bug**: 0
- **Docs**: 0

**By Area**:
- **Core**: 5
- **GUI**: 6
- **Analysis**: 2
- **Export**: 2
- **Testing**: 3

**By Milestone**:
- **Milestone 2**: 6 issues (35%)
- **Milestone 3**: 4 issues (24%)
- **Milestone 4**: 2 issues (12%)
- **Milestone 5**: 3 issues (18%)
- **Milestone 6**: 2 issues (12%)

---

## Adding Issues to Project Board

Since the GitHub CLI needs additional permissions, add issues to the project board manually:

1. Go to https://github.com/users/james-westwood/projects/1
2. Click "Add item" at the bottom of any column
3. Search for issue number (e.g., "#1")
4. Repeat for issues #1-17

**Suggested Column Assignments**:
- **Backlog**: #5, #6, #9, #10, #11, #12, #14, #15, #16, #17
- **Ready**: #1, #2, #3, #4 (Milestone 2 priorities)
- **In Progress**: (none yet)

**Or run after authentication**:
```bash
gh auth refresh -h github.com -s project
for i in {1..17}; do
  gh project item-add 1 --owner james-westwood --url https://github.com/james-westwood/playchitect/issues/$i
done
```

---

## Next Steps

1. **Add issues to project board** (manually or after auth refresh)
2. **Start with issue #1** (Intensity Analyzer) - highest priority
3. **Track progress** by moving issues through: Backlog → Ready → In Progress → Review → Testing → Done
4. **Close issues** with "Fixes #N" in commit messages
5. **Update milestones** as issues complete

---

## Labels Reference

**Priority**:
- `priority-critical` - Must be done for milestone
- `priority-high` - Important for milestone
- `priority-medium` - Nice to have
- `priority-low` - Future consideration

**Type**:
- `type-feature` - New functionality
- `type-bug` - Bug fix
- `type-refactor` - Code improvement
- `type-test` - Testing work
- `type-docs` - Documentation

**Area**:
- `area-core` - Core business logic
- `area-cli` - CLI interface
- `area-gui` - GUI interface
- `area-analysis` - Audio analysis
- `area-export` - Export functionality
- `area-testing` - Testing infrastructure

**Status**:
- `needs-testing` - Awaiting tests
- `needs-review` - Awaiting code review
- `blocked` - Blocked by other work
- `good first issue` - Good for newcomers

---

**Created**: 2026-02-19
**Last Updated**: 2026-02-19
