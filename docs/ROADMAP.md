# Playchitect Development Roadmap

## Overview

10-week MVP development plan to build intelligent DJ playlist management with K-means clustering.

## Milestones

### ✅ Milestone 1: Foundation & Core Refactoring (Weeks 1-2) - COMPLETE

**Status**: Complete (2026-02-19)

**Goals**:
- Establish project infrastructure
- Extract existing functionality into modules
- Create testing framework

**Deliverables**:
- ✅ GitHub repository with project board
- ✅ Modular core: `audio_scanner.py`, `metadata_extractor.py`
- ✅ Basic CLI with existing functionality
- ✅ Unit tests for extracted modules (>85% coverage target)
- ✅ CI/CD pipeline (GitHub Actions - TODO)

**Test Coverage**: 92% (audio_scanner), 61% (metadata_extractor)

---

### 🚧 Milestone 2: Intelligent Analysis Engine (Weeks 3-4) - NEXT

**Goals**:
- Implement K-means clustering
- Add librosa-based intensity analysis
- Create smart track selection logic

**Deliverables**:
- [ ] Intensity analyzer (spectral centroid, high-freq energy, RMS, percussiveness)
- [ ] K-means clustering with auto-K determination (elbow method)
- [ ] Cluster splitting for target playlist length
- [ ] Smart first/last track selector
- [ ] Analysis result caching system
- [ ] Enhanced CLI with intelligent features

**Critical Files**:
- `playchitect/core/intensity_analyzer.py` - Librosa integration
- `playchitect/core/clustering.py` - K-means implementation
- `playchitect/core/track_selector.py` - Intro/outro detection

**Key Algorithms**:
- Hardness Score: `0.4 * spectral_centroid + 0.3 * hf_energy + 0.2 * rms + 0.1 * percussiveness`
- Feature Vector: `[normalized_bpm, brightness, hf_energy, rms, percussiveness]`
- First Track Criteria: Low intensity + intro >30s + no immediate kick drum

---

### 📋 Milestone 3: GTK4 GUI Development (Weeks 5-7)

**Goals**:
- Build native GNOME interface with libadwaita
- Integrate GNOME Sushi for preview
- Create visualization widgets

**Deliverables**:
- [ ] Main application window (Adw.ApplicationWindow)
- [ ] Track list widget (Gtk.ColumnView with sorting/filtering)
- [ ] Cluster visualization panel
- [ ] Analysis progress indicators
- [ ] GNOME Sushi integration (spacebar preview)
- [ ] Preferences dialog
- [ ] Drag-and-drop track reordering

**Critical Files**:
- `playchitect/gui/app.py` - Adw.Application entry point
- `playchitect/gui/windows/main_window.py` - Primary user interface
- `playchitect/gui/widgets/track_list.py` - Main track display widget

**Design Requirements**:
- Follow GNOME HIG guidelines
- Support adaptive layout (mobile-friendly)
- Dark mode support via libadwaita
- Keyboard navigation for all features
- ARIA labels for screen readers

---

### 📋 Milestone 4: Export & Integration (Week 8)

**Goals**:
- Complete export functionality
- Integrate with system

**Deliverables**:
- [ ] M3U playlist exporter (refactored from existing)
- [ ] CUE sheet generator with proper timing
- [ ] Desktop file and icon (SVG + PNG)
- [ ] MIME type associations for M3U/CUE
- [ ] File manager context menu integration

**Critical Files**:
- `playchitect/core/export.py` - M3U and CUE exporters
- `packaging/flatpak/com.github.jameswestwood.Playchitect.desktop`

---

### 📋 Milestone 5: Testing & Quality Assurance (Week 9)

**Goals**:
- Comprehensive test coverage
- Automation and pre-commit hooks

**Deliverables**:
- [ ] Unit tests (>85% coverage for core modules)
- [ ] Integration tests (CLI workflows)
- [ ] GUI smoke tests (layout, accessibility, GNOME HIG compliance)
- [ ] Pre-commit hooks enhanced with GUI tests
- [ ] Performance benchmarking (10k+ tracks)

**Critical Files**:
- `.pre-commit-config.yaml` - Enhanced configuration
- `tests/gui/test_gui_layout.py` - GUI smoke tests

**GUI Smoke Test Criteria**:
- All required widgets present and rendered
- Proper spacing and alignment (GNOME HIG)
- Keyboard navigation functional
- ARIA labels present for accessibility
- No layout regressions

---

### 📋 Milestone 6: Packaging & Distribution (Week 10)

**Goals**:
- Publish to Flathub and PyPI
- Release v1.0.0

**Deliverables**:
- [ ] Flatpak manifest with all dependencies
- [ ] PyPI package with entry points
- [ ] User guide with screenshots
- [ ] Developer documentation
- [ ] Flathub submission
- [ ] PyPI production release

**Critical Files**:
- `packaging/flatpak/com.github.jameswestwood.Playchitect.yml`
- `setup.py` - PyPI distribution configuration

---

## Timeline

```
Week 1-2:  ✅ Milestone 1 - Foundation (COMPLETE)
Week 3-4:  🚧 Milestone 2 - Analysis Engine (NEXT)
Week 5-7:  📋 Milestone 3 - GUI Development
Week 8:    📋 Milestone 4 - Export & Integration
Week 9:    📋 Milestone 5 - Testing & QA
Week 10:   📋 Milestone 6 - Packaging & Distribution
```

## Success Metrics

**Development**:
- Code coverage: >85% for core modules
- Build success rate: >95%
- All pre-commit hooks passing
- Zero critical bugs in production

**Quality**:
- 100% pre-commit hook pass rate
- All GUI smoke tests passing
- <5 minor bugs per release
- Performance: 1000 tracks analyzed in <10 minutes

**Adoption** (Post-Launch):
- Flathub downloads: 100+ in first month
- PyPI downloads: 50+ in first month
- GitHub stars: 25+ in first quarter

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines.

## Questions?

Open an issue on GitHub: https://github.com/james-westwood/playchitect/issues
