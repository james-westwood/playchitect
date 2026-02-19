# Gemini Review & Roadmap Feedback — Milestone 2

## Verdict: APPROVE (With Technical Recommendations)

The project foundation is exceptionally solid. Milestone 1 was executed with high engineering standards (TDD, type safety, modularity). As we transition into the **Intelligent Analysis Engine (Milestone 2)**, the following technical steering is recommended to ensure both performance and reliability.

---

### 1. Priority: Close the Coverage Gap
**Target**: `playchitect/core/metadata_extractor.py` (Current: 61%)
**Action**: Before deep-diving into Librosa analysis, bring this module's coverage to **>85%**. Stable metadata extraction is critical for the caching and clustering logic that follows.

### 2. Performance Optimization Strategy
To hit the target of **1,000 tracks in <10 minutes**, implement these three "High-ROI" optimizations:

#### A. Fast File Hashing (Disk I/O)
*   **Problem**: MD5 hashing of full files (especially FLAC/WAV) is too slow for large libraries.
*   **Solution**: Implement a "Fast Hash" in `intensity_analyzer.py` that checks `file_size + mtime + first_1mb_data`. This provides 99.9% confidence with near-zero overhead.

#### B. Architectural Parallelism
*   **Recommendation**: Use Python's `multiprocessing` via `ProcessPoolExecutor`.
*   **Why**: Audio analysis (STFT, HPSS) is "embarrassingly parallel." Utilizing all CPU cores is the most efficient way to scale. **Do not introduce Rust or Go yet**; exhaust Python's parallel capabilities first.

#### C. Efficient Audio Loading
*   **Tactic**: Load audio at a reduced sample rate (e.g., `sr=22050`). BPM and intensity features do not require 44.1kHz fidelity. This significantly reduces CPU and RAM usage during analysis.

### 3. Algorithm Refinements

#### Clustering Robustness (`clustering.py`)
*   **Feature Scaling**: Always use `StandardScaler` (scikit-learn) before K-means. BPM (80-170) and RMS (-60 to 0) operate on vastly different scales.
*   **Auto-K Selection**: Supplement the "Elbow Method" with the **Silhouette Score**. It is easier to automate programmatically as it provides a clear peak for the optimal number of clusters.

#### Optional Dependencies (`Essentia`)
*   **Issue #5**: Keep Essentia/MusiCNN as an **optional/extra** dependency. Its system-level requirements (`libessentia`) can complicate Flatpak and PyPI distribution. Maintain a lightweight core by making it a "Plugin" feature.

---

### What's Good
- **TDD Rigor**: 92% coverage on `audio_scanner.py` is excellent.
- **Hardness Score**: The specific weighting formula is a clear and testable implementation goal.
- **GNOME Strategy**: Integration with GNOME Sushi (`#10`) demonstrates a strong commitment to native UX.

### Summary Checklist for Claude
- [ ] Increase `metadata_extractor.py` coverage to >85%.
- [ ] Implement `FastHash` for the analysis cache.
- [ ] Use `ProcessPoolExecutor` for batch analysis.
- [ ] Add `StandardScaler` to the clustering pipeline.
- [ ] Use `sr=22050` (or lower) for Librosa loading.

---
**Reviewer**: Gemini CLI (Senior Python Engineer)
**Date**: 2026-02-19
