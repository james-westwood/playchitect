# Milestone 2 MVP: BPM-Only Clustering - COMPLETE

## Summary

Implemented MVP Phase of Milestone 2 with BPM-only K-means clustering. Provides functional playlist generation while laying groundwork for intensity features.

---

## What Was Built

### Core Modules

#### 1. `playchitect/core/clustering.py`
- **PlaylistClusterer** class with K-means clustering
- Auto-K determination using elbow method
- Support for target playlist size by **track count** OR **duration**
- Cluster splitting for oversized playlists
- **Coverage**: 91% (103 lines, 9 missed)

**Key Features**:
- Normalizes BPM values using StandardScaler
- Balances elbow method with user constraints
- Handles edge cases (single track, no BPM, few tracks)

#### 2. `playchitect/core/export.py`
- **M3UExporter** class for playlist generation
- BPM range labels in filenames (e.g., `Playlist 1 [128-130bpm].m3u`)
- Relative path support
- CUE exporter placeholder (Milestone 4)

#### 3. Updated `playchitect/cli/commands.py`
- `scan` command with `--target-tracks` and `--target-duration` options
- Full integration: scan → extract → cluster → export
- Progress bars and cluster statistics
- Clear error messages

### Test Coverage

**14 new unit tests** in `tests/unit/test_clustering.py`:
- Initialization validation
- Edge cases (empty, single track, no BPM)
- Clustering behavior verification
- Reproducibility testing
- Cluster splitting

**Results**: ✅ 41 tests passing, 1 skipped (integration test), 49% overall coverage

---

## CLI Usage

```bash
# Create playlists with 25 tracks each
uv run playchitect scan ~/Music/Techno --target-tracks 25 --output ~/Playlists

# Create playlists of 60 minutes each
uv run playchitect scan ~/Music/House --target-duration 60 --output ~/Playlists

# Custom playlist name prefix
uv run playchitect scan ~/Music/Drum-and-Bass -t 30 -n "DnB Mix"
```

**Output Example**:
```
Clustering tracks by BPM...
Using K=4 clusters

Created 4 clusters:
  Cluster 1: 23 tracks, BPM: 122.5 ± 3.2, Duration: 95.3 min
  Cluster 2: 28 tracks, BPM: 130.8 ± 2.1, Duration: 112.7 min
  Cluster 3: 19 tracks, BPM: 138.2 ± 4.5, Duration: 78.1 min
  Cluster 4: 25 tracks, BPM: 145.1 ± 3.8, Duration: 102.4 min

✓ Successfully created 4 playlists:
  DnB Mix 1 [122-125bpm].m3u
  DnB Mix 2 [130-132bpm].m3u
  DnB Mix 3 [138-142bpm].m3u
  DnB Mix 4 [145-148bpm].m3u
```

---

## Pre-Trained Model Research

### 1. Essentia + MusiCNN ⭐ (RECOMMENDED FOR PHASE 3)

**What**: Open-source C++/Python library with pre-trained TensorFlow models

**Capabilities**:
- Auto-tagging (50+ music tags: genre, mood, instruments)
- Music embeddings (128-dimension vectors for transfer learning)
- Real-time analysis
- 790k parameters, state-of-the-art performance

**Integration Plan**:
- Phase 3: Extract embeddings as additional clustering features
- Use pre-trained genre classifier for auto-detection
- Combine embeddings with BPM + intensity for richer clustering

**Sources**: [Essentia models](https://essentia.upf.edu/models.html), [MusiCNN paper](https://www.researchgate.net/publication/341084851_Tensorflow_Audio_Models_in_Essentia)

### 2. Spotify Audio Features API

**What**: Pre-computed audio features (energy, danceability, valence, tempo, loudness)

**Validation Study**: Energy and valence correlate well with human perception

**Integration Plan**:
- Optional Phase 4 feature
- Fetch Spotify features for tracks in their catalog
- Use as fallback when librosa analysis is slow

**Limitation**: Only works for tracks in Spotify's catalog

**Sources**: [Spotify API](https://developer.spotify.com/documentation/web-api/reference/get-audio-features), [Validation study](https://sciety.org/articles/activity/10.31234/osf.io/8gfzw_v2)

### 3. Audio Foundation Models (HuBERT, wav2vec 2.0, WavLM)

**What**: Large pre-trained models from speech processing

**Capabilities**: Zero-shot music classification without fine-tuning

**Trade-off**: Heavy computational requirements (multi-GB models)

**Integration Plan**: Phase 4+ if needed for genre classification

**Sources**: [Music Genre Classification with LLMs](https://arxiv.org/html/2410.08321v1)

---

## Revised Implementation Plan

### ✅ Phase 1 (MVP): BPM-Only Clustering - COMPLETE

**Delivered**:
- K-means clustering on BPM only
- Target size by track count or duration
- M3U export with BPM labels
- Full CLI integration

**Timeline**: Completed 2026-02-19

---

### 🚧 Phase 2: Intensity Analysis (NEXT - 1 week)

**Goal**: Add multi-dimensional intensity features

**Features to Implement** (in priority order):

#### 1. **RMS Energy** (Overall Loudness)
```python
rms = librosa.feature.rms(y=y)[0]
rms_mean = np.mean(rms)  # Weight by frame (louder = more important)
```

#### 2. **Spectral Centroid** (Brightness)
```python
cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
# Weight by RMS (louder frames count more)
rms_weights = rms / np.sum(rms)
brightness = np.sum(cent * rms_weights) / (sr / 2)  # Normalize to 0-1
```

#### 3. **Bass Energy** (Separated Analysis) ⭐
Per your requirement:
- **Sub-bass**: 20-60Hz (sub-kick, rumble)
- **Kick fundamental**: 60-120Hz (main kick energy)
- **Bass harmonics**: 120-250Hz (bass notes)

```python
S = np.abs(librosa.stft(y))
freqs = librosa.fft_frequencies(sr=sr)

sub_bass = np.mean(S[(freqs >= 20) & (freqs < 60), :])
kick_energy = np.mean(S[(freqs >= 60) & (freqs < 120), :])
bass_harmonics = np.mean(S[(freqs >= 120) & (freqs < 250), :])
```

#### 4. **Percussiveness** (HPSS Ratio)
```python
y_harmonic, y_percussive = librosa.effects.hpss(y)
perc_ratio = np.sqrt(np.mean(y_percussive**2)) / np.sqrt(np.mean(y**2))
```

#### 5. **Onset Strength** (Beat Intensity)
```python
onset_env = librosa.onset.onset_strength(y=y, sr=sr)
onset_mean = np.mean(onset_env)
```

**Feature Vector** (8 dimensions):
```
[normalized_bpm, rms_energy, brightness, sub_bass, kick_energy,
 bass_harmonics, percussiveness, onset_strength]
```

**Weighting Strategy** ⭐ (Per your requirements):
1. **Genre-specific PCA**: Learn optimal weights per genre from user's library
2. **User override**: Advanced settings dropdown to manually adjust weights
3. **Default**: Equal weights (0.125 each) as fallback

**Timeline**: 1 week

---

### 📋 Phase 3: Pre-trained Model Integration (3-4 days)

**Goal**: Add Essentia/MusiCNN embeddings for richer clustering

**Implementation**:
1. Install Essentia with TensorFlow support
2. Load pre-trained MusiCNN model
3. Extract 128-dimension embeddings per track
4. Concatenate with intensity features: `[8 intensity features + 128 embedding features]`
5. Auto-detect genres using pre-trained classifier

**Benefits**:
- Genre-aware clustering
- Better semantic similarity (tracks that "feel" similar)
- Foundation for mixed-genre playlists

**Timeline**: 3-4 days

---

### 📋 Phase 4: Genre-Aware Multi-Clustering (3-4 days)

**Goal**: Support mixed-genre DJ sets

**Implementation**:
1. Auto-detect genres using Essentia model
2. Run separate K-means per genre
3. Support "mixed genre" mode with adjusted BPM scaling
4. User can manually assign/override genres

**Use Cases**:
- Pure techno set: Single-genre clustering
- Eclectic set: Mixed-genre with BPM transitions
- Genre-specific energy profiles

**Timeline**: 3-4 days

---

## Key Decisions Made

### 1. MVP Scope: BPM-Only First ✅
**Rationale**: Validate clustering approach before adding complexity

### 2. Target Size: Tracks OR Duration ✅
**Rationale**: DJs think in both units depending on context

### 3. Bass Energy: 3-Way Split ✅
**Rationale**: Techno emphasizes different bass ranges (sub vs kick vs harmonics)

### 4. Feature Weighting: PCA + User Override ✅
**Rationale**: Learn from data per genre, but allow manual tuning for DJ preferences

### 5. Pre-trained Models: Essentia First ✅
**Rationale**: Open-source, lightweight, proven for music analysis

---

## Technical Achievements

### Performance
- K-means scales efficiently (O(n*k*i) where i is iterations)
- Elbow method runs quickly for k=2-10
- StandardScaler prevents BPM from dominating

### Code Quality
- Type hints throughout
- Comprehensive docstrings
- 91% test coverage on clustering module
- All pre-commit hooks passing

### Extensibility
- Easy to add new features to clustering
- Pluggable weighting strategies
- Cluster splitter for future use

---

## Next Steps

### Immediate (This Week)
1. Implement `intensity_analyzer.py` with 5 core features
2. Update clustering to use 8-dimensional feature vectors
3. Add genre-specific PCA weighting
4. Create intensity analysis tests

### Short Term (Next Week)
1. Integrate Essentia/MusiCNN
2. Add genre auto-detection
3. Implement mixed-genre clustering mode
4. Create advanced settings for feature weights

### Medium Term (Milestone 3)
1. Build GTK4 GUI
2. Visualize clusters and features
3. Interactive weight adjustment
4. GNOME Sushi integration

---

## Open Questions Resolved

**Q**: Should we use high-frequency energy (>8kHz) for intensity?
**A**: For techno, bass energy (20-250Hz split into 3 bands) is more important. High-freq captures hi-hats/cymbals, not track intensity.

**Q**: What about feature weighting (0.4, 0.3, 0.2, 0.1)?
**A**: Use PCA per genre to learn weights from data, with user override capability.

**Q**: How to handle mixed-genre sets?
**A**: Phase 4 will add genre-aware clustering with separate K-means per genre or adjusted BPM scaling for mixed mode.

**Q**: Pre-trained models worth it?
**A**: Yes! Essentia/MusiCNN embeddings will significantly improve clustering quality.

---

## Files Changed

**Added**:
- `playchitect/core/clustering.py` (277 lines)
- `playchitect/core/export.py` (73 lines)
- `tests/unit/test_clustering.py` (233 lines)
- `docs/MILESTONE2_MVP_COMPLETE.md` (this file)

**Modified**:
- `playchitect/cli/commands.py` (+82 lines)
- `pyproject.toml` (+5 lines for mypy config)

**Test Results**:
- 41 passing, 1 skipped
- Coverage: 49% overall, 91% on clustering module

---

**Status**: Milestone 2 MVP Complete ✅
**Next**: Phase 2 - Intensity Analysis
**Last Updated**: 2026-02-19
