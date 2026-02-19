# Research: Genre-Specific Feature Weighting (Issue #3)

**Status**: Research complete — pending Gemini review before implementation
**Date**: 2026-02-19
**Relates to**: [Issue #3](https://github.com/james-westwood/playchitect/issues/3)

---

## TL;DR

PCA alone is **not the right tool** for feature weighting in K-means. It finds directions of maximum variance, not maximum cluster separability — these are different things. The recommended approach is a three-layer system: PCA communality weights as a starting point, refined by **EWKM** (Entropy-Weighted K-Means, Jing et al. 2007) which learns per-cluster weights during clustering, with a clean fallback hierarchy for small libraries.

---

## How PCA-Based Weights Actually Work

The issue brief says "use PCA" but the standard practice is not to cluster in PCA space — it's to derive **per-feature importance weights** from PCA communalities, then apply those as a weighted Euclidean distance in standard K-Means.

The derivation:

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)          # (n_tracks, 8) — non-negotiable

pca = PCA(n_components=None)
pca.fit(X_scaled)

# Retain components explaining 90% of variance
n_components = np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.90) + 1

evr = pca.explained_variance_ratio_[:n_components]

# Loadings = correlation between original features and each PC
loadings = pca.components_[:n_components].T * np.sqrt(pca.explained_variance_[:n_components])
# shape: (n_features, n_components)

# Weighted communality: "how much of each feature's variance is captured by the PC structure?"
importance = np.sum(loadings**2 * evr[np.newaxis, :], axis=1)
weights = importance / importance.sum()   # sums to 1.0

# Apply in K-Means as scaled Euclidean distance:
# d(x,c)^2 = sum_j w[j] * (x[j] - c[j])^2
# Implementation shortcut:
X_weighted = X_scaled * np.sqrt(weights)[np.newaxis, :]
# → run standard KMeans on X_weighted
```

---

## Why PCA Alone Is Insufficient

**1. Variance ≠ separability.**
PCA maximises explained variance globally. In a library that is 80% ambient and 20% techno, PCA will weight features that discriminate _within_ the large ambient blob. The features that actually separate techno from ambient (kick energy, onset strength) may appear as minor PCs and receive low weight — exactly the opposite of what we want.

**2. Global weights miss genre context.**
A techno cluster should weight kick energy heavily; the ambient cluster should weight onset strength near-zero as its key discriminator. A single global weight vector cannot capture this.

**3. Minimum sample size.**
PCA eigenvalues are unstable below ~40 tracks (5× the number of features). Below this threshold, bootstrap resampling shows wide confidence intervals on loadings — the weights will vary substantially between runs on similar data.

**4. Feature correlation in this design.**
The three bass-band features (`sub_bass`, `kick_energy`, `bass_harmonics`) are derived from adjacent spectral bands and will be strongly correlated (+0.5 to +0.8). PCA will merge them into 1–2 PCs. Their combined weight could dominate the distance metric. The effective dimensionality of the 8-feature vector is likely **4–5 independent dimensions**, not 8.

Expected correlation pairs:

| Feature Pair | Expected Correlation | Implication |
|---|---|---|
| sub_bass + kick_energy | +0.5 to +0.8 | Adjacent bands; both low-end energy |
| kick_energy + bass_harmonics | +0.4 to +0.7 | Adjacent bands; similar instruments |
| rms_energy + onset_strength | +0.5 to +0.7 | Both capture energy density |
| brightness + sub_bass_energy | -0.4 to -0.7 | High treble = less bass energy |
| percussiveness + onset_strength | +0.5 to +0.8 | Percussion produces strong onsets |
| bpm + onset_strength | +0.3 to +0.6 | Faster music has more onsets/second |

---

## Recommended Algorithm: Adaptive Weighted K-Means

A three-layer approach, degrading gracefully based on library size:

```
n_tracks < 20:          uniform weights (0.125 each)
20 ≤ n_tracks < 40:     heuristic genre weights (if genre known), else uniform
40 ≤ n_tracks < 80:     PCA communality weights + bootstrap stability check
n_tracks ≥ 80:          PCA communality weights → EWKM per-cluster refinement
```

### Layer 1 — PCA Communality Weights (global, initial)

As above, with a **bootstrap stability check**: resample 200×, compute weights each time (with Procrustes alignment to handle PC sign/order flips), report 95% CI per feature. If max CI width > 0.3, fall back to heuristic weights. This prevents overfitting to small libraries.

### Layer 2 — EWKM Per-Cluster Weight Refinement

EWKM (Jing, Ng, Huang 2007, IEEE TKDE) co-optimises cluster assignments **and** per-cluster feature weights simultaneously.

**Objective** (minimise):
```
sum_k sum_{i∈C_k} sum_j  w[k,j] · (x_i[j] - centroid_k[j])²
  +  γ · sum_k sum_j  w[k,j] · log(w[k,j])
```

The second term is an entropy regulariser that prevents degenerate solutions where one feature gets all the weight. The per-cluster weight update is a closed-form softmax:

```python
def ewkm_weight_update(X, labels, centroids, gamma=1.0):
    n_clusters, n_features = centroids.shape
    weights = np.zeros((n_clusters, n_features))
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() == 0:
            weights[k] = 1.0 / n_features
            continue
        # Within-cluster dispersion per feature
        D = np.mean((X[mask] - centroids[k])**2, axis=0)
        # Softmax with temperature gamma (lower D → higher weight)
        log_w = -D / gamma
        log_w -= log_w.max()   # numerical stability
        weights[k] = np.exp(log_w) / np.exp(log_w).sum()
    return weights
```

The full EWKM loop alternates between:
1. Assign each track to nearest centroid under current per-cluster weights
2. Update centroids (arithmetic mean)
3. Update per-cluster weights (softmax over inverse dispersions)

This naturally discovers that **kick_energy** dominates within a techno cluster and **onset_strength near-zero** dominates within an ambient cluster — without any genre labels.

### Layer 3 — Heuristic Genre Profiles (fallback)

When the library is too small for reliable PCA weights, or when the user explicitly specifies a genre:

```python
# Order: [bpm, rms_energy, brightness, sub_bass, kick_energy, bass_harmonics, percussiveness, onset_strength]
GENRE_WEIGHTS = {
    "techno":  [0.20, 0.18, 0.10, 0.12, 0.20, 0.08, 0.07, 0.05],
    "house":   [0.18, 0.12, 0.12, 0.08, 0.15, 0.15, 0.10, 0.10],
    "ambient": [0.05, 0.25, 0.20, 0.08, 0.05, 0.12, 0.05, 0.20],
    "dnb":     [0.25, 0.12, 0.08, 0.15, 0.10, 0.10, 0.12, 0.08],
}
```

Rationale by genre:

| Genre | Key discriminating features | Notes |
|---|---|---|
| **Techno** | BPM + kick_energy (equal, 0.20 each) | Four-on-the-floor kick is genre-signature; tight BPM range 130–145 |
| **House** | bass_harmonics + BPM (0.15 each) | Walking basslines as important as kick; slightly wider BPM range |
| **Ambient** | rms_energy (0.25) + brightness (0.20) | Near-zero onset and low RMS define the genre; BPM nearly irrelevant |
| **DnB** | BPM (0.25, strongest single discriminator) | 160–180 BPM is genre-defining; sub_bass and percussiveness secondary |

**Genre detection without labels** — check in order:
1. ID3 TCON tag via mutagen (already in stack)
2. Parent folder name (`/techno/`, `/house/`, etc.)
3. BPM range heuristic: `< 100` → ambient, `118–132` → house, `132–155` → techno, `≥ 160` → DnB

---

## Module Design

```
playchitect/core/weighting.py
├── WeightProfile           dataclass: weights + metadata (source, genre, n_tracks, ci_width)
├── learn_weights_pca()     data-driven weights with bootstrap stability check
├── get_heuristic_weights() genre-specific expert profiles
├── get_uniform_weights()   fallback: 1/8 each
├── ewkm_refine()           per-cluster weight optimisation
└── select_weights()        dispatch: picks strategy based on n_tracks + genre hint
```

`clustering.py` change: `cluster_by_features()` calls `select_weights()` to get a `WeightProfile`, scales features by `sqrt(weights)`, runs KMeans, then optionally calls `ewkm_refine()`.

### Full Pseudocode

```python
def select_weights(X_scaled, genre=None):
    n = len(X_scaled)
    if genre is not None:
        return WeightProfile(weights=get_heuristic_weights(genre), source="heuristic")
    if n >= 40:
        w = learn_weights_pca(X_scaled, n_bootstrap=200, ci_threshold=0.3)
        if w is not None:
            return WeightProfile(weights=w, source="pca")
    if n >= 20 and genre is not None:
        return WeightProfile(weights=get_heuristic_weights(genre), source="heuristic")
    return WeightProfile(weights=get_uniform_weights(8), source="uniform")

def cluster_by_features_weighted(metadata_dict, intensity_dict, genre=None):
    # ... build X as before (n_tracks, 8) ...
    X_scaled = StandardScaler().fit_transform(X)
    profile = select_weights(X_scaled, genre)
    X_weighted = X_scaled * np.sqrt(profile.weights)
    labels = KMeans(n_clusters=k).fit_predict(X_weighted)
    if len(X) >= 80:
        labels, per_cluster_weights = ewkm_refine(X_scaled, labels, gamma=1.0)
    return build_cluster_results(...)
```

---

## Alternatives Considered and Rejected

| Approach | Verdict |
|---|---|
| Variance-based weighting | Collapses to uniform after StandardScaler — useless |
| Correlation drop + equal weights | Loses information; arbitrary threshold |
| Laplacian Score (He et al. 2005) | Theoretically superior to PCA for unsupervised clustering, but requires a k-NN graph and is harder to implement. Good future upgrade path. |
| UMAP / t-SNE | Visualisation only; non-linear, produces no per-feature weights |
| Mahalanobis distance | Fully decorrelates features but all features become equal weight post-transform |
| Supervised RF importance | Best when ≥30% of library has genre labels; worth adding as upgrade path in a later milestone |

---

## Open Questions for Review

1. **EWKM gamma hyperparameter**: γ = 1.0 is proposed as the default. Without ground truth labels, CV uses silhouette score as a proxy. Is γ ∈ {0.5, 1.0, 2.0} a reasonable search space, or should γ scale with the feature dispersion range?

2. **Bass-band feature redundancy**: `sub_bass`, `kick_energy`, `bass_harmonics` will be strongly correlated. Should we (a) keep all three and let EWKM down-weight redundant ones per cluster, (b) merge into a single `low_end_energy` composite before this module, or (c) apply a VIF check at weight-learning time and drop features with VIF > 5?

3. **Bootstrap Procrustes alignment**: Correct CI computation for PCA weights requires aligning PC sign/order across resamples. This is non-trivial. Alternative: compare learned weights against uniform via KL divergence — if KL < threshold, treat as uninformative and fall back. Is this simpler alternative acceptable?

4. **Module boundary**: EWKM tightly couples weight optimisation with K-means assignment. Should EWKM live in `clustering.py` rather than `weighting.py`, with `weighting.py` handling only the initial weight computation?

---

## References

- Jing, L., Ng, M.K., Huang, J.Z. (2007). An Entropy Weighting k-Means Algorithm for Subspace Clustering of High-Dimensional Sparse Data. *IEEE TKDE* 19(8). doi:10.1109/TKDE.2007.1048
- Amorim, R.C. (2016). A Survey on Feature Weighting Based K-Means Algorithms. *J. Classification* 33, 210–242.
- He, X., Cai, D., Niyogi, P. (2005). Laplacian Score for Feature Selection. *NIPS 18*.
- Hair, J.F. et al. (1998). *Multivariate Data Analysis* (5th ed.) — sample size guidelines for PCA.
- Ahn, J., Marron, J.S. (2010). The maximal data piling direction for discrimination. *Biometrika* 97(1).
