"""
Feature weighting strategies for K-means clustering.

Provides three strategies in a fallback hierarchy:
  1. PCA communality weights  (≥40 tracks, bootstrap-validated)
  2. Heuristic genre weights  (any size, when genre is known)
  3. Uniform weights          (always available)

Optionally followed by EWKM per-cluster weight refinement (≥80 tracks).

Reference: Jing, Ng, Huang (2007) — An Entropy Weighting k-Means Algorithm
for Subspace Clustering of High-Dimensional Sparse Data. IEEE TKDE 19(8).
"""

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# Feature names in the order they appear in the 8-dimensional feature vector.
# Defined here to avoid circular imports with clustering.py; re-exported from
# clustering.py for backwards compatibility.
FEATURE_NAMES: tuple[str, ...] = (
    "bpm",
    "rms_energy",
    "brightness",
    "sub_bass",
    "kick_energy",
    "bass_harmonics",
    "percussiveness",
    "onset_strength",
)

# ── Thresholds ───────────────────────────────────────────────────────────────

_MIN_TRACKS_PCA: int = 40  # Minimum tracks for reliable PCA weights
_MIN_TRACKS_EWKM: int = 80  # Minimum tracks to run EWKM refinement
_N_BOOTSTRAP: int = 200  # Bootstrap resamples for PCA stability check
_CI_THRESHOLD: float = 0.3  # Max 95% CI width before falling back
_PCA_VARIANCE_TARGET: float = 0.90  # Retain PCs explaining this fraction of variance
_EWKM_GAMMA: float = 1.0  # Entropy regularisation temperature
_EWKM_MAX_ITER: int = 100

# ── Genre profiles ───────────────────────────────────────────────────────────
# Order matches FEATURE_NAMES: bpm, rms_energy, brightness, sub_bass,
# kick_energy, bass_harmonics, percussiveness, onset_strength

_GENRE_PROFILES: dict[str, list[float]] = {
    # Techno: four-on-the-floor kick and tight BPM are genre-defining
    "techno": [0.20, 0.18, 0.10, 0.12, 0.20, 0.08, 0.07, 0.05],
    # House: walking bass harmonics as important as kick; warmer feel
    "house": [0.18, 0.12, 0.12, 0.08, 0.15, 0.15, 0.10, 0.10],
    # Ambient: near-zero onset/energy define the genre; BPM nearly irrelevant
    "ambient": [0.05, 0.25, 0.20, 0.08, 0.05, 0.12, 0.05, 0.20],
    # DnB: BPM is the strongest single discriminator (160-180 is genre-defining)
    "dnb": [0.25, 0.12, 0.08, 0.15, 0.10, 0.10, 0.12, 0.08],
}

SUPPORTED_GENRES: tuple[str, ...] = tuple(_GENRE_PROFILES.keys())


# ── Public types ─────────────────────────────────────────────────────────────


@dataclass
class WeightProfile:
    """Feature weights for K-means distance computation."""

    weights: np.ndarray  # shape (8,), sums to 1.0
    source: str  # "pca", "heuristic", or "uniform"
    genre: str | None = None
    n_tracks: int = 0
    ci_width: float | None = None  # max bootstrap CI width; None if not computed

    def as_dict(self) -> dict[str, float]:
        """Return weights as a named dict for logging/reporting."""
        return {name: float(self.weights[i]) for i, name in enumerate(FEATURE_NAMES)}


# ── Public API ───────────────────────────────────────────────────────────────


def select_weights(
    X_scaled: np.ndarray,
    genre: str | None = None,
    random_state: int = 42,
) -> WeightProfile:
    """
    Select the best available weight strategy for the given data.

    Fallback hierarchy:
      1. PCA communality weights (≥40 tracks, bootstrap-stable)
      2. Heuristic genre weights (when genre is known)
      3. Uniform weights (1/8 each)

    Args:
        X_scaled: Standardised feature matrix, shape (n_tracks, 8)
        genre: Optional genre hint — one of 'techno', 'house', 'ambient', 'dnb'
        random_state: Seed for reproducible bootstrap

    Returns:
        WeightProfile with selected weights and source metadata
    """
    n_tracks = len(X_scaled)

    if n_tracks >= _MIN_TRACKS_PCA:
        profile = learn_weights_pca(X_scaled, random_state=random_state)
        if profile is not None:
            top = _fmt_top3(profile)
            logger.info(f"PCA weights (ci={profile.ci_width:.3f}, n={n_tracks}): {top}")
            return profile
        logger.warning("PCA bootstrap CI too wide; falling back to heuristic/uniform")

    if genre is not None:
        profile = get_heuristic_weights(genre, n_tracks=n_tracks)
        logger.info(f"Heuristic weights for genre '{genre}': {_fmt_top3(profile)}")
        return profile

    logger.info(f"Using uniform weights (n_tracks={n_tracks} < {_MIN_TRACKS_PCA})")
    return get_uniform_weights(n_tracks=n_tracks)


def learn_weights_pca(
    X_scaled: np.ndarray,
    n_bootstrap: int = _N_BOOTSTRAP,
    ci_threshold: float = _CI_THRESHOLD,
    variance_target: float = _PCA_VARIANCE_TARGET,
    random_state: int = 42,
) -> WeightProfile | None:
    """
    Derive per-feature weights from PCA communalities with bootstrap validation.

    Communality for feature j (retaining k components):
        h_j = sum_l ( (components[l,j] * sqrt(eigenvalue[l]))^2 * evr[l] )

    Weights are normalised so sum(w) = 1.0 and applied as a scaled Euclidean
    distance in K-Means: d(x,c)^2 = sum_j w[j] * (x[j] - c[j])^2.
    Implementation: X_weighted = X_scaled * sqrt(w), then standard KMeans.

    Communalities are sign-invariant (squared loadings), so no Procrustes
    alignment is needed across bootstrap resamples.

    Args:
        X_scaled: Standardised feature matrix (n_tracks, n_features)
        n_bootstrap: Number of bootstrap resamples for stability check
        ci_threshold: Max allowed 95% CI width; returns None if exceeded
        variance_target: Retain enough PCs to explain this fraction of variance
        random_state: Seed for reproducible bootstrap

    Returns:
        WeightProfile if weights are stable, None if bootstrap CI too wide
    """
    n_tracks, n_features = X_scaled.shape

    # Fit on full dataset to determine n_components
    pca_full = PCA(n_components=n_features)
    pca_full.fit(X_scaled)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, variance_target)) + 1
    n_components = min(n_components, n_features)

    weights = _communality_weights(pca_full, n_components)

    # Bootstrap stability check
    rng = np.random.default_rng(random_state)
    boot_weights = np.zeros((n_bootstrap, n_features))
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_tracks, size=n_tracks)
        pca_b = PCA(n_components=n_components)
        pca_b.fit(X_scaled[idx])
        boot_weights[b] = _communality_weights(pca_b, n_components)

    ci_width = float(
        np.max(np.percentile(boot_weights, 97.5, axis=0) - np.percentile(boot_weights, 2.5, axis=0))
    )

    if ci_width > ci_threshold:
        logger.debug(f"PCA CI width {ci_width:.3f} > threshold {ci_threshold}; unstable")
        return None

    return WeightProfile(
        weights=weights,
        source="pca",
        n_tracks=n_tracks,
        ci_width=ci_width,
    )


def get_heuristic_weights(genre: str, n_tracks: int = 0) -> WeightProfile:
    """
    Return expert heuristic weights for a known genre.

    Args:
        genre: One of 'techno', 'house', 'ambient', 'dnb'
        n_tracks: Track count (stored in profile metadata only)

    Returns:
        WeightProfile with normalised genre weights

    Raises:
        ValueError: If genre is not in SUPPORTED_GENRES
    """
    key = genre.lower()
    if key not in _GENRE_PROFILES:
        raise ValueError(f"Unknown genre '{genre}'. Supported: {', '.join(SUPPORTED_GENRES)}")

    raw = np.array(_GENRE_PROFILES[key], dtype=float)
    return WeightProfile(
        weights=raw / raw.sum(),
        source="heuristic",
        genre=key,
        n_tracks=n_tracks,
    )


def get_uniform_weights(n_tracks: int = 0) -> WeightProfile:
    """Return uniform weights (1/n_features for all features)."""
    n = len(FEATURE_NAMES)
    return WeightProfile(
        weights=np.ones(n) / n,
        source="uniform",
        n_tracks=n_tracks,
    )


def ewkm_refine(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    gamma: float = _EWKM_GAMMA,
    max_iter: int = _EWKM_MAX_ITER,
) -> tuple[np.ndarray, np.ndarray]:
    """
    EWKM per-cluster weight refinement (Jing, Ng, Huang 2007).

    Minimises:
        sum_k sum_{i in C_k} sum_j  w[k,j] * (x[i,j] - c[k,j])^2
          + gamma * sum_k sum_j  w[k,j] * log(w[k,j])

    The entropy term (gamma) prevents degenerate all-weight-on-one-feature
    solutions. Per-cluster weight update (closed form):
        w[k,j] = exp(-D[k,j] / gamma) / sum_l exp(-D[k,l] / gamma)
    where D[k,j] is the mean squared distance on feature j within cluster k.
    Features with low within-cluster dispersion receive high weight.

    Args:
        X_scaled: Standardised feature matrix (n_tracks, n_features)
        labels: Initial cluster assignments (n_tracks,)
        centroids: Initial centroids in X_scaled space (n_clusters, n_features)
        gamma: Entropy regularisation temperature (higher = more uniform weights)
        max_iter: Maximum iterations

    Returns:
        Tuple of (labels, per_cluster_weights) where per_cluster_weights has
        shape (n_clusters, n_features) and each row sums to 1.0
    """
    n_clusters, n_features = centroids.shape
    labels = labels.copy()
    centroids = centroids.copy()
    per_cluster_weights = np.full((n_clusters, n_features), 1.0 / n_features)

    for iteration in range(max_iter):
        old_labels = labels.copy()

        # Step 1: Assign tracks to nearest centroid (vectorised)
        # diff: (n_tracks, n_clusters, n_features)
        diff = X_scaled[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        # dists: (n_tracks, n_clusters)
        dists = np.sum(per_cluster_weights[np.newaxis, :, :] * diff**2, axis=2)
        labels = np.argmin(dists, axis=1)

        # Step 2: Update centroids
        for k in range(n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                centroids[k] = X_scaled[mask].mean(axis=0)

        # Step 3: Update per-cluster weights
        per_cluster_weights = _ewkm_weight_update(X_scaled, labels, centroids, gamma)

        if np.all(labels == old_labels):
            logger.debug(f"EWKM converged after {iteration + 1} iterations")
            break

    return labels, per_cluster_weights


# ── Private helpers ───────────────────────────────────────────────────────────


def _communality_weights(pca: PCA, n_components: int) -> np.ndarray:
    """
    Compute normalised communality-based feature weights from a fitted PCA.

    h_j = sum_l ( (components[l,j] * sqrt(eigenvalue[l]))^2 * evr[l] )
    w_j = h_j / sum(h)
    """
    evr = pca.explained_variance_ratio_[:n_components]
    loadings = pca.components_[:n_components].T * np.sqrt(pca.explained_variance_[:n_components])
    importance = np.sum(loadings**2 * evr[np.newaxis, :], axis=1)
    total = float(importance.sum())
    if total > 0:
        return importance / total
    return np.ones(len(importance)) / len(importance)


def _ewkm_weight_update(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Compute per-cluster weights via softmax over within-cluster feature dispersions."""
    n_clusters, n_features = centroids.shape
    weights = np.zeros((n_clusters, n_features))

    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() == 0:
            weights[k] = 1.0 / n_features
            continue
        D = np.mean((X[mask] - centroids[k]) ** 2, axis=0)
        log_w = -D / gamma
        log_w -= log_w.max()  # numerical stability
        exp_w = np.exp(log_w)
        weights[k] = exp_w / exp_w.sum()

    return weights


def _fmt_top3(profile: WeightProfile) -> str:
    """Format top-3 features by weight for log messages."""
    top = sorted(profile.as_dict().items(), key=lambda x: x[1], reverse=True)[:3]
    return ", ".join(f"{k}={v:.3f}" for k, v in top)
