# Research: Audio Mood & Emotion Detection

This document outlines the strategy for implementing mood-aware track analysis and clustering in Playchitect using Essentia and MusiCNN.

## Current Infrastructure
Playchitect already employs the **MSD-MusiCNN** (`msd-musicnn-1.pb`) model for semantic embeddings. This model provides 50 labels that include basic mood descriptors as a by-product:
- **Available Mood Tags**: `beautiful`, `chillout`, `Mellow`, `chill`, `ambient`, `party`, `easy listening`, `sexy`, `catchy`, `sad`, `happy`.

## Recommended Models

### 1. Moods MIREX (The "Easy Win")
- **Model**: `moods_mirex-msd-musicnn-1.pb`
- **Size**: 95 KB (Header only).
- **Backbone**: `msd-musicnn-1` (Already used).
- **Output**: 5 mutually exclusive clusters:
    - **Cluster 1**: Passionate, rousing, confident, boisterous, rowdy.
    - **Cluster 2**: Rollicking, cheerful, fun, sweet, amiable.
    - **Cluster 3**: Literate, poignant, wistful, bittersweet, brooding.
    - **Cluster 4**: Humorous, silly, quirky, whimsical, witty.
    - **Cluster 5**: Aggressive, fiery, tense, intense, volatile.
- **Why**: Extremely lightweight and requires zero additional audio loading time if integrated into the existing MusiCNN pipeline.

### 2. MTG-Jamendo Mood & Theme (High Resolution)
- **Model**: `mtg_jamendo_moodtheme-discogs-effnet-1.pb`
- **Size**: 2.7 MB + 18.4 MB (Discogs-EffNet backbone).
- **Output**: 56 multi-label sigmoid classes.
- **Relevant Tags**: `calm`, `dark`, `deep`, `dramatic`, `energetic`, `epic`, `groovy`, `happy`, `heavy`, `inspiring`, `meditative`, `melancholic`, `motivational`, `party`, `powerful`, `relaxing`, `romantic`, `sad`, `upbeat`, `uplifting`.
- **Why**: Much more accurate for DJ-specific "vibe" mapping (e.g., "dark" vs. "uplifting").

## Implementation Strategy

1. **Short-Term (Milestone 7)**:
    - Integrate **Moods MIREX** into `EmbeddingExtractor`.
    - Map the 5 MIREX clusters to internal "Vibe" labels.
    - Use these labels as secondary features in the K-means clustering logic.

2. **Long-Term (Milestone 9)**:
    - Add **MTG-Jamendo** as an optional `[mood]` dependency.
    - Implement a "Mood Profile" visualization in the GUI.
    - Allow users to filter or cluster specifically by deep labels (e.g., "Find tracks that are >70% Dark").

## Model File Comparison

| File | Size | Notes |
| :--- | :--- | :--- |
| `msd-musicnn-1.pb` | 3.2 MB | Backbone (already implemented) |
| `moods_mirex-msd-musicnn-1.pb` | 95 KB | MIREX head |
| `discogs-effnet-bs64-1.pb` | 18.4 MB | EffNet backbone (Jamendo only) |
| `mtg_jamendo_moodtheme-effnet-1.pb` | 2.7 MB | Jamendo head |
