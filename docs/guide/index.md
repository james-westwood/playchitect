# What is Playchitect?

Playchitect transforms DJ playlist creation from rigid BPM-based grouping to **intelligent multi-dimensional clustering**.

Traditional DJ software groups tracks solely by tempo (e.g., "all 125 BPM tracks"). The problem is that a 125 BPM ambient intro sounds nothing like a 125 BPM hard techno track. Playchitect solves this by analyzing how tracks *actually sound*, creating playlists that feel coherent and flow naturally.

## The Core Problem

Most DJ library management tools stop at BPM and Key. While essential, these metrics ignore the texture, energy, and "vibe" of a track.

*   **BPM**: Tells you the speed, not the intensity.
*   **Key**: Tells you the harmonic compatibility, not the mood.
*   **Genre tags**: Often inconsistent or too broad (e.g., "Techno" can mean anything from Dub Techno to Hardgroove).

## The Solution

Playchitect uses **K-means clustering** on a rich set of audio features extracted via `librosa`. It analyzes:

*   **BPM**: Tempo is still important, but it's just one factor.
*   **Spectral Centroid**: Measures "brightness" (treble content). High centroid = bright/airy; Low centroid = dark/muffled.
*   **High-Frequency Energy**: Specific energy above 8kHz (hi-hats, shakers).
*   **RMS Energy**: The overall "loudness" or density of the track.
*   **Percussiveness**: The strength of the beat/transients.
*   **Bass Energy**: Low-frequency content.

By combining these into a multi-dimensional feature vector, Playchitect can distinguish between:
*   **Atmospheric/Deep**: Low brightness, low percussiveness.
*   **Driving/Peak**: High RMS, high percussiveness, high brightness.
*   **Hypnotic/Tool**: Steady RMS, medium brightness, high percussiveness.

## Key Features

*   **Intelligent Clustering**: Automatically groups tracks that belong together based on sonic character.
*   **Intensity Sequencing**: "Ramp mode" orders tracks within a playlist by their hardness score, creating a smooth energy arc from warm-up to peak.
*   **Opener/Closer Selection**: Identifies tracks with long intros (good for starting) or specific outro characteristics.
*   **Native GNOME GUI**: A modern GTK4 + libadwaita interface that fits perfectly on Linux desktops.
*   **DJ-Ready Export**: Generates standard `.m3u` playlists and frame-accurate `.cue` sheets.
