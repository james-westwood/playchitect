"""Pytest fixtures for performance benchmarking."""

import random
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from mutagen import File as MutagenFile
from mutagen.flac import FLAC, FLACNoHeaderError
from mutagen.id3 import ID3NoHeaderError

SYNTHETIC_SAMPLE_RATE = 22050  # Hz — standard audio sample rate; kept low for smaller files


@pytest.fixture(scope="module")
def synthetic_library(tmp_path_factory: pytest.TempPathFactory):
    """
    A fixture that generates a synthetic music library of N tiny audio files
    with fake metadata for benchmarking purposes.
    """

    def _factory(n_tracks: int, track_duration_seconds: float = 0.5) -> Path:
        base_dir = tmp_path_factory.mktemp(f"synthetic_library_{n_tracks}_tracks")
        sample_rate = SYNTHETIC_SAMPLE_RATE

        for i in range(n_tracks):
            artist = f"Artist_{random.randint(1, 100)}"
            album = f"Album_{random.randint(1, 50)}"
            title = f"Track_{i:05d}"
            genre = random.choice(["Electronic", "Techno", "House", "Ambient"])
            bpm = random.randint(100, 140)

            track_dir = base_dir / artist / album
            track_dir.mkdir(parents=True, exist_ok=True)
            file_path = track_dir / f"{title}.flac"

            # Generate stereo sine wave audio (minimal content)
            time = np.linspace(
                0, track_duration_seconds, int(sample_rate * track_duration_seconds), endpoint=False
            )
            frequency_left = random.uniform(200, 800)
            frequency_right = random.uniform(200, 800)
            audio_data_left = 0.5 * np.sin(2 * np.pi * frequency_left * time)
            audio_data_right = 0.5 * np.sin(2 * np.pi * frequency_right * time)
            audio_data = np.stack((audio_data_left, audio_data_right), axis=-1)

            # Save as FLAC
            sf.write(file_path, audio_data, sample_rate)

            # Add metadata using mutagen
            try:
                audio = MutagenFile(file_path)
                if audio is None:
                    audio = FLAC(file_path)
            except (ID3NoHeaderError, FLACNoHeaderError):
                audio = FLAC(file_path)

            audio["artist"] = artist
            audio["album"] = album
            audio["title"] = title
            audio["genre"] = genre
            audio["bpm"] = str(bpm)

            audio.save()

        return base_dir

    return _factory


@pytest.fixture
def mock_numpy_array_input() -> np.ndarray:
    """Fixture to generate a mock numpy array representing raw audio samples."""
    return np.random.rand(22050 * 5, 2).astype(np.float32)  # 5 seconds of stereo audio


@pytest.fixture
def mock_features_array_input() -> np.ndarray:
    """Fixture to generate a mock numpy array of features for clustering."""
    return np.random.rand(1000, 10)  # 1000 tracks, 10 features each
