"""
Utility for scoped warning suppression.
"""

import logging
import warnings
from collections.abc import Generator
from contextlib import contextmanager

# Loggers that emit noisy backend-negotiation messages during audio loading.
_NOISY_AUDIO_LOGGERS: tuple[str, ...] = ("audioread", "soundfile", "librosa")


@contextmanager
def suppress_librosa_warnings() -> Generator[None]:
    """
    Context manager to surgically suppress noisy librosa and audioread warnings.

    Specifically targets backend transition warnings (Issue #94) without
    affecting global warning state.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
        warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
        warnings.filterwarnings("ignore", category=UserWarning, module="audioread")
        warnings.filterwarnings("ignore", message="PySoundFile failed.*")
        yield


@contextmanager
def suppress_audio_log_warnings() -> Generator[None]:
    """
    Context manager that raises the log level of noisy audio-backend loggers
    to ERROR for the duration of the block.

    This suppresses WARNING-level records emitted by ``audioread`` and
    ``soundfile`` during backend negotiation (e.g. "PySoundFile failed",
    "Trying audioread…") without altering the global logging configuration.

    Do not use this to hide *unexpected* errors — only wrap calls where
    backend-negotiation chatter is known and harmless.
    """
    loggers = [logging.getLogger(name) for name in _NOISY_AUDIO_LOGGERS]
    original_levels = [lg.level for lg in loggers]
    try:
        for lg in loggers:
            if lg.level < logging.ERROR:
                lg.setLevel(logging.ERROR)
        yield
    finally:
        for lg, level in zip(loggers, original_levels):
            lg.setLevel(level)
