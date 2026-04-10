"""
Utility for scoped warning and stderr suppression during audio analysis.

Handles three layers of noise:
- Python ``warnings`` from librosa/audioread (Issue #94)
- Python ``logging`` from soundfile/audioread backends (Issue #94)
- C-level stderr from mpg123 and similar decoders (Issue #95)
"""

import logging
import os
import warnings
from collections.abc import Generator
from contextlib import contextmanager

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

    Suppresses WARNING-level records emitted by ``audioread`` and
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


@contextmanager
def suppress_c_stderr() -> Generator[None]:
    """
    Context manager that redirects C-level stderr (fd 2) to /dev/null.

    C extensions like mpg123 write decoding diagnostics directly to file
    descriptor 2, bypassing Python's warnings/logging entirely. This captures
    and discards that noise during ``librosa.load()`` calls (Issue #95).

    Thread-safe: only affects the calling thread's process, and is only active
    for the duration of the context manager.
    """
    original_stderr_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(original_stderr_fd, 2)
        os.close(devnull)
        os.close(original_stderr_fd)
