"""
Utility for scoped warning suppression.
"""

import warnings
from collections.abc import Generator
from contextlib import contextmanager


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
