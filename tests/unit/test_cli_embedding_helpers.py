"""
Unit tests for the embedding diagnostic helpers in playchitect.cli.commands.

These cover the pure helpers directly, without invoking the ``scan`` command
or requiring essentia to be installed. The end-to-end wiring of the same
helpers through the CLI is covered separately by
``tests/integration/test_cli_embedding_diagnostics.py``.
"""

from collections import Counter

import pytest

from playchitect.cli.commands import (
    EMBEDDING_REASON_TOO_SHORT,
    EMBEDDING_WARNING_LOG_LIMIT,
    _classify_embedding_failure,
    _dominant_embedding_failure,
    _report_embedding_outcome,
)
from playchitect.core.embedding_extractor import (
    _MUSICNN_MIN_AUDIO_SECONDS,
    AudioTooShortForEmbeddingError,
)

OTHER_REASON = "DecodeError"
SECOND_REASON = "MemoryError"


class _StubDecodeError(RuntimeError):
    """Stand-in for a decode failure raised by the audio backend."""


class _NestedTooShortError(AudioTooShortForEmbeddingError):
    """A more specific too-short error, to check isinstance-based bucketing."""


class TestClassifyEmbeddingFailure:
    """Tests for _classify_embedding_failure."""

    def test_too_short_error_maps_to_the_too_short_reason(self) -> None:
        exc = AudioTooShortForEmbeddingError("clip is 1.20s, needs 2.960s")

        assert _classify_embedding_failure(exc) == EMBEDDING_REASON_TOO_SHORT

    def test_subclass_of_too_short_error_also_maps_to_the_too_short_reason(self) -> None:
        # The check is isinstance-based, so a narrower too-short error still
        # lands in the actionable bucket rather than being split off.
        exc = _NestedTooShortError("clip is 0.50s")

        assert _classify_embedding_failure(exc) == EMBEDDING_REASON_TOO_SHORT

    def test_generic_exception_buckets_by_type_name(self) -> None:
        assert _classify_embedding_failure(ValueError("bad frame count")) == "ValueError"

    def test_plain_runtime_error_is_not_treated_as_too_short(self) -> None:
        # AudioTooShortForEmbeddingError subclasses RuntimeError, so a bare
        # RuntimeError must not be swept into the too-short bucket.
        reason = _classify_embedding_failure(RuntimeError("model failed to load"))

        assert reason == "RuntimeError"
        assert reason != EMBEDDING_REASON_TOO_SHORT

    def test_same_type_with_different_messages_classifies_identically(self) -> None:
        # The whole point of bucketing by type: ten distinct decode messages
        # must aggregate into one countable reason.
        messages = [
            "cannot decode /music/a.flac: invalid block",
            "cannot decode /music/b.flac: truncated stream",
            "cannot decode /music/c.flac: unsupported sample rate 96000",
        ]
        reasons = {_classify_embedding_failure(_StubDecodeError(m)) for m in messages}

        assert reasons == {"_StubDecodeError"}
        assert len(reasons) == 1

    def test_message_text_never_leaks_into_the_reason(self) -> None:
        # Reasons are used as Counter keys, so they must not carry per-track
        # detail such as file paths.
        exc = _StubDecodeError("cannot decode /music/private_set/track_01.flac")

        assert "/music" not in _classify_embedding_failure(exc)

    def test_subclass_of_a_bucketed_type_gets_its_own_bucket(self) -> None:
        # type(exc).__name__ resolves to the concrete class, so subclasses do
        # not collapse into their parent's bucket.
        assert _classify_embedding_failure(_StubDecodeError("boom")) == "_StubDecodeError"
        assert _classify_embedding_failure(_StubDecodeError("boom")) != "RuntimeError"

    @pytest.mark.parametrize(
        "exc",
        [
            AudioTooShortForEmbeddingError("too short"),
            _NestedTooShortError("too short"),
            ValueError("bad"),
            _StubDecodeError("bad"),
            RuntimeError(""),
            Exception(""),
        ],
    )
    def test_always_returns_a_non_empty_string(self, exc: Exception) -> None:
        reason = _classify_embedding_failure(exc)

        assert isinstance(reason, str)
        assert reason.strip() != ""


class TestDominantEmbeddingFailure:
    """Tests for _dominant_embedding_failure."""

    def test_returns_the_clear_winner_and_its_count(self) -> None:
        failures = Counter({OTHER_REASON: 7, EMBEDDING_REASON_TOO_SHORT: 2})

        assert _dominant_embedding_failure(failures) == (OTHER_REASON, 7)

    def test_single_entry_counter_returns_that_entry(self) -> None:
        assert _dominant_embedding_failure(Counter({OTHER_REASON: 4})) == (OTHER_REASON, 4)

    def test_three_way_spread_returns_the_largest(self) -> None:
        failures = Counter({OTHER_REASON: 3, EMBEDDING_REASON_TOO_SHORT: 5, SECOND_REASON: 1})

        assert _dominant_embedding_failure(failures) == (EMBEDDING_REASON_TOO_SHORT, 5)

    def test_three_way_spread_does_not_favour_too_short_when_outnumbered(self) -> None:
        # The too-short preference is a tie-break only; it must never beat a
        # strictly more common cause.
        failures = Counter({OTHER_REASON: 9, EMBEDDING_REASON_TOO_SHORT: 8, SECOND_REASON: 2})

        assert _dominant_embedding_failure(failures) == (OTHER_REASON, 9)

    def test_exact_tie_resolves_in_favour_of_too_short(self) -> None:
        failures = Counter({OTHER_REASON: 4, EMBEDDING_REASON_TOO_SHORT: 4})

        assert _dominant_embedding_failure(failures) == (EMBEDDING_REASON_TOO_SHORT, 4)

    def test_exact_tie_favours_too_short_regardless_of_insertion_order(self) -> None:
        # Counter iteration follows insertion order, so the tie-break has to
        # hold with the too-short reason recorded either first or last.
        too_short_first: Counter[str] = Counter()
        too_short_first[EMBEDDING_REASON_TOO_SHORT] = 3
        too_short_first[OTHER_REASON] = 3

        too_short_last: Counter[str] = Counter()
        too_short_last[OTHER_REASON] = 3
        too_short_last[EMBEDDING_REASON_TOO_SHORT] = 3

        assert _dominant_embedding_failure(too_short_first) == (EMBEDDING_REASON_TOO_SHORT, 3)
        assert _dominant_embedding_failure(too_short_last) == (EMBEDDING_REASON_TOO_SHORT, 3)

    def test_tie_between_two_non_too_short_reasons_is_deterministic(self) -> None:
        # With neither reason actionable, the key collapses to (count, False)
        # for both and max() keeps the first one it saw, i.e. the reason that
        # was recorded first. The rule is arbitrary but stable -- this test
        # pins the observed behaviour so a change is noticed, not because
        # either winner is inherently more correct.
        first_recorded: Counter[str] = Counter()
        first_recorded[OTHER_REASON] = 2
        first_recorded[SECOND_REASON] = 2
        assert _dominant_embedding_failure(first_recorded) == (OTHER_REASON, 2)

        reversed_order: Counter[str] = Counter()
        reversed_order[SECOND_REASON] = 2
        reversed_order[OTHER_REASON] = 2
        assert _dominant_embedding_failure(reversed_order) == (SECOND_REASON, 2)

    def test_repeated_calls_on_the_same_counter_are_stable(self) -> None:
        failures = Counter({OTHER_REASON: 2, SECOND_REASON: 2, EMBEDDING_REASON_TOO_SHORT: 1})

        results = {_dominant_embedding_failure(failures) for _ in range(10)}

        assert len(results) == 1

    def test_empty_counter_raises_value_error(self) -> None:
        # Documented behaviour, not a design goal: max() over an empty
        # Counter raises. The scan call site only reaches this helper after
        # at least one failure has been recorded, so the path is unreachable
        # in production, but the behaviour is pinned here so any future
        # caller knows an empty Counter is not handled.
        with pytest.raises(ValueError):
            _dominant_embedding_failure(Counter())

    def test_does_not_mutate_the_counter(self) -> None:
        failures = Counter({OTHER_REASON: 3, EMBEDDING_REASON_TOO_SHORT: 1})

        _dominant_embedding_failure(failures)

        assert failures == Counter({OTHER_REASON: 3, EMBEDDING_REASON_TOO_SHORT: 1})


class TestReportEmbeddingOutcomeSuccess:
    """Tests for _report_embedding_outcome on successful runs."""

    def test_nothing_attempted_emits_nothing(self, capsys: pytest.CaptureFixture[str]) -> None:
        _report_embedding_outcome(attempted=0, succeeded=0, failures=Counter())

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_full_success_reports_both_counts(self, capsys: pytest.CaptureFixture[str]) -> None:
        _report_embedding_outcome(attempted=12, succeeded=12, failures=Counter())

        captured = capsys.readouterr()
        assert "12" in captured.out
        assert captured.err == ""

    def test_full_success_does_not_raise(self) -> None:
        # No SystemExit: the run must continue.
        _report_embedding_outcome(attempted=5, succeeded=5, failures=Counter())

    def test_partial_success_reports_succeeded_and_attempted(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _report_embedding_outcome(
            attempted=10, succeeded=7, failures=Counter({EMBEDDING_REASON_TOO_SHORT: 3})
        )

        out = capsys.readouterr().out
        assert "7" in out
        assert "10" in out

    def test_partial_success_emits_no_error_output(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A partial failure is survivable; the run continues on what it got.
        _report_embedding_outcome(
            attempted=10, succeeded=7, failures=Counter({EMBEDDING_REASON_TOO_SHORT: 3})
        )

        assert capsys.readouterr().err == ""

    def test_partial_success_does_not_exit(self) -> None:
        _report_embedding_outcome(attempted=4, succeeded=1, failures=Counter({OTHER_REASON: 3}))

    def test_single_success_out_of_many_is_still_survivable(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Boundary between "partial" and "total" failure: exactly one success.
        _report_embedding_outcome(attempted=100, succeeded=1, failures=Counter({OTHER_REASON: 99}))

        captured = capsys.readouterr()
        assert "100" in captured.out
        assert captured.err == ""


class TestReportEmbeddingOutcomeTotalFailure:
    """Tests for _report_embedding_outcome when nothing could be embedded."""

    def test_zero_successes_exits_with_status_one(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            _report_embedding_outcome(
                attempted=6, succeeded=0, failures=Counter({EMBEDDING_REASON_TOO_SHORT: 6})
            )

        assert excinfo.value.code == 1

    def test_zero_successes_reports_the_attempted_count(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            _report_embedding_outcome(
                attempted=6, succeeded=0, failures=Counter({EMBEDDING_REASON_TOO_SHORT: 6})
            )

        err = capsys.readouterr().err
        assert "6" in err
        assert "0" in err

    def test_zero_successes_names_the_dominant_reason_and_its_count(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        failures = Counter({EMBEDDING_REASON_TOO_SHORT: 5, OTHER_REASON: 2})

        with pytest.raises(SystemExit):
            _report_embedding_outcome(attempted=7, succeeded=0, failures=failures)

        err = capsys.readouterr().err
        assert EMBEDDING_REASON_TOO_SHORT in err
        assert "5" in err
        assert "7" in err

    def test_zero_successes_names_a_non_too_short_dominant_reason(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        failures = Counter({OTHER_REASON: 8, EMBEDDING_REASON_TOO_SHORT: 1})

        with pytest.raises(SystemExit):
            _report_embedding_outcome(attempted=9, succeeded=0, failures=failures)

        err = capsys.readouterr().err
        assert OTHER_REASON in err
        assert "8" in err

    def test_zero_successes_cites_the_minimum_duration_when_clips_were_too_short(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            _report_embedding_outcome(
                attempted=3, succeeded=0, failures=Counter({EMBEDDING_REASON_TOO_SHORT: 3})
            )

        err = capsys.readouterr().err
        assert f"{_MUSICNN_MIN_AUDIO_SECONDS:.3f}" in err

    def test_minimum_duration_is_cited_even_when_too_short_is_not_dominant(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The line is gated on the too-short count being non-zero, not on it
        # winning: a user with one short clip among many still benefits.
        failures = Counter({OTHER_REASON: 9, EMBEDDING_REASON_TOO_SHORT: 1})

        with pytest.raises(SystemExit):
            _report_embedding_outcome(attempted=10, succeeded=0, failures=failures)

        err = capsys.readouterr().err
        assert f"{_MUSICNN_MIN_AUDIO_SECONDS:.3f}" in err

    def test_minimum_duration_is_suppressed_when_nothing_was_too_short(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Printing a duration hint when no clip was short would misdirect the
        # user towards a fix that cannot help them.
        failures = Counter({OTHER_REASON: 4, SECOND_REASON: 2})

        with pytest.raises(SystemExit):
            _report_embedding_outcome(attempted=6, succeeded=0, failures=failures)

        err = capsys.readouterr().err
        assert f"{_MUSICNN_MIN_AUDIO_SECONDS:.3f}" not in err
        assert "MusiCNN needs at least" not in err

    def test_suppressing_the_duration_line_keeps_the_rest_of_the_diagnostic(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        failures = Counter({OTHER_REASON: 4})

        with pytest.raises(SystemExit):
            _report_embedding_outcome(attempted=4, succeeded=0, failures=failures)

        err = capsys.readouterr().err
        assert OTHER_REASON in err
        assert "4" in err
        assert "--use-embeddings" in err

    def test_zero_successes_suggests_rerunning_without_the_flag(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            _report_embedding_outcome(
                attempted=2, succeeded=0, failures=Counter({EMBEDDING_REASON_TOO_SHORT: 2})
            )

        err = capsys.readouterr().err
        assert "--use-embeddings" in err

    def test_zero_successes_writes_the_diagnostic_to_stderr_not_stdout(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            _report_embedding_outcome(
                attempted=2, succeeded=0, failures=Counter({EMBEDDING_REASON_TOO_SHORT: 2})
            )

        captured = capsys.readouterr()
        assert "--use-embeddings" not in captured.out
        assert "--use-embeddings" in captured.err

    def test_zero_successes_still_emits_the_summary_line_on_stdout(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            _report_embedding_outcome(
                attempted=2, succeeded=0, failures=Counter({EMBEDDING_REASON_TOO_SHORT: 2})
            )

        assert "2" in capsys.readouterr().out

    def test_zero_successes_logs_the_aggregate_at_error_level(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        failures = Counter({OTHER_REASON: 3})

        with caplog.at_level("ERROR", logger="playchitect.cli.commands"):
            with pytest.raises(SystemExit):
                _report_embedding_outcome(attempted=3, succeeded=0, failures=failures)

        assert OTHER_REASON in caplog.text
        assert "3" in caplog.text


class TestEmbeddingHelperConstants:
    """Tests for the module constants the helpers depend on."""

    def test_too_short_reason_is_a_non_empty_human_readable_string(self) -> None:
        assert isinstance(EMBEDDING_REASON_TOO_SHORT, str)
        assert EMBEDDING_REASON_TOO_SHORT.strip() != ""

    def test_too_short_reason_cannot_collide_with_a_type_name_bucket(self) -> None:
        # Type-name buckets are Python identifiers; the too-short reason
        # contains spaces, so the two keyspaces cannot overlap in the Counter.
        assert not EMBEDDING_REASON_TOO_SHORT.isidentifier()

    def test_warning_log_limit_is_a_small_positive_int(self) -> None:
        # At least one warning must reach the user, and the cap has to be
        # finite or the aggregate summary drowns.
        assert isinstance(EMBEDDING_WARNING_LOG_LIMIT, int)
        assert 1 <= EMBEDDING_WARNING_LOG_LIMIT <= 10
