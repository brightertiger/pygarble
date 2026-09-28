"""Pure statistics behind the non-parametric strategies."""

import zlib

import pytest

from pygarble.gibberish.measures import (
    Lcg,
    bucket,
    cross_parsing,
    ngram_rank_distance,
    permutation_gap,
    primed_compression,
    standardised,
    windows,
)


def test_lcg_outputs_are_fixed():
    rng = Lcg(0)
    assert [rng.below(10) for _ in range(4)] == [0, 1, 6, 4]
    assert rng.state == 7401132627792533940
    rng = Lcg(20260929)
    assert [rng.below(1000) for _ in range(4)] == [37, 772, 391, 356]


def test_lcg_below_stays_in_range():
    rng = Lcg(123)
    assert all(0 <= rng.below(7) < 7 for _ in range(1000))
    assert Lcg(5).below(1) == 0


def test_lcg_seed_wraps_to_64_bits():
    assert Lcg(2**64 + 9).state == Lcg(9).state


def test_lcg_shuffle_is_fixed_fisher_yates():
    items = list("abcde")
    Lcg(0).shuffle(items)
    assert items == ["c", "d", "b", "e", "a"]


def test_cross_parsing_counts_phrases_by_hand():
    # "abc" + "ab": the second phrase stops at the end of the window.
    assert cross_parsing("abcab", "abc") == pytest.approx(2 / 5)
    # Every character is absent, so each is a phrase of length one.
    assert cross_parsing("xyz", "abc") == pytest.approx(1.0)
    # Longer prefixes are found after the first occurrence.
    assert cross_parsing("aba", "abxaba") == pytest.approx(1 / 3)
    assert cross_parsing("the cat", "the cat sat") == pytest.approx(1 / 7)


def test_cross_parsing_mixes_found_and_missing_characters():
    # "ab" | "q" | "ba" against a reference without "q".
    assert cross_parsing("abqba", "abba") == pytest.approx(3 / 5)


def test_primed_compression_uses_raw_deflate_with_dictionary():
    window = "the quick brown fox"
    dictionary = b"jumps over the lazy dog the quick brown fox"
    compressor = zlib.compressobj(
        9, zlib.DEFLATED, -15, 9, zlib.Z_DEFAULT_STRATEGY, dictionary
    )
    size = len(compressor.compress(window.encode()) + compressor.flush())
    assert primed_compression(window, dictionary) == size / len(window)


def test_primed_compression_rewards_text_found_in_dictionary():
    window = "the quick brown fox jumps"
    primed = primed_compression(window, b"the quick brown fox jumps over")
    unprimed = primed_compression(window, b"zzzzzzzzzzzzzzzzzzzzzzzzzzzzzz")
    assert primed < unprimed


def test_ngram_rank_distance_by_hand():
    # Profile of " ab ": " " (2), then " a", " ab", "a", ... (1 each).
    assert ngram_rank_distance("ab", {" ": 0, "a": 1, "b": 2}) == (
        pytest.approx(6 / 9)
    )
    assert ngram_rank_distance("ab", {" ": 0, " a": 1, " ab": 2}) == 0.0
    assert ngram_rank_distance("ab", {" ab": 0, " a": 1, " ": 2}) == (
        pytest.approx(4 / 9)
    )


def test_ngram_rank_distance_short_profile():
    # " a " has five distinct n-grams; a larger table keeps all five.
    ranks = {g: r for r, g in enumerate([" ", " a", " a ", "a", "a ", "y"])}
    assert ngram_rank_distance("a", ranks) == 0.0
    ranks = {g: r for r, g in enumerate(["x", "y", " ", " a", " a ", "a"])}
    assert ngram_rank_distance("a", ranks) == pytest.approx(14 / 30)


def test_permutation_gap_is_deterministic():
    log_probs = {"th": -1.0, "he": -1.0, "e ": -1.0, " c": -2.0}
    first = permutation_gap("the cat sat", log_probs, -10.0)
    assert first == permutation_gap("the cat sat", log_probs, -10.0)
    assert first < 0.0


def test_permutation_gap_leaves_spaces_in_place():
    # Only pairs touching a space cost anything, and there are fewer of
    # them if a space moves to an edge, so any moved space shows up.
    window = "ab cde f"
    log_probs = {}
    for c in "abcdef":
        log_probs[c + " "] = log_probs[" " + c] = -1.0
    gap = permutation_gap(window, log_probs, 0.0, shuffles=20)
    assert gap == pytest.approx(0.0, abs=1e-12)
    # The letters themselves do move.
    log_probs["ab"] = 5.0
    assert permutation_gap(window, log_probs, 0.0, shuffles=20) < 0.0


def test_permutation_gap_values_are_whole_shuffles():
    # "ab" shuffles to "ab" (gap 0) or "ba" (gap -2) each time.
    log_probs = {"ab": -1.0, "ba": -3.0}
    gap = permutation_gap("ab", log_probs, -10.0, shuffles=8)
    assert gap * 8 / -2 == pytest.approx(round(gap * 8 / -2))
    assert -2.0 <= gap <= 0.0


def test_permutation_gap_english_below_gibberish():
    from pygarble.data import BIGRAM_LOG_PROBS, DEFAULT_LOG_PROB

    english = permutation_gap(
        "the weather is nice today and we went for a walk",
        BIGRAM_LOG_PROBS,
        DEFAULT_LOG_PROB,
    )
    mash = permutation_gap(
        "xqzv kjhq wpfd zzqx lkjh vbnm", BIGRAM_LOG_PROBS, DEFAULT_LOG_PROB
    )
    assert english < -0.5
    assert mash > english


def test_windows_short_text_is_one_window():
    assert windows("hello world") == ["hello world"]
    text = "x" * 127
    assert windows(text) == [text]


def test_windows_split_greedily_and_drop_short_tail():
    # 25 five-character slots fill 124 characters; five words remain.
    text = " ".join(["abcd"] * 30)
    assert windows(text) == [" ".join(["abcd"] * 25)]
    text = " ".join(["abcd"] * 50)
    assert windows(text) == [" ".join(["abcd"] * 25)] * 2


def test_windows_keep_long_tail():
    text = " ".join(["abcd"] * 38)
    assert windows(text) == [
        " ".join(["abcd"] * 25),
        " ".join(["abcd"] * 13),
    ]


def test_windows_long_single_word_is_its_own_window():
    word = "x" * 200
    tail = " ".join(["abcd"] * 20)
    assert windows(word + " " + tail) == [word, tail]
    assert windows("abc " + "y" * 130 + " abc") == ["abc", "y" * 130]
    assert windows(word) == [word]


def test_bucket_edges():
    assert [bucket(n) for n in (0, 15, 16, 31, 32, 63, 64, 1000)] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]


def test_standardised_uses_bucket_row():
    table = [(0.1, 0.2), (0.2, 0.4), (0.3, 0.9), (0.0, 1.0)]
    assert standardised(0.5, 1, table) == pytest.approx(1.5)
    assert standardised(0.2, 1, table) == 0.0
    assert standardised(0.0, 3, table) == 0.0
    assert standardised(-0.3, 2, table) == pytest.approx(-1.0)
