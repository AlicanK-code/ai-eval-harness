import math

from evalharness.stats import bootstrap_ci, paired_diff, percentile


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert math.isnan(percentile([], 50))


def test_bootstrap_is_seeded_and_brackets_mean():
    values = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    lo, hi = bootstrap_ci(values, seed=7)
    assert (lo, hi) == bootstrap_ci(values, seed=7)
    assert lo < sum(values) / len(values) < hi


def test_constant_values_have_zero_width():
    assert bootstrap_ci([1.0] * 20) == (1.0, 1.0)


def test_paired_diff_uses_shared_cases_only():
    a = {"x": 0.0, "y": 0.5, "only_a": 1.0}
    b = {"x": 1.0, "y": 1.0, "only_b": 0.0}
    d = paired_diff(a, b)
    assert d["n"] == 2
    assert d["mean_diff"] == 0.75
