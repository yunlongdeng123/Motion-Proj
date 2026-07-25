import pytest

from scripts.validate_n1_kinematic_review import wilson_lower_bound


def test_wilson_lower_bound_known_values() -> None:
    assert wilson_lower_bound(0, 0) is None
    assert wilson_lower_bound(20, 20) == pytest.approx(0.8388748419)
    assert wilson_lower_bound(16, 20) == pytest.approx(0.5839825677)


def test_wilson_lower_bound_monotonic_in_successes() -> None:
    values = [wilson_lower_bound(successes, 20) for successes in range(21)]
    assert all(left <= right for left, right in zip(values, values[1:]))
