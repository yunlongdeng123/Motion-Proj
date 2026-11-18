"""T2原生入口的CPU许可/域外初值回归；不加载native。"""
import numpy as np
import pytest
from run_native_columns import compose_colors, bytes_equal


def fixture():
    rows = np.array([10, 30, 40, 80])
    parent = np.full((4, 3), .1, np.float32)
    eligible = np.array([1, 1, 0, 0], bool)
    A0 = np.full((4, 3), .4, np.float32)
    T2 = A0.copy()
    T2[0] = [.6, .3, .2]
    cache = dict(active_world_rows=rows, filtered_rgb_T=T2, filtered_rgb_A0=A0)
    return rows, parent, eligible, cache


def test_exterior_eligible_uses_filtered_A0_in_both_columns():
    rows, parent, eligible, cache = fixture()
    output = compose_colors(parent, rows, eligible, cache)
    assert bytes_equal(output['T2_mid'][1], cache['filtered_rgb_A0'][1])
    assert not bytes_equal(output['T2_mid'][1], parent[1])
    assert bytes_equal(output['A0_mid'][eligible], cache['filtered_rgb_A0'][eligible])
    for value in output.values():
        assert bytes_equal(value[~eligible], parent[~eligible])
    assert bytes_equal(parent, np.full((4, 3), .1, np.float32))


def test_identity_reorder_and_nonfinite_fails_closed():
    rows, parent, eligible, cache = fixture()
    cache['active_world_rows'] = rows[::-1]
    with pytest.raises(ValueError, match='identity'):
        compose_colors(parent, rows, eligible, cache)
    cache['active_world_rows'] = rows
    cache['filtered_rgb_T'][0, 0] = np.nan
    with pytest.raises(ValueError, match='invalid'):
        compose_colors(parent, rows, eligible, cache)


def test_byte_comparison_preserves_nan_payload_semantics():
    a = np.array([np.nan, 1.], np.float32)
    assert bytes_equal(a, a.copy())
    assert not bytes_equal(a, a.astype(np.float64))
