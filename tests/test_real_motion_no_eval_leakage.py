import pytest

from motion_proj.data.real_motion_targets import (
    GENERATED_EVALUATION_SCOPE,
    REAL_TARGET_SCOPE,
    RealMotionTargetError,
    assert_target_scope,
)


def test_real_future_targets_are_explicitly_forbidden_in_generated_evaluation():
    assert_target_scope(REAL_TARGET_SCOPE)
    with pytest.raises(RealMotionTargetError):
        assert_target_scope(GENERATED_EVALUATION_SCOPE)

