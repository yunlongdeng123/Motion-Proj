import pytest
import torch

from motion_proj.data.real_motion_targets import RealMotionTargetError, timestamps_to_seconds


def test_microsecond_timestamps_become_seconds():
    delta = timestamps_to_seconds(torch.tensor([1_000_000, 1_500_000, 2_100_000]))
    assert delta.tolist() == pytest.approx([0.5, 0.6])


@pytest.mark.parametrize(
    "timestamps",
    [
        [1000, 1500],  # 毫秒值误当微秒。
        [1_000_000, 1_000_000],
        [2_000_000, 1_500_000],
    ],
)
def test_timestamp_unit_or_monotonicity_errors_fail_closed(timestamps):
    with pytest.raises(RealMotionTargetError):
        timestamps_to_seconds(timestamps)

