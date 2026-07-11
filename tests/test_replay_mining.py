from types import SimpleNamespace

from motion_proj.replay.mine import replay_is_eligible


def _result(energy=True, eligible=0.8):
    return SimpleNamespace(diagnostics={"energy_decreased": energy,
                                        "eligible_fraction": eligible})


def test_replay_requires_high_drift_energy_drop_and_70pct_eligible():
    assert replay_is_eligible(2.0, _result(), 1.0)
    assert not replay_is_eligible(0.9, _result(), 1.0)
    assert not replay_is_eligible(2.0, _result(energy=False), 1.0)
    assert not replay_is_eligible(2.0, _result(eligible=0.699), 1.0)
