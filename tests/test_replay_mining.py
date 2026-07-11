from types import SimpleNamespace

from omegaconf import OmegaConf

from motion_proj.replay.mine import _generation_settings, _summarize_rows, replay_is_eligible


def _result(energy=True, eligible=0.8):
    return SimpleNamespace(diagnostics={"energy_decreased": energy,
                                        "eligible_fraction": eligible})


def test_replay_requires_high_drift_energy_drop_and_70pct_eligible():
    assert replay_is_eligible(2.0, _result(), 1.0)
    assert not replay_is_eligible(0.9, _result(), 1.0)
    assert not replay_is_eligible(2.0, _result(energy=False), 1.0)
    assert not replay_is_eligible(2.0, _result(eligible=0.699), 1.0)


def test_replay_diagnostics_preserve_generation_settings_and_gate_values():
    cfg = OmegaConf.create({"cache": {"num_inference_steps": 7, "decode_chunk_size": 2}})
    assert _generation_settings(cfg) == {"num_inference_steps": 7, "decode_chunk_size": 2}

    summary = _summarize_rows([
        {"kept": False, "reject_reason": "eligible", "eligible_fraction": 0.62},
        {"kept": True, "reject_reason": None, "eligible_fraction": 0.75},
    ])
    assert summary["kept"] == 1
    assert summary["rejected"]["eligible"] == 1
    assert summary["eligible_fraction"] == {
        "mean": 0.685,
        "min": 0.62,
        "max": 0.75,
        "n": 2,
    }
