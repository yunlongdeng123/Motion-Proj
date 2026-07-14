from types import SimpleNamespace

from omegaconf import OmegaConf

from motion_proj.replay.mine import (
    _generation_settings,
    _summarize_rows,
    replay_energy_decreased,
    replay_is_eligible,
)


def _result(energy=True, eligible=0.8, before=2.0, after=1.0):
    return SimpleNamespace(
        diagnostics={"energy_decreased": energy, "eligible_fraction": eligible},
        energy_before={"total": before},
        energy_after={"total": after},
    )


def test_replay_requires_high_drift_energy_drop_and_70pct_eligible():
    assert replay_is_eligible(2.0, _result(), 1.0, drift_after=1.0)
    assert not replay_is_eligible(0.9, _result(), 1.0, drift_after=0.5)
    assert not replay_is_eligible(2.0, _result(before=1.0, after=1.0), 1.0, drift_after=2.0)
    assert not replay_is_eligible(2.0, _result(eligible=0.699), 1.0, drift_after=1.0)


def test_replay_rejects_vacuous_zero_track_energy_gate():
    vacuous = _result(energy=True, before=39.0, after=39.0)
    assert not replay_energy_decreased(vacuous, drift_before=39.0, drift_after=39.0)
    assert not replay_is_eligible(30.0, vacuous, 1.0, drift_after=30.0)


def test_replay_accepts_static_drift_reduction_after_reaudit():
    result = _result(before=10.0, after=10.0)
    assert replay_energy_decreased(result, drift_before=12.0, drift_after=8.0)
    assert replay_is_eligible(12.0, result, 1.0, drift_after=8.0)


def test_replay_diagnostics_preserve_generation_settings_and_gate_values():
    cfg = OmegaConf.create({
        "cache": {"num_inference_steps": 7, "decode_chunk_size": 2},
        "model": {"generation": {
            "protocol": "svd_official_v1", "fps": 7, "motion_bucket_id": 127,
            "noise_aug_strength": 0.02, "min_guidance_scale": 1.0,
            "max_guidance_scale": 3.0,
        }},
    })
    assert _generation_settings(cfg) == {
        "num_inference_steps": 7, "decode_chunk_size": 2,
        "protocol": "svd_official_v1", "fps": 7, "fps_time_id": 6,
        "motion_bucket_id": 127, "noise_aug_strength": 0.02,
        "min_guidance_scale": 1.0, "max_guidance_scale": 3.0,
    }

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
