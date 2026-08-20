from __future__ import annotations

from scripts.localize_worldsim_v521_failures import bootstrap_scene_rate, mse_from_psnr


def test_psnr_to_mse_is_exact() -> None:
    assert mse_from_psnr(20.0) == 0.01
    assert mse_from_psnr(None) is None


def test_scene_bootstrap_is_deterministic_and_requires_two_scenes() -> None:
    first = bootstrap_scene_rate([0.0, 0.5, 1.0], samples=1000)
    second = bootstrap_scene_rate([0.0, 0.5, 1.0], samples=1000)
    assert first == second
    assert first["status"] == "done"
    assert bootstrap_scene_rate([0.5])["status"] == "undefined_insufficient_denominator"
