from scripts.finalize_worldsim_v3_a0 import markdown_table, parse_scene_record


def test_parse_scene_record() -> None:
    scene, source, actor = parse_scene_record("scene-0230=/source/run,/actor/run")
    assert scene == "scene-0230"
    assert source.parts[-2:] == ("source", "run")
    assert actor.parts[-2:] == ("actor", "run")


def test_markdown_table_renders_abstain() -> None:
    row = {
        "scene": "scene-0242",
        "global_psnr": 29.1,
        "global_ssim": 0.906,
        "global_lpips": 0.113,
        "high_actor_psnr": 19.8,
        "high_actor_ssim": 0.665,
        "high_actor_lpips_tight_crop": 0.153,
        "high_boundary_psnr": 23.3,
        "high_boundary_ssim": 0.795,
        "boundary_status": "ABSTAIN",
        "boundary_actor_psnr": None,
        "boundary_band_psnr": None,
        "background_gaussians": 843756,
        "rigid_gaussians": 86255,
        "train_seconds": 2006.2,
        "train_peak_gpu_mib": 12783,
    }
    rendered = markdown_table([row])
    assert "scene-0242" in rendered
    assert "ABSTAIN" in rendered
    assert "843,756 / 86,255" in rendered
