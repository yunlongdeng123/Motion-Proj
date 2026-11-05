"""Read historical dual-actor controls with the new fixed checkpoint metrics."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from evaluate_dual import ENGINE, native_camera, load_component, read_json
from native_ownership import CAMPAIGN, Ownership
from readout_native import readout, PILOTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--frames', choices=['pilots', 'all'], required=True)
    options = parser.parse_args()
    diag = CAMPAIGN / 'scene008_static_origin_diag_20261104_r1/source'
    sys.path[:0] = [str(ENGINE), str(diag)]
    import torch
    from omegaconf import OmegaConf
    from scene import GaussianModel, Dynamic_GaussianModel, EnvLight
    from scene.cameras import Camera
    from dual_actor_runtime import install_identity_serialization
    torch.set_num_threads(1)
    install_identity_serialization()
    config = read_json(options.model / 'effective_config.json')
    args = OmegaConf.create(config)
    static, dynamic = GaussianModel(args), Dynamic_GaussianModel(args)
    env = EnvLight(resolution=args.env_map_res).cuda()
    for key, model in (('chkpnt', static), ('dynamic_chkpnt', dynamic), ('env_light_chkpnt', env)):
        load_component(torch, options.model / f'{key}{options.iteration}.pth', model, options.iteration,
                       time_duration=config['time_duration'] if model is not env else None)
    entries = {int(row['img_name'][-3:]): row for row in read_json(options.model / 'cameras.json')}
    timestamp = read_json(options.model / 'actual_scene_binding.json')['timestamps']
    frames = PILOTS if options.frames == 'pilots' else tuple(range(89, 138))
    cameras = [native_camera(Camera, torch, raw, entries[raw], timestamp[raw-89])[0] for raw in frames]
    scene = SimpleNamespace(gaussians=static, dynamic_gaussians=dynamic, getTrainCameras=lambda: cameras)
    options.output.mkdir(parents=True, exist_ok=True)
    readout(Ownership(), scene, args, torch.zeros(3, device='cuda'), env, options.output, options.iteration, frames)


if __name__ == '__main__':
    main()
