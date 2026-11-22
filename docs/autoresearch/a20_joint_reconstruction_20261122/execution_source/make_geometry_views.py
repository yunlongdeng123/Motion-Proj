"""CPU diagnostic shading from the actual Gaussian expected-depth buffer."""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

root = Path('/root/autodl-tmp/drivingeditor_repro/campaign/a20_joint_reconstruction_20261122_r1')
execution, stage, label = sys.argv[1:4]
run = root / execution / stage
specs = json.loads((run/'synthetic_cameras.json').read_text())['specs']
records = []
for spec in specs:
    folder = run/'teacher_cache'/spec['key'] if label == 'teacher' else run/'evaluation'/label/spec['key']/'actor'
    with np.load(folder/'render_float.npz') as z:
        alpha = z['alpha'].squeeze()
        depth = z['depth_sum_m'].squeeze() / np.maximum(alpha, 1e-8)
    height, width = alpha.shape
    focal = width/(2*np.tan(np.deg2rad(spec['fov_degrees'])/2))
    yy, xx = np.mgrid[:height, :width]
    points = np.stack([(xx-(width-1)/2)*depth/focal,
                       (yy-(height-1)/2)*depth/focal, depth], axis=-1)
    dx = np.gradient(points, axis=1)
    dy = np.gradient(points, axis=0)
    normal = np.cross(dx, dy)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-10)
    normal *= np.where(np.sum(normal*points, axis=-1, keepdims=True)>0, -1, 1)
    light = np.array([-.3, -.4, -1.]); light /= np.linalg.norm(light)
    shade = .22 + .78*np.clip(normal@light, 0, 1)
    stable = alpha >= .95
    valid = stable.copy()
    valid[1:-1,1:-1] &= stable[2:,1:-1]&stable[:-2,1:-1]&stable[1:-1,2:]&stable[1:-1,:-2]
    value = np.where(valid, shade, .35*alpha)
    rgb = np.repeat(np.clip(value*210,0,255).astype(np.uint8)[...,None],3,axis=-1)
    Image.fromarray(rgb).save(folder/'expected_depth_grey.png')
    Image.fromarray(np.uint8((normal*.5+.5)*255)*valid[...,None]).save(folder/'expected_depth_normal.png')
    records.append(dict(key=spec['key'], stable_pixels=int(stable.sum()), normal_valid_pixels=int(valid.sum())))
(run/('teacher_expected_depth_geometry_diagnostic.json' if label == 'teacher' else 'expected_depth_geometry_diagnostic.json')).write_text(json.dumps(dict(
    method='perspective expected Gaussian depth normals; no mesh, smoothing or geometry edits',
    actual_geometry_updates=0, views=records),indent=2))
print(json.dumps(records))
