"""Read-only frame-156 source attribution and saved-output parity audit."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys, json, time
import numpy as np
from PIL import Image

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/waymo_val065_native10k_20261108_r1')
sys.path.insert(0, str(ROOT / 'source'))
from editable_scene import Scene, BANK
from native_runtime import native_rows

def main():
    import torch
    from utils.v4_road_render import merge_rows, raster
    out = ROOT / 'audit_frame156_20261109_r1'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    scene = Scene(with_road=True)
    raw = 156
    cam = scene.camera(raw)
    labels = np.array(Image.open(BANK.parent / 'masks_all_resolved_r2/156/labels.png'))
    target = labels == 10
    Image.fromarray(labels).save(out / 'labels.png')
    groups = ['all_static'] + ['source_dynamic_' + str(i) for i in scene.cfg['followup_scene_spec'].get('dynamic_actor_ids', [])]
    actors = sorted(set(scene.source_ids.cpu().tolist()))
    groups = ['all_static'] + ['source_dynamic_' + str(i) for i in actors]
    assert len(groups) <= 22
    records = []
    points = [(250, 300), (245, 265), (260, 285), (240, 230)]

    def save(name, pack, group_names, reference):
        rgb = pack['rgb'].detach().cpu().numpy()
        alpha = pack['alpha'].detach().cpu().numpy()
        mass = pack['extra'][:len(group_names)].detach().cpu().numpy()
        error = float(abs(mass.sum(0) - alpha[0]).max())
        assert error < 3e-6, (name, error)
        saved = np.load(reference)['rgb']
        parity = float(abs(saved-rgb).max())
        assert parity < 2e-5, (name, parity)
        Image.fromarray(np.uint8(rgb.transpose(1,2,0).clip(0,1)*255+.5)).save(out / (name+'.png'))
        np.savez_compressed(out / (name+'.npz'), rgb=rgb, alpha=alpha, group_alpha=mass)
        rec = dict(name=name, group_names=group_names, target_pixels=int(target.sum()),
                   mean_mass_on_target=mass[:,target].mean(1).tolist(),
                   rgb_saved_maxabs=parity, group_sum_maxabs=error,
                   pixel_contributions=[dict(x=x,y=y,rgb=rgb[:,y,x].tolist(),
                                             groups=mass[:,y,x].tolist()) for x,y in points])
        records.append(rec)
        return rgb

    with torch.no_grad():
        s = native_rows(cam,scene.static,scene.pipe)
        d = native_rows(cam,scene.dynamic,scene.pipe,True)
        for row, static in [(s,True),(d,False)]:
            f = row['features'].new_zeros((len(row['means3D']),24))
            f[:,:2] = row['features'][:,:2]
            if static:
                f[:,2] = 1
            else:
                for i, actor in enumerate(actors):
                    f[:,3+i] = scene.source_ids == actor
            row['features'] = f
        values = merge_rows(s,d)
        delete_mask = torch.cat([torch.ones(len(scene.sids),device='cuda',dtype=torch.bool),scene.source_ids!=10])
        native_rgb = {}
        for state in ['keep','delete']:
            rows = dict(values,mask=values['mask'] & (delete_mask if state=='delete' else torch.ones_like(delete_mask)))
            pack = raster(cam,scene.pipe,0,rows,torch.zeros(3,device='cuda'))
            pack['rgb'] += (1-pack['alpha']) * scene.env(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
            native_rgb[state] = save('native_'+state,pack,groups,ROOT/'formal10k_eval/156'/f'{state}.npz')
        delta = abs(native_rgb['keep']-native_rgb['delete'])
        native_delta = dict(rgb_maxabs=float(delta.max()), changed_pixels_above_1_over_255=int((delta.max(0)>1/255).sum()), target_rgb_mae=float(delta[:,target].mean()))
        del s,d,values,pack

        road = scene.road
        for mode in ['semantic','final']:
            scene.road = road if mode=='final' else None
            if mode=='final':
                z = torch.load(str(ROOT/'road_gates/checkpoint001000.pt'),map_location='cuda',weights_only=False)
                scene.gate_ids = torch.tensor(np.load(ROOT/'road_gates/contributors.npz')['ids'],device='cuda')
                scene.gates = z['gate'].sigmoid()
            for state in ['keep','delete']:
                pack = scene.render(raw,state=='delete',cam=cam)
                save(mode+'_'+state,pack,['target_assigned','other_static','other_dynamic','road'],ROOT/(mode+'_all_eval' if mode=='semantic' else 'final_all_eval')/'156'/f'{state}.npz')
        assert scene.unchanged()

    # Read the original arrays, including their road-evidence coverage, without fitting.
    roadview = np.load(ROOT/'world_road/views/156.npz')
    coverage = {key:int(roadview[key].sum()) for key in ['real_domain','inferred_domain','observed_road','inside']}
    result = dict(task_id='WAYMO-VAL065-A9-DELETE-AUDIT-20261109', raw=raw,
                  target_track='NTjq-OhZ_97OMJ4xwJAxXg',target_native_id=10,target_alias='a9',
                  records=records,native_delete_delta=native_delta,road_view_mask_counts=coverage,
                  training_updates=0,optimizer_created=False,parent_parameters_unchanged=True,
                  automatic_shutdown=False,seconds=time.monotonic()-started)
    (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__ == '__main__':
    main()
