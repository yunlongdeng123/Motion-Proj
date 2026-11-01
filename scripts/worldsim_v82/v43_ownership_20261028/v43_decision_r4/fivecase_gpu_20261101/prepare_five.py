"""五例固定来源、小域候选准备；不读独立编辑参考或编辑结果。"""
import os
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[k] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys, json, gc
from pathlib import Path
import numpy as np
from PIL import Image
import torch

REPO = Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
CODE = REPO / 'scripts/worldsim_v82/v43_ownership_20261028/v43_decision_r4'
sys.path.insert(0, str(CODE))
import prepare_cpu as old
from r4_core import canonical_hash, enumerate_routes, freeze_family, register_tasks

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/r4_fivecase_gpu_20261101_r1')
PRIOR = Path('/root/autodl-tmp/codex_tmp/v43_case15_isolation_20261101_r1')
PERSIST = ROOT.parent.parent / 'persistent_world_20261027'
LEGACY = ROOT.parent.parent / 'r15_baseline_owner5_20261018/runs/scene0008_089_137_hybrid_30000_seed0_GN11_resume20k'
def read(p): return json.loads(Path(p).read_text())
def write(p, x): Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def main():
    output = ROOT / 'cpu'
    output.mkdir(exist_ok=False)
    inputs = read(ROOT / 'sensor_inputs/manifest.json')
    specs = []
    for r in inputs:
        run = LEGACY if r['scene_key'] == 'waymo-train-008' else PERSIST / 'native_scene_training_v1' / r['context'] / 'formal30k'
        specs.append(dict(id=r['case'], event=str(run / 'checkpoint_events/iteration_030000.json')))
    write(output / 'run_contract.json', dict(task_id='WS-V43-R4-FIVECASE-GPU-20261101-R1', checkpoint_inputs=specs,
        cases=[r['case'] for r in inputs], input_role='seen real-world development; provisional masks; no counterfactual GT',
        stage='E0 and E2 fixed representation; no new geometry/color/opacity/model training',
        candidate_radius_m=.25, groups='first/middle/last anchor first-hit-disjoint',
        responsibility_refinement=dict(min_total_alphaT=1e-4, body_ratio=.95, role='inherited development scope, not admission or owner truth'),
        operations=dict(none=True, delete=True, shift_world_m=[0,1.5,0], rotate_world_z_deg=30),
        formal_eligibility='unknown, no independently calibrated thresholds',
        legacy_dynamic_association='must freeze legal contribution/motion subset; never edit entire five-car layer',
        planned_native_route_outputs=5*8*4*3, full_sequence_control='two fixed endpoint routes x49frames x4q, extra separately counted',
        learning_gate='no transferable real-data head without valid reference roles and independently identifiable paired signal', human_verdict=None))
    for r, spec in zip(inputs, specs):
        dst = output / r['case']; dst.mkdir()
        event = old.read(spec['event'], 'original_checkpoint_event')
        src = Path(event['source_path']); transform = old.matrix(src/'transform_matrix.txt', 'frozen_gauge')
        cameras = old.read(Path(event['model_path'])/'cameras.json', 'frozen_training_cameras')
        prep = old.read(src/'preparation_manifest.json', 'original_preparation')
        frames_by_raw = {x['source_index']: x for x in prep['frames']}
        cameras_by_id = {x['img_name']: x for x in cameras}
        associated = r['dynamic_actor_index'] is not None
        masks = src/'dynamic_actor_assets'/('actor_%d'%r['dynamic_actor_index'])/'body_masks' if associated else PRIOR/'prepared_source_masks'/r['case']/'body_masks'
        payload, loaded = old.load_checkpoint(event)
        static_sha = next(x['sha256'] for x in loaded if x['name']=='static')
        frames, context = [], []
        with np.load(ROOT/'sensor_inputs'/(r['case']+'.npz')) as z:
            for p in r['frames']:
                raw = p['raw']; f = frames_by_raw[raw]; fid = f['id']; cam = cameras_by_id[fid]
                mask_path = masks/(fid+'.png')
                mask = np.asarray(Image.open(mask_path).convert('L').resize((cam['width'],cam['height']), Image.Resampling.NEAREST))>0
                # Cameras.json is the actual480x320 frozen render domain; mask resize only, no hand editing.
                c2w = old.camera_world(cam, transform)
                world = z['world_'+str(raw)]; model = world@transform[:3,:3].T+transform[:3,3]
                native_c2w = np.eye(4); native_c2w[:3,:3] = cam['rotation']; native_c2w[:3,3] = cam['position']
                cv = (model-native_c2w[:3,3])@native_c2w[:3,:3]
                uv = np.zeros((len(world),2)); valid = np.isfinite(cv).all(1)&(cv[:,2]>1e-4)
                uv[valid] = cv[valid,:2]/cv[valid,2,None]*[cam['fx'],cam['fy']]+[cam['cx'],cam['cy']]
                pix = np.floor(uv).astype(np.int64)
                valid &= (pix[:,0]>=0)&(pix[:,0]<mask.shape[1])&(pix[:,1]>=0)&(pix[:,1]<mask.shape[0])
                ids = np.flatnonzero(valid); ids = ids[mask[pix[ids,1],pix[ids,0]]]
                frames.append(dict(id=fid, model_time=f['model_time'], timestamp_s=p['timestamp_micros']/1e6,
                    camera_to_world=c2w, center=np.asarray(p['center_world']), rotation=np.asarray(p['rotation_world']),
                    body_pixels=int(mask.sum()), measured_depth_pixels=len(np.unique(pix[ids],axis=0)), support_world=world[ids],
                    support_source_rows=z['source_'+str(raw)][ids], support_ref=str(ROOT/'sensor_inputs'/(r['case']+'.npz')), mask_ref=str(mask_path)))
                context.append(dict(frame_id=fid, raw=raw, model_time=f['model_time'], camera_to_world=c2w.tolist(),
                    actor_center_world=p['center_world'], actor_rotation_world=p['rotation_world'], dimensions=p['dimensions'],
                    mask_ref=str(mask_path), original_body_pixels=int(mask.sum()), measured_target_returns=len(ids)))
        role=dict(role='reviewed_measured_real_observations', role_detail='annotated track + fixed provisional DEVA mask + decoded measured OBB returns',
            original_input_contract=str(ROOT/'sensor_inputs/manifest.json'), limitations=['OBB points are not instance/surface GT', 'mask identity provisional; no edited GT',r['numeric_limit']])
        candidate,mappings=old.candidates(payload,frames,transform,r['case'],'waymo_context_'+r['context'],role,static_sha,.25)
        evidence=old.legal_evidence(frames,role)
        components={x['name']:x['sha256'] for x in loaded}
        frozen=dict(component_sha256=components, checkpoint_bundle_sha256=canonical_hash([dict(name=n,sha256=components[n]) for n in sorted(components)]),
            event_sha256=old.receipt(spec['event'],'event')['sha256'],gauge_sha256=old.receipt(src/'transform_matrix.txt','gauge')['sha256'],
            preparation_sha256=old.receipt(src/'preparation_manifest.json','prep')['sha256'],camera_sha256=old.receipt(Path(event['model_path'])/'cameras.json','camera')['sha256'])
        routes=enumerate_routes(candidate)
        write(dst/'candidate_manifest.json',candidate);write(dst/'shared_routes.json',routes)
        write(dst/'frozen_H0.json',freeze_family(candidate));write(dst/'legal_preedit_evidence.json',evidence)
        write(dst/'tasks.json',register_tasks(r['case'],candidate,evidence,routes,static_sha,frozen))
        write(dst/'camera_track_context.json',dict(role=role,frames=context,world_to_model=transform.tolist(),
            target_track=r['track'],target_dynamic_actor_index=r['dynamic_actor_index'],scene_key=r['scene_key']))
        write(dst/'checkpoint_cpu_receipt.json',dict(components=loaded))
        for m in mappings:
            gid=m.pop('group_id');np.savez_compressed(dst/('row_mapping_'+gid+'.npz'),**m)
        print(json.dumps(dict(case=r['case'],candidate_rows=freeze_family(candidate)['candidate_row_count'],routes=len(routes),frames=len(frames))),flush=True)
        del payload,frames;gc.collect()
if __name__=='__main__':main()
