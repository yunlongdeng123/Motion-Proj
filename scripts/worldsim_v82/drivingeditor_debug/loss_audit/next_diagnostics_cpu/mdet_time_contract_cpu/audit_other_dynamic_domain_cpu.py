"""Existing annotations/camera input only: no point classification or detector calls."""
from pathlib import Path
import hashlib
import itertools
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from PIL import Image

BASE = Path('/root/autodl-tmp/drivingeditor_repro')
ROOT = BASE / 'campaign/vgpu_bugfix_20261012'
INIT = ROOT / 'loss_audit/next_diagnostics_cpu/static_initialisation_actor_cpu'
OUT = ROOT / 'loss_audit/next_diagnostics_cpu/mdet_time_contract_cpu'
DATA = BASE / 'data/waymo_paper_scenes/scene0017_000_048'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    raw_path = INIT / 'raw49_labels_layout.json'
    original_audit = INIT / 'static_initialisation_actor_audit.json'
    assert sha(raw_path) == 'fa0448b99abc28c22f40fc7722ecb1d68dbcc68dbe88ae44b5aed0357b894806'
    assert sha(original_audit) == 'df97e5d48ee2be553b54e127dbeff2be12dafa8561c93d6bc87bc630059b8ecd'
    raw = json.loads(raw_path.read_text())
    audit = json.loads(original_audit.read_text())
    target_id = audit['target_identity']['Waymo_track_id']
    prep = json.loads((DATA/'preparation_manifest.json').read_text())
    inputs = {}
    rows = []
    for fi, (frame, proof) in enumerate(zip(prep['frames'],raw['frames'])):
        assert frame['id'] == proof['frame'] and fi == proof['source_index']
        name = frame['id']
        cp = DATA/'calib'/(name+'.txt')
        mp = DATA/'dynamic_masks/dynamic_mask_0'/(name+'.png')
        for p in (cp,mp):
            actual = sha(p)
            assert actual == audit['pins'][str(p)]
            inputs[str(p)] = actual
        cal = {l.split(':',1)[0]:np.fromstring(l.split(':',1)[1],sep=' ') for l in cp.read_text().splitlines() if ':' in l}
        tr = cal['Tr_velo_to_cam_0'].reshape(3,4)
        proj = cal['P0'].reshape(3,4)
        mask = np.asarray(Image.open(mp)) > 0
        h,w = mask.shape
        assert [w,h] == frame['image_size']
        for label in proof['labels']:
            center = np.asarray(label['center'])
            half = np.asarray(label['size_lwh'])/2
            c,s = np.cos(label['heading']),np.sin(label['heading'])
            local = np.asarray(list(itertools.product((-1,1),repeat=3))) * half
            corners = np.column_stack((local[:,0]*c-local[:,1]*s,local[:,0]*s+local[:,1]*c,local[:,2])) + center
            points = np.vstack((center,corners))
            camera = np.column_stack((points,np.ones(9))) @ tr.T
            zpositive = camera[:,2]>.01
            uv = np.full((9,2),np.nan)
            hp = np.column_stack((camera[zpositive],np.ones(zpositive.sum()))) @ proj.T
            uv[zpositive] = hp[:,:2]/hp[:,2:3]
            center_inside = bool(zpositive[0] and 0<=uv[0,0]<w and 0<=uv[0,1]<h)
            center_mask = False
            if center_inside:
                px = np.floor(uv[0]+.5).astype(int)
                if 0<=px[0]<w and 0<=px[1]<h:
                    center_mask = bool(mask[px[1],px[0]])
            bbox_intersects = None
            bbox = None
            if zpositive[1:].all():
                lo,hi = uv[1:].min(axis=0),uv[1:].max(axis=0)
                bbox = [float(lo[0]),float(lo[1]),float(hi[0]),float(hi[1])]
                bbox_intersects = bool(hi[0]>=0 and lo[0]<w and hi[1]>=0 and lo[1]<h)
            elif not zpositive[1:].any():
                bbox_intersects = False
            speed = float(np.linalg.norm(label['speed_xy']))
            rows.append({'source_index':fi,'frame':name,'track_id':label['id'],'selected_actor':label['id']==target_id,
                         'native_type':label['type'],'speed_xy_mps':speed,'num_lidar_points_in_native_box':label['num_lidar_points_in_box'],
                         'center_body_m':label['center'],'center_FRONT_pinhole_eligible':center_inside,
                         'center_in_existing_selected_body_mask':center_mask,'all8corners_positive_depth':bool(zpositive[1:].all()),
                         'projected_native_box_bbox_intersects_FRONT':bbox_intersects,'projected_native_box_bbox':bbox})
    other = [r for r in rows if not r['selected_actor']]
    positive = [r for r in other if r['speed_xy_mps']>0]
    eligible = [r for r in positive if r['center_FRONT_pinhole_eligible']]
    tracks=[]
    for track in sorted(set(r['track_id'] for r in other)):
        rr=[r for r in other if r['track_id']==track]
        v=np.asarray([r['speed_xy_mps'] for r in rr])
        tracks.append({'track_id':track,'native_type':rr[0]['native_type'],'rows':len(rr),
                       'speed_xy_mps_min_median_max':[float(v.min()),float(np.median(v)),float(v.max())],
                       'annotated_nonzero_speed_rows':sum(r['speed_xy_mps']>0 for r in rr),
                       'annotated_nonzero_speed_FRONT_center_frames':[r['source_index'] for r in rr if r['speed_xy_mps']>0 and r['center_FRONT_pinhole_eligible']],
                       'nonzero_speed_FRONT_center_outside_selected_mask_frames':[r['source_index'] for r in rr if r['speed_xy_mps']>0 and r['center_FRONT_pinhole_eligible'] and not r['center_in_existing_selected_body_mask']],
                       'projected_bbox_intersects_FRONT_frames':[r['source_index'] for r in rr if r['projected_native_box_bbox_intersects_FRONT'] is True],
                       'native_box_lidar_points_min_median_max':[min(r['num_lidar_points_in_native_box'] for r in rr),float(np.median([r['num_lidar_points_in_native_box'] for r in rr])),max(r['num_lidar_points_in_native_box'] for r in rr)]})
    tracks.sort(key=lambda r:(-len(r['annotated_nonzero_speed_FRONT_center_frames']),-r['speed_xy_mps_min_median_max'][1],r['track_id']))
    result={'status':'CPU_EXISTING_NATIVE_ANNOTATION_OTHER_OBJECT_DOMAIN_CONTEXT_ONLY',
            'GPU_training_detector_or_input_rewrite_started':False,'allow_training':False,
            'native_annotation_used_as_detector_or_training_input':False,'selected_actor_id':target_id,
            'summary':{'all_native_object_frame_rows':len(rows),'other_native_object_frame_rows':len(other),'other_unique_tracks':len(tracks),
                       'other_rows_with_annotated_nonzero_speed':len(positive),'other_unique_tracks_with_annotated_nonzero_speed':len(set(r['track_id'] for r in positive)),
                       'other_nonzero_speed_CENTER_in_FRONT_pinhole_rows':len(eligible),'other_nonzero_speed_CENTER_in_FRONT_unique_tracks':len(set(r['track_id'] for r in eligible)),
                       'other_nonzero_speed_CENTER_in_FRONT_outside_selected_body_mask_rows':sum(not r['center_in_existing_selected_body_mask'] for r in eligible),
                       'other_nonzero_speed_projected_bbox_intersects_FRONT_rows':sum(r['projected_native_box_bbox_intersects_FRONT'] is True for r in positive),
                       'other_nonzero_speed_nearplane_straddling_bbox_unresolved_rows':sum(r['projected_native_box_bbox_intersects_FRONT'] is None for r in positive)},
            'speed_rule':'Report native speed_xy norm and exact nonzero field values only; no tuned moving-class threshold. Tiny annotation noise/absent-field default0 is not resolved as a physical moving/static ground truth.',
            'visibility_method':'Existing native body-coordinate oriented box center/8corners projected through unchanged Tr0/P0 and native image dimensions. Body mask sampled at center only. No new raw conversion/point label assignment.',
            'limits':['Projected center/box eligibility is not actual optical visibility, pixel extent identity or proof of detector-moving returns; occlusion is untested.',
                      'Native box association is not per-return membership and cannot classify road points or drive training partitions.',
                      'The cached labels omit optional-speed HasField bits; exact0 includes possible absent/default fields. Nonzero annotated velocities are preserved, not detector labels.',
                      'Current Md contains one selected car body and no explicit moving-shadow domain. Other moving-scene support remains unbound even if a native box is camera-ineligible.',
                      'Therefore absence of other visible movers is not established; global M-detector partition cannot silently become semantically consistent with the current selected-body mask.'],
            'I_operator_precondition':'Before materializing I0, bind the actual detector output role and object/domain scope to the unchanged Md, disclose unmatched other-moving/shadow points and freeze unknown/grouping policy. No GT filtering or mask expansion is authorized. If domain agreement is unresolved, I0 readiness stays false; a detector-output audit may proceed separately when root authorized.',
            'tracks':tracks,'all_object_frame_rows':rows,'input_SHA256':inputs,
            'canonical_raw_layout_SHA256':sha(raw_path),'canonical_initialisation_audit_SHA256':sha(original_audit),
            'source_script_SHA256':sha(__file__),
            'all_audited_original_inputs_unchanged_after':all(sha(p)==v for p,v in inputs.items())}
    OUT.mkdir(parents=True,exist_ok=True)
    dest=OUT/'other_dynamic_domain_followup_cpu.json'
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'summary':result['summary'],'top_FRONT_tracks':tracks[:5],'output':str(dest),'sha256':sha(dest)},ensure_ascii=False))

if __name__=='__main__':
    main()
