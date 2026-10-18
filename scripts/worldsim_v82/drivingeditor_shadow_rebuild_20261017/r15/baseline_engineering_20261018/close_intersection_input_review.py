"""Close the reviewed 20-ID input contract; GPU release stays with A controller."""
from pathlib import Path
import json, hashlib, datetime, os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from PIL import Image
E = Path(__file__).resolve().parent
P = E/'all_vehicle_pipeline'
S = P/'inputs/scene0008_089_137'

def main():
    receipt = json.loads((E/'lidar_camera_intersection_input_receipt.json').read_text())
    actors = receipt['selected']
    ids = [a['R15_actor'] for a in actors]
    assert len(ids) == len(set(ids)) == 20
    assert all(n > 0 for n in receipt['dynamic_real_seed_points'])
    assert receipt['actual_CPU_cameras'] == 49 and receipt['additional_source_frames_used'] is False
    assert receipt['GT_boxes_filled_as_masks'] is False
    assert receipt['GT_boxes_used_to_filter_individual_initialization_points'] is False
    manifest = json.loads((S/'preparation_manifest.json').read_text())
    assert manifest['selected_real_track_ids'] == [a['raw_laser_track_id'] for a in actors]
    assert manifest['selected_camera_track_ids'] == [a['raw_camera_track_id'] for a in actors]
    selected = json.loads((P/'reviewed_selections/scene0008_089_137.json').read_text())
    assert [a['R15_actor'] for a in selected['actors']] == ids
    records = []
    for raw in range(89,138):
        fid = f'0008{raw:03d}'
        labels = np.asarray(Image.open(S/'intersection_labels'/(fid+'.png')))
        union = np.asarray(Image.open(S/'dynamic_masks/dynamic_mask_0'/(fid+'.png'))) > 0
        assert labels.shape == union.shape == (1280,1920)
        assert np.array_equal(union,labels > 0), fid
        for i in range(20):
            mask = np.asarray(Image.open(S/f'dynamic_actor_assets/actor_{i}/body_masks'/(fid+'.png'))) > 0
            assert np.array_equal(mask,labels == i+1), (fid,i)
        rgb = S/'image/image_0'/(fid+'.png')
        assert rgb.resolve().is_relative_to(E.parent/'inputs/scene0008_089_137')
        records.append(dict(frame_id=fid,full_RGB_sha256=hashlib.sha256(rgb.read_bytes()).hexdigest(),
                            canonical_union_pixels=int(union.sum()),all20_native_masks_exact=True))
    out = dict(status='PASSED_REVIEWED20_FULL49_INPUT_CONTRACT',
        scope='LIDAR_AND_CAMERA_INTERSECTION_ONLY', frames=49, selected_actor_ids=ids,
        visual_review=dict(full49_contact_sheets_reviewed=list(range(7)),
            additional_full_resolution_anchor_reviewed='0008128', spatial_cropping=False,
            observations='Selected masks cover corresponding vehicle bodies in the reviewed complete views. No broad building/road false-positive observed. Small distant boundaries remain uncertain; this is input admission, not reconstruction quality approval.'),
        physical_prior_points=receipt['dynamic_real_seed_points'],
        limitations=['Sparse physical priors: a41 has19 retained points; point counts are not asset quality.',
                     'Predicted masks, including camera-box-prompted instances, are not pixel ground truth.',
                     '13 camera-only tracks remain RGB context and can appear in static background.',
                     'No new observation frames or generative completion used.'],
        per_frame_checks=records,training_updates=0,human_verdict=None,
        at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    gate = dict(status='PASSED_LIDAR_CAMERA_INTERSECTION_INPUT_BINDING',
        scope='LIDAR_AND_CAMERA_INTERSECTION_ONLY', frames=49,selected_actor_ids=ids,
        unresolved_selected_vehicles=[],same_identity_set_for_initialization_masks_and_future_stage2=True,
        all49_full_RGB_retained=True,selected_lidar_camera_coverage_reviewed=True,
        all_selected_actors_have_real_initialization=True,
        excluded_camera_only_actors_disclosed_as_context=True,all_scene_vehicle_coverage_claimed=False,
        review_receipt=str(E/'intersection_input_review_receipt.json'),
        GPU_release='A recovery controller releases only after A30K+30K and full49 evaluation',
        checkpoint_interval=5000,second_stage_updates=0,human_verdict=None)
    assert not (E/'intersection_input_passed.json').exists()
    (E/'intersection_input_review_receipt.json').write_text(json.dumps(out,indent=2))
    (E/'intersection_input_passed.json').write_text(json.dumps(gate,indent=2))
    assert not (E/'all_vehicle_ready.json').exists(),'A resource-chain release must remain separate'
    print(json.dumps(dict(status=out['status'],actors=len(ids),frames=49,GPU_training_released=False)))

if __name__ == '__main__':main()
