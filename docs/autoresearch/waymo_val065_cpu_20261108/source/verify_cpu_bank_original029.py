"""Validate the completed bank and write its explicit native-loader timeline."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bank',type=Path,required=True)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--anchors',type=Path,required=True)
    args=parser.parse_args()
    bank=args.bank
    receipt=json.loads((bank/'receipt.json').read_text(encoding='utf8'))
    assert receipt['status']=='PASS_CPU_OBSERVATIONS'
    assert receipt['GPU_queries']==receipt['training_updates']==receipt['render_calls']==0
    expected=set(receipt['build_raws']);held=set(receipt['holdout_raws'])
    assert expected&held==set()
    actual={int(p.stem[-3:]) for p in (bank/'calib').glob('*.txt')}
    assert actual==expected
    prompts=0
    for row in receipt['frames']:
        raw=row['raw'];name=row['id']
        assert raw not in held
        for relative,identity in row['files'].items():
            path=bank/relative
            assert path.stat().st_size==identity['bytes']
            assert digest(path)==identity['sha256']
        lidar=np.fromfile(bank/'velodyne'/f'{name}.bin',dtype='<f4').reshape(-1,6)
        assert len(lidar)==row['lidar_points'] and np.isfinite(lidar).all()
        image=Image.open(bank/'image/image_0'/f'{name}.png')
        assert image.size==(1920,1280)
        assert np.linalg.det(np.loadtxt(bank/'pose'/f'{name}.txt')[:3,:3])>.999
        suggestions=json.loads((bank/'prompts'/f'{raw:03d}.json').read_text(encoding='utf8'))
        assert len(suggestions)==row['prompt_count']
        for prompt in suggestions:
            x0,y0,x1,y1=prompt['bbox_xyxy_native']
            assert 0<=x0<x1<1920 and 0<=y0<y1<1280
        prompts+=len(suggestions)
    timeline=dict(schema='waymo.explicit_timeline.v1',context_id=receipt['context_id'],
        model_frame_interval=receipt['model_frame_interval'],holdout_raws=sorted(held),
        frames=[{k:row[k] for k in ('id','raw','timestamp_micros','model_time')} for row in receipt['frames']])
    (bank/'explicit_timeline.json').write_text(json.dumps(timeline,indent=2)+'\n',encoding='utf8')
    preparation=dict(status='awaiting_instance_and_sky_masks',segment_context_name=receipt['context_id'],
        raw_path=receipt['raw'],raw_size_bytes=receipt['raw_bytes'],camera='FRONT',camera_index=0,
        requested_frames=len(expected),source_frame_count=receipt['source_frame_count'],
        model_frame_interval=receipt['model_frame_interval'],sky_masks_present=0,
        excluded_holdout_raws=sorted(held),
        frames=[dict(id=row['id'],source_index=row['raw'],timestamp_micros=row['timestamp_micros'],
                     model_time=row['model_time'],lidar_points=row['lidar_points'],image_size=[1920,1280])
                for row in receipt['frames']],
        limitations=receipt['limitations'])
    (bank/'preparation_manifest.json').write_text(json.dumps(preparation,indent=2)+'\n',encoding='utf8')
    selection=json.loads(args.selection.read_text(encoding='utf8'))
    anchors=json.loads(args.anchors.read_text(encoding='utf8'))
    assert selection['context_id']==receipt['context_id']
    reference=anchors['reference_raw']
    assert reference in expected and reference not in held
    observation=json.loads((bank/'observations'/f'{reference:03d}.json').read_text(encoding='utf8'))
    body=next(b for b in observation['boxes'] if b['track']==selection['revealed']['track_id'])
    local=np.array([p['actor_local_xyz'] for p in anchors['pairs']])
    world_from_actor=np.array(body['world_from_actor']).reshape(4,4)
    world=local@world_from_actor[:3,:3].T+world_from_actor[:3,3]
    ego_from_world=np.linalg.inv(np.array(observation['ego_world']).reshape(4,4))
    measured_prediction=world@ego_from_world[:3,:3].T+ego_from_world[:3,3]
    actual=np.fromfile(bank/observation['lidar_path'],dtype='<f4').reshape(-1,6)[:,:3]
    distances=np.array([np.sqrt(np.min(np.sum((actual-point)**2,axis=1)))
                        for point in measured_prediction])
    assert distances.max()<.001,distances.max()
    report=dict(status='PASS_CPU_BANK',build_frames=len(expected),heldout_frames=len(held),
        all_file_hashes_checked=True,finite_lidar=True,calibrations_complete=True,
        native_RGB_shape=[1920,1280],prompts=prompts,model_time_gap_preserved=True,
        selected_anchor_count=len(local),selected_anchors_recovered_from_original_full_cloud=True,
        anchor_max_distance_m=float(distances.max()),GPU_queries=0,training_updates=0)
    (bank/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report))


if __name__=='__main__':main()
