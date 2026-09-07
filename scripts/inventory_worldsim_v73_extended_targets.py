"""按fit Actor已知轨迹与本地LiDAR载荷盘点长时段标签，模型输入不变。"""
import argparse,json
from pathlib import Path
import subprocess,sys,time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v73.actor_rays import ActorRayDataset
from motion_proj.worldsim_v73.native_data import interpolate_pose


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--native-run',type=Path,required=True)
    parser.add_argument('--actor-data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); started=time.monotonic()
    data=Path('/root/autodl-tmp/data/worldsim_v4/drivestudio_raw_trainval')
    scenes={s['scene_id']:s for s in torch.load(args.native_run/'build_observations.pt',map_location='cpu',weights_only=False) if s['role']=='fit'}
    entries=[r for r in json.loads((args.actor_data/'index.json').read_text())['cases'] if r['role']=='fit']
    sensor=ActorRayDataset(data,set(scenes)); index=sensor.index
    scene_tokens={s['name']:s['token'] for s in index.scenes if s['name'] in scenes}
    rows=[]; scene_rows=[]
    for name,scene in scenes.items():
        samples=sorted([s for s in index.samples if s['scene_token']==scene_tokens[name]],key=lambda s:s['timestamp'])
        build={v['sample_id'] for v in scene['views']}
        positions=[i for i,s in enumerate(samples) if s['token'] in build]
        records=[]
        for i,sample in enumerate(samples):
            scan=index.sample_data.get(index.data_by_sample_channel.get((sample['token'],'LIDAR_TOP'),''))
            if scan is None: continue
            records.append({'sample_id':sample['token'],'sample_index':i,'timestamp_us':int(scan['timestamp']),
                'payload_available':(data/scan['filename']).is_file(),'build_input':sample['token'] in build,
                'outside_build_time_span':i<min(positions) or i>max(positions)})
        scene_rows.append({'scene':name,'log_id':scene['log_id'],'keyframes':len(records),
            'payload_available':sum(r['payload_available'] for r in records),
            'outside_window_payloads':sum(r['payload_available'] and r['outside_build_time_span'] for r in records)})
        for entry in [r for r in entries if r['scene']==name]:
            trajectory=sensor.tracks[name][entry['owner']]
            usable=[r for r in records if interpolate_pose(trajectory,r['timestamp_us']) is not None]
            extra=[r for r in usable if r['outside_build_time_span'] and r['payload_available']]
            rows.append({**entry,'track_keyframes_with_pose':len(usable),
                'available_outside_window_keyframes':len(extra),'outside_window_samples':extra,
                'missing_payload_keyframes_with_pose':sum(not r['payload_available'] for r in usable)})
        print(json.dumps(scene_rows[-1]),flush=True)
    result={'status':'done','code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'scenes':scene_rows,'actors':rows,'wall_s':time.monotonic()-started,
        'boundary':'fit-only payload/trajectory inventory; no new point labels generated, no depth/quality-based selection; current model inputs and training runs unchanged; development not expanded'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__': main()
