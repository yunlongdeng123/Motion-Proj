"""V81: 流式提取已落盘 nuScenes 子集；只按 metadata / 文件可用性选场景。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import json, argparse, collections, time
from pathlib import Path
import ijson

def stream(path):
    with path.open('rb') as f:
        yield from ijson.items(f,'item',use_float=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--out',required=True)
    p.add_argument('--max-logs',type=int,default=12)
    a=p.parse_args(); root=Path(a.root); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    meta=root/'v1.0-trainval'
    def table(n): return json.loads((meta/(n+'.json')).read_text())
    scenes={r['token']:r for r in table('scene')}; samples={r['token']:r for r in table('sample')}
    sensors={r['token']:r for r in table('sensor')}; calib={r['token']:r for r in table('calibrated_sensor')}
    available=set()
    for directory in (root/'samples').iterdir():
        if directory.is_dir(): available.update('samples/'+directory.name+'/'+r.name for r in os.scandir(directory))
    by_sample=collections.defaultdict(dict)
    for r in stream(meta/'sample_data.json'):
        if r['is_key_frame'] and r['filename'] in available:
            ch=sensors[calib[r['calibrated_sensor_token']]['sensor_token']]['channel']
            if ch.startswith('CAM') or ch=='LIDAR_TOP': by_sample[r['sample_token']][ch]=r
    by_scene=collections.defaultdict(list)
    for token, data in by_sample.items():
        if len(data)==7: by_scene[samples[token]['scene_token']].append(token)
    inventory=[]; chosen=[]; log_counts=collections.Counter()
    for scene_id in sorted(by_scene,key=lambda s: (scenes[s]['log_token'],scenes[s]['name'])):
        s=scenes[scene_id]; ts=sorted(by_scene[scene_id],key=lambda t:samples[t]['timestamp'])
        inventory.append({'scene':s['name'],'log':s['log_token'],'complete_keyframes':len(ts),'role':'DISCOVERY_LEGACY_EXPOSURE'})
        if len(ts)<5 or log_counts[s['log_token']]>=2: continue
        if s['log_token'] not in log_counts and len(log_counts)>=a.max_logs: continue
        log_counts[s['log_token']]+=1
        # 最多两个窗口，围绕中间位置按预先固定顺序选；不读取画质。
        centers=sorted(set([len(ts)//3,2*len(ts)//3]))
        for ix in centers:
            t=ts[ix]; prevs=[]; nxts=[]; cur=t
            for _ in range(2):
                cur=samples[cur]['prev']
                if not cur: break
                prevs.append(cur)
            cur=t
            for _ in range(2):
                cur=samples[cur]['next']
                if not cur: break
                nxts.append(cur)
            neighbors=[n for n in reversed(prevs)]+nxts
            ref=[n for n in neighbors if 'LIDAR_TOP' in by_sample.get(n,{})]
            views=[n for n in neighbors if len(by_sample.get(n,{}))==7]
            chosen.append({'window_id':s['name']+'_'+t[:8], 'sample_token':t,'scene':s['name'],'scene_token':scene_id,'log':s['log_token'], 'role':'DISCOVERY','reference_samples':ref,'context_samples':views,'description':s['description']})
    selected={w['sample_token'] for w in chosen}
    selected.update(n for w in chosen for n in w['reference_samples']+w['context_samples'])
    kept={t:by_sample[t] for t in selected}; poses_needed={r['ego_pose_token'] for d in kept.values() for r in d.values()}
    poses={r['token']:r for r in stream(meta/'ego_pose.json') if r['token'] in poses_needed}
    annotations=collections.defaultdict(list)
    for r in stream(meta/'sample_annotation.json'):
        if r['sample_token'] in selected: annotations[r['sample_token']].append(r)
    index={'root':str(root),'selection':f'log_then_scene_metadata_order; max{a.max_logs} logs,2 scenes/log,2 windows/scene; all DISCOVERY; no predictions accessed', 'windows':chosen,'samples':{t:samples[t] for t in selected},'sample_data':kept,'calibrated':calib,'poses':poses,'annotations':annotations,'inventory':inventory}
    (out/'index.json').write_text(json.dumps(index))
    (out/'v81_scene_registry.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in inventory))
    summary={'available_scenes':len(inventory),'available_logs':len({r['log'] for r in inventory}),'selected_windows':len(chosen),'selected_scenes':len({w['scene'] for w in chosen}),'selected_logs':len(log_counts),'selected_samples':len(selected),'incomplete_reference_windows':sum(len(w['reference_samples'])<2 for w in chosen)}
    (out/'index_summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary),flush=True)
if __name__=='__main__': main()
