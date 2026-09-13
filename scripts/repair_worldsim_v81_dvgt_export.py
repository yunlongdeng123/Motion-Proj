"""保留原生推理资产，仅更正DVGT官方坐标、训练尺度和首相机ego时间。"""
import json,shutil,sys,numpy as np
from pathlib import Path
from motion_proj.worldsim_v81.geometry import transform
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_worldsim_v81_inference import select_views,dvgt_camera_points

def main():
    atlas=Path('/root/autodl-tmp/runs/worldsim_v81/WS-V81-CPU-01/20260913-cpu-r2');run=Path('/root/autodl-tmp/runs/worldsim_v81/WS-V81-GPU-P2-01')
    index=json.loads((atlas/'index.json').read_text());target=run/'input_manifests';target.mkdir(exist_ok=True)
    for p in (atlas/'input_manifests').glob('*.json'):
        m=json.loads(p.read_text());m['original_manifest']=str(p)
        for v in m['views']+m['context_views']:
            sd=index['sample_data'][v['sample_token']][v['camera']]
            v['world_from_ego_camera']=transform(index['poses'][sd['ego_pose_token']]).tolist()
            v['camera_ego_timestamp_us']=index['poses'][sd['ego_pose_token']]['timestamp']
        (target/p.name).write_text(json.dumps(m))
    jobs=[json.loads(l) for l in (run/'execution_queue.jsonl').read_text().splitlines()]
    for j in jobs:j['manifest']=str(target/Path(j['manifest']).name)
    (run/'execution_queue.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in jobs))
    audit=[]
    for rp in sorted((run/'dvgt').rglob('result.json')):
        r=json.loads(rp.read_text())
        if r.get('export_contract')=='dvgt_rdf_scale0.1_camera_ego_v1':continue
        m=json.loads((target/(r['window']+'.json')).read_text());views=select_views(m,r['variant'],r.get('anchor_camera'))
        old=run/'retired_contract_r0'/rp.parent.relative_to(run);old.mkdir(parents=True,exist_ok=True)
        shutil.copy2(rp,old/'historical_result.json')
        for p in rp.parent.glob('*depth_z_m.npy'):shutil.copy2(p,old/p.name)
        with np.load(rp.parent/'native_outputs.npz') as native:
            points=native['points'][0];points=points.reshape(-1,*points.shape[-3:])
            for i,v in enumerate(views):
                if v['sample_token']!=m['target_sample']:continue
                depth=dvgt_camera_points(points[i],views[0],v)[...,2].astype('float32')
                np.save(rp.parent/(v['camera']+'_depth_z_m.npy'),depth)
        r.update(export_contract='dvgt_rdf_scale0.1_camera_ego_v1',metric_scale=10.,scale_source='official_gt_scale_factor_0.1_no_lidar_fit',manifest=str(target/(r['window']+'.json')))
        rp.write_text(json.dumps(r,indent=2));audit.append(str(rp))
    (run/'contract_repair.json').write_text(json.dumps({'failure_id':'V81-F03','category':'implementation_export_not_model_failure','reexported':audit,'raw_outputs_unchanged':True,'historical_export_root':str(run/'retired_contract_r0'),'official_sources':['configs/_base_/default_dataset.yaml','tools/data_process/dataloaders/base_loader.py','tools/data_process/dataloaders/nuscene_loader.py','dvgt/datasets/scene_dataset.py','dvgt/models/architectures/dvgt1.py']},indent=2))
    print('reexported',len(audit))

if __name__=='__main__':main()
