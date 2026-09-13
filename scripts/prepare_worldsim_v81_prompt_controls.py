"""相同 INPUT_PROMPT 点数、不同空间覆盖；不冒充 DriveMVS。"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from motion_proj.worldsim_v81.geometry import transform,apply,project,remove_boxes,roi_mask,coverage

def spread(ids,uv,k):
    coords=uv[ids];chosen=[int(np.argmin(np.linalg.norm(coords-coords.mean(0),axis=1)))];dist=np.linalg.norm(coords-coords[chosen[0]],axis=1)
    for _ in range(k-1):
        j=int(np.argmax(dist));chosen.append(j);dist=np.minimum(dist,np.linalg.norm(coords-coords[j],axis=1));dist[chosen]=-1
    return ids[chosen]

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run)
    idx=json.loads((run/'index.json').read_text());root=Path(idx['root']);rows={r['roi_id']:r for r in map(json.loads,(run/'v81_roi_registry.jsonl').read_text().splitlines())};selected=json.loads((run/'case_selection.json').read_text());registry=[]
    for case in selected:
        row=rows[case['roi_id']];m=json.loads((run/'input_manifests'/f"{row['window_id']}.json").read_text());s=m['target_sample'];record=idx['sample_data'][s]['LIDAR_TOP'];c=idx['calibrated'][record['calibrated_sensor_token']];T=transform(idx['poses'][record['ego_pose_token']])@transform(c)
        raw=np.fromfile(root/record['filename'],np.float32).reshape(-1,5)[:,:3];world=remove_boxes(apply(raw.astype(float),T),idx['annotations'].get(s,[]))
        v=next(v for v in m['views'] if v['camera']==row['camera']);uv,z,xyz=project(world,np.array(v['world_from_camera']),np.array(v['K']));visible=roi_mask(uv,z,[0,0,1600,900]);inside=roi_mask(uv,z,row['box']);ins=np.flatnonzero(inside);outs=np.flatnonzero(visible&~inside);allids=np.flatnonzero(visible)
        k=min(32,len(ins),len(outs));rng=np.random.default_rng(8101)
        if k<8:registry.append({'roi_id':row['roi_id'],'status':'INSUFFICIENT_INPUT_PROMPT','count':k});continue
        variants={'inside_spread':spread(ins,uv,k),'outside_concentrated':outs[np.argsort(np.linalg.norm(uv[outs]-uv[outs].mean(0),axis=1))[:k]],'roi_dropout_outside_spread':spread(outs,uv,k),'random_same_count':rng.choice(allids,k,replace=False)}
        folder=run/'interventions'/row['roi_id'];folder.mkdir(exist_ok=True)
        for name,ids in variants.items():
            path=folder/f'prompt_{name}.npz';np.savez_compressed(path,world_points=world[ids].astype('float32'),uv=uv[ids].astype('float32'),input_sample=np.array(s),input_file=np.array(record['filename']),selected_indices_after_static_filter=ids)
            registry.append({'roi_id':row['roi_id'],'variant':name,'count':k,'points_inside_roi':int(inside[ids].sum()),'roi_spatial_coverage':coverage(uv[ids],row['box']),'path':str(path),'input_role':'INPUT_PROMPT','heldout_points_used':False,'method':'INPUT_ONLY_NOT_DRIVEMVS','status':'READY'})
        if row['cohort']=='C10':
            fig,axs=plt.subplots(2,2,figsize=(12,8));rgb=Image.open(v['image'])
            for ax,(name,ids) in zip(axs.flat,variants.items()):
                ax.imshow(rgb);ax.scatter(uv[ids,0],uv[ids,1],s=9,c='#ff512f');x0,y0,x1,y1=row['box'];ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color='#11d0b0',lw=2));ax.set_title(f'{name}: {k} points, {inside[ids].sum()} in ROI');ax.axis('off')
            fig.suptitle('Same point count, different anchor placement — input control only');fig.tight_layout();fig.savefig(run/'figures/prompt_placement.png',dpi=130);plt.close(fig)
    (run/'prompt_interventions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in registry));print(json.dumps({'variants':len(registry),'ready':sum(r['status']=='READY' for r in registry)}))
if __name__=='__main__':main()
