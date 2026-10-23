"""R7 incremental source expansion. Immutable R5 inputs; no CUDA or geometry/active-unit changes."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from scipy.ndimage import label
from PIL import Image
from prepare_v41_revision import native_rgb,witness_at,sample_rgb


def main():
    p=argparse.ArgumentParser()
    for key in ('previous','output','engine'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();started=time.perf_counter()
    sys.path.insert(0,str(args.engine))
    from utils.v41_sources import EVIDENCE_VERSION,SOURCE_POLICY,source_consensus,first_two,validate_label_sources
    if args.output.exists():raise FileExistsError('new output only')
    old=json.loads((args.previous/'manifest.json').read_text());meta=json.loads(Path(old['source_bundle']).read_text())
    if old['version']!='V41_FIXED_DENSE_EVIDENCE_R5' or len(old['frame_order'])!=49:raise ValueError('immutable R5 required')
    for folder in ('frames','consensus','overlays','comparison'): (args.output/folder).mkdir(parents=True,exist_ok=True)
    evidence=[];rgb=[]
    for fid in old['frame_order']:
        source=args.previous/'frames'/f'{fid}_evidence.npz'
        with np.load(source,allow_pickle=False) as z:evidence.append((z['box_clear_witness'].copy(),z['measured_patch_depth_m'].copy()))
        shutil.copy2(source,args.output/'frames'/source.name)
        shutil.copy2(args.previous/'overlays'/f'{fid}.png',args.output/'overlays'/f'{fid}.png')
        rgb.append(native_rgb(meta['frames'][fid]['original_rgb']))
    rows=[]
    for i,fid in enumerate(old['frame_order']):
        with np.load(args.previous/'frames'/f'{fid}.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
        legacy=data['deletion_valid'].copy()
        candidates=data['deletion_candidate'] & ~legacy
        yy,xx=np.where(candidates);pixels=np.flatnonzero(candidates)
        frame=meta['frames'][fid];K=np.asarray(frame['intrinsic']);c2w=np.linalg.inv(frame['world_to_camera'])
        rays=np.c_[xx+.5,yy+.5,np.ones(len(xx))] @ np.linalg.inv(K).T
        world=(rays*data['fixed_surface_depth_m'][candidates,None]) @ c2w[:3,:3].T+c2w[:3,3]
        colors=np.full((49,len(xx),3),np.nan,np.float32)
        uvs=np.zeros((49,len(xx),2),np.float32)
        for j,source in enumerate(old['frame_order']):
            if i==j:continue
            clear,depth=evidence[j]
            valid,uv=witness_at(world,meta['frames'][source],clear,depth)
            colors[j,valid]=sample_rgb(rgb[j],uv[valid]);uvs[j,valid]=uv[valid]
        result=source_consensus(colors,i)
        before_spatial=np.zeros((320,480),bool);before_spatial[candidates]=result['accepted']
        added=np.zeros_like(before_spatial)
        for actor in range(1,6):
            components,n=label(before_spatial & (data['actor_labels']==actor),structure=np.ones((3,3),bool))
            sizes=np.bincount(components.ravel());keep=sizes>=SOURCE_POLICY['min_new_component_pixels'];keep[0]=False
            added |= keep[components]
        accepted=added[candidates]
        np.savez_compressed(args.output/'consensus'/f'{fid}.npz',candidate_pixels=pixels,
            colors=colors,source_uv=uvs,accepted_after_spatial=accepted,**result)
        data['deletion_legacy_valid']=legacy
        data['deletion_label_kind']=np.where(legacy,1,np.where(added,2,0)).astype(np.uint8)
        data['deletion_validation_source_ids']=np.full((320,480,2),-1,np.int16)
        fit_ids,check_ids=first_two(result['fit_source_mask']),first_two(result['validation_source_mask'])
        data['deletion_source_ids'][added]=fit_ids[accepted]
        data['deletion_validation_source_ids'][added]=check_ids[accepted]
        for slot in (0,1):
            data['deletion_source_uv'][added,slot]=uvs[fit_ids[accepted,slot],np.flatnonzero(accepted)]
        for key in ('fit_count','fit_inliers','fit_span','validation_count','validation_inliers','validation_mae'):
            field=np.zeros((320,480),np.float32 if key.endswith('mae') else np.int16)
            field[added]=result[key][accepted];data['deletion_'+key]=field
        data['deletion_valid']=legacy|added
        data['deletion_target_rgb'][:,added]=result['target'][accepted].T
        data['deletion_domains']=np.stack([data['deletion_valid'] & (data['actor_labels']==a) for a in range(1,6)])
        validate_label_sources(data,i)
        np.savez_compressed(args.output/'frames'/f'{fid}.npz',**data)
        overlay=rgb[i].copy()
        for mask,color in ((legacy,[.1,.8,.25]),(added,[0,.8,1])):overlay[mask]=.45*overlay[mask]+.55*np.array(color)
        Image.fromarray(np.uint8(overlay.clip(0,1)*255)).save(args.output/'comparison'/f'{fid}.png')
        row=dict(old['frames'][i]); actors={}
        for a,actor in enumerate((5,26,18,6,37)):
            domain=data['actor_labels']==a+1;subset=domain[candidates]
            actors[str(actor)]=dict(target_pixels=int(domain.sum()),legacy=int((legacy&domain).sum()),
                rejected_candidates=int(subset.sum()),fit_consensus=int((result['fit_ok']&subset).sum()),
                heldout_accepted=int((result['accepted']&subset).sum()),added=int((added&domain).sum()),
                total=int((data['deletion_valid']&domain).sum()))
        row.update(deletion_verified_pixels=int(data['deletion_valid'].sum()),
            deletion_by_actor={a:r['total'] for a,r in actors.items()},source_expansion=actors,
            new_source_labels=int(added.sum()),spatial_rejected=int(before_spatial.sum()-added.sum()),
            source_validation_mae_mean=float(result['validation_mae'][accepted].mean()) if accepted.any() else None)
        rows.append(row);print(json.dumps(dict(frame=fid,actors=actors)),flush=True)
    totals={a:{k:sum(r['source_expansion'][a][k] for r in rows) for k in rows[0]['source_expansion'][a]} for a in ('5','26','18','6','37')}
    manifest=dict(old,version=EVIDENCE_VERSION,derived_from=str(args.previous.resolve()),frames=rows,
        policy=dict(old['policy'],source_expansion=SOURCE_POLICY),source_expansion_totals=totals,
        totals=dict(old['totals'],deletion_verified_pixels=sum(r['deletion_verified_pixels'] for r in rows)),
        CPU_seconds=time.perf_counter()-started,source_contract='R5 legacy + disjoint-block validated consensus additions',
        input_roles=dict(old['input_roles'],RGB='R5 labels retained; new colors fitted only from other-frame fit split; validation blocks gate only'),
        GPU_used=False,optimizer_updates=0,quality_accepted=False)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(totals,indent=2))


if __name__=='__main__':main()
