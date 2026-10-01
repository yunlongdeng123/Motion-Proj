"""只读检查r6的数据分布、实际损失权重和可训练范围。"""
from pathlib import Path
import json, math, sys
from collections import Counter, defaultdict
import numpy as np
from PIL import Image
from safetensors import safe_open

TASK=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
ROOT=TASK/'r7'
REPO=Path('/root/autodl-tmp/motion_proj_v77')
OFFICIAL=Path('/root/autodl-tmp/external/worldsim_v75_downstream_bench/DriveEditor')
sys.path.insert(0,str(REPO/'scripts/worldsim_v77/target_protected'))
from iteration6.train_pilot import selected

def contextual(k):
    # 只扩展原网络已有的条件、时空attention及入口/出口；不增加架构。
    main=k.startswith('model.diffusion_model.') and '_3d' not in k
    return (main and (('.attn1.' in k or '.attn2.' in k) and any(v in k for v in ('.to_q.','.to_k.','.to_v.','.to_out.'))
                     or k.startswith('model.diffusion_model.input_blocks.0.0.')
                     or k.startswith('model.diffusion_model.out.')
                     or k=='model.diffusion_model.null_emb'))

def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def protected_mask(folder,i,shape,kind):
    paths=sorted((folder/'protected').glob(f'{i:03}*.png'))
    if not paths:assert kind=='background',(folder,i)
    masks=[np.asarray(Image.open(p))>0 for p in paths]
    return np.logical_or.reduce(masks) if masks else np.zeros(shape,dtype=bool)

def main():
    catalog=json.loads((TASK/'r6/dataset_catalog.json').read_text());rows=[];frames={}
    for c in catalog['cases']:
        folder=Path(c['folder']);pair=json.loads((folder/'pair_manifest.json').read_text());stats=[]
        for i in range(c['frame_count']):
            h=np.asarray(Image.open(folder/'model_hole'/f'{i:03}.png'))>0
            b=protected_mask(folder,i,h.shape,c['type'])
            x=np.asarray(Image.open(folder/'X'/f'{i:03}.png')).astype('float32')/127.5-1
            y=np.asarray(Image.open(folder/'Y'/f'{i:03}.png')).astype('float32')/127.5-1
            leak=int(np.count_nonzero((x!=y).any(-1)&~h));x[h]=0;y[h]=0
            assert leak==0 and np.array_equal(x,y)
            s={'hole_fraction':float(h.mean()),'protected_hidden_canvas_fraction':float((h&b).mean()),
               'protected_hidden_ratio':float((h&b).sum()/b.sum()) if b.any() else None,
               'protected_visible_pixels':int((b&~h).sum()),'protected_total_pixels':int(b.sum())}
            stats.append(s);frames[(c['dataset_id'],i)]=s
        rows.append(dict(c,source_id=pair.get('source_id'),donor_source_id=pair.get('donor_source_id'),
                         frame_statistics=stats,mean_hole_fraction=float(np.mean([s['hole_fraction'] for s in stats])),
                         mean_protected_hidden_canvas_fraction=float(np.mean([s['protected_hidden_canvas_fraction'] for s in stats])),
                         min_protected_visible_pixels=min(s['protected_visible_pixels'] for s in stats)))
    steps=[json.loads(s) for s in (TASK/'r6/training/steps.jsonl').read_text().splitlines()];weighted=[]
    for s in steps:weighted += [frames[(s['last_case'],s['last_window']+i)] for i in range(10)]
    census=defaultdict(lambda:{'tensors':0,'parameters':0,'example_keys':[]})
    with safe_open(str(OFFICIAL/'checkpoints/model.safetensors'),framework='pt') as f:
        for k in f.keys():
            if not k.startswith('model.diffusion_model.') or '_3d' in k:continue
            shape=f.get_slice(k).get_shape();n=math.prod(shape)
            group=('spatial_self_attention' if selected(k) else 'temporal_attention' if '.time_stack.' in k and '.attn' in k
                   else 'cross_attention' if '.attn2.' in k else 'other_main_network')
            for name in [group,'official_main_unfrozen' if not any(v in k for v in ('time_embed.','label_emb.')) else 'official_frozen_embeddings'] + (['contextual_control'] if contextual(k) else []):
                v=census[name];v['tensors']+=1;v['parameters']+=n
                if len(v['example_keys'])<4:v['example_keys'].append(k)
    train=[c for c in rows if c['split']=='train'];val=[c for c in rows if c['split']=='validation']
    summary={'case_counts':dict(Counter(c['split'] for c in rows)),
             'train_type_counts':dict(Counter(c['type'] for c in train)),
             'train_receiver_scenes':dict(Counter(c['receiver_scene'] for c in train)),
             'train_unique_receiver_sources':len({c['source_id'] for c in train}),
             'train_unique_receiver_donor_pairs':len({(c['source_id'],c['donor_source_id']) for c in train}),
             'val_receiver_scenes':dict(Counter(c['receiver_scene'] for c in val)),
             'training_step_case_counts':dict(Counter(s['last_case'] for s in steps)),
             'training_seen_frames':len(weighted),
             'step_weighted_hole_fraction':float(np.mean([s['hole_fraction'] for s in weighted])),
             'step_weighted_protected_hidden_canvas_fraction':float(np.mean([s['protected_hidden_canvas_fraction'] for s in weighted])),
             'loss_spatial_weight':'mask_fuse=0, ((mask+1)/2*(5-1)+1)=3 everywhere; normalize by spatial mean => 1 everywhere',
             'synthetic_RGB_leak':0,'GT_role':'real Y unchanged; X masked before condition; no actor RGB artifact reaches condition',
             'parameter_census':dict(census),'limits':'pixel fractions approximate supervision share; VAE spatial mixing means not exact latent loss attribution; technical input pass does not prove statistical coverage or generation learning'}
    dump(ROOT/'data_and_module_audit.json',{'summary':summary,'cases':rows});print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
