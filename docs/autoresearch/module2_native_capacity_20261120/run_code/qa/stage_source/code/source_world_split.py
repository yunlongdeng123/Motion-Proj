"""Source-only world split. NumPy/SciPy only; never imports a renderer or Torch.

No teacher RGB is read to select this split. Existing F0 provenance is reported,
not rebuilt. Held labels remain preexposed through frozen prior/source history.
"""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
from scipy import ndimage as ndi

Q = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1')
HIGH = Q/'gpu_20261119/quality_bench_20261120_r1/highres'
NEW = Q/'gpu_20261119/native_capacity_20261120_r1'
SENTINEL = np.iinfo(np.int32).min

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4*1024**2),b''): h.update(part)
    return h.hexdigest()

def load_keys(path, keys):
    with np.load(path,allow_pickle=False) as z: return {k:z[k] for k in keys}

def world_partition(xyz, *, block_m=2., origin=(0.,0.), seed=101, scene='scene065'):
    xy=np.asarray(xyz,np.float64)[...,:2]
    if not np.isfinite(xy).all() or block_m<=0: raise ValueError('finite canonical XY and positive block size required')
    blocks=np.floor((xy-np.asarray(origin))/float(block_m)).astype(np.int64)
    unique,inverse=np.unique(blocks.reshape(-1,2),axis=0,return_inverse=True)
    held=[]; records=[]
    for bx,by in unique:
        text=f'{scene}|{int(seed)}|{int(bx)}|{int(by)}'
        digest=hashlib.sha256(text.encode('utf-8')).digest()
        residue=int.from_bytes(digest[:8],'big')%5
        held.append(residue==0)
        records.append(dict(block_xy=[int(bx),int(by)],hash_text=text,sha256=digest.hex(),modulo5=residue,held=residue==0))
    return blocks, np.asarray(held,bool)[inverse].reshape(xy.shape[:-1]), records

def strict_core(context, radius):
    return ndi.minimum_filter(np.asarray(context,np.uint8),size=2*int(radius)+1,mode='constant',cval=0).astype(bool)

def source_corner_masses(uv, ledger):
    uv=np.asarray(uv,np.float64);lo=np.floor(uv).astype(np.int64);f=uv-lo
    H,W=ledger['original_source'].shape
    result={k:np.zeros(len(uv),np.float64) for k in ['corner_mass','common_valid','known','unknown','seed','teacher_H_permission','old_A0_valid']}
    result['source']=np.zeros((len(uv),4),np.float64)
    keys=dict(common_valid='common_material_valid',known='original_known',unknown='original_unknown',seed='base_seed_ledger_mask',teacher_H_permission='teacher_H_allow',old_A0_valid='old_A0_valid')
    for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
        x=lo[:,0]+dx;y=lo[:,1]+dy
        inside=(x>=0)&(x<W)&(y>=0)&(y<H)
        w=(f[:,0] if dx else 1-f[:,0])*(f[:,1] if dy else 1-f[:,1])*inside
        xx=np.clip(x,0,W-1);yy=np.clip(y,0,H-1)
        result['corner_mass']+=w
        s=ledger['original_source'][yy,xx]
        if not np.isin(s,[0,1,2,3]).all(): raise ValueError('unexpected original source code')
        for code in range(4): result['source'][:,code]+=w*(s==code)
        for out,key in keys.items(): result[out]+=w*ledger[key][yy,xx]
    return result

def summarize_source(masses, sparse_mask):
    n=int(sparse_mask.sum())
    return dict(pixels=n,measurement='center-UV four-corner bilinear support; not a full physical-Sigma material attribution',
                original_source_weight_sum={str(c):float(masses['source'][sparse_mask,c].sum()) for c in range(4)},
                mean_support_mass={k:(float(v[sparse_mask].mean()) if n else None) for k,v in masses.items() if k!='source'},
                pixels_with_any_seed=int((masses['seed'][sparse_mask]>1e-12).sum()),
                pixels_with_any_teacher_H_permission=int((masses['teacher_H_permission'][sparse_mask]>1e-12).sum()),
                prior_exposure='preexposed/unknown history; zero new target-label gradient is not a blind test')

def prepare_frame(maps, masks, alpha, ledger, *, sigmas=(1.,2.,4.), alpha_min=.95, seed=101):
    shape=tuple(maps['surface_hit'].shape);ids=maps['pixel_ids'].astype(np.int64)
    if ids.ndim!=1 or len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=np.prod(shape)): raise ValueError('unique camera-local sparse pixel_ids required')
    if not np.array_equal(np.flatnonzero(maps['surface_hit'].ravel()),np.sort(ids)): raise ValueError('pixel_ids do not exactly cover canonical hits')
    if maps['surface_world_xyz'].shape!=(len(ids),3) or maps['center_uv'].shape!=(len(ids),2): raise ValueError('canonical sparse map mismatch')
    if alpha.shape!=shape or not np.isfinite(alpha).all(): raise ValueError('fixed alpha shape/finite mismatch')
    blocks,held_sparse,block_records=world_partition(maps['surface_world_xyz'],seed=seed)
    code=np.zeros(np.prod(shape),np.uint8);code[ids]=np.where(held_sparse,2,1);code=code.reshape(shape)
    bx=np.full(np.prod(shape),SENTINEL,np.int32);by=bx.copy();bx[ids]=blocks[:,0];by[ids]=blocks[:,1]
    out=dict(pixel_ids=ids,world_code=code,world_block_x=bx.reshape(shape),world_block_y=by.reshape(shape),canonical_hit=maps['surface_hit'],fixed_alpha=alpha,
             K=masks['K_vehicle'],source_material_valid=masks['source_material_valid'],image_hw=np.asarray(shape),
             partition_code=np.asarray('0=unmapped,1=seen,2=held'),block_size_m=np.asarray(2.),origin_xy=np.asarray([0.,0.]),seed=np.asarray(seed))
    masses=source_corner_masses(maps['center_uv'],ledger)
    for key,value in masses.items(): out['F0_sparse_'+key+'_mass']=value.astype(np.float32)
    rows={}
    for role in ['A','C','S']:
        domain=np.asarray(masks[role+'_pre_alpha'],bool)
        if domain.shape!=shape:raise ValueError('source role mask shape')
        seen=domain&(code==1);held=domain&(code==2);unmapped=domain&(code==0)
        # No alpha/source threshold is applied to the RGB supervision partition.
        out[role+'_pre_alpha']=domain;out[role+'_seen']=seen;out[role+'_held']=held;out[role+'_unmapped']=unmapped
        row=dict(expected_pre_alpha=int(domain.sum()),seen=int(seen.sum()),held=int(held.sum()),unmapped=int(unmapped.sum()),
                 seen_blocks=int(len(np.unique(blocks[seen.ravel()[ids]],axis=0))),held_blocks=int(len(np.unique(blocks[held.ravel()[ids]],axis=0))),
                 stable_seen=int((seen&(alpha>=alpha_min)).sum()),stable_held=int((held&(alpha>=alpha_min)).sum()),
                 source_valid_seen=int((seen&masks['source_material_valid']).sum()),source_valid_held=int((held&masks['source_material_valid']).sum()),
                 detail={},F0={})
        assert row['seen']+row['held']+row['unmapped']==row['expected_pre_alpha']
        for split,mask in [('seen',seen),('held',held)]:
            context=mask&(alpha>=alpha_min)&~masks['K_vehicle']
            out[f'{role}_{split}_context']=context
            entry=dict(context=int(context.sum()),HF={})
            for sigma in sigmas:
                radius=int(3*float(sigma)+.5);core=strict_core(context,radius)
                label=str(float(sigma)).rstrip('0').rstrip('.')
                out[f'{role}_{split}_HF_core_sigma{label}']=core
                entry['HF'][label]=dict(radius=radius,core=int(core.sum()),guard=int((context&~core).sum()),status='defined' if core.any() else 'N/A_no_strict_core')
            sobel=strict_core(context,1);out[f'{role}_{split}_Sobel_core']=sobel
            # r2 is retained for appearance-edge/NMS evaluation; Sobel training itself is r1.
            out[f'{role}_{split}_appearance_edge_core']=strict_core(context,2)
            entry['Sobel']=dict(radius=1,core=int(sobel.sum()),guard=int((context&~sobel).sum()),status='defined' if sobel.any() else 'N/A_no_strict_core')
            row['detail'][split]=entry;row['F0'][split]=summarize_source(masses,mask.ravel()[ids])
        row['held_domain_interpretation']='prior-conditioned spatial diagnostic only' if row['held_blocks']>=3 else 'N/A_for_domain_generalization; fewer_than3_nonempty_held_blocks'
        rows[role]=row
    return out,dict(shape_HW=list(shape),canonical_pixels=len(ids),world_blocks=block_records,domains=rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--highres',type=Path,default=HIGH);p.add_argument('--ledger',type=Path,default=Q/'permission/final_r1/teacher_evidence.npz')
    p.add_argument('--output',type=Path,default=NEW/'qa/source_split_r1');args=p.parse_args()
    if args.output.exists():raise FileExistsError('preserve prior split; new output directory required')
    began=time.perf_counter();ledger_keys=['original_source','original_known','original_unknown','base_seed_ledger_mask','teacher_H_allow','old_A0_valid','common_material_valid']
    ledger=load_keys(args.ledger,ledger_keys);sources={str(args.ledger):sha(args.ledger)};args.output.mkdir(parents=True)
    rows=[];block_labels={}
    for raw in [86,110]:
        paths=dict(maps=args.highres/f'maps/{raw:03d}.npz',targets=args.highres/f'targets/{raw:03d}.npz',basis=args.highres/f'native_r1/{raw:03d}/road_basis.npz')
        m=load_keys(paths['maps'],['pixel_ids','center_uv','surface_world_xyz','surface_hit'])
        t=load_keys(paths['targets'],['A_pre_alpha','C_pre_alpha','S_pre_alpha','K_vehicle','source_material_valid'])
        alpha=load_keys(paths['basis'],['alpha'])['alpha']
        out,r=prepare_frame(m,t,alpha,ledger)
        for rec in r['world_blocks']:
            key=tuple(rec['block_xy'])
            if key in block_labels and block_labels[key]!=rec['held']:raise AssertionError('crossview world block mismatch')
            block_labels[key]=rec['held']
        dest=args.output/f'{raw:03d}.npz';np.savez_compressed(dest,**out)
        r.update(raw=raw,output=str(dest),output_sha256=sha(dest));rows.append(r)
        for path in paths.values():sources[str(path)]=sha(path)
        print(json.dumps(dict(raw=raw,domains={k:{q:r['domains'][k][q] for q in ['expected_pre_alpha','seen','held','unmapped']} for k in ['A','C','S']})),flush=True)
    receipt=dict(schema='source_world_supervision_split.v1',complete=True,frames=rows,source_SHA256=sources,code_SHA256=sha(__file__),seconds=time.perf_counter()-began,
        assignment=dict(scene='scene065',origin_xy=[0.,0.],world_block_m=[2.,2.],seed=101,encoding='scene065|101|bx|by UTF-8; first8 SHA256 bytes big-endian modulo5 equals0',nominal_held_fraction=.2,observed_held_blocks=sum(block_labels.values()),observed_blocks=len(block_labels),salt_retries=0),
        teacher_RGB_read=False,teacher_labels_used_to_select_split=False,parameter_isolation=False,physical_Sigma_coupling_forbidden=False,
        context_rule='For training, teacher context is seen-world source domain & fixed alpha>=.95 & ~K; held values are masked BEFORE HF/Sobel. Strict finite-kernel cores are saved for every sigma. Held-only context is supplied for diagnostic evaluation.',
        unmapped_rule='Cannot assign missing canonical surface to world blocks. Retain counts/full evaluation denominator; RGB training handling must be registered, do not silently call these pixels seen or held.',
        prior='F0, teacher-derived seeds, permissions and prior development remain preexposed; no trueblind or effective generalization claim',
        source_count_scope='four-corner CENTER support only; not full Sigma/kernel source attribution',
        GPU_executed=False,native_calls=0,optimizer_updates=0,production_asset_modified=False)
    (args.output/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
    print(json.dumps(dict(completed=True,output=str(args.output),seconds=receipt['seconds'])))

if __name__=='__main__':main()
