"""CPU-only R8 road/BG pool and fixed target binding, without training edits."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,ast,copy,hashlib,json,os
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,q):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):
    p=Path(p).resolve();return dict(path=p.as_posix(),sha256=sha(p))
def load(p):
    with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def candidate_ids(arrays):
    if 'source_candidate_ids' in arrays:
        return np.asarray(arrays['source_candidate_ids'], np.int64)
    n = len(arrays['xyz']); raw = np.asarray(arrays.get('source_raw', np.zeros(n)), np.int64)
    uv = np.asarray(arrays.get('source_uv', arrays.get('source_pixel_xy', np.zeros((n, 2)))), np.float64)
    xy = np.rint(uv).astype(np.int64)
    return raw * 100000000 + xy[:, 1] * 10000 + xy[:, 0]
def runtime_candidate_function(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='candidate_ids')
    ns={'np':np};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(path),'exec'),ns);return ns['candidate_ids']
def default(key,z,n):
    if key=='source_candidate_ids':return candidate_ids(z)
    if key=='coeffs':return z['coefficients'].copy()
    if key in ['teacher_color','fused_teacher_color']:return z['rgb'].copy()
    if key=='independent_teacher_support_count':return np.zeros(n,np.uint8)
    if key=='geometry_cell_flat_row_ids':return np.full(n,-1,np.int64)
    if key=='source_is_real_background':return np.zeros(n,bool)
    if key=='prior_xyz':return z['xyz'].copy()
    if key=='prior_world_xyz':return z['world_xyz'].copy()
    if key in ['normal_local','prior_normal']:return z.get('normal_local',z.get('prior_normal',np.tile([0,0,1.],(n,1)))).astype(np.float32)
    if key=='confidence':return np.ones(n,np.float32)
    raise AssertionError(('No justified initial_tensors-compatible default',key))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--road-pool',type=Path,required=True);ap.add_argument('--road-targets',type=Path,required=True);ap.add_argument('--runtime-source',type=Path,required=True);a=ap.parse_args();R=a.root.resolve()
    bg_manifest_path=R/'manifest.json';bg_manifest=read(bg_manifest_path);assert bg_manifest['complete']and bg_manifest['source_photo_only']and not bg_manifest['teacher_generated']
    bg_pool_path=Path(bg_manifest['candidate_pool']['path']);assert sha(bg_pool_path)==bg_manifest['candidate_pool']['sha256']
    old=load(a.road_pool);bg=load(bg_pool_path);nr=len(old['xyz']);nb=len(bg['xyz'])
    exact=runtime_candidate_function(a.runtime_source);road_ids=exact(old);assert same(road_ids,candidate_ids(old))
    bg_ids=exact(bg);assert same(bg_ids,bg['source_candidate_ids'])and (bg_ids>=10**12).all()
    assert not np.intersect1d(road_ids,bg_ids).size and len(np.unique(road_ids))==nr and len(np.unique(bg_ids))==nb
    assert not bg['source_independent_teacher'].any()and bg['source_is_real_background'].all()
    old['source_candidate_ids']=road_ids;old['source_is_real_background']=np.zeros(nr,bool)
    fields=sorted(set(old)|set(bg));combined={};defaults=[]
    for key in fields:
        x=old.get(key);y=bg.get(key)
        if x is None:x=default(key,old,nr);defaults.append(dict(group='road',field=key))
        if y is None:y=default(key,bg,nb);defaults.append(dict(group='real_background',field=key))
        assert x.ndim and y.ndim and len(x)==nr and len(y)==nb and x.shape[1:]==y.shape[1:]and x.dtype==y.dtype,key
        combined[key]=np.concatenate([x,y],0)
    for key,value in load(a.road_pool).items():assert same(combined[key][:nr],value),('Original road field changed',key)
    assert same(combined['source_candidate_ids'][:nr],road_ids)and same(combined['source_candidate_ids'][nr:],bg_ids)
    assert combined['source_independent_teacher'][:nr].all()and not combined['source_independent_teacher'][nr:].any()
    assert not combined['source_is_real_background'][:nr].any()and combined['source_is_real_background'][nr:].all()
    assert all(np.isfinite(v).all()for v in combined.values())
    pool=R/'initialization/combined_candidate_pool.npz';assert not pool.exists(),'Preserve a prior combined pool; do not overwrite';np.savez_compressed(pool,**combined)
    targets=copy.deepcopy(read(a.road_targets));original=read(a.road_targets);bg_rows={int(x['raw']):x for x in bg_manifest['frames']};train=targets['training_frames'];assert sorted(bg_rows)==sorted(train)
    counts=[]
    for row in targets['frames']:
        raw=int(row['raw'])
        if raw not in train:assert not row['categories'];continue
        source=bg_rows[raw]['real_visible'];p=Path(source['path']);assert sha(p)==source['sha256'];z=load(p);assert z['support'].dtype==bool and np.array_equal(z['loss_weight'],z['support'].astype(np.float32))
        row['background_completion']=copy.deepcopy(source);row['categories']['real_visible_background']=copy.deepcopy(source)
        row['background_completion_weight']=1.;row['background_completion_source']='same-position original-photo canonical linear RGB; not an independent generated teacher'
        before=next(x for x in original['frames']if x['raw']==raw)
        for key,value in before['categories'].items():assert row['categories'][key]==value
        counts.append(dict(raw=raw,real_visible_background_pixels=int(z['support'].sum()),weight=1.0))
    targets.setdefault('category_weights',{}).update(real_visible_background=1.)
    targets.update(background_completion_manifest=pin(bg_manifest_path),background_seed=copy.deepcopy(bg_manifest['seed']),background_candidate_pool=copy.deepcopy(bg_manifest['candidate_pool']),combined_candidate_pool=pin(pool),background_independent_generated_teacher=False)
    dest=R/'config/targets8.remote.json'
    if dest.exists():
        backup=dest.with_name(dest.name+'.before_background_binding.'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));backup.write_bytes(dest.read_bytes())
    write(dest,targets)
    record=dict(schema='scene065.r8.combined_inputs.v1',complete=True,CPU_only=True,GPU_calls=0,road_pool=pin(a.road_pool),BG_pool=bg_manifest['candidate_pool'],BG_seed=bg_manifest['seed'],BG_source_manifest=pin(bg_manifest_path),candidate_ID_reference_runtime=pin(a.runtime_source),
        combined_pool=pin(pool),target_bindings=pin(dest),original_road_rows=nr,real_BG_rows=nb,total_rows=nr+nb,all_original_road_fields_byte_equal=True,original_road_candidate_IDs_exact=True,road_candidate_IDs_SHA256=hashlib.sha256(road_ids.tobytes()).hexdigest(),BG_candidate_IDs_preserved=True,BG_is_independent_teacher=False,BG_is_real_background=True,defaults=defaults,background_target_counts=counts,
        road_categories_unchanged=True,heldout_categories_empty=True,runtime_objective_driver_modified=False)
    write(R/'analysis/combined_inputs_receipt.json',record);print(json.dumps(record,ensure_ascii=False))
if __name__=='__main__':main()
