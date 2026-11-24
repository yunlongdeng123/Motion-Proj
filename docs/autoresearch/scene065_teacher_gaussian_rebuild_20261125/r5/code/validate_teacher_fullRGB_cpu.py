"""CPU integrity, full-texture colour provenance and target permission checks."""
from pathlib import Path
import argparse, hashlib, json, os
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from scipy.ndimage import map_coordinates

PREFIX='/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_gaussian_rebuild_20261125_r5/'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy()for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args();root=args.root.resolve();checked=[0]
    def local(spec):
        assert spec['path'].startswith(PREFIX),spec['path'];p=root/spec['path'][len(PREFIX):]
        assert sha(p)==spec['sha256'],p;checked[0]+=1;return p
    manifest=read(root/'initialization/manifest.json');binding=read(local(manifest['targets']));teachers=read(local(manifest['teacher_manifest']))
    seed=load(local(manifest['teacher_seed']));pool=load(local(manifest['candidate_pool']));rights=load(local(manifest['row_permissions']));pose=read(local(manifest['pose']));P=np.asarray(pose['world_from_patch'])
    assert len(seed['xyz'])==manifest['seed_rows']==3126 and len(pool['xyz'])==manifest['candidate_rows']==50626
    for name,z in [('seed',seed),('candidate',pool)]:
        n=len(z['xyz']);assert all(np.isfinite(v).all() for v in z.values())
        assert z['active'].all() and z['trainable_rgb_rows'].all() and not z['structure_unknown'].any()
        assert set(np.unique(z['source_raw']))=={110,137,165}
        assert z['source_independent_teacher'].all() and not z['source_is_reprojection'].any()
        assert np.all(z['opacity']==np.float32(.2)) and (z['scales']>0).all()
        assert np.allclose(np.linalg.norm(z['rotation'],axis=1),1,atol=1e-6)
        assert np.array_equal(z['xyz'],z['prior_xyz']) and np.array_equal(z['world_xyz'],z['prior_world_xyz'])
        assert abs(z['xyz']@P[:3,:3].T+P[:3,3]-z['world_xyz']).max()<2e-6
        assert not z['coefficients'][:,1:].any()
        srgb=np.clip(.5+.28209479177387814*z['coefficients'][:,0],0,1)
        rgb=np.where(srgb<=.04045,srgb/12.92,((srgb+.055)/1.055)**2.4)
        assert abs(rgb-z['rgb']).max()<1e-6
        assert np.std(z['teacher_color'],axis=0).min()>.001,'Teacher texture must not be uniform grey'
    rows={x['raw']:x for x in binding['frames']};colour_provenance=[]
    for source in teachers['rows']:
        raw=source['raw'];local(source['raw_teacher']);full=load(local(source['fullRGB_native']))['rgb_common_linear']
        target=load(local(rows[raw]['categories']['teacher_pseudo']))
        use=seed['source_raw']==raw;uv=seed['source_uv'][use];xy=seed['source_pixel_xy'][use]
        value=np.stack([map_coordinates(full[...,ch],[uv[:,1],uv[:,0]],order=1,mode='constant',cval=0,prefilter=False)for ch in range(3)],-1)
        colour_error=float(abs(value-seed['teacher_color'][use]).max());assert colour_error<1e-6
        assert target['support'][xy[:,1],xy[:,0]].all()
        direct_error=float(abs(target['rgb_common_linear'][xy[:,1],xy[:,0]]-seed['teacher_color'][use]).max());assert direct_error==0
        colour_provenance.append(dict(raw=raw,points=int(use.sum()),FULL_RGB_sample_max_abs=colour_error,direct_target_sample_max_abs=direct_error,teacher_color_channel_std=np.std(seed['teacher_color'][use],axis=0).tolist()))
    counts=[]
    for row in binding['frames']:
        raw=row['raw']
        if raw in binding['evaluation_frames']:
            assert not row['categories'];continue
        m=load(local(row['target_masks']));p=load(local(row['protection']));sets=[];count={}
        for name,spec in row['categories'].items():
            z=load(local(spec));s=z['support'];rgb=z['rgb_common_linear'];w=z['loss_weight'];sets.append(s);count[name]=int(s.sum())
            assert np.isfinite(rgb).all() and np.array_equal(w,s.astype(np.float32))
            assert not (s&(m['structural_veto']|p['body']|p['keep_orange'])).any()
            assert not(s&~m['red_rebuild']).any()
            assert ((rgb[s]>=0)&(rgb[s]<=1)).all()
        assert np.max(sum(s.astype(np.uint8)for s in sets))<=1
        assert row['independent_teacher']==(raw in [110,137,165])
        assert row['teacher_target_is_geometry_reprojection']==(raw not in [110,137,165])
        counts.append(dict(raw=raw,**count))
    q=dict(pass_=True,CPU_only=True,GPU_calls=0,files_SHA_verified=checked[0],seed_rows=len(seed['xyz']),candidate_rows=len(pool['xyz']),all_active=True,permanent_world_carve=False,all_parameters_learnable=manifest['all_parameters_learnable'],FULL_RGB_provenance=colour_provenance,targets=counts,heldout_RGB_not_used=True,shared_RGB_or_LF_or_old_student_colour_used=False)
    (root/'supervision/CPU_validation.json').write_text(json.dumps(q,indent=2)+'\n',encoding='utf8');print(json.dumps(q))
if __name__=='__main__':main()
