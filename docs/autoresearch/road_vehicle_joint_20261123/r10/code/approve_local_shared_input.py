"""Record root's actual five-view shared-material input review."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):return dict(path=str(p),sha256=sha(p))
def main():
    p=argparse.ArgumentParser()
    for k in ['r10','r9','numeric','visual','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();fp=a.r10/'local_shared_continuous_v1/final_atlas.npz'
    assert sha(fp)=='eb86cb6d61b711366d8c30052fd02b823433a6e940b80d6f4e0463685e38a5dd'
    for gp in [a.numeric,a.visual]:
        q=read(gp);assert q['complete'] and q['decision']=='PASS'
        assert q.get('final_atlas_sha256',q.get('field_sha256'))==sha(fp)
    with np.load(fp,allow_pickle=False) as f,np.load(a.r9/'joint_shared_continuous_v2/final_atlas.npz',allow_pickle=False) as old:
        e=f['permission_mask']&~old['permission_mask'];rgb=f['atlas_linear']
        assert e.sum()==12328 and np.isfinite(rgb).all() and rgb.min()>=0 and rgb.max()<=1
        assert rgb[~e].tobytes()==old['atlas_linear'][~e].tobytes()
        assert f['real_locks'].tobytes()==old['real_locks'].tobytes()
        assert rgb[f['real_locks']].tobytes()==old['atlas_linear'][old['real_locks']].tobytes()
        counts=dict(extension_texels=int(e.sum()),extension_teacher_dominant_kind3=int((e&(f['source_kind']==3)).sum()),
            extension_direct_teacher_support=int((e&f['teacher_directly_observed']).sum()),real_locks=int(f['real_locks'].sum()))
    sp=a.r10/'local_shared_surface_review_v1/receipt.json';s=read(sp)
    assert s['final_atlas_sha256']==sha(fp) and len(s['views'])==5
    assert all(x['unavailable_material_pixels']==0 and x['surface_dark_maxRGB_below40']==0 for x in s['views'])
    out=dict(schema='R10.root_shared_material_input_review.v1',complete=True,decision='PASS',final_atlas_sha256=sha(fp),
        local_patch_appearance_accepted=True,student_fit_authorized=True,currently_authorized_optimizer_updates=0,
        current_action='build the persistent prototype and render the zero-step joint control; inspect actual joint output before any optional fit',
        source_field=pin(fp),numeric_review=pin(a.numeric),independent_visual_review=pin(a.visual),direct_surface_review=pin(sp),counts=counts,
        root_visual_views=[86,110,134,179,198],root_visual_observation='Within the exact declared asset scope, the photographic road texture has no obvious returning vehicle black rim, large colour blocks or stripe artefacts. Display cut-outs are outside the asset scope, not black RGB material.',
        root_numeric_parent_and_real_lock_byteexact=True,source_interpretation='Similar-photograph seed grain conditioned on the existing shared boundary; jointly calibrated raw teachers add weak continuous colour influence. Not a teacher-dominant reconstruction.',
        LF_edge_p95_is_reported_without_HF_blurring=True,
        direct_surface_is_not_Gaussian_joint_render=True,whole_scene_quality_accepted=False,default_promotion=False,
        geometry_approval_is_separate=True,new_diffusion_calls=0,Gaussian_updates=0,GPU_forward=0,GPU_backward=0)
    assert not a.output.exists();a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(complete=True,decision='PASS',counts=counts)))

if __name__=='__main__':main()
