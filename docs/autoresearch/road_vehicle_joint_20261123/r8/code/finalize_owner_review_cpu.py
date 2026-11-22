"""Make review figures/receipts from existing owner buffers only; no renderer."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
from PIL import Image, ImageDraw


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    r=a.root;o=r/'audit/owner_unification_CPU_v1';cfg=read(r/'baseline_contract.json')
    raw_results=read(o/'owner_unification_audit.json');owner=int(cfg['actors'][0]['owner_id'])
    sheet=Image.new('RGB',(2040,600),(15,23,34));d=ImageDraw.Draw(sheet)
    labels=['correct pinhole photo','actual shared RGB','centers: red Ktouch / amber held or unknown',
            'old native dynamic alphaT','static unowned alphaT','unified actor alphaT (PNG estimate)']
    for row,raw in enumerate([86,110]):
        cache=load(Path(cfg['camera_audit_cache'])/f'{raw:03d}/view_cache.npz')
        mask=load(r/f'audit/rectified_masks_v1/{raw:03d}/rectified_masks.npz')['actor_visible'].astype(bool)
        y,x=np.nonzero(mask);box=(int(x.min())-18,int(y.min())-18,int(x.max())+19,int(y.max())+19)
        photo=Image.fromarray(np.rint(np.clip(cache['photo_pinhole_srgb'],0,1)*255).astype('uint8'))
        j=load(r/f'baseline_A0_five_views/{raw:03d}/joint_render.npz')
        rgb=Image.fromarray(np.rint(np.clip(j['rgb'].transpose(1,2,0),0,1)*255).astype('uint8'))
        measured=r/f'owner_measurement_A0_v1/{raw:03d}'
        images=[photo,rgb,Image.open(o/f'{raw:03d}/native_same_instance_center_markers.png'),
                Image.open(measured/f'native_dynamic_owner{owner}_alphaT.png').convert('RGB'),
                Image.open(measured/'native_static_unowned_alphaT.png').convert('RGB'),
                Image.open(o/f'{raw:03d}/unified_owner_alphaT_PNG_estimate.png')]
        for col,(image,label) in enumerate(zip(images,labels)):
            crop=image.crop(box);crop.thumbnail((330,225));sheet.paste(crop,(col*340+5,row*300+38))
            d.text((col*340+5,row*300+8),f'{raw:03d} {label}',fill=(228,237,247))
        q=raw_results['frame_results'][str(raw)]['exact_unified_owner_integral']
        d.text((8,row*300+273),f"strictV low-single-field pixels {q['low_owner_trueV_pixels']} | unified mean alphaT {q['unified_mean_alphaT']:.6f} | RGB/alpha unchanged | old K guard retained",fill=(183,210,232))
    sheet.save(o/'same_instance_owner_review.png')
    # Correct a stale raw-semantic inventory sentence; preserve the original.
    doc=r/'audit/baseline_A0_coverage_v1/README.md'
    old='原native owner21拓扑仍保留静态1465、动态5052行，其中1909行为K-positive保留；只有共享渲染贡献测量才能判断当前可见来源。'
    new='旧CPU清单的静态1465、动态5052仅为显式语义21行，不包含v44_factory将动态semantic0恢复为源实例身份的source_fallback。实际086加载动态owner21为27691行；共享贡献测量中只有412行对严格V低归属域有正贡献。后续实际逐行测量及同实例归属核对见../owner_unification_CPU_v1/README.md，不能把旧6517清单当作实际加载总数。'
    text=doc.read_text(encoding='utf-8')
    if old in text:
        backup=doc.with_name(doc.name+'.before_owner_inventory_correction')
        if not backup.exists(): backup.write_bytes(doc.read_bytes())
        doc.write_text(text.replace(old,new),encoding='utf-8')
    files=[o/'owner_unification_audit.json',o/'same_instance_native_candidates.npz',o/'same_instance_owner_review.png',o/'README.md',o/'architecture_components.svg',
           r/'audit/joint_ownership_interface_v1/joint_ownership_plan.json',r/'audit/joint_ownership_interface_v1/native_update_candidates.npz',
           r/'code/audit_owner_unification_cpu.py',r/'code/joint_ownership_interface.py',r/'code/finalize_owner_review_cpu.py']
    sources=[]
    for raw in [86,110]:
        for name in ['actual_contributing_rows.npz','receipt.json']:
            p=r/f'owner_measurement_A0_v1/{raw:03d}'/name;sources.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
    result=dict(schema='R8.owner_unification_review_delivery.v1',complete=True,CPU_only=True,newF=0,newB=0,updates=0,
        prior_actual_measurement_F=2,prior_actual_measurement_B=2,R8_cumulative_F=27,R8_cumulative_B=2,
        figures_not_asset_changes=True,geometry_color_opacity_unchanged=True,whole_actor_deletion=False,
        actual_row_sources=sources,files=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in files],
        proposal='unify same-instance semantic readout; preserve providers and old K-positive guard; no training before R9 and complete K evidence',
        native_gate_candidates_after_old_guard=0,all_five_view_K_evidence_ready=False,training_enabled=False)
    (o/'owner_review_delivery.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(complete=True,receipt=str(o/'owner_review_delivery.json'),sha256=sha(o/'owner_review_delivery.json'),figure=str(o/'same_instance_owner_review.png'))))


if __name__=='__main__': main()
