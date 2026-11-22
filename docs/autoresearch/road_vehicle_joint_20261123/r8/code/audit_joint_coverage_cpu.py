"""Independent existing-render coverage/occlusion audit; no torch/GPU/optimization."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,hashlib,time,numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt

R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_vehicle_joint_dggt_20261123_r8')
SOURCE=R/'baseline_A0_five_views';OUT=R/'audit/baseline_A0_coverage_v1'
CACHE=R.parent/'road_shared_appearance_20261122_r7/projection_cache_v1'
FRAMES=[86,110,134,179,198]
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def png(path,x):Image.fromarray(np.rint(np.clip(x,0,1)*255).astype(np.uint8)).save(path)
def image(x):return x.transpose(1,2,0) if x.ndim==3 and x.shape[0]==3 else x


def main():
    started=time.time();OUT.mkdir(parents=True,exist_ok=False);allrows=[];panels=[];inputs=[]
    for raw in FRAMES:
        folder=OUT/f'{raw:03d}';folder.mkdir()
        joint_path=SOURCE/f'{raw:03d}/joint_render.npz';component_path=SOURCE/f'{raw:03d}/component_render.npz'
        mask_path=R/f'audit/rectified_masks_v1/{raw:03d}/rectified_masks.npz';view_path=CACHE/f'{raw:03d}/view_cache.npz'
        j=load(joint_path);c=load(component_path);m=load(mask_path);v=load(view_path)
        for path in [joint_path,component_path,mask_path,view_path]:inputs.append(dict(path=str(path),sha256=sha(path)))
        a=j['alpha'].squeeze();ra=c['road_alpha'].squeeze();aa=c['actor_alpha'].squeeze();ja=j['actor_alphaT'].squeeze();jr=j['road_alphaT'].squeeze();jc=j['context_alphaT'].squeeze()
        rgb=image(j['rgb']);photo=v['photo_pinhole_srgb'];valid=m['RGB_valid'].astype(bool);K=m['protected'].astype(bool)
        assert np.array_equal(valid,v['photo_pinhole_valid'])
        original=v['teacher_edit_requested'].astype(bool);expand=distance_transform_edt(~original)<=8
        scopes={'original_C':v['requested_C'].astype(bool),'original_S':v['requested_S'].astype(bool),'original_CS':original,
                'full_CS_dilation8':expand,'full_CS_dilation8_Kfree':expand&~K,'K_rectified':K,
                'actor_visible_proxy':m['actor_visible_proxy'].astype(bool)}
        if 'actor_visible' in m:scopes['actor_visible_frozen_V']=m['actor_visible'].astype(bool)
        # Original photo semantic labels are categorical in Brown coordinates.
        semantic_path=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/semantics')/f'{raw:03d}.png'
        semimage=np.asarray(Image.open(semantic_path));bu=v['photo_source_brown_uv']
        xx=np.floor((bu[...,0]+.5)*semimage.shape[1]/1920).astype(int);yy=np.floor((bu[...,1]+.5)*semimage.shape[0]/1280).astype(int)
        sem=np.full(a.shape,255,np.uint8);inside=valid&(xx>=0)&(xx<semimage.shape[1])&(yy>=0)&(yy<semimage.shape[0]);sem[inside]=semimage[yy[inside],xx[inside]]
        inputs.append(dict(path=str(semantic_path),sha256=sha(semantic_path)))
        scopes['observed_road_semantic0_CS8']=expand&(sem==0)&valid&~K
        frame=[]
        for name,scope in scopes.items():
            n=int(scope.sum());sv=scope&valid
            record=dict(scope=name,pixels=n,RGB_valid_pixels=int(sv.sum()),Brown_invalid_pixels=int((scope&~valid).sum()),
                original_mesh_unprojectable_pixels=int((scope&~v['projection_valid']).sum()),K_pixels=int((scope&K).sum()),
                joint_RGB_L1_on_correct_pinhole=float(abs(rgb[sv]-photo[sv]).mean()) if sv.any() else None,
                RGB_metric_role='same-frame disagreement; original C/S may intentionally remove actor10, not road ground truth',thresholds={})
            for threshold in [.5,.95]:
                record['thresholds'][str(threshold)]=dict(scene_total_alpha_low=int((scope&(a<threshold)).sum()),
                    road_isolated_alpha_low=int((scope&(ra<threshold)).sum()),
                    road_isolated_low_scene_sufficient=int((scope&(ra<threshold)&(a>=threshold)).sum()),
                    road_low_scene_sufficient_context_dominant=int((scope&(ra<threshold)&(a>=threshold)&(jc>=ja)&(jc>=jr)).sum()),
                    actor_isolated_alpha_low=int((scope&(aa<threshold)).sum()),
                    actor_isolated_sufficient_joint_actor_low=int((scope&(aa>=threshold)&(ja<threshold)).sum()),
                    actor_sufficient_joint_low_context_dominant=int((scope&(aa>=threshold)&(ja<threshold)&(jc>ja)).sum()))
            frame.append(record)
        # Coverage proposals are separate from mixture/occlusion diagnostics. No color-based selection.
        trueV=m.get('actor_visible',np.zeros(a.shape,bool)).astype(bool);proxy=m['actor_visible_proxy'].astype(bool)
        road_proposal=expand&valid&~K&(a<.95)&(ja<.05)
        actor_true_proposal=trueV&valid&~K&(aa<.95)
        actor_proxy_proposal=proxy&valid&~K&(aa<.95)
        np.savez_compressed(folder/'pixel_proposals.npz',road_scene_alpha95_holes=road_proposal,
            actor_frozenV_isolated_alpha95_holes=actor_true_proposal,actor_proxy_isolated_alpha95_holes=actor_proxy_proposal,
            CS8=expand,K=K,RGB_valid=valid,original_mesh_unprojectable=~v['projection_valid'])
        inspected=expand|proxy|K;reason=np.zeros(a.shape,np.uint8)
        reason[inspected]=10
        reason[expand&(ra<.95)&(a>=.95)]=8;reason[expand&(ra<.5)&(a>=.95)]=9
        reason[proxy&(aa>=.95)&(ja<.95)]=6;reason[proxy&(aa>=.95)&(ja<.5)]=7
        reason[proxy&(aa<.95)]=4;reason[proxy&(aa<.5)]=5
        reason[inspected&(a<.95)]=2;reason[inspected&(a<.5)]=3;reason[inspected&~valid]=1
        palette=np.asarray([[15,23,34],[130,130,130],[246,126,104],[245,45,65],[245,192,80],[238,134,31],[99,126,224],[76,89,180],[59,194,176],[37,147,126],[54,65,81]],np.float32)/255
        colour=palette[reason];overlay=np.clip(rgb,0,1).copy();take=inspected&(reason!=10);overlay[take]=.3*overlay[take]+.7*colour[take]
        png(folder/'reason_overlay.png',overlay);png(folder/'reason_classes.png',colour);png(folder/'photo_correct_pinhole.png',photo)
        for name,value in [('scene_total_alpha',a),('road_isolated_alpha',ra),('actor_isolated_alpha',aa),('joint_actor_alphaT',ja),('joint_road_alphaT',jr)]:png(folder/(name+'.png'),value)
        ys,xs=np.nonzero(expand|proxy);box=(max(int(xs.min())-8,0),max(int(ys.min())-8,0),min(int(xs.max())+9,a.shape[1]),min(int(ys.max())+9,a.shape[0]))
        panels.append((raw,box,[photo,np.clip(rgb,0,1),overlay,np.repeat(a[...,None],3,2),np.repeat(ra[...,None],3,2),np.repeat(aa[...,None],3,2)]))
        row=dict(raw=raw,scopes=frame,visible_evidence='frozen conservative V' if 'actor_visible' in m else 'instance proxy only, not frozen V',
            proposal_counts=dict(road_scene_alpha95=int(road_proposal.sum()),actor_frozenV_isolated_alpha95=int(actor_true_proposal.sum()),actor_proxy_isolated_alpha95=int(actor_proxy_proposal.sum())),
            owner_alpha_sum_max_abs=float(abs(jr+ja+jc-a).max()),source_native_render_unchanged=True)
        write(folder/'metrics.json',row);allrows.append(row)
    sheet=Image.new('RGB',(6*360,5*235),(15,23,34));draw=ImageDraw.Draw(sheet)
    titles=['correct pinhole photo','shared native RGB','coverage / occlusion','scene alpha','road-only alpha','actor-only alpha']
    for i,(raw,box,images) in enumerate(panels):
        for col,(title,array) in enumerate(zip(titles,images)):
            im=Image.fromarray(np.rint(np.clip(array,0,1)*255).astype(np.uint8)).crop(box);im.thumbnail((360,205));sheet.paste(im,(col*360,i*235+28));draw.text((col*360+4,i*235+6),f'{raw:03d} {title}',fill=(233,240,249))
    sheet.save(OUT/'five_view_coverage_occlusion.png')
    result=dict(schema='R8.independent_corrected_mask_coverage_occlusion_CPU.v1',complete=True,CPU_only=True,new_nativeF=0,new_nativeB=0,optimizer_updates=0,new_teacher_calls=0,
        thresholds=[.5,.95],frames=allrows,inputs=inputs,reason_legend={'1':'Brown invalid','2':'scene alpha .5-.95','3':'scene alpha below .5','4':'actor-only alpha .5-.95','5':'actor-only alpha below .5','6':'actor only sufficient but joint actor below .95','7':'actor sufficient but joint actor below .5','8':'road-only .5-.95 while scene sufficient','9':'road-only below .5 while scene sufficient','10':'other inspected pixels'},
        denominator_policy='All original requested and CS8 denominators retained. Mesh-unprojectable and Brown-invalid counted explicitly. L1 alone uses valid Brown-corrected target pixels.',
        StageB_proposals='pixel domains only; no change applied. Must wait R9 shared-input acceptance and root hole/occlusion review. K actual unified-render contribution protection must be recomputed before updates.',
        black_rim_not_used_for_sigma_selection=True,not_claimed_3D_restoration=True,seconds=time.time()-started)
    write(OUT/'coverage_occlusion_audit.json',result)
    print(json.dumps(dict(output=str(OUT),frames=[dict(raw=x['raw'],proposal_counts=x['proposal_counts']) for x in allrows]),ensure_ascii=False))


if __name__=='__main__':main()
