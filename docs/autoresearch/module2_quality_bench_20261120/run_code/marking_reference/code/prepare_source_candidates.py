"""仅真实T_A来源的标线候选资格；不读取预测RGB、不导入Torch/native。"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,time
import numpy as np
from scipy import ndimage as ndi
from PIL import Image,ImageDraw,ImageFont

B=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1/gpu_20261119/quality_bench_20261120_r1')
OUT=B/'marking_reference/source_only_r1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def display(a):return Image.fromarray(np.clip(np.floor(a*255+.5),0,255).astype(np.uint8),'RGB')

def main():
    start=time.time();OUT.mkdir(parents=True,exist_ok=False)
    write(OUT/'started.json',dict(UTC=datetime.now(timezone.utc).isoformat(),selection='source-only before native predicted RGB inspection',
        eligibility='A_pre_alpha & surface_hit & fixedroadalpha>=.95 & ~K_vehicle',
        candidate='encoded-sRGB luma>=.65 and max(RGB)-min(RGB)<=.15; candidate only, never automatic lane GT',
        source_material_valid='diagnostic only; not a hidden veto on real-A',native_calls=0,optimizer_updates=0,prediction_RGB_arrays_read=False))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16);rows=[]
    for raw in (86,110):
        target=B/'highres/targets'/f'{raw:03d}.npz';packet=B/'highres/rendered/M11'/f'{raw:03d}.npz'
        with np.load(target,allow_pickle=False)as z:
            # No T_C/T_S data are read. The native T_A source has the registered real-observation role.
            rgb=z['T_A'].transpose(1,2,0).astype(np.float64);A=z['A_pre_alpha'];hit=z['surface_hit'];K=z['K_vehicle'];material=z['source_material_valid']
        with np.load(packet,allow_pickle=False)as z:
            # The only method packet array read is fixed geometry coverage, not any predicted color.
            alpha=z['road_alpha'].astype(np.float64)
        shape=(1280,1920)
        if rgb.shape!=(*shape,3)or any(a.shape!=shape for a in (A,hit,K,material,alpha)):raise ValueError('Native source shape mismatch')
        if any(a.dtype!=bool for a in (A,hit,K,material))or not np.isfinite(alpha).all():raise ValueError('Invalid source masks/alpha')
        eligible=A&hit&(alpha>=.95)&~K
        if not np.isfinite(rgb[eligible]).all():raise ValueError('Nonfinite real A source inside eligible domain')
        luma=rgb@np.array([.2126,.7152,.0722]);spread=rgb.max(-1)-rgb.min(-1)
        candidate=eligible&(luma>=.65)&(spread<=.15)
        labels,n=ndi.label(candidate,structure=np.ones((3,3),np.uint8));objects=ndi.find_objects(labels)
        context={r:ndi.minimum_filter(eligible.astype(np.uint8),2*r+1,mode='constant',cval=0).astype(bool)for r in (2,3,6,12)}
        components=[]
        for i,sl in enumerate(objects,1):
            if sl is None:continue
            local=labels[sl]==i;yy,xx=np.nonzero(local);yy=yy+sl[0].start;xx=xx+sl[1].start;npx=len(xx)
            pts=np.c_[xx,yy].astype(float);center=pts.mean(0);centered=pts-center
            if npx>=2:
                eig,axes=np.linalg.eigh(centered.T@centered/max(npx,1));proj=centered@axes[:,::-1]
                extent=np.ptp(proj,axis=0)+1
            else:eig=np.array([0.,0.]);extent=np.array([1.,1.])
            components.append(dict(component_id=i,pixels=npx,bbox_xyxy=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],
                center_xy=center.tolist(),principal_extent_px=extent.tolist(),PCA_eigenvalues=eig.tolist(),
                full_context_pixels={str(r):int(context[r][yy,xx].sum())for r in context},
                source_material_valid_pixels=int(material[yy,xx].sum()),role='bright-neutral real-A candidate, not semantic lane GT',
                verification_status='PENDING_SOURCE_VISUAL_REVIEW'))
        components.sort(key=lambda c:(-c['pixels'],c['component_id']))
        original=display(rgb);original.save(OUT/f'{raw:03d}_T_A.png')
        # Overlay is a visualization only. All candidate computation used original float T_A.
        overlay=np.asarray(original).copy().astype(float);overlay[~eligible]*=.30
        overlay[eligible]=overlay[eligible]*.70+np.array([0,150,220])*.30
        overlay[candidate]=np.array([255,224,32]);overlay[K]=overlay[K]*.35+np.array([190,30,45])*.65
        ov=Image.fromarray(np.clip(overlay,0,255).astype(np.uint8));dr=ImageDraw.Draw(ov)
        for c in components[:12]:
            x0,y0,x1,y1=c['bbox_xyxy'];dr.rectangle((x0,y0,x1-1,y1-1),outline=(255,235,120),width=2)
            dr.text((x0,max(0,y0-18)),f"#{c['component_id']} n={c['pixels']}",font=font,fill=(255,245,160))
        ov.save(OUT/f'{raw:03d}_A_candidate_overlay.png')
        for c in components[:12]:
            box=c['bbox_xyxy'];pad=32;crop=[max(0,box[0]-pad),max(0,box[1]-pad),min(1920,box[2]+pad),min(1280,box[3]+pad)]
            w,h=crop[2]-crop[0],crop[3]-crop[1]
            canvas=Image.new('RGB',(2*w+12,h+28),(10,17,27));canvas.paste(original.crop(crop),(0,28));canvas.paste(ov.crop(crop),(w+12,28));dd=ImageDraw.Draw(canvas)
            dd.text((3,3),f'real T_A raw{raw} candidate#{c["component_id"]}',font=font,fill=(235,240,250));dd.text((w+15,3),'A/hit/alpha95 + bright candidate',font=font,fill=(235,240,250))
            name=f'{raw:03d}_candidate_{c["component_id"]:04d}.png';canvas.save(OUT/name);c['preview_file']=name;c['preview_box_xyxy']=crop
        np.savez_compressed(OUT/f'{raw:03d}_source_masks.npz',eligible_real_A=eligible,bright_neutral_candidate=candidate,component_labels=labels,
            A_pre_alpha=A,surface_hit=hit,K_vehicle=K,source_material_valid=material,road_alpha=alpha)
        row=dict(raw=raw,width=1920,height=1280,target_source=str(target),target_SHA256=sha(target),coverage_source=str(packet),coverage_source_SHA256=sha(packet),
            prediction_packet_keys_read=['road_alpha'],teacher_keys_read=['T_A','A_pre_alpha','surface_hit','K_vehicle','source_material_valid'],
            T_A_role='real_observation; root-registered native original observation target',
            counts=dict(A_pre_alpha=int(A.sum()),surface_hit_in_A=int((A&hit).sum()),eligible_real_A=int(eligible.sum()),
                bright_candidates=int(candidate.sum()),candidate_components=n,eligible_material_unknown=int((eligible&~material).sum())),
            components=components,thresholds=dict(luma_min=.65,RGB_spread_max=.15,road_alpha_min=.95),
            no_morphological_bridge_or_color_threshold_search=True,shape_reference_status='N_A_PENDING_SOURCE_VISUAL_QUALIFICATION',
            lane_GT_created=False,human_verdict=None)
        write(OUT/f'{raw:03d}_candidates.json',row);rows.append(row)
        print('SOURCE_CANDIDATES',raw,row['counts'],'largest',components[:4],flush=True)
    write(OUT/'receipt.json',dict(completed=True,rows=rows,seconds=time.time()-start,code_SHA256=sha(__file__),
        candidate_selection_frozen_before_source_preview=True,predicted_RGB_inspected=False,C_or_S_generated_line_used=False,
        bench_domains_or_metrics_modified=False,source_only_qualification=True,native_calls=0,GPU_operations=0,optimizer_updates=0,power_operations=0))

if __name__=='__main__':main()
