"""CPU fixed-domain interpretation of actual native ideal read controls; no optimization."""
from pathlib import Path
from types import SimpleNamespace
import csv,json,sys
import numpy as np
from PIL import Image,ImageDraw

B=Path('/root/autodl-tmp/drivingeditor_repro/campaign');Q=B/'road_teacher_detail_cpu_20261119_r1'
C=B/'road_capacity_cpu_20261117_r1';G=B/'road_capacity_gpu_20261118_r1'
O=Q/'gpu_20261119/ideal_read_probe';FRAMES=(86,110,134,179,198)
sys.path.insert(0,str(G/'code'));sys.path.insert(0,str(C/'ideal_code'))
from ideal_capacity_metrics import DetailMetrics
from ideal_world_teacher import evaluate_world

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def metric(a,b,m):
    e=np.asarray(a,np.float64)[...,m]-np.asarray(b,np.float64)[...,m]
    return dict(pixels=int(m.sum()),RMSE=float(np.sqrt(np.mean(e*e))) if m.any() else None,MAE=float(np.abs(e).mean()) if m.any() else None)

def main():
    cfg=json.loads((C/'ideal_targets_r2/field_config.json').read_text());names=['marking_w120','marking_w240']
    geos={r:load(C/'ideal_targets_r2/geometry'/f'{r:03d}.npz') for r in FRAMES}
    masks={r:g['fixed_evaluation_domain']&g['canonical_mesh_hit_valid']&(g['fixed_alpha']>.9) for r,g in geos.items()}
    if sum(m.sum() for m in masks.values())!=37206:raise RuntimeError('Frozen D_color denominator changed')
    target={name:{r:load(C/'ideal_targets_r2'/name/f'{r:03d}.npz')for r in FRAMES}for name in names}
    data=SimpleNamespace(root=C/'ideal_targets_r2',field=cfg,cases=names,masks=masks,geometry=geos,target=lambda name:target[name])
    detail=DetailMetrics(data);rows=[];images={};normalized={};source=[];png=[]
    for mode in ('road','joint'):
        for method in ('center_source_grid','average_GH31','pixel_box'):
            for name in names:normalized[(mode,method,name)]={}
    for raw in FRAMES:
        pixel=load(O/f'{raw:03d}_pixel.npz');actual=load(O/'rendered'/f'{raw:03d}.npz');M=masks[raw]
        for mode in ('road','joint'):
            basis=load(Q/'gpu_20261119/native_qualification_r2/basis'/f'{raw:03d}'/f'{mode}_basis.npz');w=basis['eligible_weight'].astype(np.float64)
            if mode=='road': images[raw]={}
            for index,name in enumerate(names,1):
                truth=pixel['independent_world_center_rgb'][index].transpose(2,0,1).astype(np.float64)-pixel['independent_world_center_rgb'][0].transpose(2,0,1)
                paint=target[name][raw]['paint_fraction'];asphalt=pixel['independent_world_center_rgb'][0].transpose(2,0,1).astype(np.float64)
                delta_actual=actual[f'avg_{mode}_rgb'][index].astype(np.float64)-actual[f'avg_{mode}_rgb'][0].astype(np.float64)
                deltas=dict(average_GH31=delta_actual,
                    pixel_box=w[None]*(pixel['rgb'][index]-pixel['rgb'][0]).transpose(2,0,1),
                    center_source_grid=w[None]*(pixel['center_rgb'][index]-pixel['center_rgb'][0]).transpose(2,0,1))
                d=np.asarray(cfg['paint_srgb'])[:,None,None]-asphalt
                for method,delta in deltas.items():
                    norm=np.divide(delta,w[None],out=np.zeros_like(delta),where=w[None]>1e-12)
                    # Reuse the original detail evaluator on the paint increment. The real readonly term cancels.
                    normalized[(mode,method,name)][raw]=(geos[raw]['fixed_alpha'][None]*(asphalt+norm)).astype(np.float32)
                    n=metric(delta,w[None]*truth,M)
                    den=float(np.sum((w[None]*truth[:,M])**2)) if False else float(np.sum((w[None]*truth)[:,M]**2))
                    gain=float(np.sum(delta[:,M]*(w[None]*truth)[:,M])/den) if den>0 else None
                    row=dict(raw=raw,mode=mode,case=name,method=method,increment_RGB=n,
                        target_weight='Actual eligible '+mode+' alphaT; readonly asphalt and paint columns subtract exactly in real arithmetic',
                        target_center_sampling=True,material_zero_pixels_in_D=int(((pixel['mass']<=1e-12)&M).sum()),
                        eligible_weight_zero_in_D=int(((w==0)&M).sum()),eligible_weight_mean_in_D=float(w[M].mean()),increment_projection_gain=gain)
                    rows.append(row)
                    if mode=='road':
                        fraction=np.divide(np.sum(norm*d,axis=0),np.sum(d*d,axis=0),out=np.zeros_like(w),where=np.sum(d*d,axis=0)>1e-12)
                        images[raw][(name,method)]=np.clip(fraction,0,1)
                source.append(dict(raw=raw,mode=mode,case=name,grid_center_vs_continuous_center=metric(
                    (pixel['center_rgb'][index]-pixel['center_rgb'][0]).transpose(2,0,1),truth,M),
                    pixel_box_vs_grid_center=metric((pixel['rgb'][index]-pixel['rgb'][0]).transpose(2,0,1),
                        (pixel['center_rgb'][index]-pixel['center_rgb'][0]).transpose(2,0,1),M)))
        canvas=Image.new('RGB',(480*3,342*2),(5,8,12));draw=ImageDraw.Draw(canvas)
        for j,name in enumerate(names):
            for i,label in enumerate(('independent center paint','average_GH31','pixel_box')):
                a=target[name][raw]['paint_fraction'] if i==0 else images[raw][(name,label)]
                # Display the same frozen D_color. Everything outside remains black.
                rgb=np.rint(np.repeat(np.where(M,a,0)[...,None],3,axis=2)*255).astype(np.uint8)
                canvas.paste(Image.fromarray(rgb),(i*480,j*342+22));draw.text((i*480+5,j*342+4),f'{name}: {label}',fill=(225,230,240))
        path=O/'previews'/f'{raw:03d}_paint_increment.png';path.parent.mkdir(exist_ok=True);canvas.save(path);png.append(str(path))
    details={}
    for key,imgs in normalized.items():
        details['/'.join(key)]=detail.evaluate(key[-1],imgs)
    for row in rows:
        d=details['/'.join((row['mode'],row['method'],row['case']))]
        frame=next(r for r in d['frames'] if r['raw']==row['raw'])
        row['fixed_detail_metrics_after_readonly_cancellation']=frame
    result=dict(schema='ideal_native_reader_control.v1',completed=True,scientific_training_updates=0,
        reference='Old independent ideal world function, actual canonical G_s, same 12/24cm world width/phase and native sRGB. Current source grid is 2cm; independent pixel-center truth is not GS self-rendered.',
        role='Zero-update synthetic reader/filter control, not actual-data quality, not optimization/representation upper-bound certificate.',
        fixed_D_color_pixels=sum(int(m.sum())for m in masks.values()),fixed_D_full_pixels=sum(int(g['fixed_evaluation_domain'].sum())for g in geos.values()),
        source_filter='Common valid normalization; GH31 average across fixed world covariance then true native per-GS RGB raster. Pixel uses cached true G_s and uncapped anisotropic pixel-box, followed by actual road/joint eligible unit weights.',
        comparison='Native paint minus native no-paint asphalt versus matching pixel delta. Details use delta/actual eligible weight only for diagnosis, with zero-weight pixels reported and kept in fixed domain; no training unpremultiplication.',
        sampling_convention='Independent center-sample target retained. Pixel-box and 2cm bilinear source may attenuate near-3px paint; this is reported, not a hidden changed GT.',
        numerical_identity='Native basis was separately tested with actual constant/unit color passes, max direct-readonly reconstruction 9.5367e-7; no bitexact claim.',
        metrics=rows,source_grid_and_pixel_footprint_diagnostics=source,detail_by_mode_method_case=details,previews=png,
        native_forward_calls=30,native_backward_calls=0,optimizer_updates=0,human_verdict=None,production_promoted=False)
    (O/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (O/'metrics.csv').open('w',newline='')as f:
        cols=['raw','mode','case','method','increment_RMSE','increment_MAE','projection_gain','contrast_retention','edge_F1','center_error_max_px','detail_pass','zero_weight_D']
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for r in rows:
            d=r['fixed_detail_metrics_after_readonly_cancellation'];w.writerow(dict(raw=r['raw'],mode=r['mode'],case=r['case'],method=r['method'],
                increment_RMSE=r['increment_RGB']['RMSE'],increment_MAE=r['increment_RGB']['MAE'],projection_gain=r['increment_projection_gain'],
                contrast_retention=d['contrast_retention'],edge_F1=d['edge_F1'],center_error_max_px=d['line_center_error_max_px'],detail_pass=d['passed'],zero_weight_D=r['eligible_weight_zero_in_D']))
    for name in names:
        for method in ('center_source_grid','average_GH31','pixel_box'):
            d=details['/'.join(('road',method,name))]
            print(name,method,[(r['raw'],r['contrast_retention'],r['edge_F1'],r['passed'])for r in d['frames']])

if __name__=='__main__':main()
