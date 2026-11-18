"""CPU-only, output-independent road texture diagnosis of frozen 2-D teacher PNGs.

Run: python work/pixel_diagnosis/diagnose_pixels.py
Input originals are never edited. All ROI decisions use masks and clean/reference
RGB only; teacher output is read only after masks / patch coordinates are frozen.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"
import argparse, hashlib, json, math, platform, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter
from skimage.color import rgb2lab
import cv2
cv2.setNumThreads(1)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BASE = Path(r"C:/Users/dengyunlong/Documents/Codex/2026-11-07/https-arxiv-org-html-2509-19937/outputs/road_mask_decoupling_20261116/assets/readability_20261118")
OLDRESULT = BASE.parents[3] / "work/road_mask_decoupling_20261116/evaluation/results_r1/results.json"
HP_NATIVE_SIGMAS = (1.25, 2.5, 5.0, 10.0)
PSD_BANDS = ((.025, .0625), (.0625, .125), (.125, .2), (.2, .5))
PATCH = 80
STRIDE = 40

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_rgb(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)

def resize_rgb(a, wh, method=Image.Resampling.BILINEAR):
    # Exact historical PIL uint8 quantization, rather than float interpolations.
    a = np.clip(np.rint(a), 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(a).resize(wh, method), dtype=np.float32)

def block4(a):
    h,w = a.shape[:2]
    return a.reshape(h//4,4,w//4,4,3).mean(axis=(1,3), dtype=np.float32)

def mask_grid(m, wh):
    # Keep a pixel only if >99.9% of its native support is certified.
    return cv2.resize(m.astype(np.float32), wh, interpolation=cv2.INTER_AREA) > .999

def lum(a):
    return a @ np.asarray([.2126,.7152,.0722],dtype=np.float32)

def structure_mask(reference):
    # Exclude white/yellow paint and strong low-frequency edges (lane / crack /
    # shadow edges), solely from clean RGB in synthetic cases and observation
    # in the proxy reference. The rule does NOT inspect generated RGB.
    y=lum(reference)
    r,g,b=np.moveaxis(reference,-1,0)
    paint=((y>130)&((np.max(reference,axis=-1)-np.min(reference,axis=-1))<45)) | ((r>130)&(g>115)&(b<.8*g))
    low=gaussian_filter(y,2.5,mode="reflect")
    dy,dx=np.gradient(low)
    coarse_edge=np.hypot(dx,dy)>2.0
    bad=paint|coarse_edge
    return distance_transform_edt(~bad)>12, {"paint_pixels":int(paint.sum()),"coarse_edge_pixels":int(coarse_edge.sum())}

def choose_tiles(mask, patch=PATCH, stride=STRIDE):
    h,w=mask.shape
    tiles=[]
    for y in range(0,h-patch+1,stride):
        for x in range(0,w-patch+1,stride):
            if mask[y:y+patch,x:x+patch].all():
                tiles.append((x,y,patch))
    return tiles

def hp_metrics(img, mask, spacing, output_distance):
    y=lum(img)
    results={}
    for sig_native in HP_NATIVE_SIGMAS:
        sig=sig_native/spacing
        # Do not call a subpixel Gaussian residual a resolved texture band.
        if sig<.49:
            continue
        support=max(12.0,4*sig_native)
        safe=mask & (output_distance>support)
        if safe.sum()<64:
            results[str(sig_native)]={"valid_pixels":int(safe.sum()),"status":"insufficient"}
            continue
        residual=y-gaussian_filter(y,sig,mode="reflect",truncate=4)
        v=residual[safe].astype(np.float64)
        clip=float(np.quantile(np.abs(v),.95))
        results[str(sig_native)]={"valid_pixels":int(safe.sum()),"rms_Y_0_255":float(np.sqrt(np.mean(v*v))),
            "winsor95_rms_Y_0_255":float(np.sqrt(np.mean(np.minimum(np.abs(v),clip)**2))),
            "abs_p95":clip,"top5_energy_fraction":float(np.sum(v[np.abs(v)>clip]**2)/max(np.sum(v*v),1e-20))}
    return results

def color_metrics(img, ref, mask):
    n=int(mask.sum())
    if n<64:
        return {"valid_pixels":n,"status":"insufficient"}
    a=img[mask].astype(np.float64); b=ref[mask].astype(np.float64)
    la=rgb2lab((a/255).reshape(-1,1,3)).reshape(-1,3)
    lb=rgb2lab((b/255).reshape(-1,1,3)).reshape(-1,3)
    de=np.linalg.norm(la-lb,axis=-1)
    return {"valid_pixels":n,"rgb_mae_0_255":float(np.abs(a-b).mean()),"mean_RGB":a.mean(axis=0).tolist(),
      "reference_mean_RGB":b.mean(axis=0).tolist(),"mean_RGB_bias":(a-b).mean(axis=0).tolist(),
      "mean_Lab_bias":(la-lb).mean(axis=0).tolist(),"deltaE76_mean":float(de.mean()),
      "deltaE76_median":float(np.median(de)),"deltaE76_of_mean_Lab":float(np.linalg.norm((la-lb).mean(axis=0)))}

def patch_psd(img, tiles, spacing):
    spectra=[]; energies=[]; rms=[]
    yimg=lum(img)
    radial_edges=np.linspace(0,.5,41)
    for x,y,p in tiles:
        x,y,p=int(round(x/spacing)),int(round(y/spacing)),int(round(p/spacing))
        a=yimg[y:y+p,x:x+p].astype(np.float64)
        if a.shape!=(p,p): continue
        yy,xx=np.mgrid[:p,:p]
        xx=(xx-(p-1)/2)/p;yy=(yy-(p-1)/2)/p
        design=np.stack([np.ones_like(xx),xx,yy],axis=-1).reshape(-1,3)
        a=a-(design@np.linalg.lstsq(design,a.ravel(),rcond=None)[0]).reshape(a.shape)
        window=np.outer(np.hanning(p),np.hanning(p))
        power=np.abs(np.fft.fft2(a*window))**2/(p*p*np.sum(window**2))
        fy=np.fft.fftfreq(p,d=spacing);fx=np.fft.fftfreq(p,d=spacing)
        rad=np.hypot(fy[:,None],fx[None,:])
        nyq=.5/spacing
        band=[]
        for lo,hi in PSD_BANDS:
            if lo>=nyq-1e-9: band.append(None); continue
            ok=(rad>=lo)&(rad<min(hi,nyq))
            band.append(float(power[ok].sum()))
        energies.append(band);rms.append(float(np.sqrt(np.mean(a*a))))
        spectra.append([float(np.mean(power[(rad>=lo)&(rad<hi)])) if lo<nyq and np.any((rad>=lo)&(rad<hi)) else None
                        for lo,hi in zip(radial_edges[:-1],radial_edges[1:])])
    out={"patches":len(energies),"native_patch_wh":PATCH,"spacing_native_px":spacing,"nyquist_cycles_per_native_px":.5/spacing,
         "band_edges_cycles_per_native_px":[list(v) for v in PSD_BANDS]}
    if not energies: out["status"]="insufficient";return out
    out["band_power_mean_Y_squared"]= [float(np.mean([e[i] for e in energies])) if energies[0][i] is not None else None for i in range(len(PSD_BANDS))]
    out["band_power_median_Y_squared"]= [float(np.median([e[i] for e in energies])) if energies[0][i] is not None else None for i in range(len(PSD_BANDS))]
    out["patch_detrended_rms_mean"]=float(np.mean(rms))
    out["radial_frequency_midpoints"]=((radial_edges[:-1]+radial_edges[1:])/2).tolist()
    out["radial_power_mean"]=[float(np.mean([s[i] for s in spectra if s[i] is not None])) if any(s[i] is not None for s in spectra) else None for i in range(len(radial_edges)-1)]
    return out

def measure(img, ref, native_mask, output_dist, spacing, tiles):
    wh=(img.shape[1],img.shape[0]); mask=mask_grid(native_mask,wh)
    dist=cv2.resize(output_dist.astype(np.float32),wh,interpolation=cv2.INTER_NEAREST)
    return {"wh":list(wh),"color":color_metrics(img,ref,mask),"HP":hp_metrics(img,mask,spacing,dist),"PSD":patch_psd(img,tiles,spacing)}

def divide(a,b):
    return float(a/b) if a is not None and b is not None and b>1e-12 else None

def paired_summary(c):
    m=c['metrics'];out={}
    if c['domain']=='synthetic_visible_2d':
        for arm in ['A','B']:
            d={}
            for grid,teacher,ref in [('model768',arm+'_model','GT_model_bilinear'),('native1920',arm+'_native','GT_native'),('sample480',arm+'_sample_historical','GT_sample_area')]:
                a,b=m[teacher],m[ref]
                d[grid]={"deltaE76_mean":a['color'].get('deltaE76_mean'),"mean_Lab_bias":a['color'].get('mean_Lab_bias'),
                    "HP_RMS_ratio":{s:divide(a['HP'][s].get('rms_Y_0_255'),b['HP'].get(s,{}).get('rms_Y_0_255')) for s in a['HP']},
                    "HP_winsor95_ratio":{s:divide(a['HP'][s].get('winsor95_rms_Y_0_255'),b['HP'].get(s,{}).get('winsor95_rms_Y_0_255')) for s in a['HP']},
                    "PSD_band_power_ratio":[divide(v,w) for v,w in zip(a['PSD'].get('band_power_mean_Y_squared',[]),b['PSD'].get('band_power_mean_Y_squared',[]))]}
            out[arm]=d
        out['resolution_only_native_HP_ratio']={s:divide(m['GT_model_bilinear_up_native']['HP'][s].get('rms_Y_0_255'),m['GT_native']['HP'][s].get('rms_Y_0_255')) for s in m['GT_native']['HP']}
        out['resolution_only_sample_HP_ratio']={s:divide(m['GT_model_bilinear_sample_historical']['HP'][s].get('rms_Y_0_255'),m['GT_sample_area']['HP'][s].get('rms_Y_0_255')) for s in m['GT_sample_area']['HP']}
        out['native_teacher_vs_resolution_matched_GT_HP_ratio']={arm:{s:divide(m[arm+'_native']['HP'][s].get('rms_Y_0_255'),m['GT_model_bilinear_up_native']['HP'][s].get('rms_Y_0_255')) for s in m['GT_native']['HP']} for arm in ['A','B']}
    return out

def preview_case(case, reference, observation, raw_a, raw_b, mask, tiles, proxy_tiles=None):
    # A scientific contact sheet, with the fixed mask and all selected patches.
    yy,xx=np.where(mask)
    if not len(xx): return
    x0=max(0,int(xx.min())-80);x1=min(mask.shape[1],int(xx.max())+81)
    y0=max(0,int(yy.min())-80);y1=min(mask.shape[0],int(yy.max())+81)
    columns=[('Clean GT' if case['domain']=='synthetic_visible_2d' else 'Observation (proxy only)',reference),('Teacher A native',raw_a),('Teacher B native',raw_b)]
    panels=[]
    for label,a in columns:
        im=Image.fromarray(a.astype(np.uint8));draw=ImageDraw.Draw(im)
        for x,y,p in tiles: draw.rectangle((x,y,x+p-1,y+p-1),outline=(255,210,55),width=3)
        if proxy_tiles:
            for x,y,p in proxy_tiles: draw.rectangle((x,y,x+p-1,y+p-1),outline=(0,240,220),width=3)
        im=im.crop((x0,y0,x1,y1));im.thumbnail((560,400),Image.Resampling.LANCZOS)
        panels.append((label,im))
    cw=max(v.width for _,v in panels);ch=max(v.height for _,v in panels)
    canvas=Image.new('RGB',(cw*3,ch+65),(20,24,31));draw=ImageDraw.Draw(canvas)
    draw.text((8,5),case['id']+' | yellow: all fixed target tiles; cyan: same-height proxy tiles',fill=(235,235,235))
    for i,(label,im) in enumerate(panels):
        draw.text((i*cw+8,25),label,fill=(235,235,235));canvas.paste(im,(i*cw,50))
    canvas.save(HERE/(case['id']+'_fixed_tiles.png'))
    Image.fromarray(mask.astype(np.uint8)*255).save(HERE/(case['id']+'_analysis_mask.png'))

def main():
    manifest=json.loads((BASE/'source_manifest.json').read_text(encoding='utf-8-sig'))
    old=json.loads(OLDRESULT.read_text(encoding='utf-8-sig'))
    oldcases={c['id']:c for c in old['cases']}
    output={"schema":"teacher_pixel_diagnosis.v1","python":sys.version,"cpu_only":True,"threads":1,
      "source_manifest":str(BASE/'source_manifest.json'),"source_manifest_sha256":sha(BASE/'source_manifest.json'),
      "selection":{"primary":"target_core & W_output & certified road, excluding retained K; native distance to W exterior >12 px; clean-reference paint/coarse-edge exclusion with 12 px distance", "paint_rule":"Y>130 and channel range<45, or R>130 G>115 B<0.8G; Y=0.2126R+0.7152G+0.0722B, RGB uint8", "coarse_edge_rule":"norm gradient of Gaussian Y sigma=2.5 native >2.0/px; remove within12 native px", "tiles":"all globally anchored80x80 native tiles, stride40, fully inside fixed safe mask; never inspect teacher RGB for selection", "HP":"Y-Gaussian(Y), sRGB intensity0..255; per-scale W-boundary distance>max(12,4sigma_native); ordinary and95% winsorized RMS", "PSD":"all fixed tiles, linear-plane detrending, Hann periodogram, density units Y squared; radial bands capped at axis Nyquist", "aggregation":"case macro averages; 4 synthetic cases share only2 base frames, no independent n=4 significance"},
      "sampling":{"native":"PIL BILINEAR uint8 768x512 ->1920x1280; verified against raw_native", "historical_sample":"reshape(H/4,4,W/4,4,3).mean((1,3)) float32 ->480x320", "direct_area":"cv2.INTER_AREA float32; contrast uses exact full frame no crop", "frequency_limits_cycles_per_native_px":{"native_raster":.5,"model_independent_axis_bandwidth":.2,"sample_axis_nyquist":.125}},"cases":[]}
    filepins=[]
    for case in manifest['cases']:
        cid=case['id'];print('CASE',cid,flush=True)
        common=dict(np.load(BASE/case['common']))
        roles=dict(np.load(BASE/case['arms']['A']['roles']))
        obs=load_rgb(BASE/case['observation'])
        base=roles['target_core']&common['W_output']&~common['K']
        dist=distance_transform_edt(common['W_output'])
        target_safe=base&(dist>12)
        entry={"id":cid,"domain":case['domain'],"counts":{"target_core":int(roles['target_core'].sum()),"W":int(common['W_output'].sum())},"metrics":{},"frozen_target_tiles":[],"source_pins":[],"audit":{}}
        m=entry['metrics']
        if case['domain']=='synthetic_visible_2d':
            frame='000' if 'raw000' in cid else '050'
            clean=HERE/'truth'/('scene081_raw'+frame+'_clean.jpg')
            cert=HERE/'truth'/('scene081_raw'+frame+'_certificate.png')
            pins=oldcases[cid]['GT_source_pins']
            for local, pin in [(clean,pins['clean_rgb_for_scoring_only']),(cert,pins['visible_road_certificate_for_scoring_only'])]:
                got=sha(local)
                if got!=pin['sha256']: raise RuntimeError('GT pin mismatch: '+str(local))
                p={"local":str(local),"remote":pin['path'],"sha256":got};entry['source_pins'].append(p)
            reference=load_rgb(clean);certmask=np.asarray(Image.open(cert))>0
            if certmask.ndim==3:certmask=certmask[...,0]
            structural_ok,scount=structure_mask(reference)
            mask=target_safe&certmask&structural_ok
            entry['counts'].update({"certificate_on_target":int((base&certmask).sum()),"safe_pixels":int(mask.sum()),**scount})
            tiles=choose_tiles(mask)
            entry['frozen_target_tiles']=[list(t) for t in tiles]
            gtmod=resize_rgb(reference,(768,512));gtarea=cv2.resize(reference,(768,512),interpolation=cv2.INTER_AREA)
            gtup=resize_rgb(gtmod,(1920,1280));gtsamp=block4(reference);gtupsamp=block4(gtup)
            variants=[('GT_native',reference,reference,1),('GT_model_bilinear',gtmod,gtmod,2.5),('GT_model_area',gtarea,gtmod,2.5),
              ('GT_model_bilinear_up_native',gtup,reference,1),('GT_sample_area',gtsamp,gtsamp,4),
              ('GT_model_bilinear_sample_historical',gtupsamp,gtsamp,4),
              ('GT_model_area_sample_historical',block4(resize_rgb(gtarea,(1920,1280))),gtsamp,4)]
            for name,img,ref,spacing in variants:m[name]=measure(img,ref,mask,dist,spacing,tiles)
            araw=None;braw=None
            for arm in ['A','B']:
                a=load_rgb(BASE/case['arms'][arm]['raw_model']);native=load_rgb(BASE/case['arms'][arm]['raw']);final=load_rgb(BASE/case['arms'][arm]['final'])
                predicted=resize_rgb(a,(1920,1280))
                entry['audit'][arm]={"raw_native_vs_PIL_bilinear_uint8_maxdiff":float(np.abs(predicted-native).max()),"final_vs_raw_inside_W_maxdiff":float(np.abs(final[common['W_output']]-native[common['W_output']]).max()),"final_vs_observation_outside_W_maxdiff":float(np.abs(final[~common['W_output']]-obs[~common['W_output']]).max())}
                hist=block4(native);direct=cv2.resize(a,(480,320),interpolation=cv2.INTER_AREA)
                for name,img,ref,spacing in [(arm+'_model',a,gtmod,2.5),(arm+'_native',native,reference,1),(arm+'_sample_historical',hist,gtsamp,4),(arm+'_sample_direct_area',direct,gtsamp,4)]:m[name]=measure(img,ref,mask,dist,spacing,tiles)
                smask=mask_grid(mask,(480,320))
                entry['audit'][arm]['historical_sample_vs_direct_area']=color_metrics(hist,direct,smask)
                if arm=='A':araw=native
                else:braw=native
            preview_case(case,reference,obs,araw,braw,mask,tiles)
            entry['paired_summary']=paired_summary(entry)
        else:
            # Real target has no clean RGB or asphalt certificate. Same-y visible
            # context is a proxy only; no error / accuracy claim is made.
            structural_ok,scount=structure_mask(obs)
            road=common['Cmat_A']&~common['H_B']&~common['K']
            for key in ['shadow_possible','shadow_uncertain','shadow_boundary_guard','foreground_uncertainty']:
                road&=~roles[key]
            proxy_ok=(distance_transform_edt(road)>12)&structural_ok
            mask=target_safe&roles['road_support']
            target_tiles=choose_tiles(mask);all_refs=choose_tiles(proxy_ok)
            rows={}
            for t in all_refs:rows.setdefault(t[1],[]).append(t)
            pairs=[]
            for t in target_tiles:
                candidates=rows.get(t[1],[])
                if candidates:
                    r=min(candidates,key=lambda u:(abs(u[0]-t[0]),u[0]))
                    pairs.append((t,r))
            tiles=[p[0] for p in pairs];refs=[p[1] for p in pairs]
            entry['frozen_target_tiles']=[list(t) for t in tiles]
            entry['frozen_proxy_tiles']=[list(t) for t in refs]
            entry['counts'].update({'safe_target_pixels':int(mask.sum()),'target_tiles_before_proxy_match':len(target_tiles),'matched_tiles':len(pairs),'safe_visible_proxy_pixels':int(proxy_ok.sum())})
            entry['limitation']='Unlabelled real hidden road; same-height nearest certified visible-road patch is a texture/material proxy, not same-location truth. Generated lines or structures in target cannot be fully excluded without output-dependent selection.'
            entry['proxy_horizontal_distance_native_px']=[abs(t[0]-r[0]) for t,r in pairs]
            # These PSDs use exactly corresponding physical locations / supports;
            # all color differences are patch aggregate, not image reconstruction.
            for grid,spacing,reference in [('native',1,obs),('model',2.5,resize_rgb(obs,(768,512))),('sample',4,block4(obs))]:
                m['proxy_'+grid]={'PSD':patch_psd(reference,refs,spacing)}
                for arm in ['A','B']:
                    a=load_rgb(BASE/case['arms'][arm]['raw_model'] if grid=='model' else BASE/case['arms'][arm]['raw'])
                    if grid=='sample':a=block4(a)
                    m[arm+'_'+grid]={'PSD':patch_psd(a,tiles,spacing)}
                    if pairs:
                        target_pixels=[];ref_pixels=[]
                        for t,r in pairs:
                            x,y,p=[int(round(v/spacing)) for v in t];rx,ry,rp=[int(round(v/spacing)) for v in r]
                            target_pixels.append(a[y:y+p,x:x+p].reshape(-1,3));ref_pixels.append(reference[ry:ry+rp,rx:rx+rp].reshape(-1,3))
                        aa=np.concatenate(target_pixels);bb=np.concatenate(ref_pixels)
                        la=rgb2lab((aa/255).reshape(-1,1,3)).reshape(-1,3);lb=rgb2lab((bb/255).reshape(-1,1,3)).reshape(-1,3)
                        m[arm+'_'+grid]['proxy_color']={'mean_RGB_bias':(aa-bb).mean(0).tolist(),'mean_Lab_bias':(la-lb).mean(0).tolist(),'deltaE76_of_mean_Lab':float(np.linalg.norm((la-lb).mean(0)))}
            preview_case(case,obs,obs,load_rgb(BASE/case['arms']['A']['raw']),load_rgb(BASE/case['arms']['B']['raw']),mask,tiles,refs)
        for p in [case['observation'],case['common'],case['arms']['A']['roles']]+[case['arms'][a][k] for a in ['A','B'] for k in ['raw_model','raw','final']]:
            filepins.append({'relative':p,'sha256':sha(BASE/p)})
        output['cases'].append(entry)
        (HERE/'pixel_diagnosis.json').write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding='utf-8')
    output['source_files_sha256']=filepins
    output['script_sha256']=sha(__file__)
    synth=[c for c in output['cases'] if c['domain']=='synthetic_visible_2d']
    macro={}
    for arm in ['A','B']:
        macro[arm]={}
        for grid in ['model768','native1920','sample480']:
            ss=[c['paired_summary'][arm][grid] for c in synth]
            macro[arm][grid]={"case_count":len(ss),"base_frame_count":2,"deltaE76_mean_macro":float(np.mean([s['deltaE76_mean'] for s in ss])),
             "mean_Lab_bias_macro":np.mean([s['mean_Lab_bias'] for s in ss],axis=0).tolist(),
             "HP_RMS_ratio_macro":{s:float(np.mean([v['HP_RMS_ratio'][s] for v in ss if v['HP_RMS_ratio'][s] is not None])) for s in ss[0]['HP_RMS_ratio']},
             "HP_winsor95_ratio_macro":{s:float(np.mean([v['HP_winsor95_ratio'][s] for v in ss if v['HP_winsor95_ratio'][s] is not None])) for s in ss[0]['HP_winsor95_ratio']},
             "PSD_band_power_ratio_macro":[float(np.mean([v['PSD_band_power_ratio'][i] for v in ss if v['PSD_band_power_ratio'][i] is not None])) if any(v['PSD_band_power_ratio'][i] is not None for v in ss) else None for i in range(4)]}
    output['synthetic_macro']=macro
    (HERE/'pixel_diagnosis.json').write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding='utf-8')
    make_chart(output)
    print(json.dumps({'synthetic_macro':macro,'tiles':{c['id']:len(c['frozen_target_tiles']) for c in output['cases']}},indent=2),flush=True)

def make_chart(output):
    plt.style.use('dark_background')
    fig,axs=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    synth=[c for c in output['cases'] if c['domain']=='synthetic_visible_2d']
    colors={'A':'#f7c75e','B':'#6ed8da'}
    labels=['000 v0','000 v1','050 v0','050 v1']
    x=np.arange(4)
    for arm,off in [('A',-.13),('B',.13)]:
        vals=[c['paired_summary'][arm]['model768']['deltaE76_mean'] for c in synth]
        axs[0].bar(x+off,vals,width=.25,label=arm,color=colors[arm])
        vals=[c['paired_summary'][arm]['model768']['HP_RMS_ratio']['2.5'] for c in synth]
        axs[1].plot(x,vals,'o-',label=arm,color=colors[arm])
        c=synth[0]
        p=c['metrics'][arm+'_model']['PSD']
        axs[2].semilogy(p['radial_frequency_midpoints'],p['radial_power_mean'],label='teacher '+arm,color=colors[arm])
    axs[0].set_title('Same-location color error at 768x512');axs[0].set_ylabel('mean Delta E 76');axs[0].set_xticks(x,labels);axs[0].legend()
    axs[1].set_title('Texture RMS / clean GT at 768x512');axs[1].set_ylabel('HP sigma=1 model px ratio');axs[1].set_xticks(x,labels);axs[1].axhline(1,color='#ccc',ls='--',lw=1);axs[1].legend()
    p=synth[0]['metrics']['GT_model_bilinear']['PSD']
    axs[2].semilogy(p['radial_frequency_midpoints'],p['radial_power_mean'],label='clean GT 768',color='white')
    axs[2].axvline(.125,ls='--',color='#c5a1ff',label='480 sample Nyquist');axs[2].set_xlim(.025,.205)
    axs[2].set_title('Fixed tiles PSD: 000 v0');axs[2].set_xlabel('cycles / native pixel');axs[2].set_ylabel('mean radial periodogram');axs[2].legend(fontsize=8)
    fig.savefig(HERE/'paired_summary.png',dpi=170);plt.close(fig)

if __name__=='__main__':main()
