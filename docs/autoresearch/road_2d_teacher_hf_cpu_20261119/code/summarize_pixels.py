"""Derive concise stage diagnostics and a geometry-selected pixel contact sheet.
Run after diagnose_pixels.py. Inputs and old assets remain read-only.
"""
import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[k]="1"
import csv, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from diagnose_pixels import HERE,BASE,load_rgb,resize_rgb,block4

def ratio(a,b):return float(a/b) if b else None
def main():
    d=json.loads((HERE/'pixel_diagnosis.json').read_text())
    manifest=json.loads((BASE/'source_manifest.json').read_text(encoding='utf-8-sig'))
    expected={p['relative']:p['sha256'] for p in manifest['files']}
    pins=d['source_files_sha256']
    assert all(p['sha256']==expected[p['relative']] for p in pins), 'Frozen source digest mismatch'
    cmeta={c['id']:c for c in manifest['cases']}
    result={'schema':'teacher_stage_diagnosis_compact.v1','reproduce':['python work/pixel_diagnosis/diagnose_pixels.py','python work/pixel_diagnosis/summarize_pixels.py'],
      'integrity':{'original_source_files_verified':len(pins),'truth_files_verified':4,'raw_native_exact_PIL_upsample_all':all(a['raw_native_vs_PIL_bilinear_uint8_maxdiff']==0 for c in d['cases'] if c['domain']=='synthetic_visible_2d' for a in c['audit'].values()),'old_assets_modified':False},
      'notes':[
       'Four synthetic cases use only two clean base frames (000 and050). They are paired descriptive evidence, not four independent scenes; no significance claim.',
       'All metrics use original images, never report screenshots. Source digests and clean truth/certificate digests are verified.',
       'No generated RGB selects the masks or80x80 native tiles. The fixed target_core/W/certificate mask is eroded from output boundaries; clean-reference white/yellow paint and coarse edges are excluded. All globally anchored, fully certified tiles are included.',
       'GT768 uses Pillow BILINEAR antialiased downsampling with uint8 quantization. It is an explicit resolution control, NOT an exact reproduction of the actual input builder four-neighbor bilinear RGB sampling. AREA768 is a second resolution-control sensitivity.',
       'PSD: Y=0.2126R+0.7152G+0.0722B on0..255 sRGB; every80x80 native support uses a linear-plane detrend and Hann window. Power sums have the same Y-squared units across native/model/sample grids. Frequencies are cycles per native pixel.',
       'Stage3 total-energy ratio compares teacher768 radial0.025<=f<0.2 to historical sample480 radial0.025<=f<0.125, each capped at its axis Nyquist, on the same physical tiles. This compares differently available bandwidth; it is NOT a same-band ratio, RMS loss, percent authentic details, or information-theoretic loss.',
       'Stage3 shared-band ratio integrates the same0.025<=f<0.125 on both grids. This is the preferred estimate of additional attenuation within the sample-representable band. Discrete grids/Hann windows differ, so these are PSD diagnostics, not an analytic transfer-function measurement.',
       'Raw periodogram means can be dominated by generated lane/seam/shadow structures, especially050. Median tile-band powers and95% winsorized HP RMS are sensitivity summaries alongside, never replacements for, the primary means.',
       'Every synthetic arm/case has HP RMS greater than the resolution-matched clean control at768, including95% winsorization. Extra high-frequency energy does not establish authentic asphalt material: generated noise, false structure, and wrong spectra can all increase it.',
       'Real hidden RGB has no GT. Same-height nearest visible-road patch excludes mask-labelled shadows, retained objects, paint and reference coarse edges, but horizontal separations160–800 native pixels can cross material/lighting changes. Proxy color/PSD differences are not hidden-area accuracy or causal evidence.',
       'real_scene081_raw062 has no fully valid80x80 target tile under the fixed rule; no patch texture conclusion is made for that case.',
       'Only CPU local single-thread diagnostics and read-only SCP of four pinned truth files were used. No inference, training, GPU,3D fitting, or remote writes.'
      ],'synthetic_cases':[],'real_proxy_cases':[],'hypotheses':{
       '1_large_hole_limited_material_context':{'status':'plausible_not_causally_established','evidence':'A/B fixed-W comparisons change hidden/context masks, but texture/color direction is case dependent. Existing arrays cannot isolate hole area from material reliability. A mask/context controlled teacher experiment would be required.'},
       '2_generation_resolution_limits_granularity':{'status':'supported_as_bandwidth_bottleneck_not_full_explanation','evidence':'GT resolution control alone removes most native near-pixel HP energy;768 independent Nyquist0.2 cyc/native becomes0.125 at480. Yet teacher768 spectrum and color remain mismatched to GT768.'},
       '3_attention_LoRA_prefers_global_not_fine_material':{'status':'not_identifiable_from_frozen_outputs','evidence':'Outputs show mismatch but there is no matched attention/full/convolution adapter ablation. No claim that attention LoRA caused the mismatch.'},
       '4_training_steps_insufficient':{'status':'not_identifiable_from_frozen_outputs','evidence':'One trained endpoint and A/B inference arms do not provide a checkpoint learning curve or longer-training control. More steps could improve, saturate or degrade.'}
      }}
    csvrows=[];synthetic=[c for c in d['cases'] if c['domain']=='synthetic_visible_2d']
    canvas=Image.new('RGB',(6*245,4*280+40),(20,24,31));draw=ImageDraw.Draw(canvas)
    draw.text((8,7),'Fixed pixel crops | nearest target tile to mask centroid; all panels show the same80x80 native support; nearest-neighbor display zoom only',fill=(240,240,240))
    for ci,c in enumerate(synthetic):
        m=c['metrics'];ps=c['paired_summary'];cid=c['id']
        n=m['GT_native']['PSD']['band_power_mean_Y_squared'];u=m['GT_model_bilinear_up_native']['PSD']['band_power_mean_Y_squared']
        res={'id':cid,'fixed_tile_count':len(c['frozen_target_tiles']),'fixed_mask_pixels':c['counts']['safe_pixels'],
          'resolution_control_native_to_bilinear768_to_native':{'HP_RMS_retention_at_sigma1p25_native':ps['resolution_only_native_HP_ratio']['1.25'],
           'HP_energy_retention_at_sigma1p25_native':ps['resolution_only_native_HP_ratio']['1.25']**2,
           'PSD_total_energy_retention_f0p025_to0p5':ratio(sum(u),sum(n)),
           'PSD_band_energy_retention':[ratio(a,b) for a,b in zip(u,n)]},
          'area768_vs_bilinear768_control':{'deltaE76_mean':m['GT_model_area']['color']['deltaE76_mean'],
           'HP_sigma1model_RMS_ratio':ratio(m['GT_model_area']['HP']['2.5']['rms_Y_0_255'],m['GT_model_bilinear']['HP']['2.5']['rms_Y_0_255'])},'arms':{}}
        for arm in ['A','B']:
            t=m[arm+'_model'];s=m[arm+'_sample_historical'];a=m[arm+'_sample_direct_area'];gt=m['GT_model_bilinear']
            src=t['PSD']['band_power_mean_Y_squared'][:3];dst=s['PSD']['band_power_mean_Y_squared'][:2];area=a['PSD']['band_power_mean_Y_squared'][:2]
            srcmed=t['PSD']['band_power_median_Y_squared'][:3];dstmed=s['PSD']['band_power_median_Y_squared'][:2]
            total=ratio(sum(dst),sum(src));shared=ratio(sum(dst),sum(src[:2]))
            rv={'model768_vs_clean_control':{'deltaE76_mean':t['color']['deltaE76_mean'],'mean_Lab_bias':t['color']['mean_Lab_bias'],
              'HP_RMS_ratio_sigma1model':ps[arm]['model768']['HP_RMS_ratio']['2.5'],
              'HP_winsor95_RMS_ratio_sigma1model':ps[arm]['model768']['HP_winsor95_ratio']['2.5'],
              'PSD_band_power_ratio_mean':ps[arm]['model768']['PSD_band_power_ratio'],
              'PSD_band_power_ratio_median':[ratio(a,b) if a is not None else None for a,b in zip(t['PSD']['band_power_median_Y_squared'],gt['PSD']['band_power_median_Y_squared'])],
              'HP_RMS_ratio_sigma1model_vs_AREA768':ratio(t['HP']['2.5']['rms_Y_0_255'],m['GT_model_area']['HP']['2.5']['rms_Y_0_255'])},
              'native_vs_actual_clean':{'HP_RMS_ratio_sigma1p25native':ps[arm]['native1920']['HP_RMS_ratio']['1.25']},
              'historical480_vs_teacher768':{'PSD_total_band_energy_retention':total,'PSD_total_band_energy_reduction_percent':100*(1-total),
               'PSD_shared_band_energy_retention':shared,'PSD_shared_band_energy_reduction_percent':100*(1-shared),
               'teacher768_PSD_energy_fraction_f0p125_to0p2':ratio(src[2],sum(src)),
               'PSD_total_band_retention_from_median_band_powers_sensitivity':ratio(sum(dstmed),sum(srcmed)),
               'PSD_shared_band_retention_from_median_band_powers_sensitivity':ratio(sum(dstmed),sum(srcmed[:2])),
               'direct_AREA480_PSD_shared_band_retention':ratio(sum(area),sum(src[:2])),
               'historical480_vs_direct_AREA480_RGB_MAE_0_255':c['audit'][arm]['historical_sample_vs_direct_area']['rgb_mae_0_255']}}
            res['arms'][arm]=rv
            csvrows.append({'case':cid,'arm':arm,'tiles':res['fixed_tile_count'],
               'DE76_768':rv['model768_vs_clean_control']['deltaE76_mean'],
               'HP_RMS_ratio_768_sigma1':rv['model768_vs_clean_control']['HP_RMS_ratio_sigma1model'],
               'HP_winsor95_RMS_ratio_768_sigma1':rv['model768_vs_clean_control']['HP_winsor95_RMS_ratio_sigma1model'],
               'PSD_shared_band_energy_reduction_768to480_pct':100*(1-shared),'PSD_total_band_energy_reduction_768to480_pct':100*(1-total)})
        result['synthetic_cases'].append(res)
        # Geometry-only fixed crop selection, prior to reading teacher RGB.
        mask=np.asarray(Image.open(HERE/(cid+'_analysis_mask.png')))>0
        yy,xx=np.where(mask);cx,cy=xx.mean(),yy.mean()
        tile=min(c['frozen_target_tiles'],key=lambda t:((t[0]+t[2]/2-cx)**2+(t[1]+t[2]/2-cy)**2,t[1],t[0]))
        res['fixed_pixel_crop_native_xywh']=list(tile)
        x,y,p=tile;meta=cmeta[cid]
        clean=load_rgb(Path(c['source_pins'][0]['local']));oracle=resize_rgb(resize_rgb(clean,(768,512)),(1920,1280))
        aa=load_rgb(BASE/meta['arms']['A']['raw']);bb=load_rgb(BASE/meta['arms']['B']['raw'])
        panels=[('Clean native',clean[y:y+p,x:x+p]),('GT bilinear768 -> native',oracle[y:y+p,x:x+p]),('Teacher A native',aa[y:y+p,x:x+p]),('Teacher B native',bb[y:y+p,x:x+p]),('Teacher A historical480',block4(aa)[y//4:(y+p)//4,x//4:(x+p)//4]),('Teacher B historical480',block4(bb)[y//4:(y+p)//4,x//4:(x+p)//4])]
        for j,(label,img) in enumerate(panels):
            draw.text((j*245+3,ci*280+43),cid.replace('visible_scene081_','')+' '+label,fill=(235,235,235))
            im=Image.fromarray(np.clip(np.rint(img),0,255).astype(np.uint8)).resize((240,240),Image.Resampling.NEAREST)
            canvas.paste(im,(j*245,ci*280+62))
    canvas.save(HERE/'fixed_pixel_crops.png')
    for c in d['cases']:
        if c['domain']!='real_unlabelled':continue
        result['real_proxy_cases'].append({'id':c['id'],'matched_tiles':len(c['frozen_target_tiles']),
          'proxy_distance_native_px':{'min':min(c['proxy_horizontal_distance_native_px']) if c['proxy_horizontal_distance_native_px'] else None,'max':max(c['proxy_horizontal_distance_native_px']) if c['proxy_horizontal_distance_native_px'] else None},
          'limitation':c['limitation'],'model_proxy_mean_Lab_bias':{a:c['metrics'][a+'_model'].get('proxy_color',{}).get('mean_Lab_bias') for a in ['A','B']}})
    result['synthetic_macro']=d['synthetic_macro']
    result['assets']={'main_full_json':str(HERE/'pixel_diagnosis.json'),'paired_csv':str(HERE/'paired_synthetic.csv'),'fixed_pixel_crops':str(HERE/'fixed_pixel_crops.png'),'summary_plot':str(HERE/'paired_summary.png'),'all_case_fixed_tile_previews':str(HERE/'*_fixed_tiles.png')}
    (HERE/'compact_results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    with (HERE/'paired_synthetic.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    print(json.dumps({'files_verified':len(pins),'ranges':{k:[min(r[k] for r in csvrows),max(r[k] for r in csvrows)] for k in ['HP_RMS_ratio_768_sigma1','HP_winsor95_RMS_ratio_768_sigma1','PSD_shared_band_energy_reduction_768to480_pct','PSD_total_band_energy_reduction_768to480_pct']},'area768_control_HP_ratio_range':[min(c['area768_vs_bilinear768_control']['HP_sigma1model_RMS_ratio'] for c in result['synthetic_cases']),max(c['area768_vs_bilinear768_control']['HP_sigma1model_RMS_ratio'] for c in result['synthetic_cases'])]},indent=2))

if __name__=='__main__':main()
