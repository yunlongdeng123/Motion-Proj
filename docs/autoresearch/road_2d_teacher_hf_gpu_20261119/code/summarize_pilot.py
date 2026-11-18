"""CPU summary of a completed, gated48-prediction teacher pilot score.

This program never opens truth images, generated images, model weights or GPU.
Run only after score_pilot.py has produced a complete score JSON:
  python summarize_pilot.py --score scores.json --predictions predictions.json \
    --manifest prepared/pilot_manifest.json --out-dir summary

Production mode rejects incomplete scores and labelled CPU fixtures. Optional
--allow-cpu-mock is for validating this summary against legacy-array fixtures.
No method parameters, training budget, output masks or model outputs are changed.
"""
from __future__ import annotations
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, csv, hashlib, json, math
from collections import defaultdict
from pathlib import Path
import statistics

IDS=['visible_scene081_raw000_v0','visible_scene081_raw000_v1','visible_scene081_raw050_v0','visible_scene081_raw050_v1','scene065_raw086','scene065_raw110']
ARMS=['full768_T0','full768_base','crop768_T0','full1152_T0']
SEEDS=[101,202]
REFERENCE_ARM='full768_T0'
DOMAINS={'synthetic_certified_hidden_target':'synthetic_certified_hidden_target',
         'known_visible_road_preservation_only':'real_known_visible_control'}
BASE_FRAMES={sid:('scene081_raw000' if 'raw000' in sid else 'scene081_raw050') for sid in IDS[:4]}
BASE_FRAMES.update({'scene065_raw086':'scene065_raw086','scene065_raw110':'scene065_raw110'})
COMMON_BANDS=[[.025,.0625],[.0625,.125],[.125,.2]]
SIGMAS=[2.5,5.,10.]

class SummaryError(RuntimeError):pass
def require(value,message):
    if not value:raise SummaryError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pin(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def write(path,data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def number(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
def mean(values):
    values=[v for v in values if number(v)]
    return statistics.mean(values) if values else None
def abslogratio(value):return abs(math.log2(max(value,1e-12))) if number(value) and value>=0 else None
def csv_write(path,rows):
    if not rows:return
    names=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader();writer.writerows(rows)

def validate_score(d,allow_cpu_mock=False):
    require(d.get('schema')=='road_2d_teacher_postfreeze_CPU_score.v1' and d.get('complete') is True,'score incomplete or schema mismatch')
    mock=d.get('mode')=='CPU_MOCK_VALIDATION_ONLY'
    require(not mock or allow_cpu_mock,'CPU mock is not a GPU pilot result; --allow-cpu-mock required')
    require(mock or d.get('mode')=='FROZEN_PREDICTION_CPU_SCORING','unrecognized scoring mode')
    require(d.get('scored_jobs')==48 and d.get('GT_opened_after_gate') is True,'48 gated scores required')
    gate=d.get('gate',{})
    require(gate.get('passed') is True and gate.get('completed_jobs')==48,'freeze gate did not pass')
    require(all(gate.get(k) is True for k in ['actual_W_outside_exact','actual_K_exact','actual_native_transform_exact']),'actual pixel/transform gates missing')
    require(gate.get('GT_opened_before_gate') is False,'truth ordering not established')
    require(gate.get('CPU_mock') is mock,'score/gate CPU mock label mismatch')
    cases={c['id']:c for c in d['cases']}
    require(len(d['cases'])==6 and set(cases)==set(IDS),'six frozen cases required')
    expected={(sid,arm,seed) for sid in IDS for arm in ARMS for seed in SEEDS}
    actual={(j['case_id'],j['arm'],j['seed']) for j in d['jobs']}
    require(len(d['jobs'])==48 and len(actual)==48 and actual==expected,'score combinations are missing, duplicate or extra')
    # The native support must be identical across every arm/seed. Model grids
    # sample that same physical support but contain different numbers of pixels.
    native_counts=defaultdict(set)
    for j in d['jobs']:
        sid=j['case_id'];role=j['metric_role']
        require(role in DOMAINS and cases[sid]['metric_role']==role,'metric role mismatch')
        require((sid in IDS[:4])==(role=='synthetic_certified_hidden_target'),'real hidden GT must not be invented')
        require(j['view']==j['arm'].rsplit('_',1)[0],'arm/view mismatch')
        for grid in ['native','same_view']:
            m=j[grid];spacing=m['spacing_native_px']
            require(number(spacing) and spacing>0,'invalid pixel spacing')
            if grid=='native':
                require(spacing==1 and m['grid_wh']==[1920,1280],'native grid/unit mismatch')
                native_counts[sid].add(m['valid_pixels'])
            else:
                expected_spacing=1920/1152 if j['view']=='full1152' else 2.5
                require(abs(spacing-expected_spacing)<1e-12,'same-view physical spacing mismatch')
            require(m['PSD']['band_edges_cycles_per_native_px'][:3]==COMMON_BANDS,'physical PSD bands changed')
            require(abs(m['PSD']['axis_nyquist_cycles_per_native_px']-.5/spacing)<1e-12,'PSD frequency unit mismatch')
            for key,h in m['HP'].items():
                if h.get('status')=='insufficient':continue
                require(float(key) in SIGMAS and abs(h['sigma_native_px']-float(key))<1e-12,'HP physical sigma mismatch')
                require(abs(h['sigma_this_grid_px']*spacing-h['sigma_native_px'])<1e-10,'HP grid/native sigma mismatch')
            require(number(m['color']['deltaE76_mean']) and number(m['color']['RGB_MAE_0_255']),'nonfinite color score')
    require(all(len(v)==1 for v in native_counts.values()),'native analysis mask varies across arm/seed')
    return mock

def flatten_job(j,grid):
    m=j[grid];domain=DOMAINS[j['metric_role']]
    row={'case_id':j['case_id'],'base_frame':BASE_FRAMES[j['case_id']],'domain':domain,'arm':j['arm'],'seed':j['seed'],'view':j['view'],'grid':grid,
      'width':m['grid_wh'][0],'height':m['grid_wh'][1],'spacing_native_px':m['spacing_native_px'],'valid_pixels':m['valid_pixels'],
      'deltaE76_mean':m['color']['deltaE76_mean'],'RGB_MAE_0_255':m['color']['RGB_MAE_0_255']}
    for name,v in zip(['L','a','b'],m['color']['mean_Lab_bias']):row['mean_Lab_bias_'+name]=v
    for sigma in SIGMAS:
        key=str(sigma);label=str(sigma).replace('.','p');h=m['HP'].get(key,{})
        ratio=h.get('RMS_ratio');winsor=h.get('winsor95_RMS_ratio')
        row['HP_RMS_ratio_sigma_'+label+'_native']=ratio
        row['HP_abs_log2_RMS_ratio_sigma_'+label+'_native']=abslogratio(ratio)
        row['HP_winsor95_RMS_ratio_sigma_'+label+'_native']=winsor
        row['HP_abs_log2_winsor95_RMS_ratio_sigma_'+label+'_native']=abslogratio(winsor)
        row['HP_pattern_RMSE_sigma_'+label+'_native']=h.get('HP_pattern_RMSE_Y_0_255')
    psd=m['PSD'];ratios=psd['band_power_ratio'];errors=[]
    for i,(lo,hi) in enumerate(psd['band_edges_cycles_per_native_px']):
        ratio=ratios[i];name=f'PSD_energy_ratio_f{lo:g}_to_{hi:g}_cycles_per_native_px'
        row[name]=ratio
        if i<3:errors.append(abslogratio(ratio))
    require(all(v is not None for v in errors),'common PSD band unavailable; do not compare unequal bandwidth')
    row['PSD_common3_mean_abs_log2_energy_ratio']=mean(errors)
    row['PSD_all_available_mean_abs_log2_energy_ratio_descriptive_only']=psd['mean_abs_log2_band_power_ratio']
    row['PSD_fixed_native_tile_count']=psd['fixed_native_tile_count']
    return row

def value_columns(rows):
    prefix=('deltaE','RGB_MAE','mean_Lab_bias_','HP_','PSD_energy_ratio_','PSD_common3_')
    return [k for k in rows[0] if k.startswith(prefix)]

def make_tables(d):
    rows=[flatten_job(j,grid) for j in d['jobs'] for grid in ['native','same_view']]
    fields=value_columns(rows);groups=defaultdict(list)
    for row in rows:groups[(row['case_id'],row['arm'],row['grid'])].append(row)
    case_rows=[]
    for (sid,arm,grid),rs in groups.items():
        require(len(rs)==2 and {r['seed'] for r in rs}==set(SEEDS),'two seeds required for every case/arm/grid')
        out={k:rs[0][k] for k in ['case_id','base_frame','domain','arm','grid']}
        out.update(seed_count=2,seed_statistic='arithmetic mean; range is seed range, not confidence interval')
        for f in fields:out[f]=mean([r.get(f) for r in rs])
        for f in ['deltaE76_mean','PSD_common3_mean_abs_log2_energy_ratio','HP_abs_log2_RMS_ratio_sigma_2p5_native']:
            vals=[r[f] for r in rs if number(r[f])];out[f+'_seed_min']=min(vals) if vals else None;out[f+'_seed_max']=max(vals) if vals else None
        case_rows.append(out)
    # Case means first; also retain the two source-frame averages explicitly.
    frame_groups=defaultdict(list)
    for row in case_rows:frame_groups[(row['base_frame'],row['domain'],row['arm'],row['grid'])].append(row)
    frame_rows=[]
    for (frame,domain,arm,grid),rs in frame_groups.items():
        row={'base_frame':frame,'domain':domain,'arm':arm,'grid':grid,'task_count':len(rs),'seed_count_per_task':2}
        row.update({f:mean([r.get(f) for r in rs]) for f in fields});frame_rows.append(row)
    macro_groups=defaultdict(list)
    for row in frame_rows:macro_groups[(row['domain'],row['arm'],row['grid'])].append(row)
    macro_rows=[]
    for (domain,arm,grid),rs in macro_groups.items():
        out={'domain':domain,'arm':arm,'grid':grid,'base_frame_count':len(rs),'task_count':sum(r['task_count'] for r in rs),'seeds_per_task':2,
             'aggregation':'average seeds, then task variants per base frame, then equal base frames; descriptive only'}
        out.update({f:mean([r.get(f) for r in rs]) for f in fields});macro_rows.append(out)
    lookup={(r['case_id'],r['arm'],r['seed'],r['grid']):r for r in rows}
    contrasts=[]
    error_fields=[f for f in fields if f in ['deltaE76_mean','RGB_MAE_0_255','PSD_common3_mean_abs_log2_energy_ratio'] or f.startswith('HP_abs_log2_')]
    for row in rows:
        if row['arm']==REFERENCE_ARM:continue
        ref=lookup[(row['case_id'],REFERENCE_ARM,row['seed'],row['grid'])]
        out={k:row[k] for k in ['case_id','base_frame','domain','arm','seed','grid']};out['reference_arm']=REFERENCE_ARM
        out['interpretation']='negative descriptive error delta is closer to this grid reference; no automatic method winner'
        for f in error_fields:
            a,b=row.get(f),ref.get(f);out[f+'_delta']=a-b if number(a) and number(b) else None
        contrasts.append(out)
    return rows,case_rows,frame_rows,macro_rows,contrasts

def normalize_known_metrics(m):
    """teacher_pilot.known_metrics uses normalized RGB/Y; return explicit units."""
    return {'pixels':m['pixels'],'RGB_MAE_0_255':m['rgb_mae']*255,'signed_RGB_bias_0_255':[v*255 for v in m['signed_rgb_bias']],
      'signed_luminance_bias_0_255':m['signed_luminance_bias']*255,'HP_sigma_native_px':m['highpass_sigma_native_px'],
      'HP_RMS_reference_Y_0_255':m['highpass_rms_reference']*255,'HP_RMS_prediction_Y_0_255':m['highpass_rms_prediction']*255,
      'HP_RMS_ratio':m['highpass_rms_ratio']}

def frozen_ledger_controls(d,predictions_path,manifest_path,mock):
    if predictions_path is None:
        return {'loaded':False,'reason':'predictions ledger not supplied; no VAE completeness or timing claim'}
    p=read(predictions_path)
    require(sha(predictions_path)==d['gate']['predictions']['sha256'],'frozen predictions digest mismatch')
    require(p.get('complete') is True and p.get('completed_diffusion_calls')==p.get('attempted_diffusion_calls')==48,'prediction ledger not complete48')
    require(p.get('GT_opened') is False and p.get('training_updates')==0,'generation truth/training flags changed')
    require((p.get('CPU_mock') is True)==mock,'prediction/score mock label mismatch')
    vae=p.get('VAE_controls',[])
    require(mock or len(vae)==4,'production completion requires all four VAE controls')
    manifest=None
    if manifest_path is not None:
        manifest=read(manifest_path)
        require(sha(manifest_path)==d['gate']['manifest']['sha256'],'frozen prepared manifest digest mismatch')
        require(manifest.get('GT_opened') is False and manifest.get('training_updates')==0,'prepared truth/training flags changed')
    reference={r['id']:r for r in (manifest or {}).get('vae_controls',[])}
    if not mock and manifest is not None:
        require(len(reference)==4 and {c['id'] for c in vae}==set(reference),'VAE control combinations mismatch')
    controls=[]
    for control in vae:
        row={'id':control['id'],'metric_role':'legally_visible_road_reconstruction_control_not_hidden_truth',
             'seconds':control['seconds'],'VAE_latent_mode':control.get('VAE_latent_mode'),'UNet_calls':control.get('UNet_calls')}
        for name in ['known_road_vs_native_observation','VAE_increment_over_resampling']:
            row[name]=normalize_known_metrics(control[name])
        if control['id'] in reference:row['resampling_only_vs_native_observation']=normalize_known_metrics(reference[control['id']]['resampling_only'])
        controls.append(row)
    timing=defaultdict(list)
    for job in p['jobs']:
        if number(job.get('seconds')):timing[job['arm']].append(job['seconds'])
    return {'loaded':True,'predictions':pin(predictions_path),'VAE_controls_completed':len(controls),'VAE_controls':controls,
       'timing':{'wall_seconds':p.get('wall_seconds'),'model_load_seconds':p.get('model_load_seconds'),
        'diffusion_seconds_sum':sum(v for values in timing.values() for v in values),
        'VAE_seconds_sum':sum(c['seconds'] for c in controls),
        'per_arm_mean_diffusion_seconds':{arm:mean(values) for arm,values in timing.items()},
        'peak_GPU_allocated_GiB':p.get('peak_GPU_allocated_bytes',0)/(1024**3) if p.get('peak_GPU_allocated_bytes') is not None else None},
       'limits':['VAE control masks differ from the main spectral scoring mask; do not combine their absolute errors as one metric.',
         'VAE_increment_over_resampling compares decoded RGB directly with the resampling control. It is not obtained by subtracting two MAEs and is not an additive causal decomposition.']}

def plot(case_rows,path,mock):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.style.use('dark_background');fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    colors=['#6dd9cd','#f3c36a','#9fafff','#f598b0'];labels=['Full768 + T0','Full768 base','Crop768 + T0','Full1152 + T0']
    lookup={(r['case_id'],r['arm'],r['grid']):r for r in case_rows}
    x=np.arange(4)
    for col,grid in enumerate(['native','same_view']):
        for row,field,label in [(0,'deltaE76_mean','Mean Delta E76'),(1,'PSD_common3_mean_abs_log2_energy_ratio','Mean |log2 PSD energy ratio|')]:
            ax=axes[row,col]
            for ai,arm in enumerate(ARMS):
                rs=[lookup[(sid,arm,grid)] for sid in IDS[:4]];ys=[r[field] for r in rs]
                low=[y-r[field+'_seed_min'] for y,r in zip(ys,rs)];high=[r[field+'_seed_max']-y for y,r in zip(ys,rs)]
                ax.errorbar(x+(ai-1.5)*.035,ys,yerr=[low,high],marker='o',lw=1.1,capsize=3,label=labels[ai],color=colors[ai])
            ax.set_xticks(x,['000 v0','000 v1','050 v0','050 v1']);ax.set_ylabel(label);ax.set_title(('Native1920 fixed support' if grid=='native' else 'Each view vs exact same-view GT')+(' | color' if row==0 else ' | common physical PSD bands'))
            ax.grid(axis='y',alpha=.15)
    axes[0,0].legend(fontsize=8)
    title='CPU MOCK VALIDATION ONLY' if mock else 'Frozen48 pilot | synthetic certified targets only'
    fig.suptitle(title+'\nFour tasks reuse two base frames; points=two-seed mean, whiskers=seed range; no automatic winner',fontsize=11)
    fig.savefig(path,dpi=170);plt.close(fig)

def summarize(score_path,out_dir,predictions_path=None,manifest_path=None,allow_cpu_mock=False,make_plot=True):
    d=read(score_path);mock=validate_score(d,allow_cpu_mock)
    out_dir=Path(out_dir)
    require(not out_dir.exists() or not any(out_dir.iterdir()),'preserve existing summary; use an empty output directory')
    rows,cases,frames,macro,contrasts=make_tables(d)
    ledger=frozen_ledger_controls(d,predictions_path,manifest_path,mock)
    result={'schema':'road_2d_teacher_48_comparison.v1','complete':True,'mode':d['mode'],'score':pin(score_path),'CPU_only':True,
      'gate':d['gate'],'automatic_winner_selected':False,'common_PSD_bands_cycles_per_native_px':COMMON_BANDS,
      'sample_counts':{'synthetic_tasks':4,'synthetic_base_frames':2,'real_known_visible_control_cases':2,'arms':4,'seeds_per_task':2,'predictions':48,'grid_metric_rows':96},
      'notes':[
        'Native and same_view results are separate. Native uses the same1920x1280 mask per case across every arm/seed; same_view compares each raw generation grid with clean RGB sampled by the exact frozen builder transform, including the crop offset.',
        'HP sigmas2.5/5/10 are native pixels, not image-grid pixels. Full768/crop768 spacing2.5 native px/model px; full1152 spacing5/3. Do not pool grid-dependent filter units.',
        'Primary PSD error uses identical radial bands0.025–0.0625,0.0625–0.125,0.125–0.2 cycles/native pixel for every arm/grid. The scorer all-available-band scalar has3 bands at768,4 at1152 and5 atnative; that scalar is retained only as descriptive metadata and is not the comparison criterion.',
        'Lower RGB MAE/DeltaE describes closer color. HP RMS and PSD energy are closest to their reference at ratio1; higher energy is not better material. Absolute log2 ratios describe magnitude mismatch symmetrically, but do not establish authentic granularity or remove false generated structure.',
        'PSD uses energy ratios; HP uses RMS amplitude ratios. Their log values are not combined into a composite score. Color, spectrum and visible artifacts must be considered together.',
        'All four synthetic tasks reuse only two clean source frames. Tables average the two seeds within a task, then variants within a base frame, then the two base frames. No independence assumption, significance test, confidence interval or universal scene-generalization claim is made.',
        'Same numerical seeds across view sizes do not pair identical latent pixels. Paired contrasts hold task/seed labels fixed and remain descriptive.',
        'The two real cases score known visible road preservation only. They provide no hidden-road color or texture accuracy and are never pooled with synthetic hidden-target truth.',
        'Generated outputs do not select masks or patches. False lines and seams can inflate RMS/PSD;95% winsorized HP is a separate sensitivity statistic, not a replacement for raw metrics.',
        'Native error is the primary shared-grid comparison; same_view error helps distinguish generation-scale mismatch from inverse resampling effects. Different reference sampling across views limits interpretation of isolated same_view scalar changes.',
        'All CSV/JSON results are descriptive. No automatic winner, checkpoint selection, new training or method alteration is performed.'
      ],'macro_equal_base_frames':macro,'case_seed_means':cases,'base_frame_seed_means':frames,'paired_contrasts':contrasts,'ledger_controls':ledger}
    out_dir.mkdir(parents=True,exist_ok=True)
    for name,data in [('per_job.csv',rows),('case_seed_means.csv',cases),('base_frame_means.csv',frames),('macro_equal_base_frames.csv',macro),('paired_vs_full768_T0.csv',contrasts)]:csv_write(out_dir/name,data)
    write(out_dir/'comparison.json',result)
    if make_plot:plot(cases,out_dir/'synthetic_color_spectrum.png',mock)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--score',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--predictions',type=Path);p.add_argument('--manifest',type=Path);p.add_argument('--allow-cpu-mock',action='store_true');p.add_argument('--no-plot',action='store_true')
    a=p.parse_args();r=summarize(a.score,a.out_dir,a.predictions,a.manifest,a.allow_cpu_mock,not a.no_plot)
    print(json.dumps({'complete':r['complete'],'mode':r['mode'],'automatic_winner_selected':False,'out_dir':str(a.out_dir.resolve())}))
if __name__=='__main__':main()
