"""CPU-only spectral transfer of frozen generated RGB into20 supervision packs.

No truth RGB, original observation, GPU, training or3D code is opened/imported.
The fixed80-native-pixel tiles are common across every source arm, seed, output
grid and the historical bilinear-native/block-mean path, separately for C/S.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,csv,hashlib,importlib.util,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

ARMS=['full768_T0','full1152_T0'];SEEDS=[101,202];RAWS=[86,110];ROLES=['C','S']
def require(v,message):
    if not v:raise RuntimeError(message)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def qload(path):
    keys=['target_C_srgb','C','S','q_C_2d','q_S_2d','teacher_grid_C_valid','teacher_grid_S_valid','source_model_valid','output_wh','teacher_wh','native_wh','native_from_model','teacher_native_window_xyxy']
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in keys}
def ratio(a,b):return float(a/b) if a is not None and b is not None and b>1e-20 else None

def bilinear_all_axis(a,target,axis):
    """Strict positive-weight PIL-upsample bilinear support; rational centers.
    No zero-weight neighbor is included. Edge source indices are clamped as PIL.
    """
    a=np.moveaxis(a,axis,0);source=len(a);out=np.empty((target,*a.shape[1:]),bool)
    for j in range(target):
        numerator=(2*j+1)*source-target;denominator=2*target
        low=numerator//denominator;remainder=numerator-low*denominator
        lo=max(0,min(source-1,low));hi=max(0,min(source-1,low+1))
        out[j]=a[lo] if remainder==0 else a[lo]&a[hi]
    return np.moveaxis(out,0,axis)

def historical_support(source_mask):
    native=bilinear_all_axis(bilinear_all_axis(source_mask,1920,1),1280,0)
    return native.reshape(320,4,480,4).all(axis=(1,3))

def historical_rgb(raw):
    native=np.asarray(Image.fromarray(raw).resize((1920,1280),Image.Resampling.BILINEAR),dtype=np.float32)
    return native.reshape(320,4,480,4,3).mean(axis=(1,3),dtype=np.float32)/255

def bounds(tile,wh):
    x,y,p=tile;s=1920/wh[0]
    require(abs(s-1280/wh[1])<1e-12,'nonisotropic full frame')
    values=[x/s,y/s,p/s];require(all(abs(v-round(v))<1e-7 for v in values),'noninteger fixed tile alignment')
    return tuple(int(round(v)) for v in values)
def tile_valid(mask,tile):
    x,y,p=bounds(tile,(mask.shape[1],mask.shape[0]));return mask[y:y+p,x:x+p].shape==(p,p) and bool(mask[y:y+p,x:x+p].all())
def candidate_tiles(mask):
    return [(x,y,80) for y in range(0,1280-79,40) for x in range(0,1920-79,40) if mask[y:y+80,x:x+80].all()]

def energy_row(group,role,label,image,mask,tiles,source_power,band_power):
    wh=[image.shape[1],image.shape[0]];spacing=1920/wh[0]
    powers=band_power(image*255,tiles,spacing)
    common=[i for i,(p,s) in enumerate(zip(powers,source_power)) if p is not None and s is not None]
    shared2=ratio(sum(powers[:2]),sum(source_power[:2]))
    same=ratio(sum(powers[i] for i in common),sum(source_power[i] for i in common))
    source_total=sum(v for v in source_power if v is not None)
    missing=sum(source_power[i] for i,v in enumerate(powers) if v is None and source_power[i] is not None)
    return {'raw':group['raw'],'arm':group['arm'],'seed':group['seed'],'role':role,'path':label,'output_wh':wh,'source_wh':group['source_wh'],
      'fixed_common_native_tiles':len(tiles),'source_axis_nyquist_cycles_per_native_px':group['source_wh'][0]/3840,
      'target_axis_nyquist_cycles_per_native_px':wh[0]/3840,'source_band_power_Y_squared':source_power,'target_band_power_Y_squared':powers,
      'per_band_energy_retention':[ratio(a,b) for a,b in zip(powers,source_power)],
      'shared_0p025_to_0p125_energy_retention':shared2,'shared_0p025_to_0p125_energy_reduction_percent':100*(1-shared2) if shared2 is not None else None,
      'shared_source_target_axis_band_energy_retention':same,'shared_source_target_band_indices':common,
      'source_energy_fraction_above_target_axis_nyquist':ratio(missing,source_total),
      'total_different_available_band_energy_retention':ratio(sum(v for v in powers if v is not None),source_total),
      'permission_pixels_at_this_grid':int(mask.sum())}

def run(receipt_path,score_code,outdir,make_plot=True):
    outdir=Path(outdir);require(not outdir.exists(),'preserve existing transfer output')
    receipt=read(receipt_path)
    require(receipt['complete'] and receipt['package_count']==len(receipt['packages'])==20,'20 packs must be complete')
    require(receipt['uniform_arms']==ARMS and receipt['seeds']==SEEDS and receipt['raw_frames']==RAWS and not receipt['per_case_or_seed_selection'],'uniform paired pack contract changed')
    require(receipt['source_level_identity_checks']==8 and receipt['baseline_seed101_prior_exact_checks']==4,'source/baseline identity checks missing')
    require(sha(score_code)=='898c990128423df71705696d8cbe35066623518153f637232a2fce30a2c61ca1','frozen PSD implementation changed')
    spec=importlib.util.spec_from_file_location('frozen_pixel_score',score_code);lib=importlib.util.module_from_spec(spec);spec.loader.exec_module(lib)
    files={str(Path(receipt_path).resolve()):pin(receipt_path),str(Path(score_code).resolve()):pin(score_code)};groups={};domains={}
    for raw in RAWS:
        p=Path(receipt['native_domains_source'])/f'{raw:03d}_native_domains.npz';files[str(p)]=pin(p)
        with np.load(p,allow_pickle=False) as z:domains[raw]={r:z[r].copy() for r in ROLES}
        require(all(v.shape==(1280,1920) and v.dtype==bool for v in domains[raw].values()),'native domain mismatch')
    for row in receipt['packages']:
        key=(row['raw'],row['arm'],row['seed']);folder=Path(row['path']);manifest=read(folder/'manifest.json');packet=qload(folder/'supervision.npz')
        for p in [folder/'manifest.json',folder/'supervision.npz']:files[str(p)]=pin(p)
        require(manifest['source_composited'] is False and manifest['source_kind']=='lossless_raw_model','generated source provenance changed')
        require(packet['native_wh'].tolist()==[1920,1280] and packet['teacher_native_window_xyxy'].tolist()==[0,0,1920,1280],'only fixed full-FOV source comparison is authorized')
        require(packet['output_wh'].tolist()==row['output_wh'],'packet output mapping mismatch')
        for role in ROLES:require(np.array_equal(packet['q_'+role+'_2d'],packet[role].astype(np.float32)),'q is not exact binary permission')
        if key not in groups:
            rawpath=folder/'source/raw_model.png';files[str(rawpath)]=pin(rawpath)
            require(files[str(rawpath)]['sha256']==manifest['source_prediction_file']['sha256'],'frozen raw source digest mismatch')
            rawrgb=np.asarray(Image.open(rawpath).convert('RGB')).copy();source_wh=list(rawrgb.shape[1::-1])
            expected=np.diag([1920/source_wh[0],1280/source_wh[1],1.])
            require(np.allclose(packet['native_from_model'],expected,rtol=0,atol=1e-12),'source transform changed')
            groups[key]={'raw':key[0],'arm':key[1],'seed':key[2],'raw_rgb':rawrgb,'source_wh':source_wh,'packets':{},'hist_rgb':historical_rgb(rawrgb)}
        groups[key]['packets'][tuple(row['output_wh'])]=packet
        if row['source_level']:
            require(row['source_level_teacher_RGB_exact'] is True,'source-level receipt failed')
            require(np.array_equal(packet['target_C_srgb'],groups[key]['raw_rgb'].transpose(2,0,1).astype(np.float32)/255),'source-level identity is not exact')
    require(set(groups)=={(raw,arm,seed) for raw in RAWS for arm in ARMS for seed in SEEDS},'source group combinations mismatch')
    for group in groups.values():
        source=group['packets'][tuple(group['source_wh'])];group['source_packet']=source
        group['historical_masks']={r:historical_support(source[r]) for r in ROLES}
    selected={};support_rows=[]
    for raw in RAWS:
        related=[g for g in groups.values() if g['raw']==raw]
        for role in ROLES:
            before=candidate_tiles(domains[raw][role]);common=[]
            for tile in before:
                if all(tile_valid(p[role],tile) for g in related for p in g['packets'].values()) and all(tile_valid(g['historical_masks'][role],tile) for g in related):common.append(tile)
            selected[(raw,role)]=common
            support_rows.append({'raw':raw,'role':role,'native_permission_pixels':int(domains[raw][role].sum()),'native_anchored80_tiles_before_all_support_intersection':len(before),
              'fixed_all_arm_seed_output_history_common_tiles':[list(t) for t in common],'fixed_common_tile_count':len(common),
              'status':'quantified' if common else 'insufficient_full80_native_support_no_smaller_ROI_selected'})
    rows=[]
    for group in groups.values():
        for role in ROLES:
            tiles=selected[(group['raw'],role)]
            if not tiles:continue
            source_rgb=group['raw_rgb'].astype(np.float32)/255
            source_power=lib.band_power(source_rgb*255,tiles,1920/group['source_wh'][0])
            for wh,packet in group['packets'].items():
                rows.append(energy_row(group,role,'exact_single_area',packet['target_C_srgb'].transpose(1,2,0),packet[role],tiles,source_power,lib.band_power))
            rows.append(energy_row(group,role,'historical_PIL_bilinear_uint8_native_then_float_block4mean',group['hist_rgb'],group['historical_masks'][role],tiles,source_power,lib.band_power))
    aggregate_groups=defaultdict(list)
    for row in rows:aggregate_groups[(row['arm'],row['role'],row['path'],tuple(row['output_wh']))].append(row)
    aggregate=[]
    for (arm,role,path,wh),rs in aggregate_groups.items():
        field='shared_0p025_to_0p125_energy_retention';v=[r[field] for r in rs]
        aggregate.append({'arm':arm,'role':role,'path':path,'output_wh':list(wh),'source_case_seed_count':len(rs),'real_frame_count':len({r['raw'] for r in rs}),
          'shared_0p025_to_0p125_energy_retention_mean':float(np.mean(v)),'shared_0p025_to_0p125_energy_retention_min':min(v),'shared_0p025_to_0p125_energy_retention_max':max(v),
          'source_energy_fraction_above_target_axis_nyquist_mean':float(np.mean([r['source_energy_fraction_above_target_axis_nyquist'] for r in rs]))})
    result={'schema':'road_2d_generated_supervision_spectral_transfer.v1','complete':True,'CPU_only':True,'threads':1,'GPU_calls':0,'training_updates':0,'hidden_GT_RGB_opened':False,'original_observation_RGB_opened':False,
      'source_receipt':pin(receipt_path),'frozen_PSD_code':pin(score_code),'band_edges_cycles_per_native_px':[list(v) for v in lib.BANDS],
      'tile_support':support_rows,'rows':rows,'aggregate':aggregate,'qmask_count_changes_separate_not_quality':receipt['comparisons'],'source_pins':list(files.values()),
      'notes':['This measures transfer of the existing generated source signal, not real asphalt detail, correctness, confidence,3D image quality or new high-frequency authorization.',
        'For each raw frame and role, all arms, seeds, area output grids and the historical filter share one fixed set of globally anchored80x80-native tiles with40-native stride. Every tile has complete same-role permission and nonzero filter support in every representation. No output RGB selects the support.',
        'If S or C has no complete80-native tile, it is reported as insufficient. No smaller tile or visually chosen region substitutes for it.',
        'q mask changes and baseline/highres-only counts are reported separately. A larger highres mask never increases the analysis denominator.',
        'The PSD method is the exact frozen main-score implementation: linear-plane detrend, Hann periodogram, sRGB Y0..255, physical frequency cycles/native pixel. Primary transfer compares identical0.025–0.125 bands.768 outputs also retain a separate0.125–0.2 band;1152 source-level adds0.2–0.3.',
        'Shared-band energy ratios use identical frequency ranges; total_different_available_band_energy_retention compares different available bandwidth and must not be called same-band/RMS/percent real detail. Energy is not RMS amplitude.',
        'Source energy above target axis Nyquist cannot remain as independently represented target-grid frequencies; area filtering can alias some power, so this fraction is not an information-theoretic loss measure.',
        'Historical RGB uses uint8PIL BILINEAR source->1920x1280 then float324x4mean->480x320. Its source permission follows the exact strictly positive rational bilinear support plus full4x4 support.',
        'The two real frames and two seeds provide descriptive transfer checks. No significance test, automatic winner, hidden GT inference, exposure fit or sharpening is performed.']}
    outdir.mkdir(parents=True);write(outdir/'transfer.json',result)
    flat=[]
    for row in rows:
        r={k:v for k,v in row.items() if not isinstance(v,list)};r['output_wh']='x'.join(map(str,row['output_wh']));r['source_wh']='x'.join(map(str,row['source_wh']))
        for i,v in enumerate(row['per_band_energy_retention']):r['band'+str(i)+'_energy_retention']=v
        flat.append(r)
    if flat:
        with (outdir/'transfer.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    if make_plot:plot(result,outdir/'generated_signal_transfer.png')
    return result

def plot(result,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use('dark_background');fig,ax=plt.subplots(figsize=(10,4.6),layout='constrained')
    xs=np.arange(3);labels=['Exact area768','Exact area480','Historical480'];colors=['#69d5c6','#f6bc6b']
    for ai,arm in enumerate(ARMS):
        means=[];lo=[];hi=[]
        for width,typ in [(768,'exact_single_area'),(480,'exact_single_area'),(480,'historical_PIL_bilinear_uint8_native_then_float_block4mean')]:
            rs=[r for r in result['aggregate'] if r['arm']==arm and r['role']=='C' and r['output_wh'][0]==width and r['path']==typ]
            if not rs:means.append(np.nan);lo.append(0);hi.append(0);continue
            r=rs[0];m=r['shared_0p025_to_0p125_energy_retention_mean'];means.append(m);lo.append(m-r['shared_0p025_to_0p125_energy_retention_min']);hi.append(r['shared_0p025_to_0p125_energy_retention_max']-m)
        ax.errorbar(xs+(ai-.5)*.06,means,yerr=[lo,hi],label=arm,marker='o',capsize=4,color=colors[ai])
    ax.axhline(1,color='white',ls='--',lw=.8);ax.set_xticks(xs,labels);ax.set_ylabel('PSD energy retention | shared0.025–0.125 cycles/native px')
    ax.set_title('Generated source signal → supervision | fixed common C tiles\nTwo real frames × two seeds; ranges are observed case/seed ranges; not authentic detail recovery',fontsize=11)
    ax.legend();ax.grid(axis='y',alpha=.15);fig.savefig(path,dpi=170);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--score-code',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--no-plot',action='store_true')
    a=p.parse_args();r=run(a.receipt,a.score_code,a.out_dir,not a.no_plot);print(json.dumps({'complete':r['complete'],'rows':len(r['rows']),'tile_support':[{k:x[k] for k in ['raw','role','fixed_common_tile_count','status']} for x in r['tile_support']],'aggregate':r['aggregate']},indent=2))
if __name__=='__main__':main()
