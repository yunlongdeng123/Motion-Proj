"""Offline CPU scoring, strictly after all48 frozen pilot predictions pass scope gates.

python score_pilot.py --predictions P/predictions.json --manifest prepared/pilot_manifest.json \
  --truth-config private/truth_config.json --out scoring/results.json

Generation never imports this program or the independent truth configuration.
No winner is selected automatically. --allow-cpu-mock is solely for labelled CPU
fixture validation; its output cannot be presented as GPU experimental evidence.
Dependencies: numpy, Pillow, scipy. CPU thread count is1; CUDA is disabled.
"""
from __future__ import annotations
import os
for _name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, copy, hashlib, json, math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter

IDS=['visible_scene081_raw000_v0','visible_scene081_raw000_v1','visible_scene081_raw050_v0','visible_scene081_raw050_v1','scene065_raw086','scene065_raw110']
ARMS=['full768_T0','full768_base','crop768_T0','full1152_T0']
SEEDS=[101,202]
SIGMAS_NATIVE=(2.5,5.,10.)
BANDS=((.025,.0625),(.0625,.125),(.125,.2),(.2,.3),(.3,.5))
PATCH_NATIVE=80

class GateError(RuntimeError):pass
def require(condition,message):
    if not condition:raise GateError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.writing')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');tmp.replace(path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pin(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def rgb(path):return np.asarray(Image.open(path).convert('RGB')).copy()
def binary(path):
    a=np.asarray(Image.open(path).convert('L'));require(np.isin(a,[0,255]).all(),'nonbinary mask: '+str(path));return a>127
def roles(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def luminance(a):return a.astype(np.float32)@np.array([.2126,.7152,.0722],np.float32)

def view_geometry(plan,native_hw):
    w,h=map(int,plan['output_wh']);x0,y0,x1,y1=map(int,plan['window_xyxy'])
    require(w>0 and h>0 and 0<=x0<x1<=native_hw[1] and 0<=y0<y1<=native_hw[0],'invalid view geometry')
    sx,sy=(x1-x0)/w,(y1-y0)/h
    matrix=np.array([[sx,0,x0],[0,sy,y0],[0,0,1]],dtype=np.float64)
    require(abs(sx-sy)<1e-9,'pilot scoring expects isotropic axis-aligned scale')
    if 'native_from_input' in plan:
        require(np.allclose(plan['native_from_input'],matrix,atol=1e-12,rtol=0),'view matrix/window conflict')
    return w,h,x0,y0,x1,y1,sx,sy,matrix

def view_neighbors(plan,shape):
    # Exact _plan/_neighbors pixel-center coordinates from input_builder.py.
    # This is four-neighbor bilinear sampling WITHOUT area/antialias filtering.
    w,h,x0,y0,x1,y1,sx,sy,_=view_geometry(plan,shape)
    yy,xx=np.mgrid[:h,:w]
    x=(xx+.5)*sx+x0-.5;y=(yy+.5)*sy+y0-.5
    lx=np.floor(x).astype(np.int64);ly=np.floor(y).astype(np.int64)
    hx=lx+1;hy=ly+1
    valid=(lx>=0)&(ly>=0)&(hx<shape[1])&(hy<shape[0])
    fx=x-np.floor(x);fy=y-np.floor(y)
    lx=np.clip(lx,0,shape[1]-1);hx=np.clip(hx,0,shape[1]-1)
    ly=np.clip(ly,0,shape[0]-1);hy=np.clip(hy,0,shape[0]-1)
    return lx,ly,hx,hy,fx,fy,valid

def sample_view(array,plan,mask=False):
    lx,ly,hx,hy,fx,fy,valid=view_neighbors(plan,array.shape[:2])
    if mask:return array[ly,lx]&array[ly,hx]&array[hy,lx]&array[hy,hx]&valid
    if array.ndim==3:fx=fx[...,None];fy=fy[...,None]
    v=(1-fx)*(1-fy)*array[ly,lx]+fx*(1-fy)*array[ly,hx]+(1-fx)*fy*array[hy,lx]+fx*fy*array[hy,hx]
    return np.rint(np.clip(v,0,255)).astype(np.uint8)

def inverse_view(model,native_hw,plan):
    _,_,x0,y0,x1,y1,_,_,_=view_geometry(plan,native_hw)
    a=np.full((*native_hw,3),127,np.uint8);valid=np.zeros(native_hw,bool)
    a[y0:y1,x0:x1]=np.asarray(Image.fromarray(model).resize((x1-x0,y1-y0),Image.Resampling.BILINEAR))
    valid[y0:y1,x0:x1]=True
    return a,valid

def freeze_gate(predictions_path,manifest_path,allow_cpu_mock=False):
    """No truth path is accepted or read by this phase. Flags alone never prove scope."""
    p=read(predictions_path);m=read(manifest_path)
    require(m.get('schema')=='road_2d_teacher_pilot.v1' and m.get('complete') is True,'prepared manifest incomplete/schema')
    require(p.get('schema')=='road_2d_teacher_pilot_predictions.v1' and p.get('complete') is True,'predictions incomplete/schema')
    mock=p.get('CPU_mock') is True
    require(not mock or allow_cpu_mock,'CPU mock is not an actual GPU experiment; explicit --allow-cpu-mock required')
    require(m.get('GT_opened') is False and p.get('GT_opened') is False,'generation truth-separation flag failed')
    require(m.get('training_updates')==p.get('training_updates')==0,'unexpected training')
    require(p.get('attempted_diffusion_calls')==p.get('completed_diffusion_calls')==m.get('planned_diffusion_calls')==48,'exactly48 completed predictions required before truth')
    require(m.get('seeds')==SEEDS and m.get('arms')==ARMS,'frozen arm/seed contract changed')
    require(p.get('adapter_unchanged') is True and p.get('checkpoint_unchanged') is True,'adapter/checkpoint invariant missing')
    require(Path(p['manifest']['path']).resolve()==Path(manifest_path).resolve(),'prediction manifest path mismatch')
    require(p['manifest']['sha256']==sha(manifest_path),'prediction manifest digest mismatch')
    cases={c['id']:c for c in m['cases']}
    require(len(m['cases'])==6 and set(cases)==set(IDS),'six fixed cases required')
    expected={(sid,arm,seed) for sid in IDS for arm in ARMS for seed in SEEDS}
    planned={(j['case_id'],j['arm'],j['seed']):j for j in m['jobs']}
    actual={(j['case_id'],j['arm'],j['seed']):j for j in p['jobs']}
    require(len(m['jobs'])==len(p['jobs'])==len(planned)==len(actual)==48 and set(planned)==set(actual)==expected,'missing/duplicate/extra prediction combinations')
    checked={};loaded={};output_records=[]
    def check(record):
        key=str(Path(record['path']).resolve())
        if key not in checked:
            require(sha(key)==record['sha256'],'file digest mismatch: '+key);checked[key]=record['sha256']
        else:require(checked[key]==record['sha256'],'conflicting pin: '+key)
    for sid,c in cases.items():
        check(c['observation']);check(c['roles_native'])
        obs=rgb(c['observation']['path']);r=roles(c['roles_native']['path']);shape=tuple(c['native_hw'])
        require(shape==(1280,1920) and obs.shape==(*shape,3),'native grid must be1920x1280')
        require(all(r[k].shape==shape and r[k].dtype==bool for k in ['W_output','K','H_input','target_core','C_mat']),'native role shape/type')
        require(not (r['W_output']&(~r['H_input']|r['K'])).any(),'native W/H/K contract')
        require(not r.get('pending_output',np.zeros(shape,bool)).any(),'pending output must be empty')
        common_valid=np.ones(shape,bool)
        for name in ['full768','crop768','full1152']:
            v=c['views'][name];require(v.get('available') is True,'all three fixed views must be available')
            require(v['model_wh']==v['plan']['output_wh'],'view shape/plan mismatch')
            w,h,*_=view_geometry(v['plan'],shape)
            for key in ['I_geo','H_input','roles_model','construction']:check(v[key])
            inp=rgb(v['I_geo']['path']);hole=binary(v['H_input']['path'])
            require(inp.shape==(h,w,3) and hole.shape==(h,w),'input view shape mismatch')
            require(np.all(inp[hole]==127),'legal input neutralization failed')
            receipt=read(v['construction']['path']);matrix=view_geometry(v['plan'],shape)[-1]
            require(np.allclose(receipt['native_from_input'],matrix,atol=1e-12,rtol=0),'construction coordinate mismatch')
            _,valid=inverse_view(inp,shape,v['plan'])
            require(not (r['W_output']&~valid).any(),'view does not contain frozen W')
            common_valid&=valid
        loaded[sid]=(obs,r,common_valid)
    for key,job in actual.items():
        template=planned[key];sid,arm,seed=key;c=cases[sid];obs,r,_=loaded[sid]
        require(all(job.get(k)==template.get(k) for k in ['id','case_id','arm','seed','view','adapter']),'job metadata mismatch')
        require(job['view']==arm.rsplit('_',1)[0],'arm/view mismatch')
        require(job.get('W_outside_RGB_exact') is True and job.get('K_RGB_exact') is True,'scope receipt missing')
        for name in ['raw_model','raw_native','final_native','candidate_native_valid']:
            check(job[name]);output_records.append(job[name])
        model=rgb(job['raw_model']['path']);native=rgb(job['raw_native']['path']);final=rgb(job['final_native']['path'])
        v=c['views'][job['view']]
        require(list(model.shape[1::-1])==v['model_wh'],'raw model grid mismatch')
        require(native.shape==final.shape==obs.shape,'raw/final native shape mismatch')
        expected_native,valid=inverse_view(model,c['native_hw'],v['plan'])
        require(np.array_equal(native,expected_native),'raw inverse transform mismatch (including crop offset)')
        require(np.array_equal(binary(job['candidate_native_valid']['path']),valid),'candidate valid footprint mismatch')
        require(np.array_equal(final[~r['W_output']],obs[~r['W_output']]),'actual pixels changed outside frozen W')
        require(np.array_equal(final[r['K']],obs[r['K']]),'actual protected K pixels changed')
        require(np.array_equal(final[r['W_output']],native[r['W_output']]),'final is not exact frozen-W raw writeback')
        if template.get('legacy_exact_replay_expected'):
            require(job.get('legacy_replay_exact') and all(job['legacy_replay_exact'].values()),'legacy replay failed')
            for name,record in c['legacy_B_expected'].items():
                check(record);require(np.array_equal(rgb(job[name]['path']),rgb(record['path'])),'legacy pixel replay mismatch')
    return p,m,loaded,{'passed':True,'completed_jobs':48,'verified_unique_files':len(checked),'actual_W_outside_exact':True,'actual_K_exact':True,'actual_native_transform_exact':True,'CPU_mock':mock,'GT_opened_before_gate':False,'predictions':pin(predictions_path),'manifest':pin(manifest_path)}

def lab(a):
    v=a.astype(np.float64)/255
    v=np.where(v<=.04045,v/12.92,((v+.055)/1.055)**2.4)
    xyz=v@np.array([[.4124564,.2126729,.0193339],[.3575761,.7151522,.1191920],[.1804375,.0721750,.9503041]])
    xyz=xyz/np.array([.95047,1.,1.08883])
    f=np.where(xyz>(6/29)**3,np.cbrt(xyz),xyz/(3*(6/29)**2)+4/29)
    return np.stack([116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])],-1)

def reference_structure_ok(a):
    y=luminance(a);r,g,b=np.moveaxis(a.astype(np.float32),-1,0)
    paint=((y>130)&(np.ptp(a,axis=-1)<45))|((r>130)&(g>115)&(b<.8*g))
    dy,dx=np.gradient(gaussian_filter(y,2.5));bad=paint|(np.hypot(dx,dy)>2.)
    return distance_transform_edt(~bad)>12

def fixed_tiles(mask):
    h,w=mask.shape;p=PATCH_NATIVE
    return [(x,y,p) for y in range(0,h-p+1,40) for x in range(0,w-p+1,40) if mask[y:y+p,x:x+p].all()]

def band_power(image,tiles,spacing=1.,offset=(0,0)):
    powers=[];yimage=luminance(image)
    for x,y,p in tiles:
        xf=(x-offset[0])/spacing;yf=(y-offset[1])/spacing;pf=p/spacing
        require(all(abs(v-round(v))<1e-7 for v in [xf,yf,pf]),'tile/view coordinates need integer alignment under frozen pilot plans')
        x,y,p=map(lambda z:int(round(z)),[xf,yf,pf]);a=yimage[y:y+p,x:x+p].astype(np.float64)
        require(a.shape==(p,p),'tile outside view')
        yy,xx=np.mgrid[:p,:p];design=np.stack([np.ones_like(xx),xx/p,yy/p],-1).reshape(-1,3)
        a-= (design@np.linalg.lstsq(design,a.ravel(),rcond=None)[0]).reshape(p,p)
        window=np.outer(np.hanning(p),np.hanning(p));power=np.abs(np.fft.fft2(a*window))**2/(p*p*np.sum(window**2))
        freq=np.fft.fftfreq(p,d=spacing);rad=np.hypot(freq[:,None],freq[None,:]);nyq=.5/spacing
        powers.append([float(power[(rad>=lo)&(rad<min(hi,nyq))].sum()) if lo<nyq-1e-8 else None for lo,hi in BANDS])
    return [float(np.mean([row[i] for row in powers])) if powers and powers[0][i] is not None else None for i in range(len(BANDS))]

def metrics(pred,ref,mask,distance_native,tiles,spacing=1.,offset=(0,0),view_plan=None):
    safe=mask if view_plan is None else sample_view(mask,view_plan,mask=True)
    dist=distance_native if view_plan is None else sample_float(distance_native,view_plan)
    require(int(safe.sum())>=64,'insufficient fixed same-domain scoring mask')
    p=pred[safe].astype(np.float64);r=ref[safe].astype(np.float64)
    lp=lab(p);lr=lab(r);de=np.linalg.norm(lp-lr,axis=-1)
    output={'grid_wh':list(pred.shape[1::-1]),'valid_pixels':int(safe.sum()),'spacing_native_px':spacing,
      'color':{'RGB_MAE_0_255':float(np.abs(p-r).mean()),'mean_RGB_bias':(p-r).mean(0).tolist(),'mean_Lab_bias':(lp-lr).mean(0).tolist(),'deltaE76_mean':float(de.mean()),'deltaE76_median':float(np.median(de))},'HP':{}}
    py=luminance(pred);ry=luminance(ref)
    for sigma in SIGMAS_NATIVE:
        hmask=safe&(dist>max(12,4*sigma))
        if hmask.sum()<64:
            output['HP'][str(sigma)]={'status':'insufficient','valid_pixels':int(hmask.sum())};continue
        sg=sigma/spacing;ph=py-gaussian_filter(py,sg,truncate=4);rh=ry-gaussian_filter(ry,sg,truncate=4)
        pv=ph[hmask].astype(np.float64);rv=rh[hmask].astype(np.float64)
        pr=float(np.sqrt(np.mean(pv*pv)));rr=float(np.sqrt(np.mean(rv*rv)))
        pc=np.quantile(np.abs(pv),.95);rc=np.quantile(np.abs(rv),.95)
        pw=float(np.sqrt(np.mean(np.minimum(np.abs(pv),pc)**2)));rw=float(np.sqrt(np.mean(np.minimum(np.abs(rv),rc)**2)))
        output['HP'][str(sigma)]={'sigma_native_px':sigma,'sigma_this_grid_px':sg,'valid_pixels':int(hmask.sum()),'prediction_RMS_Y_0_255':pr,'reference_RMS_Y_0_255':rr,'RMS_ratio':pr/max(rr,1e-12),'winsor95_RMS_ratio':pw/max(rw,1e-12),'HP_pattern_RMSE_Y_0_255':float(np.sqrt(np.mean((pv-rv)**2)))}
    pp=band_power(pred,tiles,spacing,offset);rr=band_power(ref,tiles,spacing,offset)
    ratios=[float(a/max(b,1e-12)) if a is not None else None for a,b in zip(pp,rr)]
    output['PSD']={'fixed_native_tile_count':len(tiles),'fixed_tile_native_wh':PATCH_NATIVE,'band_edges_cycles_per_native_px':[list(v) for v in BANDS],
      'axis_nyquist_cycles_per_native_px':.5/spacing,'mean_band_power_prediction_Y_squared':pp,'mean_band_power_reference_Y_squared':rr,'band_power_ratio':ratios,
      'mean_abs_log2_band_power_ratio':float(np.mean([abs(math.log2(max(v,1e-12))) for v in ratios if v is not None])) if any(v is not None for v in ratios) else None}
    return output

def sample_float(a,plan):
    lx,ly,hx,hy,fx,fy,_=view_neighbors(plan,a.shape)
    return (1-fx)*(1-fy)*a[ly,lx]+fx*(1-fy)*a[ly,hx]+(1-fx)*fy*a[hy,lx]+fx*fy*a[hy,hx]

def score(predictions_path,manifest_path,truth_config_path,out,allow_cpu_mock=False):
    out=Path(out)
    require(not out.exists(),'preserve existing score file; specify a new --out')
    # This call must finish before opening the independent truth configuration.
    p,m,loaded,gate=freeze_gate(predictions_path,manifest_path,allow_cpu_mock)
    require(Path(truth_config_path).resolve() not in [Path(predictions_path).resolve(),Path(manifest_path).resolve()],'truth configuration must be independent')
    truth=read(truth_config_path)
    require(truth.get('schema')=='road_2d_teacher_scoring_truth.v1','independent truth schema mismatch')
    require(set(truth['cases'])==set(IDS),'truth configuration must explicitly name all six frozen cases')
    result={'schema':'road_2d_teacher_postfreeze_CPU_score.v1','complete':False,'CPU_only':True,'threads':1,'GT_opened_after_gate':True,
      'mode':'CPU_MOCK_VALIDATION_ONLY' if gate['CPU_mock'] else 'FROZEN_PREDICTION_CPU_SCORING','gate':gate,'truth_config':pin(truth_config_path),'cases':[],'jobs':[],
      'notes':['No automatic arm winner or checkpoint selection. All48 frozen outputs passed actual pixel-scope and inverse-transform gates before this scoring truth configuration was read.',
        'Four synthetic tasks reuse only two clean frames. Real hidden road has no GT and is not included in hidden-road accuracy aggregation.',
        'Native metrics use identical certified original-pixel masks for every arm/seed. Same-view metrics use exact builder four-neighbor pixel-center RGB sampling, including crop offset, rounded to uint8; no Pillow antialias substitution.',
        'HP sigma is always specified in native pixels and divided by native-pixels/model-pixel for each view. PSD frequencies are cycles per native pixel and capped at each grid axis Nyquist.',
        'Generated RGB never selects masks or tiles. Reference paint/coarse edges and mask-boundary neighborhoods are excluded. Generated false structures can still increase RMS/PSD;95% winsorization is separate sensitivity.',
        'PSD/HP are descriptive spectral diagnostics; higher energy is not necessarily authentic material. HP pattern RMSE includes stochastic phase disagreement.',
        'For real cases, only same-location legally visible road preservation is scored, outside H/K and labelled shadows. This is a known-region control, not error in hidden road.']}
    references={};cache={};case_lookup={c['id']:c for c in m['cases']}
    maskdir=out.parent/(out.stem+'_masks');maskdir.mkdir(parents=True,exist_ok=True)
    for sid in IDS:
        c=case_lookup[sid];obs,r,valid=loaded[sid];tc=truth['cases'][sid]
        require(tc['domain']==c['domain'],'truth/generation case domain mismatch')
        if tc['domain']=='synthetic_visible_2d':
            for key in ['clean_rgb','visible_road_certificate']:
                require(sha(tc[key]['path'])==tc[key]['sha256'],'independent truth digest mismatch')
            ref=rgb(tc['clean_rgb']['path']);cert=binary(tc['visible_road_certificate']['path'])
            require(ref.shape==obs.shape and cert.shape==obs.shape[:2],'truth/native shape mismatch')
            base=r['target_core']&r['W_output']&~r['K']&cert
            distance=distance_transform_edt(r['W_output'])
            kind='synthetic_certified_hidden_target'
        else:
            require(tc['domain']=='real_unlabelled','unsupported domain')
            require('clean_rgb' not in tc,'real hidden GT must not be invented')
            ref=obs;base=r['C_mat']&~r['H_input']&~r['K']
            for key in ['shadow_possible','shadow_uncertain','shadow_boundary_guard','foreground_uncertainty']:
                if key in r:base&=~r[key]
            distance=distance_transform_edt(base);kind='known_visible_road_preservation_only'
        mask=base&(distance>12)&reference_structure_ok(ref)&(distance_transform_edt(valid)>40)
        tiles=fixed_tiles(mask)
        require(mask.sum()>=64,'no fixed scoring support: '+sid)
        maskpath=maskdir/(sid+'.png');Image.fromarray(mask.astype(np.uint8)*255).save(maskpath)
        result['cases'].append({'id':sid,'domain':tc['domain'],'metric_role':kind,'hidden_road_GT_available':tc['domain']=='synthetic_visible_2d','fixed_native_mask':pin(maskpath),'fixed_native_mask_pixels':int(mask.sum()),'fixed_native_tiles':[list(t) for t in tiles]})
        references[sid]=(ref,mask,distance,tiles,kind)
    for j in p['jobs']:
        sid=j['case_id'];c=case_lookup[sid];ref,mask,distance,tiles,kind=references[sid]
        view=c['views'][j['view']];plan=view['plan'];sx=view_geometry(plan,c['native_hw'])[6];offset=tuple(plan['window_xyxy'][:2])
        key=(sid,j['view'],j['raw_model']['sha256'],j['raw_native']['sha256'])
        if key not in cache:
            model=rgb(j['raw_model']['path']);native=rgb(j['raw_native']['path'])
            refview=sample_view(ref,plan)
            cache[key]={'native':metrics(native,ref,mask,distance,tiles),
              'same_view':metrics(model,refview,mask,distance,tiles,sx,offset,plan)}
        result['jobs'].append({'id':j['id'],'case_id':sid,'arm':j['arm'],'seed':j['seed'],'view':j['view'],'metric_role':kind,**cache[key]})
    require(len(result['jobs'])==48,'scoring output count mismatch')
    require(sha(predictions_path)==gate['predictions']['sha256'] and sha(manifest_path)==gate['manifest']['sha256'],'freeze files changed during scoring')
    result.update(complete=True,scored_jobs=48,unique_pixel_metric_computations=len(cache),automatic_winner_selected=False)
    write(out,result)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['predictions','manifest','truth-config','out']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--allow-cpu-mock',action='store_true')
    args=parser.parse_args()
    result=score(args.predictions,args.manifest,args.truth_config,args.out,args.allow_cpu_mock)
    print(json.dumps({'complete':result['complete'],'mode':result['mode'],'scored_jobs':48,'GT_opened_after_gate':True,'out':str(args.out.resolve())}))
if __name__=='__main__':main()
