"""存量表面的双精度首交点复核；无 GPU、无模型推理、无参数拟合。"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,pathlib,time,platform
import numpy as np

def intersect(v,f,o,d,r):
    tri=np.asarray(v,dtype=np.float64)[f];e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
    first=np.full(len(r),np.inf);fid=np.full(len(r),-1,dtype=int);good={x:np.zeros(len(r),bool) for x in [.1,.2,.3,.5]};later=np.full(len(r),np.inf)
    for start in range(0,len(r),16):
        end=min(start+16,len(r));dire=np.asarray(d[start:end],dtype=np.float64)
        h=np.cross(dire[:,None,:],e2[None,:,:]);det=np.einsum('tij,ij->ti',h,e1)
        inv=np.zeros_like(det);np.divide(1,det,out=inv,where=np.abs(det)>1e-12)
        s=np.asarray(o[start:end],dtype=np.float64)[:,None,:]-tri[None,:,0,:]
        u=np.einsum('tij,tij->ti',s,h)*inv;q=np.cross(s,e1[None,:,:]);vv=np.einsum('ti,tji->tj',dire,q)*inv
        t=np.einsum('tij,ij->ti',q,e2)*inv
        valid=(np.abs(det)>1e-12)&(u>=-1e-10)&(vv>=-1e-10)&(u+vv<=1+1e-10)&(t>=0)
        t=np.where(valid,t,np.inf)
        idx=t.argmin(1);dist=t[np.arange(end-start),idx];first[start:end]=dist;fid[start:end]=np.where(np.isfinite(dist),idx,-1)
        for eps in good:
            close=np.abs(t-r[start:end,None])<=eps;good[eps][start:end]=close.any(1)
            if eps==.2:later[start:end]=np.where(close,t,np.inf).min(1)
    return first,fid,good,later

def metrics(t,r,good,eps):
    finite=np.isfinite(t);hit=finite&(np.abs(t-r)<=eps);early=finite&(t<r-eps);late=finite&(t>r+eps)
    return {'rays':len(r),'hit':int(hit.sum()),'early':int(early.sum()),'late':int(late.sum()),'miss':int((~finite).sum()),'early_with_later':int((early&good).sum()),'any_correct':int(good.sum()),'positive_ray_free_m':float(np.where(early,r-eps-t,0).mean()) if len(r) else None}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,required=True);args=ap.parse_args();root=args.root
    src=root/'sources/docs/autoresearch/worldsim_v73/paper_forensics';out=root/'audit';out.mkdir(exist_ok=True)
    start=time.monotonic();allstats={};selected={};allrows={};paths={}
    for f in src.glob('*/summary.json'):
        d=json.loads(f.read_text());allstats.update(d['statistics']);selected.update(d['selected']);allrows.update(d['rows'])
        for name in d['statistics']:paths[name]=f.parent/'cases'/(name+'.npz')
    rows=[];cached={};arrays={}
    for name in selected:
        with np.load(paths[name]) as z:a={k:z[k] for k in z.files}
        t,ids,good,later=intersect(a['vertices'],a['faces'],a['origins'],a['directions'],a['ranges'])
        cached[(name,name)]=(t,ids,good,later);arrays[name]=a
        finite=np.isfinite(t)&np.isfinite(a['first']);rid=int(a['selected_ray'])
        counts=metrics(t,a['ranges'],good[.2],.2);old=selected[name]['actor']['counts']
        row={'legacy_label':name,'actor':selected[name]['actor'],'ray':rid,'old_error_m':selected[name]['early_error_m'],'recast_error_m':float(a['ranges'][rid]-t[rid]),'first_m':float(t[rid]),'observed_m':float(a['ranges'][rid]),'later_correct_m':float(later[rid]) if np.isfinite(later[rid]) else None,'selected_is_early':bool(t[rid]<a['ranges'][rid]-.2),'selected_later_correct':bool(good[.2][rid]),'thresholds':{str(e):metrics(t,a['ranges'],good[e],e) for e in good},'first_max_abs_difference_m':float(np.abs(t[finite]-a['first'][finite]).max()) if finite.any() else None,'finite_mask_difference':int((np.isfinite(t)!=np.isfinite(a['first'])).sum()),'count_delta_vs_saved':{k:counts[k]-old[k] for k in old},'early_with_later_mask_difference':int(((t<a['ranges']-.2)&good[.2] != (a['first']<a['ranges']-.2)&a['any_hit']).sum()),'ray_unit_max_error':float(np.abs(np.linalg.norm(a['directions'],axis=1)-1).max()),'selected_triangle_q':selected[name]['face_q'],'selected_component_supported':selected[name]['component_supported']}
        np.savez_compressed(out/(name+'_recast.npz'),first=t,face_ids=ids,later=later,**{'any_'+str(e):g for e,g in good.items()})
        rows.append(row);print(json.dumps({'case':name,'rays':len(t),'count_delta':row['count_delta_vs_saved'],'error_m':row['recast_error_m']}),flush=True)
    pairs=[]
    for entry in json.loads((root/'paired/manifest.json').read_text()):
        label=entry['source_selection_method'];a=arrays[label]
        frame_sizes=[v['positive_actor_count'] for v in entry['ray_frames'] if v['role']=='heldout_time'];assert sum(frame_sizes)==len(a['ranges'])
        for name,rec in entry['surfaces'].items():
            if (label,name) in cached:t,ids,good,later=cached[(label,name)]
            else:
                with np.load(root/'paired'/rec['export']) as s:t,ids,good,later=intersect(s['vertices'],s['faces'],a['origins'],a['directions'],a['ranges'])
            row=next(v for v in allrows[name] if v['owner']==entry['actor']['owner'] and v['scene']==entry['actor']['scene'])
            counts=metrics(t,a['ranges'],good[.2],.2);frames=[];lo=0
            for n in frame_sizes:
                frames.append(metrics(t[lo:lo+n],a['ranges'][lo:lo+n],good[.2][lo:lo+n],.2));lo+=n
            pair={'case':label,'method':name,'metrics':counts,'per_heldout_frame':frames,'old_count_delta':{k:counts[k]-row['counts'][k] for k in row['counts']},'selected_ray_first_m':float(t[int(a['selected_ray'])]) if np.isfinite(t[int(a['selected_ray'])]) else None}
            pairs.append(pair)
        print('PAIRED',label,flush=True)
    d=json.loads((root/'sources/docs/autoresearch/worldsim_v73/final_confirmation/analysis.json').read_text())
    comparisons={}
    for ref,parts in d['paired_final_minus'].items():
        v=parts['external_confirmation'];h=v['hit_rate']['per_log_delta'];e=v['early_rate']['per_log_delta'];rec=v['surface_recall_02']['per_log_delta'];fre=v['free_intrusion_m']['per_log_delta']
        comparisons[ref]={'metrics':v,'joint_hit_early_logs':sum(h[k]>0 and e[k]>0 for k in h),'joint_hit_recall_early_free_logs':sum(h[k]>0 and e[k]>0 and rec[k]>0 and fre[k]>0 for k in h),'nlogs':len(h)}
    result={'status':'DONE','task':'WS-V74-H2-REDISCOVERY-01','execution':{'host':platform.node(),'cpu_only':True,'new_model_forwards':0,'optimizer_updates':0,'wall_s':time.monotonic()-start},'scope':'7 preselected historical cases plus 3 paired actors x 7 saved systems; not 7 official SOTA models','rows':rows,'pairs':pairs,'old_population':allstats,'external20_saved_comparisons':comparisons,'human_verdict':None,'failure_ledger_refs':['V73-F02','V73-F03','V73-F04','V73-F09'],'failure_ledger_delta':'V74-H2-F12'}
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False));print('DONE',result['execution'])
if __name__=='__main__':main()
