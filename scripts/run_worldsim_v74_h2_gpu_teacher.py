"""P1 GPU硬事件教师及强控制；仅合成FIT监督进入教师。"""
import argparse,json,sys,time,os
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v74_h2.synthetic import make_scene,surface_quality
from motion_proj.worldsim_v74_h2.first_event_trace import trace,metrics
from motion_proj.worldsim_v74_h2.gpu_geometry import pack,unpack,observations,geometry,features,candidate_cost,cost
from motion_proj.worldsim_v74_h2.witness_events import birth_pool,proposals,choose

FAMILIES=['multilayer','shared_support','missing_support','grazing_thin']
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def cpu(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu()
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    return x

def dataset(root,role,seed,count):
    data=[]
    for f in FAMILIES:
        for i in range(count):
            truth,initial,build,query=make_scene(f,i,seed);folder=root/role/f'{f}-{i:02d}';folder.mkdir(parents=True,exist_ok=True)
            for n,s in [('truth',truth),('initial',initial)]:s.save(folder/f'{n}.npz')
            for n,o in [('build',build),('supervision' if role!='mechanism' else 'query',query)]:np.savez_compressed(folder/f'{n}.npz',**o)
            data.append({'name':folder.name,'family':f,'folder':folder,'truth':truth,'initial':initial,'build':build,'query':query})
    return data

def generate(rows,out,teacher=True,batch=8):
    all_batches=[];results=[];diag=[];start=time.monotonic()
    for offset in range(0,len(rows),batch):
        group=rows[offset:offset+batch];s=pack([x['initial'] for x in group]);build=observations([x['build'] for x in group]);supervision=observations([x['query'] for x in group])
        full={k:torch.cat([build[k],supervision[k]],1) for k in build} if teacher else build
        birth,birth_counts=birth_pool([x['build'] for x in group]);sequence=[];hist=[cpu(s)];costs=[]
        initial_gpu=geometry(s,build)['first'].cpu().numpy()
        for j,row in enumerate(group):
            ref=trace(row['initial'],row['build'])['first'];both=np.isfinite(ref)&np.isfinite(initial_gpu[j])
            diag.append({'case':row['name'],'miss_disagreements':int((np.isfinite(ref)!=np.isfinite(initial_gpu[j])).sum()),
                         'max_finite_error_m':float(np.max(np.abs(ref[both]-initial_gpu[j,both]))) if both.any() else None})
        for step in range(8):
            pool=proposals(s,build,birth,step);value=candidate_cost(pool['states'],full)
            value=torch.where(pool['valid'],value,torch.full_like(value,float('inf')));selected=value.argmin(-1);bidx=torch.arange(len(group),device='cuda')
            target=value-value[:,:1];finite=torch.where(torch.isfinite(target),target,torch.zeros_like(target))
            record={**features(s,build),'candidate':pool['desc'],'valid':pool['valid'],
                    'target_score':finite.float(),'target_delta':pool['delta'][bidx,selected].float(),
                    'target_event':selected,'all_delta':pool['delta'].float(),'kind':pool['kind'],
                    'candidate_cost':value,'evidence':pool['evidence'],'state':s,'names':pool['names']}
            sequence.append(cpu(record));s=choose(pool,selected);hist.append(cpu(s));costs.append(value[bidx,selected].cpu().tolist())
        all_batches.append({'offset':offset,'cases':[x['name'] for x in group],'sequence':sequence,'history':hist,'birth_counts':birth_counts})
        final=unpack(s)
        for j,(row,asset) in enumerate(zip(group,final)):
            method='FIT_TEACHER' if teacher else 'C1_GPU';asset.save(row['folder']/f'{method}.npz')
            first_deg=None;prev=metrics(row['initial'],row['query']);trajectory=[]
            for step,ss in enumerate(hist[1:]):
                state=unpack({k:v[j:j+1] for k,v in ss.items()})[0];state.save(row['folder']/f'{method}_step-{step}.npz')
                q=metrics(state,row['query']);trajectory.append(q)
                if first_deg is None and (q['hit']<prev['hit']-.005 or q['early']>prev['early']+.005):first_deg=step
                prev=q
            tr=trace(asset,row['query']);np.savez_compressed(row['folder']/f'{method}_query_chain.npz',**tr)
            result={'case':row['name'],'family':row['family'],'method':method,'build':metrics(asset,row['build']),
                    'query':metrics(asset,row['query']),**surface_quality(asset,row['truth']),
                    'first_degradation':first_deg,'trajectory':trajectory,'artifact':str(row['folder']/f'{method}.npz'),
                    'evaluations':8*64,'capacity_touched':False}
            results.append(result);write(row['folder']/f'{method}_results.json',result)
        print(json.dumps({'phase':'teacher' if teacher else 'C1','complete':offset+len(group),'total':len(rows),'wall_s':time.monotonic()-start,'peak_gpu_GiB':torch.cuda.max_memory_allocated()/2**30}),flush=True)
    torch.save(all_batches,out/('teacher.pt' if teacher else 'C1_states.pt'))
    write(out/('teacher_results.json' if teacher else 'C1_results.json'),results)
    write(out/'gpu_reference_diagnostic.json',diag)
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['fit','mechanism'],default='fit');a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(7411);a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    write(a.output/'manifest.json',{'task':'WS-V74-H2-GPU-P1-01','mode':a.mode,'seed':7411,'fit_scene_seed':174211,'validation_scene_seed':174311,'mechanism_scene_seed':7411,
         'steps':8,'events':64,'gpu':torch.cuda.get_device_name(0),'dtype':'float64 geometry, float32 learning features','max_faces':4096,
         'padding_slots':16,'padding_note':'P1 initial<=3 plus <=8 births/splits, so padding16 does not reduce physical512-patch budget',
         'teacher_access':'synthetic FIT BUILD+supervision; proposal/features only BUILD','cost':'same robust first distance + free prefix + plane/radial support proxy; actual surface metric separate',
         'failure_ledger_refs':['V74-F04','V74-F09','V74-H2-F03','V74-H2-F05'],'failure_ledger_delta':'pending','human_verdict':None})
    if a.mode=='fit':
        rows=dataset(a.output,'fit',174211,20);generate(rows,a.output,True)
        valdir=a.output/'validation_teacher';valdir.mkdir();rows=dataset(a.output,'fit_val',174311,5);generate(rows,valdir,True)
    else:
        rows=dataset(a.output,'mechanism',7411,20);generate(rows,a.output,False)
    write(a.output/'completion.json',{'status':'done','wall_s':time.monotonic()-start,'peak_gpu_GiB':torch.cuda.max_memory_allocated()/2**30,'human_verdict':None})
if __name__=='__main__':main()
