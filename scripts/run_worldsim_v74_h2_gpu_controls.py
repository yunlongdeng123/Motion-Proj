"""P1同目标普通块优化及同事件池有限束搜索；每对象每轮64次硬代价。"""
import argparse,json,sys,time,os
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4')
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v74_h2.gpu_geometry import pack,unpack,observations,cost,candidate_cost
from motion_proj.worldsim_v74_h2.witness_events import birth_pool,proposals
from motion_proj.worldsim_v74_h2.synthetic import make_scene,surface_quality
from motion_proj.worldsim_v74_h2.first_event_trace import metrics,trace
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--method',choices=['C1_ORDINARY','C3_BEAM'],required=True);a=p.parse_args()
    torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();results=[]
    families=['multilayer','shared_support','missing_support','grazing_thin'];cases=[(f,i) for f in families for i in range(20)]
    write(a.output/'manifest.json',{'method':a.method,'steps':8,'max_candidates_evaluated_per_step':64,'beam_width':4 if a.method=='C3_BEAM' else 1,
        'C1':'ordinary block position/radius/rotation/split plus pairwise scale, same birth clusters; no witness-domain update proposals',
        'C3':'same full64-pool as A; bounded beam with64 total hard evaluations across parents; no continuous global optimality claim',
        'information':'BUILD only','task':'WS-V74-H2-GPU-CONTROLS-01','failure_ledger_refs':['V74-F04','V74-F09'],'failure_ledger_delta':'pending','human_verdict':None})
    for offset in range(0,len(cases),8):
        batch=cases[offset:offset+8];scene=[make_scene(f,i,7411) for f,i in batch];b=len(batch);initial=[x[1] for x in scene]
        base=pack(initial);obs=observations([x[2] for x in scene]);birth,_=birth_pool([x[2] for x in scene]);beam={k:v[:,None] for k,v in base.items()}
        paths=torch.zeros(b,1,0,device='cuda',dtype=torch.long);history=[beam];records=[]
        for step in range(8):
            width=beam['c'].shape[1];flat={k:v.flatten(0,1) for k,v in beam.items()};obsx={k:v[:,None].expand(b,width,*v.shape[1:]).flatten(0,1) for k,v in obs.items()}
            bx={k:v[:,None].expand(b,width,*v.shape[1:]).flatten(0,1) for k,v in birth.items()};pool=proposals(flat,obsx,bx,step,ordinary=a.method=='C1_ORDINARY')
            per=64//width
            if per==64:ids=torch.arange(64,device='cuda')
            else:ids=torch.cat([torch.arange(7,device='cuda'),(torch.arange(per-7,device='cuda')*5+7+step)%57+7])
            candidate={k:v[:,ids] for k,v in pool['states'].items()};value=candidate_cost(candidate,obsx);value.masked_fill_(~pool['valid'][:,ids],float('inf'))
            value=value.reshape(b,width*per);new={k:v.reshape(b,width*per,*v.shape[2:]) for k,v in candidate.items()}
            keep=1 if a.method=='C1_ORDINARY' else 4;chosen=value.topk(keep,largest=False).indices;rr=torch.arange(b,device='cuda')[:,None]
            parent=chosen//per;event=ids[chosen%per];paths=torch.cat([paths[rr,parent],event[...,None]],-1)
            beam={k:v[rr,chosen] for k,v in new.items()};history.append({k:v.detach().cpu() for k,v in beam.items()})
            records.append({'value':value.cpu(),'selected':chosen.cpu(),'parent':parent.cpu(),'event':event.cpu(),'paths':paths.cpu()})
        final=unpack({k:v[:,0] for k,v in beam.items()})
        for j,((family,i),asset,data) in enumerate(zip(batch,final,scene)):
            folder=a.output/f'{family}-{i:02d}';folder.mkdir();asset.save(folder/'final.npz');np.savez_compressed(folder/'query_chain.npz',**trace(asset,data[3]))
            torch.save({'states':[({k:v[j] for k,v in h.items()}) for h in history],'beam_records':[{k:v[j] for k,v in r.items()} for r in records]},folder/'trace.pt')
            row={'case':folder.name,'family':family,'method':a.method,'build':metrics(asset,data[2]),'query':metrics(asset,data[3]),**surface_quality(asset,data[0]),'path':paths[j,0].cpu().tolist(),'evaluations':512}
            results.append(row);write(folder/'results.json',row)
        print(json.dumps({'method':a.method,'completed':offset+b,'total':80,'wall_s':time.monotonic()-start}),flush=True)
    write(a.output/'results.json',results)
    summary=[{'family':f,'query':{k:float(np.mean([r['query'][k] for r in results if r['family']==f])) for k in ['hit','early','miss','recall_02','any_correct','free_m']}} for f in families]
    write(a.output/'summary.json',{'rows':summary,'wall_s':time.monotonic()-start,'gpu':torch.cuda.get_device_name(0),'status':'done','human_verdict':None})
if __name__=='__main__':main()
