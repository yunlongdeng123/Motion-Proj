"""缓存共同 BUILD/PCA 初始表面；所有方法使用同一输入和宽度。"""
import argparse,json,os,sys,time
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v74_h2.surface_state import initialize
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    fit=json.loads(Path('/root/autodl-tmp/data/worldsim_v74_h2/fit_blocks_r1/index.json').read_text())['episodes']
    rows=[{'dataset':r['dataset'],'case_id':r['case_id']+'__'+r['episode'],'phase':r['phase'],'input':r['input']} for r in fit]
    # DEV全204对象的BUILD初始化是共同数据准备；43/66 probe 的身份保持既有定义。
    old=json.loads(Path('/root/autodl-tmp/data/worldsim_v74/p0_r1/index.json').read_text())['cases']
    rows += [{'dataset':r['dataset'],'case_id':r['case_id'],'phase':'DEV','input':r['build_file']} for r in old if r['role']=='DEV']
    records=[]
    for i,row in enumerate(rows):
        with np.load(row['input'],allow_pickle=False) as z:obs={k:z[k] for k in ['points_actor_m','positive_actor','ambiguous_owner','origins_actor_m']}
        surface=initialize(obs,64,.15);out=a.output/row['dataset']/row['phase']/(row['case_id']+'.npz');out.parent.mkdir(parents=True,exist_ok=True);surface.save(out)
        records.append({**row,'asset':str(out),'patches':len(surface.center),'faces':8*len(surface.center),'empty':not len(surface.center)})
        if (i+1)%500==0:print(json.dumps({'initials':i+1,'total':len(rows)}),flush=True)
    result={'task':'WS-V74-H2-INITIALS-01','status':'done','width_m':.15,'max_patches':64,'records':records,
        'quality_evaluated':False,'final_loaded':False,'wall_s':time.monotonic()-start}
    (a.output/'index.json').write_text(json.dumps(result,indent=2)+'\n')
    summary={k:v for k,v in result.items() if k!='records'};summary.update(assets=len(records),empty=sum(r['empty'] for r in records))
    (ROOT/'docs/autoresearch/worldsim_v74_h2/initials_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
