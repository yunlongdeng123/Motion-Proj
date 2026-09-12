"""P1八步共享权重训练：固定教师几何轨迹上的BPTT，随后评估自身几何闭环。"""
import argparse,json,os,sys,time
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4')
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v74_h2.witness_dynamics import WitnessDynamics,learning_loss

FIELDS=['node','edge','order','active','edge_valid','candidate','all_delta','valid','evidence','target_score','target_delta','target_event']
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def load(path,per_family=None):
    batches=torch.load(path,map_location='cpu',weights_only=False);out={}
    for key in FIELDS:
        out[key]=torch.cat([torch.stack([step[key] for step in batch['sequence']],1) for batch in batches],0).cuda()
    if per_family is not None:
        names=[name for batch in batches for name in batch['cases']]
        ids=torch.tensor([i for i,n in enumerate(names) if int(n.rsplit('-',1)[1])<per_family],device='cuda')
        out={k:v[ids] for k,v in out.items()}
    return out
def batch_step(data,ids,step):return {k:v[ids,step] for k,v in data.items()}
def assess(model,data):
    h=None;losses=[];mse=[];regrets=[];n=data['node'].shape[0]
    with torch.no_grad():
        for i in range(8):
            f={k:v[:,i] for k,v in data.items()};d,score,h=model(f,h);loss,parts=learning_loss(d,score,f)
            pred=score.masked_fill(~f['valid'],1e4).argmin(-1);vals=f['target_score'].masked_fill(~f['valid'],float('inf'))
            regret=vals[torch.arange(n,device='cuda'),pred]-vals.amin(-1)
            losses.append(float(loss));mse.append(float(parts['delta_mse']));regrets.append(float(regret.mean()))
    return {'loss':sum(losses)/8,'delta_rmse':(sum(mse)/8)**.5,'event_regret':sum(regrets)/8}
def main():
    p=argparse.ArgumentParser();p.add_argument('--teacher',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--width',type=int,default=32);p.add_argument('--epochs',type=int,default=600);p.add_argument('--models',default='A,C2');p.add_argument('--memorize12',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    train=load(a.teacher/'teacher.pt',3 if a.memorize12 else None);val=train if a.memorize12 else load(a.teacher/'validation_teacher/teacher.pt');n=train['node'].shape[0]
    manifest={'task':'WS-V74-H2-GPU-TRAIN-01','teacher':str(a.teacher),'width':a.width,'epochs_cap':a.epochs,'optimizer':'Adam lr=0.002','seed':7411,
              'rollout_steps':8,'training':'BPTT over recurrent hidden state on teacher geometry trajectories; policy geometry closure evaluated separately',
              'batch_episodes':16,'fit_scenes':n,'fit_val_scenes':val['node'].shape[0],
              'inputs':'BUILD/current surface only; candidate cost targets use synthetic FIT supervision; no mechanism QUERY as input or teacher',
              'memorization_only':a.memorize12,'selection':'training loss only' if a.memorize12 else 'FIT validation loss',
              'gpu':torch.cuda.get_device_name(0),'failure_ledger_refs':['V74-H2-F05','V74-F04','V74-F09'],'failure_ledger_delta':'pending','human_verdict':None}
    write(a.output/'manifest.json',manifest);summaries=[]
    for name in a.models.split(','):
        torch.manual_seed(7411);model=WitnessDynamics(a.width,name!='C2').cuda();opt=torch.optim.Adam(model.parameters(),lr=.002)
        curve=[];best=float('inf');best_epoch=0;parameters=sum(p.numel() for p in model.parameters())
        for epoch in range(a.epochs):
            model.train();ids=torch.randperm(n,device='cuda')
            for chunk in ids.split(16):
                opt.zero_grad();hidden=None;loss=0
                for step in range(8):
                    f=batch_step(train,chunk,step);delta,score,hidden=model(f,hidden);v,_=learning_loss(delta,score,f);loss=loss+v/8
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
            if epoch%25==0 or epoch==a.epochs-1:
                model.eval();tr=assess(model,train);vr=assess(model,val)
                row={'model':name,'epoch':epoch,'train':tr,'validation':vr,'wall_s':time.monotonic()-start,'peak_gpu_GiB':torch.cuda.max_memory_allocated()/2**30}
                curve.append(row);print(json.dumps(row),flush=True);write(a.output/f'{name}_curve.json',curve)
                payload={'model':model.state_dict(),'width':a.width,'ordered':name!='C2','epoch':epoch,'config':manifest}
                if vr['loss']<best:best=vr['loss'];best_epoch=epoch;torch.save(payload,a.output/f'{name}_best.pt')
                if epoch in [0,100,300] or epoch==a.epochs-1:torch.save(payload,a.output/f'{name}_epoch-{epoch}.pt')
                # 只以FIT验证平台判断有限训练充分性，不因DEV结果扩大更新量。
                if not a.memorize12 and epoch>=200 and epoch-best_epoch>=125:break
        summaries.append({'model':name,'parameters':parameters,'best_epoch':best_epoch,'last_epoch':epoch,'best_validation_loss':best,'last':curve[-1]})
        write(a.output/'summary.json',{'models':summaries,'wall_s':time.monotonic()-start,'status':'running' if name!=a.models.split(',')[-1] else 'done','human_verdict':None})
if __name__=='__main__':main()
