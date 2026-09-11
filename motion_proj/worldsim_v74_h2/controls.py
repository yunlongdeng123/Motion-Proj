"""强控制：普通边界/位置模式搜索与同信息有限宽度束搜索。"""
import numpy as np
from .surface_state import apply,initialize
from .first_event_trace import objective
def events(s,obs,step,budget=64):
    pool=[];move=.3*(.7**(step//2));factor=1.35 if step<4 else 1.15
    for k in range(len(s.center)):
        for axis in range(3):
            for sign in [-1,1]:
                delta=np.zeros(3);delta[axis]=move*sign;pool.append([{'kind':'move','patch':k,'delta':delta.tolist()}])
        for sector in range(8):
            for scale in [1/factor,factor]:pool.append([{'kind':'radius','patch':k,'sector':sector,'factor':scale}])
        for scale in [.6,1.4]:pool.append([{'kind':'scale','patch':k,'factor':scale}])
        for axis in range(2):
            for sign in [-1,1]:
                rv=s.rotation[k,:,axis]*.08*sign;pool.append([{'kind':'rotate','patch':k,'rotvec':rv.tolist()}])
        pool.append([{'kind':'split','patch':k,'axis':step%2}])
    birth=initialize(obs,count=min(4,int(obs['positive_actor'].sum())),width=.4)
    extra=[[{'kind':'birth','center':c.tolist(),'rotation':r.tolist(),'radius':rho.tolist()}] for c,r,rho in zip(birth.center,birth.rotation,birth.radius)]
    n=max(0,budget-len(extra))
    if len(pool)>n:
        ids=(np.linspace(0,len(pool)-1,n,dtype=int)+step*7)%len(pool);pool=[pool[i] for i in ids]
    return pool+extra
def search(initial,obs,variant='C1',steps=8,event_budget=64):
    beam=[(objective(initial,obs),initial,[])];evaluation_count=1;records=[]
    width=1 if variant=='C1' else 4
    for step in range(steps):
        candidates=list(beam);per=max(1,event_budget//len(beam));round_records=[]
        for parent,(old,s,path) in enumerate(beam):
            for event in events(s,obs,step,per):
                trial=apply(s,event);value=objective(trial,obs);evaluation_count+=1
                candidates.append((value,trial,path+[event]));round_records.append({'parent':parent,'event':event,'objective':value,'delta':value-old})
        candidates.sort(key=lambda x:x[0]);beam=candidates[:width]
        records.append({'step':step,'all_candidates':round_records,'best_objective':beam[0][0],'best_path':beam[0][2],
                        'retained_objectives':[x[0] for x in beam]})
    return beam[0][1],{'variant':variant,'evaluations':evaluation_count,'rounds':records,'selected_events':beam[0][2],
                       'bounded_search':'ordinary pattern search / finite beam; not continuous global optimum'}
