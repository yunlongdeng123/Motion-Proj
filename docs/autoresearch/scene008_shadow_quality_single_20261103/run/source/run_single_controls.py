import os,subprocess,sys,json
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
RUN=Path(__file__).resolve().parent.parent
def call(actor,args,log):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
    with (RUN/log).open('x') as f:subprocess.run([sys.executable,str(RUN/'source/single_actor_controls.py'),'--actor',str(actor)]+args,cwd=RUN,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
    for actor in [5,37]:
        call(actor,['--prepare'],'single_a%d_prepare.log'%actor)
        call(actor,['--steps','32','--output','smoke'],'single_a%d_smoke.log'%actor)
        call(actor,['--steps','33','--output','smoke_restore','--resume','smoke/checkpoint000032.pt'],'single_a%d_restore.log'%actor)
        call(actor,['--steps','33','--output','smoke_repeat','--resume','smoke/checkpoint000032.pt'],'single_a%d_repeat.log'%actor)
        call(actor,['--steps','1000','--output','formal'],'single_a%d_train.log'%actor)
        call(actor,['--evaluate','--output','evaluation','--resume','formal/checkpoint001000.pt'],'single_a%d_eval.log'%actor)
    (RUN/'single_controls_complete.json').write_text(json.dumps(dict(formal_updates=2000,smoke_updates=64,replay_updates=4,total_updates=2068,GPU_shutdown=False)))
if __name__=='__main__':main()
