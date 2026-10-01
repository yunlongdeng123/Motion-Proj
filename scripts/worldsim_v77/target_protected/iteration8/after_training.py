"""单次有界控制器：等同预算训练释放GPU，再断点完成冻结三臂。"""
from pathlib import Path
import sys,time,subprocess,fcntl,json,os,traceback
P=Path(__file__).parent;sys.path.insert(0,str(P))
from asset_factory import O,read,dump
CPU='/root/autodl-tmp/envs/motionproj/bin/python';GPU='/root/autodl-tmp/envs/driveeditor/bin/python'

def main():
    lock=open(O/'after_training.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        state=read(O/'training/state.json');ps=subprocess.check_output(['ps','-eo','args='],text=True)
        active=any('iteration8/train_data_control.py' in line and line.strip().startswith('/root/autodl-tmp/envs/driveeditor/bin/python') for line in ps.splitlines())
        dump(O/'controller_state.json',{'stage':'waiting_training','steps':state['steps'],'training_alive':active,'pid':os.getpid(),'single_run':True})
        if not active:
            assert state['stage']=='complete' and state['steps']==160,'training stopped before planned160; never sample incomplete patch'
            assert read(O/'training/same_recipe_as_r7.json')['all_fixed_recipe_fields_equal']
            break
        time.sleep(10)
    for python,script,args,stage in [(GPU,'evaluate_data_control.py',['--arms','base','r7','new_data'],'fixed_three_arms'),(CPU,'summarize_results.py',[],'quantify'),(CPU,'build_effect_review.py',[],'review_encode')]:
        dump(O/'controller_state.json',{'stage':stage,'pid':os.getpid(),'single_run':True,'training_steps':160})
        with (O/(stage+'.log')).open('a') as log:
            proc=subprocess.run([python,str(P/script)]+args,stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode:raise RuntimeError(f'{script} failed with{proc.returncode}; resume engineering only, no parameter retuning')
    dump(O/'controller_state.json',{'stage':'ready_for_effect_review_and_local_delivery','training_steps':160,'pid':os.getpid(),'human_verdict':None})
if __name__=='__main__':
    try:main()
    except Exception as e:dump(O/'controller_state.json',{'stage':'failed','error':repr(e),'traceback':traceback.format_exc(),'pid':os.getpid()});raise
