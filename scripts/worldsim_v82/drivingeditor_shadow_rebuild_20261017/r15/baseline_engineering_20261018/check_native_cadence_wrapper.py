"""Exercise the actual native binding wrapper without GPU or train updates."""
from pathlib import Path
import ast,types,sys,tempfile,json,hashlib
E=Path(__file__).resolve().parent

def main():
    code=(E/'run_native_timing_safe.py').read_text()
    func=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='bind')
    schedule=list(range(5000,30001,5000));calls=[]
    class Event:
        def synchronize(self):calls.append('end.synchronize')
        def elapsed_time(self,other):calls.append('elapsed_time');return 1.
    checkpoint=types.ModuleType('paper_checkpoint')
    checkpoint.save_components=lambda path,iteration,models:calls.append(('save',iteration))
    utils=types.ModuleType('utils');utils.paper_checkpoint=checkpoint
    contracts=types.ModuleType('contracts')
    contracts.disk_forecast=lambda cfg,actual_counts=None,remaining_nodes=3:(actual_counts,remaining_nodes)
    def require_space(path,forecast):
        calls.append(('forecast',forecast));return dict(status='fixture_disk_pass',remaining_nodes=forecast[1])
    contracts.require_disk_space=require_space
    sys.modules['utils']=utils;sys.modules['contracts']=contracts
    env=dict(Path=Path,original_binding=lambda *args:calls.append('native_bind'))
    exec(compile(ast.Module(body=[func],type_ignores=[]),'run_native_timing_safe.py','exec'),env)
    with tempfile.TemporaryDirectory(dir=E,prefix='cadence_fixture_') as td:
        cfg=dict(model_path=td,test_iterations=[10000,20000,30000],paper_eval_train_each_checkpoint=True,
                 checkpoint_iterations=[10000,20000,30000],save_iterations=[10000,20000,30000])
        args=types.SimpleNamespace()
        train=types.SimpleNamespace(torch=types.SimpleNamespace(cuda=types.SimpleNamespace(Event=Event)),
                                    training=lambda x:calls.append(('training_args',x.checkpoint_iterations,x.save_iterations)))
        env['bind'](train,cfg,{})
        train.training(args)
        assert args.checkpoint_iterations==args.save_iterations==schedule
        assert json.loads((Path(td)/'effective_config.json').read_text())['checkpoint_iterations']==schedule
        assert cfg['test_iterations']==[10000,20000,30000] and cfg['paper_eval_train_each_checkpoint'] is True
        event=train.torch.cuda.Event();assert event.elapsed_time(event)==1.
        assert calls[-2:]==['end.synchronize','elapsed_time']
        models={k:types.SimpleNamespace(get_xyz=[1,2,3]) for k in ['static','dynamic']}
        for step in schedule:checkpoint.save_components(td,step,models)
        forecasts=[v[1][1] for v in calls if isinstance(v,tuple) and v[0]=='forecast']
        assert forecasts==[6,6,5,4,3,2,1]
    receipt=dict(status='PASSED_ACTUAL_WRAPPER_CPU_DURABILITY_CADENCE',checkpoint_iterations=schedule,
        save_iterations=schedule,test_iterations=[10000,20000,30000],pretraining_disk_admission_nodes=forecasts[0],
        remaining_disk_forecast_nodes=forecasts[1:],
        event_order=['end.synchronize','elapsed_time'],actual_native_evaluation_nodes_due_to_checkpoint_policy=schedule,
        training_updates=0,GPU_training_verified=False,
        wrapper_sha256=hashlib.sha256(code.encode()).hexdigest())
    (E/'native_cadence_CPU_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))
if __name__=='__main__':main()
