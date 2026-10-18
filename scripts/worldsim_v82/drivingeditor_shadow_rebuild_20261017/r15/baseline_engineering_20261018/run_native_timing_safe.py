"""Native runner with safe CUDA timing and authorized5K durability cadence.

Keep prepared source and optimization mechanisms intact. Runtime checkpoint
cadence and its disk admission follow the user's explicit5K save request.
"""
from pathlib import Path
import sys,importlib.util
T=Path('/root/autodl-tmp/drivingeditor_repro/campaign/followup_scene_prep_20261013/tools/followup')
sys.path.insert(0,str(T));import runtime_binding
original_binding=runtime_binding.bind_training_contract
def bind(training,cfg,ready):
 original_binding(training,cfg,ready)
 # Explicit latest user authorization changes durability cadence only. The
 # prepared native objective, schedules and camera protocol stay unchanged.
 import json
 schedule=list(range(5000,30001,5000));cfg['checkpoint_iterations']=schedule;cfg['save_iterations']=schedule
 output=Path(cfg['model_path'])
 (output/'effective_config.json').write_text(json.dumps(cfg,indent=2))
 (output/'checkpoint_cadence_override.json').write_text(json.dumps(dict(source='explicit human request: 每隔5k保存一次',checkpoint_iterations=schedule,save_iterations=schedule,test_iterations=cfg['test_iterations'],training_mechanisms_changed=False),ensure_ascii=False,indent=2))
 original_training=training.training
 def train_with_cadence(args):
  args.checkpoint_iterations=schedule;args.save_iterations=schedule
  # Check all six nodes before consuming B's first training update; checking
  # only at5K could otherwise lose the entire unsaved initial segment.
  admission=require_disk_space(output,disk_forecast(cfg,remaining_nodes=len(schedule)))
  (output/'disk_admission_full5k_cadence.json').write_text(json.dumps(admission,indent=2))
  return original_training(args)
 training.training=train_with_cadence
 from utils import paper_checkpoint
 from contracts import disk_forecast,require_disk_space
 original_save=paper_checkpoint.save_components
 def save_with_full_cadence_forecast(path,iteration,models):
  forecast=disk_forecast(cfg,actual_counts=(len(models['static'].get_xyz),len(models['dynamic'].get_xyz)),remaining_nodes=sum(s>=iteration for s in schedule))
  require_disk_space(Path(path),forecast)
  return original_save(path,iteration,models)
 paper_checkpoint.save_components=save_with_full_cadence_forecast
 original=training.torch.cuda.Event
 class CompletionSafeEvent(original):
  def elapsed_time(self,end_event):
   end_event.synchronize()
   return super().elapsed_time(end_event)
 training.torch.cuda.Event=CompletionSafeEvent
runtime_binding.bind_training_contract=bind
spec=importlib.util.spec_from_file_location('native_followup_event_safe',T/'run_scene_gn11.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
if __name__=='__main__':runner.main()
