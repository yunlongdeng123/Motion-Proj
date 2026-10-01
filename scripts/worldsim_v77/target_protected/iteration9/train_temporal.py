from pathlib import Path
import sys,argparse
P=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected');sys.path.insert(0,str(P/'iteration7'));import train_control as trainer
from temporal_factory import T,read,dump
O=T/'r10'
def main():
 assert read(O/'admission_result.json')['ready'];r7=read(T/'r7/encoder_fixed_lowres/training/config.json');original=trainer.save_json
 def save(path,obj):
  if path.name.startswith('validation_'):obj['scope']='fixed teacher-noised latent loss, mixed old held worlds and limited new temporal worlds; not sampled quality'
  original(path,obj)
 trainer.save_json=save;args=argparse.Namespace(root=O,steps=160,size=[320,576],modules='spatial',encoder=T/'r7/encoder_recovery/official_svd_encoder.safetensors');trainer.main(args)
 cfg=read(O/'training/config.json');keys=['trainable_tensors','lr','weight_decay','seed','steps_requested','optimizer','loss','dtype','training_resolution','module_range'];assert all(cfg[k]==r7[k] for k in keys)
 dump(O/'training/same_recipe_as_r7.json',{'all_fixed_recipe_fields_equal':True,'architecture_unchanged':True,'only_training_data_changed':True,'steps':160,'fields_checked':keys})
if __name__=='__main__':main()
