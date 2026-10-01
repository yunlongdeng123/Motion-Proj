"""同数据/同参数数量控制，空间self attention换为时间self attention。"""
from pathlib import Path
import sys,argparse,json,math
P=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected')
sys.path.insert(0,str(P/'iteration7'));sys.path.insert(0,str(P/'iteration9'))
import train_control as trainer
from temporal_factory import T,read,dump
O=T/'r14'
def selected(k):
 return k.startswith('model.diffusion_model.') and '_3d' not in k and '.time_stack.' in k and '.attn1.' in k and any(t in k for t in ['.to_q.','.to_k.','.to_v.','.to_out.'])
def main():
 assert read(O/'admission_result.json')['ready'] and read(O/'run.json')['operator']=='temporal_self_attention_only'
 trainer.selected=selected;trainer.main(argparse.Namespace(root=O,steps=160,size=[320,576],modules='temporal_self',encoder=T/'r7/encoder_recovery/official_svd_encoder.safetensors'))
 cfg=read(O/'training/config.json');old=read(T/'r10/training/config.json');fields=['lr','weight_decay','seed','steps_requested','optimizer','loss','dtype','training_resolution'];assert all(cfg[k]==old[k] for k in fields)
 assert len(cfg['trainable_tensors'])==80 and cfg['trainable_parameters']==old['trainable_parameters']==49574080 and all(selected(k) for k in cfg['trainable_tensors'])
 new_steps=[json.loads(s) for s in (O/'training/steps.jsonl').read_text().splitlines()]
 old_steps=[json.loads(s) for s in (T/'r10/training/steps.jsonl').read_text().splitlines()]
 assert len(new_steps)==len(old_steps)==160
 assert all((n['last_case'],n['last_window'])==(o['last_case'],o['last_window']) for n,o in zip(new_steps,old_steps))
 assert all(math.isfinite(n['loss']) and math.isfinite(n['grad_norm']) for n in new_steps)
 for tag in ['base','finetuned']:
  vp=O/'training'/f'validation_{tag}.json';v=read(vp);v['scope']='same r10 frozen validation cases; fixed teacher-noised latent loss, not sampled image quality';dump(vp,v)
 dump(O/'training/scope_control.json',{'equal_tensor_count':80,'equal_parameter_count':49574080,'same_recipe_fields':fields,'same_values_verified':True,'same_dataset_as_r10':True,'same_160_case_window_order_verified':True,'all_loss_and_gradient_norm_finite':True,'architecture_unchanged':True,'from_original_not_r10_continuation':True,'only_parameter_location_changed':True,'human_verdict':None})
if __name__=='__main__':main()
