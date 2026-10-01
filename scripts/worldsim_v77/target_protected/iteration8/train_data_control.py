"""复用r7有效训练器，配方相同，只换冻结数据catalog。"""
from pathlib import Path
import sys,argparse,json
P=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected')
sys.path.insert(0,str(P/'iteration7'))
import train_control as trainer
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
O=T/'r8'
def main():
    catalog=json.loads((O/'dataset_catalog.json').read_text());s=catalog['summary']
    assert s['training_scene_count']>=20 and s['training_cases']<=50 and s['max_training_cases_per_scene']<=3
    assert all(s['training_type_counts'].get(k,0)>0 for k in ['background','single_actor','dense_actors'])
    assert all(c['assistant_score']==2 and c['technical_pass'] for c in catalog['cases'])
    r7=json.loads((T/'r7/encoder_fixed_lowres/training/config.json').read_text())
    original_save=trainer.save_json
    def save(path,obj):
        if path.name.startswith('validation_'):obj['scope']='independent receiver scenes, fixed latent loss; no sampled quality claim'
        original_save(path,obj)
    trainer.save_json=save
    args=argparse.Namespace(root=O,steps=160,size=[320,576],modules='spatial',encoder=T/'r7/encoder_recovery/official_svd_encoder.safetensors')
    assert r7['steps_requested']==args.steps and r7['training_resolution']==args.size and r7['module_range']==args.modules
    trainer.main(args)
    cfg=json.loads((O/'training/config.json').read_text())
    for key in ['trainable_tensors','lr','weight_decay','seed','steps_requested','optimizer','loss','dtype','training_resolution','module_range']:
        assert cfg[key]==r7[key],key
    original_save(O/'training/same_recipe_as_r7.json',{'all_fixed_recipe_fields_equal':True,'only_training_data_changed':True,'base_not_r7_initialization':True,'steps':160,'architecture_unchanged':True})
if __name__=='__main__':main()
