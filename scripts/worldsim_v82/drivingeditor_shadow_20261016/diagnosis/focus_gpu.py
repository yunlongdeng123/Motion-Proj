import json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

base=Path(__file__).parent.parent
plan=json.loads((Path(__file__).parent/'experiment_plan.json').read_text())
root=Path(plan['remote_root']); case=plan['primary_case']
cpu=json.loads((base/'focus_cpu/focus_attribution.json').read_text())
spec=json.loads((root/case['spec_relative']).read_text())
sys.path.insert(0,str(root/'infra'))
from priority import require_gpu_slot
from runtime import load_models
from renderer import render
lock,admission=require_gpu_slot(plan['gpu_lock'])
try:
    checkpoint=root/case['checkpoint_relative']; before=(checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns)
    out=base/'focus_gpu'; out.mkdir(exist_ok=False)
    restored=torch.load(checkpoint,map_location='cuda',weights_only=False)
    config,scene,static,instances,optimizer,env,exclusion=load_models(spec,out,restore=restored)
    del restored
    assert exclusion['removed_points']==0
    camera=list(scene.getTrainCameras())[12]
    assert Path(camera.image_name).stem==case['frame_id']
    bg=static.get_xyz.new_zeros(3)
    names=list(cpu['group_ids'])
    payload=static.get_xyz.new_zeros((len(static.get_xyz),len(names)))
    for i,name in enumerate(names):
        payload[torch.tensor(cpu['group_ids'][name],device=payload.device,dtype=torch.long),i]=1
    def call(mask=None,other=None):
        return render(camera,static,None,config,bg,env_map=env,mask=mask,other=other,render_separate=False,max_features=10,is_training=False)
    def save(name,pkg):
        data={k:pkg[v].cpu().numpy() for k,v in [('RGB','render'),('raw_RGB','raw_render'),('alpha','alpha')]}
        np.savez_compressed(out/(name+'.npz'),**data)
        Image.fromarray(np.rint(np.moveaxis(data['RGB'].clip(0,1),0,-1)*255).astype(np.uint8)).save(out/(name+'.png'))
    footprint=np.asarray(Image.open(base/'focus_cpu/reviewed_artifact_footprint.png'))>0
    with torch.no_grad():
        baseline=call()
        with np.load(base/'gpu_diagnostics/target_original_Static.npz') as f:
            assert np.array_equal(f['raw_RGB'],baseline['raw_render'].cpu().numpy())
        pkg=call(other=payload)
        assert torch.equal(pkg['raw_render'],baseline['raw_render'])
        features=pkg['feature'].cpu().numpy()
        alpha=pkg['alpha'][0].cpu().numpy()
        assert np.max(abs(features[0]-alpha))<1e-6
        np.savez_compressed(out/'group_exact_contributions.npz',group_features=features,names=np.asarray(names),footprint=footprint)
        groups={name:{'mean_alpha_T':float(features[i][footprint].mean()),'share_of_total_alpha_T':float(features[i][footprint].sum()/alpha[footprint].sum()),'probes':[float(features[i,y,x]) for x,y in case['probe_pixels_xy']]} for i,name in enumerate(names)}
        for name in ['near_black_SH_max_lt_005','near_black_center_abs_height_gt_050m','near_black_center_abs_height_le_015m']:
            keep=torch.ones(len(static.get_xyz),device=payload.device,dtype=torch.bool)
            keep[torch.tensor(cpu['group_ids'][name],device=keep.device,dtype=torch.long)]=False
            save('hide_'+name+'_NOT_A_REPAIR',call(mask=keep))
        assert (checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns)==before
    result={'GPU_baseline_bitwise_match':True,'GPU_feature_pass_preserves_RGB':True,'all_group_alpha_matches_baseline':True,'groups':groups,'training_updates':0,'checkpoint_written':False,'temporary_group_hiding_is_not_a_repair':True,'footprint_pixels':int(footprint.sum()),'group_criteria_overlap':True,'admission':admission}
    (out/'focus_GPU_receipt.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
finally:
    lock.close()
