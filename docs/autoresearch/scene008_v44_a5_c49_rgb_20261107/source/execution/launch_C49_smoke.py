from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_c49_rgb_20261107_r1');OLD=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_road_fusion_integration_20261107_r1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
contract={'schema':'C49_PARENT_SHARED_RGB_BACKGROUND_CONTROL_TRAINING_V1','status':'SEALED_BEFORE_FIRST_RGB_OPTIMIZER','source_scene':'Delivered21K component v44+semantic_a5 single assembled runtime','road_asset_sha256':'3afc6e25b7b152f851b4eb0de4a73e6d219c9526d75453d99097a31ddd5774c5',
 'initializer':'existing C400 warm material, not zero-trained material','private_Adam_fresh_at_formal_step0':True,'eligible_parents':2119,'zero_mass_frozen_parents':7979,'shared_parent_rows_total':10098,'full_children_total':40392,
 'qualification_manifest_sha256':sha(R/'parent_visibility49/qualification_manifest.json'),'all49_current_BUILD':True,'sampling':'uniform shuffled49 raw89..137, private Python random seed20261107','seed':20261107,
 'loss':'.5*(KEEP mean squared RGB error + DELETE mean squared RGB error), same original Cedgeblend_rgb and legalfit_mask, no extra weights/crops','optimizer':{'name':'Adam','lr':.01,'betas':[.9,.999],'eps':1e-8},'RGB_bounds':[0,1],
 'smoke_training_seconds_min':600,'smoke_updates':'all actual updates separately recorded, never counted as formal','lifecycle':'Cwarm0 public4KD exact; real600seconds; early/full saves; publicevaluation; disposable2; fresh exactrestore+next2; same-seed fresh formal from Cwarm0',
 'formal_initial_stop':2000,'formal_maximum':10000,'formal_check_nodes':[500,2000,5000,10000],'released_initial_formal_nodes':[500,2000],
 'freeze':'all original scene/roof/identity/Gaussian/SH and Cgeometry/scales/rotations/opacity/Mip/masks fixed; only eligible shared-parentRGB optimized','node_geometric_reference':'unmodified Cwarm0, all global alpha/depth_sum/metricdepth/8groups/semanticRGB exact',
 'known_not_best_control':True,'original_C4_tire_and_integration_failures_remain':True,'automatic_quality_adoption':False,'automatic_H_or_49_release':False,'GPU_power':'KEEP_ON','shutdown':False,'reset_card':False,
 'material_runtime_sha256':'4d8932b39b8b0fd5034c1147068acf7a6b5474c62e556d2ebc79a04e319f1009','executed_core_sha256':sha(R/'source/C49_material_core.py')}
(R/'contract/C49_parent_RGB_training_v1.json').write_text(json.dumps(contract,indent=2)+'\n')
env=os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
cmd=['/root/autodl-tmp/envs/drivingeditor/bin/python','-B',str(R/'source/C49_material_core.py'),'--mode','smoke','--output',str(R/'smoke600s_r1')]
with (R/'smoke600s_r1.log').open('x') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=R,env=env,start_new_session=True,stdin=subprocess.DEVNULL)
record={'pid':p.pid,'command':cmd,'source_sha256':sha(R/'source/C49_material_core.py'),'contract_sha256':sha(R/'contract/C49_parent_RGB_training_v1.json'),'unix':time.time(),'real_training_seconds_minimum':600,'formal_started':False}
(R/'contract/smoke600s_r1_launch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
