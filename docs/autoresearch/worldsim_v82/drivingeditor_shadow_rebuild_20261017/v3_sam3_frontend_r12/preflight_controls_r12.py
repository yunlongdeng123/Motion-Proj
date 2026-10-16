from pathlib import Path
import subprocess,json,os
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');root=R/'ownership_controls_r12_sam3';py='/root/autodl-tmp/envs/drivingeditor/bin/python'
results=[]
for name in ['data_A_box','data_B_semantic_motion']:
 p=subprocess.run([py,str(R/'implementation_r12/shadow_rebuild_v3/run.py'),'preflight','--data',str(root/name),'--config',str(root/'common_config.json'),'--output',str(root/('preflight_'+name)),'--allow-partial-masks'],capture_output=True,text=True)
 (root/(name+'_preflight_stdout.txt')).write_text(p.stdout);(root/(name+'_preflight_stderr.txt')).write_text(p.stderr)
 print(name,p.returncode,p.stdout[-2200:],p.stderr[-500:]);p.check_returncode();results.append(dict(data=name,passed=True))
(root/'preflight_controls.json').write_text(json.dumps(dict(status='PASSED',results=results,CUDA_used=False,partial_mask_gap_explicit=True),indent=2))
