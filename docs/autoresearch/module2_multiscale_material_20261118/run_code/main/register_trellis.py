from pathlib import Path
import json

P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
out=P/'trellis2/execution_contract.json'
if out.exists():raise FileExistsError('保持已登记协议')
c=dict(schema='module2_TRELLIS2_fixed_geometry_inference.v1',task_id='MODULE2-MULTISCALE-MATERIAL-20261118',
 role='one zero-shot fixed road geometry material prior, no training or new geometry',
 official_implementation='/root/autodl-tmp/trellis2_official',official_commit='75fbf018',
 python='/root/autodl-tmp/envs/trellis2-road/bin/python',
 weights='/root/autodl-tmp/models/TRELLIS.2-4B-texturing',
 aux='/root/autodl-tmp/models/dinov3-vitl16-pretrain-lvd1689m',
 input_manifest=str(P/'trellis2/input_r1/manifest.json'),
 input_qualification='CPU manifest must confirm original fixed supported world mesh, legally constructed reference86/110, role/valid/unknown masks; freeze before inference',
 mesh=dict(chart_keys_i=[-99,28],chart_keys_j=[-66,61],vertices=16384,surface_supported_all=True,
   world_bbox_min=[2315.821710,1365.543100,-20.889799],world_bbox_max=[2318.369066,1368.090160,-20.600384]),
 report_domain=dict(fullS_supported_by_patch_anchor=2177,fullS_denominator=3187,new909_supported_by_patch_anchor=909,
   final_Gaussian_domain='actual fixed footprint/source/K qualification still required; do not equate mesh membership with final contribution',
   exterior='same frozen A0 reference, no full-road TRELLIS claim'),
 inference=dict(seed=101,resolution=512,steps=12,texture_resolution=1024,preprocess_image=False),
 budget=dict(inference_calls=1,training_updates=0,automatic_new_seed_or_retry=False),
 geometry_policy='call official Trellis2TexturingPipeline on supplied mesh; compare supplied/output geometry; only baseColor used for GS',
 outputs='raw generated mesh/UV/baseColor + input/source records; subsequent same S_Sigma material lookup -> native GS road/joint',
 prohibited=['SD regeneration','TRELLIS finetuning','changed camera/mesh/alpha/scales','historical PNG substitution','diagnostic134/179/198 RGB as reference'],
 preserve='all original models, scenes, inputs, evidence and ledgers',
 GPU_lease='exclusive P/gpu_lease.json owner static_handoff; grant only after CPU input qualification',
 GPU_keep_on=True,production_promoted=False,human_verdict=None)
out.write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n')
print(str(out))
