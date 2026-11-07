"""Run only AA/C official four-frame joint renders, with one GPU process at a time."""
from pathlib import Path
import ast,hashlib,json,os,subprocess,sys,time
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_road_fusion_integration_20261107_r1')
MODULE=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82/drivingeditor_dynamic_edit/road_fusion_integration.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(MODULE)=='fd755aea8110c9b7ace6a7623db4de13ba8a379724bb928199f734e161457745'
metric=ROOT/'contract/independent_evaluation_contract_v1.json';assert sha(metric)=='4560bb91f4cc3b27b89693acd0918f939d2ac07e9bcd7cbbc6e0574e885516cc'
contract={'status':'SEALED_BEFORE_CANDIDATE_GPU','pilot_raws':[94,105,128,137],'states':['keep','delete'],
 'main_baseline':str(ROOT/'baseline4'),'baseline_exact_replay':str(ROOT/'contract/baseline4_exact_replay.json'),
 'variants':['AA','C'],'adapter_sha256':sha(MODULE),'independent_evaluation_contract':str(metric),'independent_evaluation_contract_sha256':sha(metric),
 'asset_pair_audit':str(ROOT/'contract/road_AA_C_asset_pair_audit.json'),'asset_pair_audit_sha256':sha(ROOT/'contract/road_AA_C_asset_pair_audit.json'),
 'physical_prefix_and_environment_byte_exact':True,'AA_C_same_geometry_scales_rotation_opacity_only_RGB_changes':True,
 'separate_diagnostic_raws':[130],'planned_candidate_joint_forwards':16,'optimization_updates':0,'all49_auto_release':False,
 'shutdown':False,'reset_card':False,'created_unix':time.time()}
cp=ROOT/'contract/pilot4_execution_contract.json';assert not cp.exists();cp.write_text(json.dumps(contract,indent=2)+'\n')
s=(ROOT/'source/render_latest_semantic_baseline4.py').read_text()
changes={
 'contract/baseline4_protocol.json':'contract/pilot4_execution_contract.json',
 'SEALED_BEFORE_BASELINE_GPU':'SEALED_BEFORE_CANDIDATE_GPU',
 ' defunusedzz':'notused',
}
changes.pop(' defunusedzz')
for before,after in changes.items():assert s.count(before)==1;s=s.replace(before,after)
needle='def main():\n'
assert s.count(needle)==1
s=s.replace(needle,needle+" variant=sys.argv[1];assert variant in ['AA','C']\n assert sha(Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82/drivingeditor_dynamic_edit/road_fusion_integration.py'))=='fd755aea8110c9b7ace6a7623db4de13ba8a379724bb928199f734e161457745'\n")
changes={
 ' from scripts.worldsim_v82.drivingeditor_dynamic_edit.v44_integration import SemanticActor5V44Integration':
 ' from scripts.worldsim_v82.drivingeditor_dynamic_edit.v44_integration import SemanticActor5V44Integration\n from scripts.worldsim_v82.drivingeditor_dynamic_edit.road_fusion_integration import RoadFusionV44Integration,AA_MANIFEST,C_MANIFEST',
 ' assert len(rt.semantic_static_row_ids)==232 and len(rt.semantic_dynamic_row_ids)==120722':
 ' assert len(rt.semantic_static_row_ids)==232 and len(rt.semantic_dynamic_row_ids)==120722\n core=rt;rt=RoadFusionV44Integration(core,AA_MANIFEST if variant==\'AA\' else C_MANIFEST)',
 "outdir=ROOT/'baseline4'":"outdir=ROOT/(variant+'4')",
 'data=rt.raw(raw)':'data=core.raw(raw)',
 'before=base.scene(raw,delete=delete);scene=rt.scene(raw,delete=delete)':'before=core.scene(raw,delete=delete);scene=rt.scene(raw,delete=delete)',
 "n+120954":"n+40392",
 'groups.shape==(7,320,480)':'groups.shape==(8,320,480)',
 "    added_rgb=output['added_owner_rgb_T'].detach().cpu().numpy()":"    added_rgb=output['added_owner_rgb_T'].detach().cpu().numpy();newroad_rgb=output['newroad_rgb_T'].detach().cpu().numpy();assert newroad_rgb.shape==(3,320,480)",
 'depth_sum/np.maximum(alpha,1e-8)/rt.scale':'depth_sum/np.maximum(alpha,1e-8)/core.scale',
 'group_alpha_T=groups,added_owner_rgb_T=added_rgb)':'group_alpha_T=groups,added_owner_rgb_T=added_rgb,newroad_rgb_T=newroad_rgb)',
 "groups=['asset','a37','road','native_static','a5','a18','a26'],":"groups=['asset','a37','road','native_static','a5','a18','a26','newroad'],",
 'RENDERED_LATEST_SEMANTIC_BASELINE4_PENDING_CACHED_EXACT_REPLAY':'RENDERED_ROAD_AA_OR_C_FOUR_FRAME_PILOT_PENDING_STRICT_INDEPENDENT_EVALUATION',
 'runtime_receipt=rt.receipt(),seconds=':'variant=variant,runtime_receipt=rt.receipt(),seconds=',
}
for before,after in changes.items():assert s.count(before)==1,(before,s.count(before));s=s.replace(before,after)
ast.parse(s);worker=ROOT/'source/render_road_pilot4.py';assert not worker.exists();worker.write_text(s)
env=os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
statuses=[]
for variant in ['AA','C']:
 cmd=[sys.executable,'-B',str(worker),variant]
 with (ROOT/(variant+'4.log')).open('x') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,env=env,start_new_session=True)
 start={'variant':variant,'pid':p.pid,'command':cmd,'contract_sha256':sha(cp),'worker_sha256':sha(worker),'unix':time.time(),'optimization_updates':0}
 (ROOT/'contract'/('launch_'+variant+'4.json')).write_text(json.dumps(start,indent=2)+'\n');print(json.dumps(start),flush=True)
 code=p.wait();statuses.append(dict(start,exit_code=code))
 (ROOT/'pilot_pair_status.json').write_text(json.dumps({'completed':statuses,'all49_auto_release':False,'optimization_updates':0},indent=2)+'\n')
 if code!=0:raise SystemExit(code)
print(json.dumps({'status':'COMPLETE_OFFICIAL_FOUR_FRAME_AA_C_RENDER_ONLY','actual_completed_variants':['AA','C'],'optimization_updates':0}),flush=True)
