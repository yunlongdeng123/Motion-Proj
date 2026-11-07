from pathlib import Path
import ast,hashlib,json,os,subprocess,sys,time
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_road_fusion_integration_20261107_r1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
q=r/'support_qualification4_v2';receipt=json.loads((q/'receipt.json').read_text());manifest=q/'fixed_support_manifest.json'
assert receipt['joint_forwards']==8 and receipt['dummy_feature_backwards']==8 and receipt['safe_rows']>0
assert sha(manifest)==receipt['manifest_sha256']=='61127e8510bae5ed3228a5ed66497c3ae142b1e87851a9295d432c00941923b4'
fixed=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82/drivingeditor_dynamic_edit/fixed_road_support.py')
assert sha(fixed)=='b218036582cb61e9be1484b02a8c71fbec9f131f07da8a2b4ece88014d208098'
cp=r/'contract/D4_v2_execution_contract.json';assert not cp.exists()
v={'status':'SEALED_BEFORE_CANDIDATE_GPU','pilot_raws':[94,105,128,137],'states':['keep','delete'],'variant':'D_fixed_BUILD_support_v2',
 'main_baseline':str(r/'baseline4'),'C_reference':str(r/'C4_retry1'),'fixed_support_manifest':str(manifest),'fixed_support_manifest_sha256':sha(manifest),
 'safe_rows':receipt['safe_rows'],'asset_geometry_colour_opacity_parameters_unchanged':True,'only_temporary_newroad_mask_changes':True,
 'method_contract':str(r/'contract/fixed_build_support_amendment_v2.json'),'method_contract_sha256':sha(r/'contract/fixed_build_support_amendment_v2.json'),
 'independent_evaluation_contract_sha256':sha(r/'contract/independent_evaluation_contract_v1.json'),'fixed_support_source_sha256':sha(fixed),
 'full_support_occlusion_qualification_is_not_a_protection_guarantee':True,'planned_joint_forwards':8,'optimizer_steps':0,'model_updates':0,'colour_updates':0,
 'all49_auto_release':False,'shutdown':False,'reset_card':False,'created_unix':time.time()}
cp.write_text(json.dumps(v,indent=2)+'\n')
s=(r/'source/render_road_pilot4_retry1.py').read_text()
changes={
 'contract/pilot4_execution_contract.json':'contract/D4_v2_execution_contract.json',
 ' from scripts.worldsim_v82.drivingeditor_dynamic_edit.road_fusion_integration import RoadFusionV44Integration,AA_MANIFEST,C_MANIFEST':
 ' from scripts.worldsim_v82.drivingeditor_dynamic_edit.road_fusion_integration import RoadFusionV44Integration,AA_MANIFEST,C_MANIFEST\n from scripts.worldsim_v82.drivingeditor_dynamic_edit.fixed_road_support import FixedRoadSupportV44Integration',
 " core=rt;rt=RoadFusionV44Integration(core,AA_MANIFEST if variant=='AA' else C_MANIFEST)":
 " core=rt;rt=RoadFusionV44Integration(core,C_MANIFEST)\n rt=FixedRoadSupportV44Integration(rt,ROOT/'support_qualification4_v2/fixed_support_manifest.json')\n with np.load(ROOT/'support_qualification4_v2/safe_road_children.npz',allow_pickle=False) as z:assert np.array_equal(rt.fixed_support_child_row_ids,z['safe_child_row_ids'])",
 "outdir=ROOT/(variant+'4_retry1')":"outdir=ROOT/'D4_v2'",
 'variant=variant,runtime_receipt=':"variant='D_fixed_BUILD_support_v2',runtime_receipt=",
}
for before,after in changes.items():assert s.count(before)==1,(before,s.count(before));s=s.replace(before,after)
ast.parse(s);worker=r/'source/render_fixed_road_D4_v2.py';assert not worker.exists();worker.write_text(s)
env=os.environ.copy()
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[key]='1'
cmd=[sys.executable,'-B',str(worker),'C']
with (r/'D4_v2.log').open('x') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=r,env=env,start_new_session=True)
a={'status':'STARTED_D4_V2_CONDITIONAL_RELEASE_SAFE_NONEMPTY','pid':p.pid,'command':cmd,'source_sha256':sha(worker),'contract_sha256':sha(cp),
 'manifest_sha256':sha(manifest),'safe_rows':receipt['safe_rows'],'planned_joint_forwards':8,'actual_prior_forwards':39,'actual_prior_dummy_backwards':8,
 'optimization_updates':0,'unix':time.time(),'all49_auto_release':False};(r/'contract/D4_v2_launch.json').write_text(json.dumps(a,indent=2)+'\n');print(json.dumps(a))
