from pathlib import Path
import ast,subprocess,sys
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_full5_integration_20261107_r1')
s=(r/'source/prepare_launch_v44_semantic_actor5.py').read_text()
changes={
 '08bcf4ac5e58f59bb1a2f917bdfd49fd0609c8b56e1d344528811738fd2ab6ee':'c177c97bea2c73e896fa925d3ecaa8ed490bb5b810e352ed77b2af23c5610c44',
 'semantic_actor5_calibration4':'semantic_actor5_calibration49',
 "sidecar['raw_ids']==[94,105,130,137]":"sidecar['raw_ids']==list(range(89,138))",
 "calib['joint_forwards']==4 and calib['dummy_feature_backwards']==4":"calib['joint_forwards']==45 and calib['dummy_feature_backwards']==45 and calib['reused_raw_ids']==[94,105,130,137]",
 "{'static':160,'dynamic':56118}":"{'static':232,'dynamic':120722}",
 "'SEMANTIC_ACTOR5_FIXED_VISIBLE_MASS_SIDECAR_PILOT'":"'SEMANTIC_ACTOR5_ALL49_BUILD_FIT_FOUR_FRAME_PILOT'",
 "'source_dynamic_owner_rows':{'5':54686,'37':1432,'6':0,'18':0,'26':0},'static_rows':160,":"'source_dynamic_owner_rows':{row['source']:row['selected_rows'] for row in calib['source_matrix'] if row['source']!='static'},'static_rows':232,",
 "'source37_selected_rows_are_query_semantic5':True":"'source37_selected_rows_are_query_semantic5':True,'calibration_raw_ids':list(range(89,138))",
 'Derived from original BUILD actor5 masks on four fixed views; not held-out test validation':'Derived from all49 original BUILD actor5 masks; prior four measurements reused exactly, 45 added. Not held-out test validation',
 'execution_contract_semantic_actor5.json':'execution_contract_semantic_actor5_all49fit.json',
 'candidate_semantic_a5_4':'candidate_semantic_a5_all49fit_4',
 'len(rt.semantic_static_row_ids)==160 and len(rt.semantic_dynamic_row_ids)==56118':'len(rt.semantic_static_row_ids)==232 and len(rt.semantic_dynamic_row_ids)==120722',
 "n+56278":"n+120954",
 'render_v44_semantic_actor5_candidate4.py':'render_v44_semantic_actor5_all49fit_candidate4.py',
 'candidate_semantic_actor5_launch_receipt.json':'candidate_semantic_actor5_all49fit_launch_receipt.json',
 'STARTED_FIXED_SEMANTIC_ACTOR5_SIDECAR_PILOT':'STARTED_ALL49_BUILD_SEMANTIC_ACTOR5_FOUR_FRAME_PILOT',
}
for before,after in changes.items():
 assert before in s,before;s=s.replace(before,after)
ast.parse(s);dest=r/'source/launch_v44_semantic_all49fit4.py';assert not dest.exists();dest.write_text(s)
subprocess.run([sys.executable,'-B',str(dest)],check=True)
