from pathlib import Path
import json,shutil,datetime
W=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-12/w');R=W/'work/drivingeditor_resume/repo_active';B=W/'work/drivingeditor_resume/unified_viewer_20261013'
rel='docs/autoresearch/worldsim_v82/drivingeditor_followup_20261013/r1/unified_viewer_20261013';A=R/rel;A.mkdir(parents=True,exist_ok=True)
for n in ['build_receipt.json','validation.json','unified_viewer.js','four_arm_updated.js']:shutil.copy2(B/n,A/n)
for n in ['build_unified_scene_viewer.py','verify_package_unified_viewer.py','save_unified_viewer_revision.py']:shutil.copy2(W/'work/drivingeditor_resume'/n,A/n)
shutil.copy2(W/'outputs/DrivingEditor_offline_package_receipt.json',A/'offline_package_receipt.json')
four=json.loads((W/'outputs/drivingeditor_four_arm_30k/manifest.json').read_text(encoding='utf-8-sig'))
note={'status':'READONLY_SILVER_STATIC_DIAGNOSIS_WITH_ACTUAL_PIXEL_CHECKS','reviewer':'/root/silver_static_explain',
 'actual_reviewed_images':['assets/C00/static_time/010_static.png','assets/GN11/static_time/010_static.png','assets/C00/static_time/028_static.png','assets/GN11/static_time/028_static.png','assets/C00/static_time/028_alpha_s.png','assets/GN11/static_time/028_alpha_s.png','assets/GN11/observed/010_gt.png','assets/GN11/observed/010_body_mask.png'],
 'root_additionally_viewed':'assets/GN11/observed/028_static.png','image_root':'outputs/drivingeditor_four_arm_30k/',
 'fixed_pose':'0017010','target_frame':'0017028','target_model_time':0.08,
 'target_static_RGB_MAE':{a:four['arms'][a]['static_time_frames'][28]['metrics']['static_rgb_MAE_from_anchor'] for a in ['C00','GN11']},
 'observations':'Both arms have moving road/underbody residues. Road alpha remains near opaque in the reviewed target; cannot call GN11 a new whole-static collapse.',
 'code_findings':'Static SHM position and temporal opacity remain time-dependent; lifetime/velocity penalties are soft. NPS compares local features within the current rendered image, not RGB across timestamps.',
 'causal_inference_not_verified':'Body-only masks omit vehicle shadows; outside dynamic-opacity supervision and static-RGB fitting can direct shadows toward the static layer. No specific Gaussian attribution was established.',
 'human_GN11_verdict':None,'new_training_steps':0,'remote_connected':False}
(A/'silver_static_diagnosis.json').write_text(json.dumps(note,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
p=R/'docs/RESEARCH_STATUS.md';t=p.read_text(encoding='utf-8');old='本聊天`outputs/DrivingEditor_final_report.html`汇总并链接三个真实深色viewer；';new='本聊天后续统一使用`outputs/DrivingEditor_final_report.html`，三场景在同一页切换，默认真实49帧相机/时间的静态背景，合成/动态/透明度可切换；固定相机时间诊断单独为次级模式。银车明确四臂，黑车/红车仅GN11；旧四臂入口也默认observed static，旧页与资产保留。'
assert old in t;t=t.replace(old,new,1);p.write_text(t,encoding='utf-8')
p=R/'docs/v82/DRIVINGEDITOR_FOUR_ARM_30K_RESULTS.md';t=p.read_text(encoding='utf-8');anchor='## NPS与SGCM：有效性及证据边界'
paragraph='''用户截图回看（2026-10-13）：固定pose0017010、local29/raw28/t=0.08，两臂中央道路残影均从anchor位置移到左侧远端横道，近景白条顶端均有黑块；C00硬亮边更明显，GN11较软。该时刻RGB MAE分别0.0263978243/0.0243896767，变化略减不等于补景正确；锚点MAE为0仅因为自比。已查看target两臂raw alpha_s，路面主体近白，未建立对应大块alpha孔洞。实际逐帧静态图也独立保存于`assets/{arm}/observed/{index}_static.png`，与固定相机`static_time`分开。

源码机制核对：静态renderer仍使用`get_xyz_SHM(t)`以及时间marginal opacity，寿命和速度仅软正则，不是硬冻结；速度惩罚输入还有寿命衰减，位置变化仍来自raw velocity。NPS是当前渲染的局部静态速度特征L2 sum，没有比较t1/t2的RGB。body mask缺投影阴影，在mask外动态opacity强压0、静态RGB仍拟合含阴影GT的共同机制下，推断影子容易留入静态层；具体残影的逐高斯归属尚未验证。通过相对C00的工程门槛，不等于绝对静态与补景合格。[回看与统一入口修订证据](../autoresearch/worldsim_v82/drivingeditor_followup_20261013/r1/unified_viewer_20261013/)。

'''
assert paragraph not in t;t=t.replace(anchor,paragraph+anchor,1);p.write_text(t,encoding='utf-8')
p=R/'docs/v82/DRIVINGEDITOR_FOLLOWUP_20261013.json';d=json.loads(p.read_text(encoding='utf-8'))
d['delivery']['canonical_viewer']='outputs/DrivingEditor_final_report.html';d['delivery']['viewer_camera_mode_default']='observed';d['delivery']['viewer_branch_default']='static';d['delivery']['fixed_camera_is_secondary_diagnostic']=True
d['delivery']['viewer_revision_evidence']=rel;d['delivery']['viewer_revision_new_training_steps']=0
d['delivery']['viewer_revision_browser_interaction_verified']=False
d['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'evidence':rel,'new_training_steps':0,'canonical_viewer':'outputs/DrivingEditor_final_report.html'}))
