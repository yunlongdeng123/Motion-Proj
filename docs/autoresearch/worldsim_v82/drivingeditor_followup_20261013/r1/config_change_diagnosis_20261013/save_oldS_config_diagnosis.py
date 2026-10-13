"""Record a read-only historical comparison; no training, rendering or GPU imports."""
from pathlib import Path
import json,shutil,datetime
W=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-12/w');Z=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-11/z');O=W/'outputs';R=W/'work/drivingeditor_resume/repo_active'
A=R/'docs/autoresearch/worldsim_v82/drivingeditor_followup_20261013/r1/config_change_diagnosis_20261013';A.mkdir(parents=True,exist_ok=True)
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
oldpath=Z/'work/vgpu_bugfix_20261012/continuation_diagnostic/observed_static_rgb30k_to32000/extra_eval/32000/diagnostic_eval.json'
old=load(oldpath)['frames'][20];newpath=O/'drivingeditor_C00_30k/manifest.json';new=load(newpath)['frames'][20]
om=old['regions']['non_sky_outside_actor']['metrics'];nm=new['metrics']['non_sky_outside_actor']
pixels=load(W/'work/old_s_pixel_compare/raw20_pixel_diagnostic.json')
assert old['frame_id'].endswith('0017020') and new['id'].endswith('0017020') and pixels['body_masks']['exact_equal']
assert old['regions']['non_sky_outside_actor']['pixel_count']==nm['pixel_count']
summary={'status':'READONLY_OLD_S_VS_NEW_C00_CONFIG_AND_PIXEL_DIAGNOSIS','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'user_observation':'Old S static road lacks the conspicuous car shadow seen in the new configuration; dynamic still has background problems.',
 'paired_observed_frame':'image_0/0017020','model_timestamp':new['timestamp'],'GT_geometry_aligned':True,'GT_export_max_byte_difference':1,
 'body_mask_exactly_equal':True,'body_pixels':pixels['body_masks']['pixel_count'],'new_training_steps':0,'remote_connected':False,
 'saved_FP32_region_metrics':{'source_old':str(oldpath),'source_new':str(newpath),
    'old_S':{'static_psnr_db':om['static']['clamped_rgb_psnr_db'],'static_mse':om['static']['clamped_rgb_mse_vs_gt'],'dynamic_alpha_mean':om['alpha_d']['mean']},
    'new_C00':{'static_psnr_db':nm['static_psnr_db'],'static_mse':nm['static_mse'],'dynamic_alpha_mean':nm['alpha_d_mean']}},
 'configuration_changes':{
   'start':{'old_S':'same legacy30k model/Adam/RNG/sampler, then2000 updates to32000','new_C00':'from scratch30000'},
   'LR_horizon':{'old_S':50000,'new_C00':30000},
   'static_RGB':{'both':'weight1, (1-Md) applied to prediction and GT, full-image .8L1+.2DSSIM'},
   'dynamic_opacity':{'old_S':'foreground-only logarithmic term .05, no explicit outside-zero term','new_C00':'100*mean((raw_alpha_d-Md)^2), both domains'},
   'dynamic_RGB_weight':{'old_S':1,'new_C00':5},'sky_RGB_weight':{'old_S':'not active','new_C00':5},
   'absolute_static_duration_velocity':{'old_S':'not called in active paper branch despite nonzero config keys','new_C00':'active .1/.01 soft regularizers'},
   'depth':{'old_S':'Eq16 weight.1 without decay','new_C00':'same Eq16 times exp(-iteration/8000)'},
   'extra_smoothness_and_combined_alpha_entropy':{'old_S':'not active','new_C00':'restored'},
   'SGCM_NPS':{'old_S':'legacy SGCM topology inherited, NPS active; no SGCM calls during30001-32000','new_C00':'both off'},
   'terminal_static_dynamic_points':{'old_S':[2234298,1026885],'new_C00':[986061,1028966]},
   'static_opacity_reset':{'old_S':'legacy every3000; recovered from30k reset then2k with no new reset','new_C00':'only before15000'}},
 'observed_layer_allocation':'Old S dynamic covers roads/shadow regions; new C00 outside-body dynamic is near transparent while static matches the observed dark contact/shadow pixels.',
 'mechanistic_interpretation':'An unchanged body-only mask omits shadow. Complete outside-alpha supervision plus changed RGB/regularization/trajectory constrains dynamic ownership, while static RGB still targets observed shadow outside Md. This explains the direction of layer reassignment; the individual changed term has not been isolated.',
 'body_inside_boundary':'Static RGB excludes body pixels; the underlying static appearance there has no explicit static-RGB supervision. Do not call every dark contact pixel an outside-mask shadow or alpha hole.',
 'causal_identification':False,'edited_GT_available':False,'human_GN11_verdict':None,'next_training_started':False}
save(O/'DrivingEditor_oldS_configuration_diagnosis.json',summary);save(A/'summary.json',summary)
shutil.copy2(W/'work/old_s_pixel_compare/raw20_pixel_diagnostic.json',A/'raw20_PNG_ROI_diagnostic.json')
shutil.copy2(W/'work/old_s_pixel_compare/raw20_png_diagnostic.py',A/'raw20_png_diagnostic.py')
shutil.copy2(W/'work/old_s_pixel_compare/raw20_undercar_native.png',O/'DrivingEditor_old_and_new_raw20.png')
shutil.copy2(R/'scripts/worldsim_v82/drivingeditor_debug/continuation_diagnostic/observed_static_rgb30k_to32000/effective_config.yaml',A/'old_S_effective_config.yaml')
shutil.copy2(R/'compatibility/DrivingEditor-v82/frozen_baseline_to_S.patch',A/'old_S_actual_delta.patch')
shutil.copy2(Path(__file__),A/'save_oldS_config_diagnosis.py')
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
fig,ax=plt.subplots(figsize=(10,4.4));fig.patch.set_facecolor('#0b1421');ax.set_facecolor('#0b1421');ax.set_xlim(0,10);ax.set_ylim(0,4.4);ax.axis('off')
def box(x,y,w,h,text,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06,rounding_size=.10',facecolor=color,edgecolor='#54718d',linewidth=1));ax.text(x+w/2,y+h/2,text,ha='center',va='center',color='#e5eef8',fontsize=11,linespacing=1.6)
def arrow(x1,y1,x2,y2):ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops={'arrowstyle':'->','color':'#8ac5d8','lw':1.6})
box(2.3,3.5,5.4,.65,'相同49帧RGB与body mask\nmask没有完整车影标签','#1d3046')
box(.3,1.95,4.25,1.05,'旧S：前景opacity .05，无车外项\n静态RGB 1；旧30k续训2k','#213845')
box(5.45,1.95,4.25,1.05,'新C00：完整alpha MSE 100\n静态RGB 1＋恢复其他目标；从零30k','#263c35')
arrow(4.1,3.45,2.4,3.03);arrow(5.9,3.45,7.55,3.03)
box(.3,.65,4.25,.82,'观察：动态仍覆盖道路／车影\n静态视觉较干净，但背景泄漏','#213845')
box(5.45,.65,4.25,.82,'观察：车外动态覆盖接近零\n静态层贴近GT中的暗车影','#263c35')
arrow(2.4,1.88,2.4,1.49);arrow(7.55,1.88,7.55,1.49)
ax.text(5,.15,'这是符合证据的分配机制解释；多项同时变更，未建立单项因果归因。',ha='center',color='#a9bed5',fontsize=10)
fig.savefig(O/'DrivingEditor_config_allocation_components.png',dpi=160,bbox_inches='tight',facecolor=fig.get_facecolor());fig.savefig(O/'DrivingEditor_config_allocation_components.svg',bbox_inches='tight',facecolor=fig.get_facecolor());plt.close(fig)
print(json.dumps({'status':summary['status'],'FP32_metrics':summary['saved_FP32_region_metrics'],'new_training_steps':0},ensure_ascii=True))
