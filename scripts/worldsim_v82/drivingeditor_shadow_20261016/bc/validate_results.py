"""Validate completed experiment evidence separately from quality-gate success."""
import argparse,datetime,hashlib,json,math,re,subprocess
from pathlib import Path
from PIL import Image
HERE=Path(__file__).resolve().parent
OUT=HERE.parent if HERE.name=='scripts' else HERE.parents[1]/'outputs/shadow_bc_experiments'
p=argparse.ArgumentParser();p.add_argument('--visual-reviewed',action='store_true');a=p.parse_args()
read=lambda path:json.loads((OUT/path).read_text(encoding='utf-8'))
command='PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/envs/drivingeditor/bin/python /root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_BC_20261016_v1/implementation_v1/progress_probe.py'
r=subprocess.run(['ssh','wm-vgpu-1011',command],capture_output=True,text=True,encoding='utf-8',timeout=30,check=True)
probe=json.loads(r.stdout);(OUT/'evidence/final_process_check.json').write_text(json.dumps(probe,ensure_ascii=False,indent=2),encoding='utf-8')
required={'pilot_1000_v1','plane_1000_v1','deferred_1000_v1','union_1000_v1'}
checked=[row for row in probe['runs'] if row['run'] in required]
checks={'all_owned_experiments_complete_and_exited':len(checked)==4 and all(row['state']=='complete' and not row['owned_process_alive'] for row in checked)}
statuses=[read('evidence/status.json')]+[read(prefix+'/status.json') for prefix in ['plane','deferred','union']]
checks['parent_checkpoint_and_source_parameters_preserved']=all(s.get('parent_checkpoint_unchanged') and s.get('source_parameters_unchanged') for s in statuses)
summary=read('measured_summary.json');methods=summary['methods']
checks['all_measured_comparisons_present']=set(methods)=={'R0','R1','R2','R3','R4','R5','plane_R4','deferred_R4','union_R4'}
checks['all_addon_restore_images_match']=all(m.get('addon_restore_RGB_bitwise',True) for m in methods.values())
checks['delete_all_shadow_identity']=all(m['delete_all_shadow_equals_unshaded_static_max_abs']<2e-6 for m in methods.values())
checks['actual_saved_checkpoints_and_Adam_loaded']=all(read(prefix)['actual_saved_file_loaded'] and read(prefix)['Adam_state_finite_and_loaded'] for prefix in [
    'results/R1/checkpoint_structure_validation.json','results/R2/checkpoint_structure_validation.json','results/R3/checkpoint_structure_validation.json','results/R4/checkpoint_structure_validation.json','results/R5/checkpoint_structure_validation.json',
    'plane/checkpoint_validation.json','deferred/checkpoint_validation.json','union/checkpoint_validation.json'])
checks['GPU_gradients_finite']=all(all(g['finite'] for g in read(path)['gradients'].values()) for path in ['evidence/GPU_gradient_smoke.json','plane/GPU_gradient_smoke.json','deferred/GPU_gradient_smoke.json','union/GPU_gradient_smoke.json'])
def finite(value):
    if isinstance(value,float):return math.isfinite(value)
    if isinstance(value,dict):return all(finite(x) for x in value.values())
    if isinstance(value,list):return all(finite(x) for x in value)
    return True
checks['all_JSON_finite']=all(finite(json.loads(path.read_text(encoding='utf-8'))) for path in OUT.rglob('*.json'))
count=0
for path in OUT.rglob('*.png'):
    with Image.open(path) as im:im.verify()
    count+=1
checks['all_PNG_decode']=True
report=OUT/'report.html';text=report.read_text(encoding='utf-8')
refs=[x for x in re.findall(r'(?:src|href)="([^"]+)"',text) if not x.startswith(('http:','https:','#'))]
checks['all_HTML_local_references_resolve']=all((OUT/x).is_file() for x in refs)
checks['representative_montages_and_architecture_viewed']=a.visual_reviewed
review=read('evidence/visual_review.json')
checks['selected_variant_not_visually_rejected']=summary['suggested_variant_by_clean_and_shadow_gate'] not in review['rejected_variants']
assert all(checks.values()),checks
best=summary['suggested_variant_by_clean_and_shadow_gate'];m=methods[best];base=methods['R0']
gates={'clean_road_reduction_at_least_50pct':m['diagnostic_footprint_clean_linear_MAE']<=base['diagnostic_footprint_clean_linear_MAE']*.5,
       'control_change_at_most_001':m['outside_control_sRGB_MAE_vs_parent']<=.01,
       'shadow_improves_contact':m['observed_shadow_linear_MAE']<methods['R3']['observed_shadow_linear_MAE'],
       'strict_fixed_camera_time_MAE_at_most_1e_5':m['fixed_camera_mean_nonanchor_time_MAE']<=1e-5,
       'no_new_severe_localized_artifacts_in_reviewed_images':best not in review['rejected_variants']}
result={'evidence_status':'PASS','evidence_checks':checks,'representative_images_reviewed_not_all_rendered_views':True,
       'PNG_count':count,'HTML_reference_count':len(refs),'selected_variant':best,'quality_gates':gates,
       'quality_status':'ALL_LISTED_LOCAL_GATES_PASS' if all(gates.values()) else 'PARTIAL_SUCCESS_WITH_UNMET_GATES',
       'scope':'local fixed-camera reconstruction diagnostic; not full-scene, unseen-view or arbitrary-edit GT certification',
       'next_training_step_determinism_certified':False}
(OUT/'evidence/final_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
extra=f'''<section><h2>最终交付核验</h2><p>证据与运行合同检查通过；质量门槛状态：<b>{result['quality_status']}</b>。建议变体：{best}。已检查{count}张PNG完整性和{len(refs)}项本地HTML引用，并查看代表对照图与架构图。</p><p>追加变体从前一addon热启动，累计优化预算不同；不能把最终排序当成公平同预算论文基准。道路平面校正和延迟着色另有零更新探针，支持其机制判断。参考是置信度筛选的真实跨时刻观测，仍不等于独立干净背景GT。</p><p><a href="evidence/final_validation.json">完整验证与未达门槛项</a> · <a href="evidence/final_process_check.json">实际进程退出核验</a></p></section>'''
text=text.replace('</main></html>',extra+'</main></html>');report.write_text(text,encoding='utf-8')
files=[]
for path in sorted(OUT.rglob('*')):
    if path.is_file() and path.name!='manifest.json':files.append({'path':path.relative_to(OUT).as_posix(),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(OUT/'manifest.json').write_text(json.dumps({'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'validation':result},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
