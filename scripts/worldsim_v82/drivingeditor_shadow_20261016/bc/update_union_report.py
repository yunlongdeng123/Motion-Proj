"""Collect the last bounded all-time coverage ablation and produce final summary."""
import datetime,hashlib,json,subprocess,sys
from pathlib import Path
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(HERE/'update_deferred_report.py')],check=True)
OUT=HERE.parent if HERE.name=='scripts' else HERE.parents[1]/'outputs/shadow_bc_experiments'
BASE='/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_BC_20261016_v1'
def get(source,target,required=True):
    path=OUT/target;path.parent.mkdir(exist_ok=True,parents=True)
    if path.is_file() and path.stat().st_size and target.endswith('.png'):return True
    r=subprocess.run(['scp','wm-vgpu-1011:'+BASE+'/'+source,str(path)],capture_output=True,text=True,timeout=40)
    if r.returncode and required:raise RuntimeError(r.stderr)
    return r.returncode==0
def load(path):return json.loads((OUT/path).read_text(encoding='utf-8'))
get('selection_union_v1/selection_receipt.json','union/selection_receipt.json')
get('selection_union_plane_v1/selection_receipt.json','union/geometry_projection.json')
for name in ['status.json','runtime_contract.json','GPU_gradient_smoke.json']:
    get('runs/union_1000_v1/'+name,'union/'+name,name!='GPU_gradient_smoke.json')
status=load('union/status.json');final=None;summary=load('measured_summary.json')
if status['state']=='complete':
    get('runs/union_1000_v1/R4/evaluation/metrics.json','union/metrics.json')
    get('runs/union_1000_v1/R4/checkpoint_structure_validation.json','union/checkpoint_validation.json')
    get('runs/union_1000_v1/comparison.json','union/comparison.json')
    final=load('union/metrics.json');summary['methods']['union_R4']=final
    for name in ['unshaded_static','observed_composite','delete_target','delete_all_actors','translate_+1_m','road_yaw_+30_deg','fixed_camera_time_0008089','fixed_camera_time_0008137']:
        get('runs/union_1000_v1/R4/evaluation/'+name+'.png','union/'+name+'.png')
    base=summary['methods']['R0']
    review=load('evidence/visual_review.json') if (OUT/'evidence/visual_review.json').exists() else {'rejected_variants':{}}
    rejected=set(review['rejected_variants'])
    eligible=[k for k,m in summary['methods'].items() if k not in ['R0','R1','R2','R5'] and k not in rejected
       and m['diagnostic_footprint_clean_linear_MAE']<=base['diagnostic_footprint_clean_linear_MAE']*.5
       and m['outside_control_sRGB_MAE_vs_parent']<=.01]
    best=min(eligible,key=lambda k:summary['methods'][k]['observed_shadow_linear_MAE']) if eligible else None
    summary.update(suggested_variant_by_clean_and_shadow_gate=best,visually_rejected_variants=list(rejected),
        strict_static_time_pass={k:m['fixed_camera_mean_nonanchor_time_MAE']<=1e-5 for k,m in summary['methods'].items()},
        additional_C_updates=3000,pilot_C_updates=5000,further_training_scheduled=False)
    (OUT/'measured_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    canvas=Image.new('RGB',(420*3,216*2),'white');d=ImageDraw.Draw(canvas)
    paths=[('raw101 single-time','deferred/unshaded_static.png'),('raw101 all49','union/unshaded_static.png'),('raw101 original','results/R0/unshaded_static.png'),
        ('raw137 single-time','deferred/fixed_camera_time_0008137.png'),('raw137 all49','union/fixed_camera_time_0008137.png'),('raw137 original','results/R0/fixed_camera_time_0008137.png')]
    for k,(title,path) in enumerate(paths):
        im=Image.open(OUT/path).crop((40,225,180,285)).resize((420,180));x=k%3*420;y=k//3*216
        canvas.paste(im,(x,y+32));d.text((x+8,y+9),title,fill='black')
    canvas.save(OUT/'union/time_coverage_comparison.png')
    canvas=Image.new('RGB',(540*2,332*2),'white');d=ImageDraw.Draw(canvas)
    for k,(title,path) in enumerate([('single-time deferred','deferred/observed_composite.png'),('all49 deferred','union/observed_composite.png'),
        ('all49 delete target','union/delete_target.png'),('all49 translate +1m','union/translate_+1_m.png')]):
        im=Image.open(OUT/path).crop((25,205,205,305)).resize((540,300));x=k%2*540;y=k//2*332
        canvas.paste(im,(x,y+32));d.text((x+8,y+9),title,fill='black')
    canvas.save(OUT/'union/edit_comparison.png')
rows=''
for k,m in summary['methods'].items():
    rows+='<tr><td>'+k+'</td>'+''.join(f'<td>{m[x]:.6f}</td>' for x in ['diagnostic_footprint_clean_linear_MAE','observed_shadow_linear_MAE','outside_control_sRGB_MAE_vs_parent','fixed_camera_mean_nonanchor_time_MAE'])+'</tr>'
selection=load('union/selection_receipt.json')
section=f'''<section style="background:#f2f6fc;padding:18px;margin:24px 0"><h2>最终覆盖消融：跨49时刻收集同一道路区域的候选</h2>
<p><b>状态：{status['state']}，步数 {status.get('step',0)} / 1000。</b>这是最后一项追加；首轮5,000步，追加合计3,000步，后续训练没有排队。</p>
<p>候选并集包含{selection['receiver_points']:,}个道路接收点、{selection['other_opacity_only_points']:,}个仅优化透明度的其他点。其他点不等于已判定的错误几何：并集按真实49时刻的局部前景核覆盖选择，也覆盖原先被遮挡、修复后可能露出的点。新接收点从真实道路参考初始化；按stable ID继承旧addon。所有覆盖点的位置、时间权重、筛选与SH观察方向同时关闭时间自由度，防止固定位置后SH仍使用旧运动中心。</p>
<p>道路几何仍是局部LiDAR支持的平面假设，不是稠密几何真值。没有新增高斯或修改父checkpoint；新Adam独立优化。</p>
<table><tr><th>实测方法</th><th>黑块干净道路MAE</th><th>原位车影MAE</th><th>外部控制区变化</th><th>固定相机时间变化</th></tr>{rows}</table>
<p><a href="measured_summary.json">完整实测汇总</a> · <a href="union/status.json">最后一项状态</a> · <a href="union/selection_receipt.json">跨时刻选择依据</a></p>'''
if final:
    section+=f'<p class="note"><b>union_R4 已被实际图像审查否决。</b>它虽然平均指标更好，但在原诊断轮廓以外产生了严重黑片，删除目标车辆后仍存在。EWA核触及道路区域不等于整个高斯属于道路，控制区均值也不能代替局部图像检查。排除已否决变体后，当前建议为 <b>{summary["suggested_variant_by_clean_and_shadow_gate"]}</b>；严格静态时间门槛仍须单独判断。</p>'
    section+='<img src="union/time_coverage_comparison.png" alt="跨时刻覆盖前后静态道路比较"><img src="union/edit_comparison.png" alt="跨时刻覆盖后的车影、删除和平移比较">'
section+='''<p class="note">上述结果属于当前局部重建实验；置信度伪标签、已看过49帧的父模型、大yaw车体资产缺陷和无真实反事实GT这些边界仍然成立。即使时间门槛通过，也只认证该固定相机诊断区域及所检查时刻，不认证整个场景。</p></section>'''
report=OUT/'report.html';text=report.read_text(encoding='utf-8');text=text.replace('<h1>车影 B＋C：真实 GPU 实验与自验证</h1>','<h1>车影 B＋C：真实 GPU 实验与自验证</h1>'+section);report.write_text(text,encoding='utf-8')
files=[]
for path in sorted(OUT.rglob('*')):
    if path.is_file() and path.name!='manifest.json':files.append({'path':path.relative_to(OUT).as_posix(),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(OUT/'manifest.json').write_text(json.dumps({'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'union_state':status['state']},indent=2))
print(json.dumps({'union_state':status['state'],'step':status.get('step'),'metrics':final,'suggested_variant':summary.get('suggested_variant_by_clean_and_shadow_gate')},ensure_ascii=False))
