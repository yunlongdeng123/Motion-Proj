"""Collect the targeted receiver-plane experiment and extend the pilot report."""
import datetime,hashlib,html,json,subprocess,sys
from pathlib import Path
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(HERE/'update_report.py')],check=True)
OUT=HERE.parent if HERE.name=='scripts' else HERE.parents[1]/'outputs/shadow_bc_experiments'
BASE='/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_BC_20261016_v1'
def get(source,target,required=True):
    path=OUT/target;path.parent.mkdir(exist_ok=True,parents=True)
    if path.is_file() and path.stat().st_size and target.endswith('.png'):return True
    r=subprocess.run(['scp','wm-vgpu-1011:'+BASE+'/'+source,str(path)],capture_output=True,text=True,timeout=40)
    if r.returncode and required:raise RuntimeError(r.stderr)
    return r.returncode==0
def load(p):return json.loads((OUT/p).read_text())
get('receiver_diagnosis_v1/diagnosis.json','evidence/receiver_capacity_diagnosis.json')
get('selection_plane_v1/selection_receipt.json','plane/geometry_projection.json')
get('plane_zero_update_probe_v1/R4/evaluation/metrics.json','plane/zero_update_metrics.json')
get('runs/plane_1000_v1/status.json','plane/status.json')
get('runs/plane_1000_v1/runtime_contract.json','plane/runtime_contract.json')
get('runs/plane_1000_v1/GPU_gradient_smoke.json','plane/GPU_gradient_smoke.json',False)
status=load('plane/status.json');geo=load('plane/geometry_projection.json');zero=load('plane/zero_update_metrics.json')
previous=load('results/R4/metrics.json');final=None
if status['state']=='complete':
    get('runs/plane_1000_v1/R4/evaluation/metrics.json','plane/metrics.json')
    get('runs/plane_1000_v1/R4/checkpoint_structure_validation.json','plane/checkpoint_validation.json')
    get('runs/plane_1000_v1/comparison.json','plane/comparison.json')
    final=load('plane/metrics.json')
    for name in ['unshaded_static','observed_composite','delete_target','delete_all_actors','translate_+1_m','road_yaw_+30_deg','fixed_camera_time_0008089','fixed_camera_time_0008137']:
        get('runs/plane_1000_v1/R4/evaluation/'+name+'.png','plane/'+name+'.png')
    canvas=Image.new('RGB',(540*2,332*2),'white');d=ImageDraw.Draw(canvas)
    for k,(name,path) in enumerate([
        ('old R4 observed','results/R4/observed_composite.png'),('road-plane R4 observed','plane/observed_composite.png'),
        ('old R4 delete target','results/R4/delete_target.png'),('road-plane R4 delete target','plane/delete_target.png')]):
        im=Image.open(OUT/path).crop((25,205,205,305)).resize((540,300));x=k%2*540;y=k//2*332
        canvas.paste(im,(x,y+32));d.text((x+8,y+9),name,fill='black')
    canvas.save(OUT/'plane/refinement_comparison.png')
    # A separate fixed-camera comparison makes residual temporal changes visible.
    canvas=Image.new('RGB',(420*3,216*2),'white');d=ImageDraw.Draw(canvas)
    cells=[('R0','results/R0/fixed_camera_time_0008089.png'),('R4','results/R4/fixed_camera_time_0008089.png'),('plane R4','plane/fixed_camera_time_0008089.png'),
           ('R0','results/R0/fixed_camera_time_0008137.png'),('R4','results/R4/fixed_camera_time_0008137.png'),('plane R4','plane/fixed_camera_time_0008137.png')]
    for k,(label,path) in enumerate(cells):
        im=Image.open(OUT/path).crop((40,225,180,285)).resize((420,180));x=k%3*420;y=k//3*216
        canvas.paste(im,(x,y+32));d.text((x+8,y+9),label+' '+('time raw89' if k<3 else 'time raw137'),fill='black')
    canvas.save(OUT/'plane/fixed_camera_residual_review.png')
rows=''
for label,m in [('原R4，1000步',previous),('接收面投影，0步',zero),('接收面投影，再1000步',final)]:
    rows+='<tr><td>'+label+'</td>'+(''.join(f'<td>{m[k]:.6f}</td>' for k in ['diagnostic_footprint_clean_linear_MAE','observed_shadow_linear_MAE','outside_control_sRGB_MAE_vs_parent','fixed_camera_mean_nonanchor_time_MAE']) if m else '<td colspan="4">训练中</td>')+'</tr>'
section=f'''<section style="background:#f4f8ff;padding:18px;margin:24px 0"><h2>针对性追加：校正道路接收面</h2>
<p><b>追加实验状态：{html.escape(status['state'])}，步数 {status.get('step',0)} / 1000。</b>这是首轮之后的独立几何改动，保留首轮结果；原checkpoint继续只读。</p>
<p>首轮R4虽优于接触阴影，但渲染仍有斑块。只读容量测试发现：现有接收点在阴影区平均占58.5%的可见alpha*T；即使全变黑也只有约0.00262线性误差不可避免，不能把0.0404的全部误差归因于接收覆盖不足。</p>
<p>于是把5,546个已有道路接收点沿实际相机射线投到LiDAR支持的道路平面，并重建平面切向协方差，尽量保持原屏幕高斯核。没有新增点。投影中心最大误差 {geo['anchor_projected_center_max_error_px']:.2g} 像素，中位conic相对误差 {geo['median_relative_conic_error']:.4%}；三维中心移动中位数 {geo['centre_displacement_median_m']:.3f} 米，反映低视角下高度误差会放大道路对应偏差。</p>
<p>零更新时，车影误差从0.0404下降到0.0191，但深度排序变化使道路误差变大。因此追加1,000步重新拟合颜色/透明度与阴影。新Adam，不冒充原训练的逐值续训。</p>
<table><tr><th>变体</th><th>干净道路MAE</th><th>原位车影MAE</th><th>外部控制区变化</th><th>固定相机时间变化</th></tr>{rows}</table>
<p><a href="plane/geometry_projection.json">几何映射回执</a> · <a href="plane/status.json">追加实验状态</a> · <a href="evidence/receiver_capacity_diagnosis.json">只读接收能力诊断</a></p>
'''
if final:section+='<img src="plane/refinement_comparison.png" alt="道路平面修正前后原位与删除对照"><img src="plane/fixed_camera_residual_review.png" alt="固定相机在不同时刻的残余变化">'
section+='''<p class="note">现有车辆资产在±30°旋转图中本身存在明显形状/纹理瑕疵。数学上车体与阴影使用同一编辑位姿，不等于旋转后视觉质量通过。局部背景仍有未改动legacy高斯带来的时间变化；若高于原定门槛，明确记为未解决项。</p></section>'''
report=OUT/'report.html';text=report.read_text(encoding='utf-8');text=text.replace('<h1>车影 B＋C：真实 GPU 实验与自验证</h1>','<h1>车影 B＋C：真实 GPU 实验与自验证</h1>'+section)
report.write_text(text,encoding='utf-8')
files=[]
for path in sorted(OUT.rglob('*')):
    if path.is_file() and path.name!='manifest.json':files.append({'path':path.relative_to(OUT).as_posix(),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(OUT/'manifest.json').write_text(json.dumps({'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'plane_state':status['state']},indent=2))
print(json.dumps({'plane_state':status['state'],'step':status.get('step'),'metrics':final,'report':str(report)},ensure_ascii=False))
