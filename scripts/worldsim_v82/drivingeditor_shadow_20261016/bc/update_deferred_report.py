"""Collect the final deferred-shading comparison; no training or remote writes."""
import datetime,hashlib,json,subprocess,sys
from pathlib import Path
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(HERE/'update_plane_report.py')],check=True)
OUT=HERE.parent if HERE.name=='scripts' else HERE.parents[1]/'outputs/shadow_bc_experiments'
BASE='/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_BC_20261016_v1'
def get(source,target,required=True):
    path=OUT/target;path.parent.mkdir(exist_ok=True,parents=True)
    if path.is_file() and path.stat().st_size and target.endswith('.png'):return True
    r=subprocess.run(['scp','wm-vgpu-1011:'+BASE+'/'+source,str(path)],capture_output=True,text=True,timeout=40)
    if r.returncode and required:raise RuntimeError(r.stderr)
    return r.returncode==0
def load(p):return json.loads((OUT/p).read_text())
get('../deferred_zero_update_probe_v1/R4/evaluation/metrics.json','deferred/zero_update_metrics.json')
get('runs/deferred_1000_v1/status.json','deferred/status.json')
get('runs/deferred_1000_v1/runtime_contract.json','deferred/runtime_contract.json')
get('runs/deferred_1000_v1/GPU_gradient_smoke.json','deferred/GPU_gradient_smoke.json',False)
status=load('deferred/status.json');final=None
if status['state']=='complete':
    get('runs/deferred_1000_v1/R4/evaluation/metrics.json','deferred/metrics.json')
    get('runs/deferred_1000_v1/R4/checkpoint_structure_validation.json','deferred/checkpoint_validation.json')
    get('runs/deferred_1000_v1/comparison.json','deferred/comparison.json')
    final=load('deferred/metrics.json')
    for name in ['unshaded_static','observed_composite','delete_target','delete_all_actors','translate_+1_m','road_yaw_+30_deg','fixed_camera_time_0008089','fixed_camera_time_0008137']:
        get('runs/deferred_1000_v1/R4/evaluation/'+name+'.png','deferred/'+name+'.png')
    canvas=Image.new('RGB',(540*2,332*2),'white');d=ImageDraw.Draw(canvas)
    for k,(title,path) in enumerate([
        ('per-Gaussian road shadow','plane/observed_composite.png'),('deferred visible road shadow','deferred/observed_composite.png'),
        ('deferred delete target','deferred/delete_target.png'),('deferred translate +1m','deferred/translate_+1_m.png')]):
        im=Image.open(OUT/path).crop((25,205,205,305)).resize((540,300));x=k%2*540;y=k//2*332
        canvas.paste(im,(x,y+32));d.text((x+8,y+9),title,fill='black')
    canvas.save(OUT/'deferred/edit_comparison.png')
    values={k:load(f'results/{k}/metrics.json') for k in ['R0','R1','R2','R3','R4','R5']}
    values['plane_R4']=load('plane/metrics.json');values['deferred_R4']=final
    summary={'schema':'shadow_BC_measured_comparison_v1','methods':values,
      'clean_error_reduction_vs_R0':1-final['diagnostic_footprint_clean_linear_MAE']/values['R0']['diagnostic_footprint_clean_linear_MAE'],
      'shadow_error_reduction_vs_contact':1-final['observed_shadow_linear_MAE']/values['R3']['observed_shadow_linear_MAE'],
      'static_time_invariance_gate_passed':final['fixed_camera_mean_nonanchor_time_MAE']<=1e-5,
      'scope':'primary local reconstruction pilot; confidence-filtered road/shadow labels, parent saw all49',
      'unresolved':['Legacy unmodified static branch still changes over time','Vehicle asset looks poor under large yaw','No counterfactual paired edit GT','Not full-scene or multi-scene validation']}
    (OUT/'measured_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

svg='''<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="400" viewBox="0 0 1080 400"><defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#526174"/></marker></defs><rect width="1080" height="400" fill="#f7f9fc"/><g font-family="Arial,sans-serif" fill="#172b45"><text x="25" y="30" font-size="21" font-weight="bold">Refined B + C: shade visible road contributions after rasterization</text><g stroke="#b9c9db" stroke-width="1.5" fill="white"><rect x="25" y="65" width="205" height="90" rx="10"/><rect x="280" y="65" width="210" height="90" rx="10" fill="#e8f6ef"/><rect x="25" y="255" width="205" height="90" rx="10"/><rect x="280" y="255" width="210" height="90" rx="10" fill="#fff3dd"/><rect x="545" y="105" width="220" height="105" rx="10"/><rect x="820" y="105" width="235" height="105" rx="10" fill="#eaf1ff"/><rect x="820" y="260" width="235" height="70" rx="10" fill="#e8f6ef"/></g><g font-size="16" font-weight="bold"><text x="40" y="94">RGB / cameras / LiDAR</text><text x="295" y="94">C: clean road layer</text><text x="40" y="284">Actor geometry + poses</text><text x="295" y="284">B: road shadow mask</text><text x="560" y="136">Full-scene rasterizer</text><text x="835" y="136">Shade visible road RGB</text><text x="835" y="290">Edited scene RGB</text></g><g font-size="14"><text x="40" y="121">Multi-time road observations</text><text x="40" y="143">Visibility + confidence</text><text x="295" y="121">Fixed geometry / appearance</text><text x="295" y="143">Supported road plane</text><text x="40" y="310">Edited SE(3) / delete</text><text x="40" y="332">Shared world light</text><text x="295" y="310">True road query points</text><text x="295" y="332">Direction / softness / strength</text><text x="560" y="164">Unshaded road + Actor body</text><text x="560" y="188">Depth sort + alpha*T features</text><text x="835" y="164">Road contribution only</text><text x="835" y="188">Keep body + other RGB</text><text x="835" y="313">Delete / translate / rotate</text></g><g fill="none" stroke="#526174" stroke-width="2" marker-end="url(#a)"><path d="M230 110H280"/><path d="M490 110H515V140H545"/><path d="M230 300H280"/><path d="M130 255V225H515V185H545"/><path d="M765 155H820"/><path d="M490 300H790V185H820"/><path d="M940 210V260"/></g><text x="615" y="277" font-size="13">world-space visibility</text></g></svg>'''
(OUT/'deferred_architecture.svg').write_text(svg,encoding='utf-8')
rows=''
for label,m in [('接收面校正＋逐高斯着色',load('plane/metrics.json')),('仅切换延迟着色，0步',load('deferred/zero_update_metrics.json')),('延迟着色，再1000步',final)]:
    rows+='<tr><td>'+label+'</td>'+(''.join(f'<td>{m[k]:.6f}</td>' for k in ['diagnostic_footprint_clean_linear_MAE','observed_shadow_linear_MAE','outside_control_sRGB_MAE_vs_parent','fixed_camera_mean_nonanchor_time_MAE']) if m else '<td colspan="4">训练中</td>')+'</tr>'
section=f'''<section style="background:#edf7f3;padding:18px;margin:24px 0"><h2>当前方案：对道路可见贡献做延迟阴影着色</h2>
<p><b>实验状态：{status['state']}，步数 {status.get('step',0)} / 1000。</b>这是第二次针对性追加；首轮以后总计追加2,000步，没有进入完整材质重光照D。</p>
<img src="deferred_architecture.svg" alt="跨时刻道路恢复、Actor阴影、完整场景排序与道路可见贡献着色的组件图">
<p>同样的阴影作用到每个高斯中心后再混合，会受高斯位置和核覆盖影响。此变体先在完整场景深度排序中提取道路alpha*T和RGB贡献，再按道路平面上的真实查询位置计算Actor遮挡，只替换可见道路贡献。车辆、天空和其余RGB保留，因此不是覆盖整张图的屏幕黑斑。删除所有Actor时直接返回未着色图，保证联动归零。</p>
<p>零更新切换时，干净道路误差不变，车影误差从0.02490降到0.01705；再按真实渲染损失微调1,000步。原始rasterizer在sRGB表观色上混合，额外着色采用解码/衰减/编码近似，这里不宣称完整物理光输运。</p>
<table><tr><th>变体</th><th>干净道路MAE</th><th>原位车影MAE</th><th>外部控制区变化</th><th>固定相机时间变化</th></tr>{rows}</table>
<p><a href="deferred/status.json">实际状态</a> · <a href="deferred/runtime_contract.json">渲染合同</a> · <a href="measured_summary.json">全部实测指标</a></p>'''
if final:section+='<img src="deferred/edit_comparison.png" alt="逐高斯、延迟着色以及删除平移对照">'
section+='''<p class="note">结果仅支持当前局部道路实验。仍需保留静态时间残差、车辆大角度旋转资产缺陷、伪标签边界和未做全场景验证这些限制。</p></section>'''
report=OUT/'report.html';text=report.read_text(encoding='utf-8');text=text.replace('<h1>车影 B＋C：真实 GPU 实验与自验证</h1>','<h1>车影 B＋C：真实 GPU 实验与自验证</h1>'+section);report.write_text(text,encoding='utf-8')
files=[]
for path in sorted(OUT.rglob('*')):
    if path.is_file() and path.name!='manifest.json':files.append({'path':path.relative_to(OUT).as_posix(),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(OUT/'manifest.json').write_text(json.dumps({'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'deferred_state':status['state']},indent=2))
print(json.dumps({'deferred_state':status['state'],'step':status.get('step'),'metrics':final,'report':str(report)},ensure_ascii=False))
