"""Read exact owned outputs over SSH and refresh the local experiment report.

No GPU imports, training launch, recursion over runtime sidecars, or remote writes.
"""
import datetime,hashlib,html,json,shutil,subprocess
from pathlib import Path
from PIL import Image,ImageDraw

HERE=Path(__file__).resolve().parent
if HERE.name=='scripts':OUT=HERE.parent;WORK=HERE
else:OUT=HERE.parents[1]/'outputs/shadow_bc_experiments';WORK=HERE
OUT.mkdir(exist_ok=True,parents=True)
BASE='/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_BC_20261016_v1'
RUN='pilot_1000_v1';remote=BASE+'/runs/'+RUN
PY='/root/autodl-tmp/envs/drivingeditor/bin/python'
def ssh(command):
    r=subprocess.run(['ssh','wm-vgpu-1011',command],capture_output=True,text=True,encoding='utf-8',timeout=30)
    if r.returncode:raise RuntimeError(r.stderr or r.stdout)
    return r.stdout
def fetch(source,target,required=False):
    relative=target;target=OUT/target;target.parent.mkdir(exist_ok=True,parents=True)
    if relative.startswith('results/') and target.exists() and target.stat().st_size>0:return True
    r=subprocess.run(['scp','wm-vgpu-1011:'+source,str(target)],capture_output=True,text=True,timeout=60)
    if r.returncode:
        if required:raise RuntimeError(r.stderr)
        return False
    return True
def write(path,data):(OUT/path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
probe=json.loads(ssh(f'PYTHONDONTWRITEBYTECODE=1 {PY} {BASE}/implementation_v1/progress_probe.py'))
write('live_progress.json',probe)
run=next(r for r in probe['runs'] if r['run']==RUN)
for name in ['status.json','runtime_contract.json','GPU_gradient_smoke.json','comparison.json']:
    fetch(remote+'/'+name,'evidence/'+name)
for name in ['preparation_receipt.json','training_label_review.png']:
    fetch(BASE+'/training_data_v2/'+name,'evidence/'+name,True)
fetch(BASE+'/selection_v1/selection_receipt.json','evidence/selection_receipt.json',True)
fetch(BASE+'/implementation_v1/torch_model_tests.json','evidence/torch_model_tests.json',True)
fetch(BASE+'/'+RUN+'_launch.json','evidence/launch.json',True)
fetch(BASE+'/runs/smoke_v1/status.json','evidence/smoke_status.json',True)
fetch(BASE+'/runs/smoke_v1/GPU_gradient_smoke.json','evidence/smoke_GPU_gradient.json',True)
arms=['R0','R2','R3','R4','R1','R5'];done=set(run['completed_arms']);metrics={}
images=['unshaded_static','observed_composite','delete_target','delete_all_actors','translate_+1_m','road_yaw_+30_deg',
        'fixed_camera_time_0008089','fixed_camera_time_0008137']
for arm in arms:
    if arm not in done:continue
    fetch(remote+'/'+arm+'/evaluation/metrics.json',f'results/{arm}/metrics.json',True)
    metrics[arm]=json.loads((OUT/f'results/{arm}/metrics.json').read_text())
    if arm!='R0':fetch(remote+'/'+arm+'/checkpoint_structure_validation.json',f'results/{arm}/checkpoint_structure_validation.json',True)
    for name in images:fetch(remote+'/'+arm+'/evaluation/'+name+'.png',f'results/{arm}/'+name+'.png',True)

if WORK!=OUT/'scripts':
    (OUT/'scripts').mkdir(exist_ok=True)
    for p in WORK.glob('*.py'):shutil.copy2(p,OUT/'scripts'/p.name)
    source=OUT.parent/'shadow_bc_plan/architecture.svg'
    shutil.copy2(source,OUT/'architecture.svg')

def montage(kind,name):
    present=[a for a in arms if (OUT/f'results/{a}/{kind}.png').is_file()]
    if not present:return
    canvas=Image.new('RGB',(420*3,216*((len(present)+2)//3)),'white');d=ImageDraw.Draw(canvas)
    for k,arm in enumerate(present):
        im=Image.open(OUT/f'results/{arm}/{kind}.png');im=im.crop((40,225,180,285)).resize((420,180))
        x=(k%3)*420;y=(k//3)*216;canvas.paste(im,(x,y+32));d.text((x+8,y+9),arm+' '+kind,fill='black')
    canvas.save(OUT/name)
montage('unshaded_static','clean_road_comparison.png');montage('observed_composite','observed_shadow_comparison.png')
if 'R4' in done:
    names=['observed_composite','delete_target','translate_+1_m','road_yaw_+30_deg']
    canvas=Image.new('RGB',(540*2,332*2),'white');d=ImageDraw.Draw(canvas)
    for k,name in enumerate(names):
        im=Image.open(OUT/f'results/R4/{name}.png').crop((25,205,205,305)).resize((540,300))
        x=(k%2)*540;y=(k//2)*332;canvas.paste(im,(x,y+32));d.text((x+8,y+9),'R4 '+name,fill='black')
    canvas.save(OUT/'R4_edit_review.png')

labels={'R0':'原checkpoint','R1':'仅忽略车影','R2':'C：固定干净道路','R3':'C＋接触阴影','R4':'C＋方向性投影阴影','R5':'R4保留时间自由度'}
def esc(x):return html.escape(str(x))
def num(m,key):return f'{m[key]:.6f}' if key in m else '—'
rows=''
for arm in arms:
    m=metrics.get(arm,{})
    state='完成' if arm in done else ('训练中' if run.get('arm')==arm and run['state']=='running' else '等待')
    if arm=='R0':state='原模型基线' if arm in done else '等待'
    rows+=f'<tr><td>{arm} {labels[arm]}</td><td>{state}</td><td>{num(m,"diagnostic_footprint_clean_linear_MAE")}</td><td>{num(m,"observed_shadow_linear_MAE")}</td><td>{num(m,"outside_control_sRGB_MAE_vs_parent")}</td><td>{num(m,"fixed_camera_mean_nonanchor_time_MAE")}</td></tr>'
evidence=json.loads((OUT/'evidence/preparation_receipt.json').read_text())['checks']
status=json.loads((OUT/'evidence/status.json').read_text()) if (OUT/'evidence/status.json').exists() else run
observed=''.join(f'<h2>{label}</h2><img src="{path}" alt="{label}">' for path,label in [
    ('clean_road_comparison.png','完成实验臂：未着色静态道路'),('observed_shadow_comparison.png','完成实验臂：原位车辆与阴影'),('R4_edit_review.png','R4：删除、平移与绕道路法线旋转')]
    if (OUT/path).is_file())
gate=run.get('gates');gate_text='<pre>'+esc(json.dumps(gate,ensure_ascii=False,indent=2))+'</pre>' if gate else '<p>正式成对实验尚未全部完成，暂不判定哪个方案有效。</p>'
script_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'scripts').glob('*.py')}
write('evidence/local_script_hashes.json',script_hashes)
html_text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>车影 B+C 实验记录</title>
<style>body{{font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif;margin:0;color:#203047;background:#f3f6fa}}main{{max-width:1180px;background:white;margin:auto;padding:30px 38px}}h1{{font-size:29px}}h2{{font-size:22px;margin-top:32px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #d8e1eb;padding:9px;vertical-align:top;text-align:left}}th{{background:#eff4f9}}img{{max-width:100%;height:auto}}a{{color:#1763a4}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:15px}}.note{{background:#fff5e4;padding:14px}}.status{{background:#eaf5ef;padding:15px}}small{{color:#68778a}}</style><main>
<h1>车影 B＋C：真实 GPU 实验与自验证</h1>
<p><small>更新时间：{datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')} Asia/Singapore · run {RUN}</small></p>
<p class="status"><b>当前状态：{esc(run['state'])}</b>；阶段 {esc(run.get('phase'))}；实验臂 {esc(run.get('arm'))}；步数 {esc(run.get('step'))}。实际owned进程存活：{run['owned_process_alive']}。已完成：{', '.join(run['completed_arms']) or '尚无'}。</p>
<p>本轮优先判断怎样解决当前黑影，不讨论新颖性。先从跨时刻真实道路观测恢复局部背景，再拟合由Actor编辑状态和世界光照决定的阴影。</p>
<img src="architecture.svg" alt="真实观测、干净道路、Actor条件阴影与深度合成的架构图">
<h2>实际执行范围</h2><p>主场景scene0008_089_137，raw101黑车及其周围140×60像素对应道路。新增5,546个已有高斯的颜色/透明度参数，并优化4,702个已归因近黑色贡献点的透明度因子；没有新增道路点或改车体。C主方案把所选道路与异常点的SHM位置、时间marginal及筛选改为固定局部状态。原checkpoint与其全部参数保持只读，所有训练结果保存为新addon。</p>
<p>先完成每臂5步的真实CUDA接口短测，再启动5个实验臂各1,000步。方向性阴影用8个方位各300步初始化，接触阴影用300步。先250步恢复道路，再小步联合阴影；Actor和原背景参数冻结。实际损失统一使用有效点的线性RGB L1，本轮不使用SSIM，避免无效窗口泄漏。</p>
<p class="note">这是一块有实际观测支持的局部道路重建pilot。原GN11和刚体checkpoint已看过全部49帧，不能声称未见帧泛化。阴影标签由RGB变化、车辆接近、mask和几何置信度生成，不能称独立人工车影GT。任意编辑没有配对真值。</p>
<h2>数据与接口检查</h2><ul>
<li>候选道路8,317像素；近邻LiDAR支持率 {evidence['local_LiDAR_support_fraction']:.2%}，三帧局部平面残差中位数约2.5–2.7厘米。</li>
<li>raw89/raw95合成道路参考，raw113一致性检查sRGB MAE {evidence['atlas_check_sRGB_MAE']:.5f}；曝光修正后线性MAE {evidence['atlas_check_linear_MAE_after_gain']:.5f}。</li>
<li>已人工查看8个时刻的RGB与置信度标签拼图。车道线高梯度边缘从shadow伪标签中剔除；局部LiDAR不被当作稠密可见性真值。</li>
<li>GPU短测：原Static FP32渲染逐值复现；道路颜色、道路/异常透明度、光照方位/高度、半影和直射光比例梯度均非零且有限。</li>
<li>删除全部Actor时，带阴影合成与未着色Static渲染一致；阴影与车体编辑位姿一致。addon保存文件、Adam状态及模型恢复已检查；下一步训练确定性尚未认证。</li></ul>
<p><a href="evidence/smoke_GPU_gradient.json">短测梯度回执</a> · <a href="evidence/runtime_contract.json">正式运行合同</a> · <a href="evidence/preparation_receipt.json">数据准备回执</a> · <a href="evidence/selection_receipt.json">高斯选择依据</a> · <a href="live_progress.json">实际进度快照</a></p>
<img src="evidence/training_label_review.png" alt="实际RGB和clean/shadow/unknown置信度标签">
<h2>正式实验的实际指标</h2><p>清理误差：未着色Static对跨时刻道路参考，越低越好。车影误差：带Actor和阴影合成对实际阴影伪标签区观测，越低越好。控制区为局部窗口以外对父模型的变化。时间变化是固定相机下未着色Static的黑块区域变化，包含未改动的legacy分支。</p>
<table><tr><th>实验臂</th><th>状态</th><th>黑块区干净道路线性MAE</th><th>原位车影线性MAE</th><th>外部控制区sRGB变化</th><th>固定相机时间sRGB变化</th></tr>{rows}</table>
{observed}
<h2>自主评估门槛</h2>{gate_text}
<p>正式比较完成后查看实际图，检查旧影残留、新影形状、车道线、透明度与错误遮挡，不能只按loss选最优。若OBB无法拟合轮廓，再考虑独立刚体高斯遮挡代理；若道路恢复失败，先修数据或局部表示。不会直接进入完整重光照D，也不自动追加无界训练。</p>
<h2>运行与定期检查</h2><p>已建立当前对话每15分钟检查：自动核验真实status、owned PID、短日志、图像与回执；状态无实质变化保持安静，完成或失败时反馈。该检查依赖本机与桌面应用保持运行；GPU实验本身由AutoDL独立进程执行。<a href="https://learn.chatgpt.com/docs/automations?surface=app">官方定期任务说明</a></p>
<pre>远端实验根：{BASE}
正式输出：{remote}
解释器：{PY}
只读进度命令：{PY} {BASE}/implementation_v1/progress_probe.py
本地刷新报告：python scripts/update_report.py</pre>
<p>共享GPU锁与StreetCrafter/其他任务优先保持生效。不修改其他任务、共享环境或GPU电源。当前聊天不使用subagent。</p>
</main></html>'''
(OUT/'report.html').write_text(html_text,encoding='utf-8')
for p in OUT.rglob('*.png'):
    with Image.open(p) as im:im.verify()
manifest=[]
for p in sorted(OUT.rglob('*')):
    if p.is_file() and p.name!='manifest.json':manifest.append({'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
write('manifest.json',{'run':RUN,'checked_utc':probe['checked_utc'],'files':manifest,'remote_read_only_collection':True})
print(json.dumps({'run':RUN,'state':run['state'],'phase':run.get('phase'),'arm':run.get('arm'),'step':run.get('step'),'completed':run['completed_arms'],'report':str(OUT/'report.html')},ensure_ascii=False))
