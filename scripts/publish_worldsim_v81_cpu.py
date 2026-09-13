"""整理本轮轻量CPU证据；不启动GPU或后续任务。"""
from pathlib import Path
import json,shutil,subprocess,os,datetime,html,tarfile

repo=Path('/root/autodl-tmp/motion_proj');run=Path('/root/autodl-tmp/runs/worldsim_v81/WS-V81-CPU-01/20260913-cpu-r2');ev=repo/'docs/autoresearch/worldsim_v81';ev.mkdir(parents=True,exist_ok=True)
# Windows staging的换行只在本轮上传文件归一为LF，保留历史正文。
normalize=list((repo/'docs').glob('WORLDSIM_V8_1_*.md'))+[repo/'docs'/n for n in ['RESEARCH_STATUS.md','EXPERIMENTS.md','RESEARCH_FAILURES.md']]+[repo/'docs/research_failures/entries/V81-F02.md',repo/'configs/worldsim_v81/cpu_r2.json']
for p in normalize:p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
status=json.loads((run/'status.json').read_text());status['tests']='7 passed';status['cpu_native_preprocessing']='3 official loaders with actual six-camera images passed';(run/'status.json').write_text(json.dumps(status,indent=2))
validation={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'focused_tests':{'command':'CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 PYTHONPATH=. /root/autodl-tmp/envs/worldsim-v81/bin/python -m pytest tests/test_worldsim_v81_geometry.py -q','result':'7 passed in 0.52s'},'strict_parameter_contracts':['meta_dvgt.json','meta_vggt.json','meta_dggt.json'],'native_input_contracts':'cpu_input_contracts.json','sota_forward':False,'gpu_kernels':'NOT_TESTED','pip_check_inherited_conflicts':['mapanything: trimesh missing','mapanything: opencv-python-headless 4.10 vs required 4.11','nuscenes-devkit: numpy<2 vs installed 2.2.6']}
(run/'validation.json').write_text(json.dumps(validation,indent=2))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(repo))
result=subprocess.run(['/root/autodl-tmp/envs/worldsim-v81/bin/python',str(repo/'scripts/evaluate_worldsim_v81.py'),'--atlas',str(run),'--prediction-root',str(run/'NO_GPU_OUTPUTS'),'--out',str(run/'evaluator_empty_contract')],cwd=repo,env=env,text=True,capture_output=True,check=True)
print(result.stdout)
names=['atlas_summary.json','index_summary.json','thresholds.json','case_selection.json','spot_checks.json','matched_cohorts.json','status.json','validation.json','runtime_versions.json','cpu_input_contracts.json','gpu_pilot_commands.json','independent_confirmation.json','meta_dvgt.json','meta_vggt.json','meta_dggt.json']
for name in names:shutil.copy2(run/name,ev/name)
assets={'run':str(run),'data_role':'DISCOVERY','raw_registry':str(run/'v81_roi_registry.jsonl'),'parquet':str(run/'texture_overlap_lidar_prior_stats.parquet'),'references':{'path':str(run/'reference_geometry'),'files':1527},'official_assets':str(run/'official_assets.json'),'models':{}}
for method,path in {'dvgt':'/root/autodl-tmp/models/worldsim_v81/dvgt1.pt','vggt':'/root/autodl-tmp/models/eas_vggt/vggt/model.safetensors','dggt':'/root/autodl-tmp/models/worldsim_v81/model_latest_nuscenes.pt'}.items():
    p=Path(path);assets['models'][method]={'path':path,'bytes':p.stat().st_size,'strict_meta_load':'PASS','forward':'NOT_TESTED'}
(ev/'asset_index.json').write_text(json.dumps(assets,indent=2));(run/'asset_index.json').write_text(json.dumps(assets,indent=2))
dest=repo/'docs/figures/worldsim_v81';dest.mkdir(parents=True,exist_ok=True)
for name in ['architecture.png','architecture.svg','evidence_grid.png','factor_controls.png','factor_escalation_inputs.png','prompt_placement.png']:
    shutil.copy2(run/'figures'/name,dest/name)
    if name.endswith('.svg'):
        p=dest/name;p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
(ev/'README.md').write_text('''# V8.1 CPU证据索引

WS-V81-CPU-01 / 20260913-cpu-r2；CPU_COMPLETE_WAIT_GPU。完整run路径见asset_index.json，仓库只存轻量证据。原始数据、完整权重、逐ROI参考及大表均在仓库外。

27日志、34场景、68窗口、6120 ROI；1527几何候选；49复核、36混杂排除；完整四格匹配组=0。三个官方模型参数合同和真实输入预处理通过，7项检查通过，模型前向=0。

status.json为当前状态，spot_checks.json为assistant逐图观察（不是human verdict），matched_cohorts.json为空数组。gpu_pilot_commands.json默认dry，仅开GPU后显式执行。自然H1仍需新增干净对照；V8.2证据未就绪。
''')
reviews={r['roi_id']:r for r in json.loads((run/'spot_checks.json').read_text())};cards=json.loads((run/'case_selection.json').read_text())
css='body{max-width:1280px;margin:40px auto;padding:0 24px;font:16px/1.65 system-ui,sans-serif;color:#193046;background:#f3f6fa}h1{font-size:36px;line-height:1.2}h2{margin-top:40px}img{display:block;width:100%;background:white;border-radius:10px}article{background:white;padding:18px;margin:22px 0;border-radius:14px;border:1px solid #dce3eb}.status{color:#087f72;font-weight:700}.warning{padding:18px;border-left:5px solid #cc8c32;background:#fff3df}.stats{display:flex;gap:18px;flex-wrap:wrap}.stats span{background:#fff;padding:14px 22px;border-radius:10px}.stats b{display:block;font-size:28px}a{color:#1467a1}small{color:#637485}'
parts=['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V8.1 稀疏视角 × 低纹理 · CPU图谱</title><style>'+css+'</style></head><body>', '<p class="status">CPU_COMPLETE_WAIT_GPU · 2026-09-13 · seed 8101</p><h1>稀疏视角 × 低纹理<br>V8.1 候选与混杂图谱</h1><p>目标是科学发现、badcase与可视化比较；V8.2才推动方法解决。</p><div class="stats"><span><b>27</b>独立日志</span><span><b>6,120</b>ROI筛查</span><span><b>1,527</b>几何候选</span><span><b>0</b>模型推理</span></div>', '<p class="warning"><b>当前发现：参考几何和图像纹理可能来自不同深度层。</b>49项逐图观察中36项排除主实验；剔除混杂后，同日志/语义/距离的完整四格匹配组为0。因此当前不能检验H1交互效应，更不能声称SOTA已经失败。GPU用于下一步模型筛查；自然对照不足仍需补数据。</p>', '<p><a href="WORLDSIM_V8_1_SCIENTIFIC_REPORT.md">完整科学报告</a> · <a href="WORLDSIM_V8_1_CPU_HANDOFF.md">GPU交接</a> · <a href="WORLDSIM_V8_1_METHOD_AUDIT.md">官方方法审计</a></p>']
overview=[('architecture.png','流程：输入 → 筛查与控制 → 图谱 → GPU发现 → V8.2入口'),('evidence_grid.png','固定四格候选；包含前景栅栏等混杂，不能当作干净匹配组'),('factor_controls.png','CPU输入LiDAR平面控制；不是SOTA模型误差'),('factor_escalation_inputs.png','平面招牌纹理干预；真值辅助mask，仅作诊断'),('prompt_placement.png','相同点数、不同位置；当前只准备输入')]
for file,title in overview:parts.append('<article><h2>'+title+'</h2><img alt="'+title+'" src="figures/'+file+'"></article>')
parts.append('<h2>8个固定规则案例</h2><p>所有误差图都来自当前输入LiDAR的RANSAC平面控制。尚无DVGT/VGGT/DGGT推理结果；较小控制误差不等于SOTA goodcase。点击图片可看原尺寸。</p>')
for row in cards:
    rid=row['roi_id'];rev=reviews.get(rid,{});tag=rev.get('visual_screen','UNREVIEWED');note=rev.get('observation','尚未完成该案例的单独视觉验收。')
    src='figures/'+rid+'.png';parts.append('<article><h3>'+html.escape(rid)+'</h3><p><b>'+tag+'</b> · '+html.escape(note)+'</p><a href="'+src+'"><img loading="lazy" alt="'+html.escape(rid)+' CPU control" src="'+src+'"></a></article>')
parts.append('<h2>高重叠候选逐图复核</h2><p>共46项，全部为模型结果出现前的输入/reference复核。观察仅为assistant记录，human verdict保持null。</p>')
for i in range(3):parts.append(f'<article><a href="figures/high_overlap_spotcheck_{i}.png"><img loading="lazy" alt="high overlap review {i}" src="figures/high_overlap_spotcheck_{i}.png"></a></article>')
parts.append('<h2>已停下，等待开GPU</h2><p>建议2×48GB分别运行DVGT与VGGT，或1×80GB顺序；主机≥64GB RAM / 8 vCPU。先一个六视角窗口测量显存并核验GPU前向，再进入冻结队列。没有自动续跑，没有进入V8.2。</p><small>Run: WS-V81-CPU-01 / 20260913-cpu-r2 · H1–H5 NOT_TESTED · V8.2 NO_GO_PENDING_EVIDENCE</small></body></html>')
(run/'index.html').write_text('\n'.join(parts),encoding='utf-8')
docnames=['WORLDSIM_V8_1_SCIENTIFIC_REPORT.md','WORLDSIM_V8_1_CPU_HANDOFF.md','WORLDSIM_V8_1_METHOD_AUDIT.md','WORLDSIM_V8_1_FAILURE_ATLAS.md','WORLDSIM_V8_1_TO_V8_2_DECISION.md','WorldSim_V81_SparseView_LowTexture_Failure_Discovery_Plan.md']
for name in docnames:
    content=(repo/'docs'/name).read_text().replace('figures/worldsim_v81/','figures/')
    (run/name).write_text(content,encoding='utf-8')
with tarfile.open(run/'v81_cpu_report.tar','w') as tf:
    exportnames=['index.html','figures','texture_overlap_lidar_prior_stats.parquet','spot_checks.json','atlas_summary.json','matched_cohorts.json','status.json','validation.json','asset_index.json','gpu_pilot_commands.json','cpu_input_contracts.json','case_selection.json']+docnames
    for name in exportnames:tf.add(run/name,arcname=name)
print(json.dumps({'status':'PUBLISHED_CPU_EVIDENCE','export':str(run/'v81_cpu_report.tar'),'export_bytes':(run/'v81_cpu_report.tar').stat().st_size,'models':assets['models']}))
