from pathlib import Path
import json, shutil, html
BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'outputs/V43_Ownership_20261027/gpu_v43/sim_mechanism_20261030'
E=OUT/'evidence'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def esc(x):return html.escape(str(x))
smoke=read(E/'smoke_r2/smoke_summary.json')
formal=read(E/'formal_r3_10k/returned.json')
render=read(E/'edit_r3_10k/render_manifest.json')
metrics=read(E/'edit_r3_10k/reference_metrics.json')
assessment=read(OUT/'assessment.json')
heldout_manifest=read(E/'heldout_r3_10k/render_manifest.json')
heldout_metrics=read(E/'heldout_r3_10k/reference_metrics.json')
assert heldout_manifest['status']=='NATIVE_SINGLE_HELDOUT_VIEW_RENDERED'
assert read(E/'heldout_r3_10k/execution.json')['status']=='SCORED_ONE_INDEPENDENT_HELDOUT_VIEW'
assert smoke['status']=='PASS_REAL_NATIVE_SMOKE' and smoke['approximately_ten_minutes_contract']
assert formal['iteration']==10000 and formal['reason']=='budget_saved'
assert read(E/'formal_execution.json')['returncode']==0
assert read(E/'formal_r3_10k/checkpoint_events/iteration_010000.json')['status']=='complete'
for leg in ['resumed','next_stage_entry']:
 assert read(E/f'smoke_r2/{leg}/restore_verification.json')['all_passed']
expected={(f,e) for f in ['000000','000008','000015'] for e in ['none','delete','shift','rotate']}
assert {(r['frame'],r['edit']) for r in render['frames']}==expected
assert {(r['frame'],r['edit']) for r in metrics['rows']}==expected
assert len(metrics['rows'])==12 and metrics['missing_reference_count']==0 and metrics['scored']==12
assert all(r['status']=='SCORED_INDEPENDENT_REFERENCE' for r in metrics['rows'])
template=BASE/'work/v43_gt_protocol_20261030/publication/m0_report_template'
for name in ['architecture.svg','architecture.png']:
 shutil.copy2(template/name,OUT/name)
# 图中状态标签仅陈述本报告实际已完成的接口。
svg=(OUT/'architecture.svg').read_text(encoding='utf8').replace('GPU smoke待跑','GPU smoke已通过').replace('GPU smoke 待跑','GPU smoke 已通过').replace('小时级正式运行前约10分钟；成本单列；当前均等待实际凭据','已通过约10分钟及恢复入口；正式10K；额外开销单列')
(OUT/'architecture.svg').write_text(svg,encoding='utf8')
state=dict(artifact_kind='actual_run_report',task_id='WS-V43-SIM-RECON-M0-20261030',design_revision=5,
 dataset_revision='r2',native_adapter_revision='r3',observed_frames=16,formal_updates=10000,
 smoke_complete=True,formal_complete=True,native_edit_views=len(render['frames']),
 independent_reference_views_scored=metrics['scored'],new_V43_joint_model_trained=False,
 unique_gaussian_GT_assumed=False,human_verdict=None,assessment=assessment,
 automatic_autodl_shutdown=False,heldout={'status':'executed','views':1})
(OUT/'report_state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
rows=[]
for r in metrics['rows']:
 regions=r['regions'];body=regions['original_body_footprint'];road=regions['road_reference']
 b=f'{body["clamped_mae"]:.4f}' if body['pixels'] else 'N/A（空区域）'
 d=f'{road["clamped_mae"]:.4f}' if road['pixels'] else 'N/A（空区域）'
 rows.append(f'<tr><td>{r["frame"]}</td><td>{r["edit"]}</td><td>{body["pixels"]}</td><td>{b}</td><td>{d}</td></tr>')
notes=''.join('<li>'+esc(s)+'</li>' for s in assessment['findings'])
nexts=''.join('<li>'+esc(s)+'</li>' for s in assessment['next_steps'])
report='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V43 · 独立仿真与实际重建 M0</title><style>
body{max-width:1480px;margin:26px auto;padding:0 24px;background:#f5f9fc;color:#183848;font:16px/1.6 "Microsoft YaHei",sans-serif}section{background:white;border:1px solid #c9dbe6;border-radius:9px;margin:20px 0;padding:22px}h1{font-size:29px}h2{font-size:22px}table{width:100%;border-collapse:collapse}th,td{border:1px solid #cbdbe4;text-align:left;padding:8px}th{background:#e9f1f7}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}figure{margin:0}img{width:100%;display:block}figcaption{font-size:14px;color:#466273}select{padding:8px;margin:8px;border:1px solid #abc;background:white}.notice{background:#fff4dc}a{color:#156bb2}code{overflow-wrap:anywhere}small{color:#597286}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style>
<h1>V43：从独立世界到受污染高斯的第一份实际重建</h1>
<p>Revision 5 · M0 · 2026-10-30。''' +esc(assessment['lead'])+'''</p>
<section><h2>本轮完成与边界</h2><ul>'''+notes+'''</ul>
<p>程序化普通车、道路和灯光，仅作接口与机制验证。未训练新的 V43 联合模块，未在真实场景一训练分类器，也不把此单场景当泛化证据。</p>
<p><a href="report_state.json">运行状态</a> · <a href="evidence/EXPERIMENT_CONTRACT.json">锁定合同</a> · <a href="evidence/formal_r3_10k/config_delta.json">原生配置差异</a> · <a href="evidence/dataset_r2/validation.json">数据核验</a> · <a href="evidence/edit_r3_10k/render_manifest.json">实际编辑收据</a></p></section>
<section><h2>数据与真值分路</h2><img src="architecture.svg" alt="独立世界生成受限观测，原生重建，隔离真值与编辑评价">
<p>训练读取 16 帧 480×320 RGB、稀疏 LiDAR、相机和时间；当前条件显式使用 simulator oracle body/sky masks 和 ideal tracks。完整网格、稠密深度、表面 ID、隐藏表面和编辑参考隔离在 privileged 目录，未参与初始化或训练。</p>
<p>GT 在世界表面与对象上。每颗高斯不强制唯一 owner；原动静态分支不是 GT。阴影属于道路接收面上的车辆关联效应，32 samples 开关差分只作含采样误差的诊断代理，不作纯阴影标签。</p></section>
<section><h2>同帧独立参考与实际编辑</h2><label>帧<select id="frame"><option>000000</option><option selected>000008</option><option>000015</option></select></label>
<label>操作<select id="edit"><option value="none">不编辑</option><option value="delete" selected>删除原动态分支</option><option value="shift">横移 1.5m</option><option value="rotate">转向 30°</option></select></label>
<div class="grid"><figure><img id="reference"><figcaption>仿真器独立干预参考；不是待测高斯删点图</figcaption></figure><figure><img id="final"><figcaption>实际原生编辑：C_G + (1−alpha)E</figcaption></figure><figure><img id="white"><figcaption>相同高斯，统一白底诊断：C_G + (1−alpha)</figcaption></figure></div>
<p class="notice">本处操作作用于整个已学习动态分支，是当前重建的分支编辑基线。不能称为正确归属或理想编辑。白色表示透射／空洞，不能解释为背景已补好。</p>
<div class="grid"><figure><img id="C"><figcaption>高斯预乘颜色贡献 C_G</figcaption></figure><figure><img id="alpha"><figcaption>高斯 alpha</figcaption></figure><figure><img id="fill"><figcaption>实际环境回填 (1−alpha)E；显示裁剪，FP32保留远端</figcaption></figure></div>
</section>
<section><h2>独立留出相机</h2><p>中间时刻，世界Y方向偏移1m并重新朝向车辆；该相机图像未参与训练。仅一个视角，不能作泛化结论。</p>
<div class="grid"><figure><img src="evidence/heldout_r3_10k/reference_rgb.png"><figcaption>独立仿真参考</figcaption></figure><figure><img src="evidence/heldout_r3_10k/final.png"><figcaption>实际原生结果</figcaption></figure><figure><img src="evidence/heldout_r3_10k/white.png"><figcaption>相同高斯白底诊断</figcaption></figure></div>
<p><a href="evidence/heldout_r3_10k/query_camera.json">相机与坐标核验</a> · <a href="evidence/heldout_r3_10k/reference_metrics.json">留出分区统计</a> · <a href="evidence/heldout_r3_10k/execution.json">执行收据</a></p></section>
<section><h2>保存、完整恢复及计算记账</h2><table><tr><th>项目</th><th>实测</th></tr>
<tr><td>GPU smoke</td><td>两段各至少300秒训练循环墙钟；保存／评价开销扣除，完整恢复与下一阶段none/delete实际渲染通过。<a href="evidence/smoke_r2/smoke_summary.json">收据</a></td></tr>
<tr><td>正式预算</td><td>从零 10,000 更新；沿用原方法 30,000 步学习率周期，没有为了短程压缩日程；5K/10K保存。</td></tr>
<tr><td>smoke与额外开销</td><td>'''+esc(smoke.get('extra_smoke_updates'))+''' smoke更新 + '''+esc(smoke.get('extra_entry_updates'))+''' 恢复入口更新，另有317步计时探测；不计入正式10K。</td></tr>
<tr><td>输入适配修复</td><td>r1仿真delete配对含1LSB噪声，r2关闭抖动／去噪并修正网格法线；r3掩码查询floor(uv)修正半像素错配，K与姿态不变。旧现场保留。</td></tr>
<tr><td>原生不编辑一致性</td><td>3关键帧重放最大差 '''+esc(render['native_none_identity_max_abs'])+'''；训练相机/K复核通过；评价无参数更新。</td></tr>
</table><p><a href="evidence/formal_execution.json">正式耗时</a> · <a href="evidence/formal_r3_10k/cuda_memory_peak.json">显存</a> · <a href="evidence/edit_r3_10k/training_camera_restore_verification.json">相机复核</a></p></section>
<section><h2>区域误差仅辅助核验</h2><p>RGB MAE归一化到0–1；区域来自独立可见表面。原车体投影区域在删除后应与无车参考比较。这些数值不等于高斯归属准确率，也不自动把误差判成残影。</p><table><tr><th>帧</th><th>操作</th><th>原车体区域像素</th><th>原车体区域MAE</th><th>道路MAE</th></tr>'''+''.join(rows)+'''</table><p><a href="evidence/edit_r3_10k/reference_metrics.json">完整原始与裁剪后分区统计</a></p></section>
<section><h2>下一步</h2><ol>'''+nexts+'''</ol><p>真实三场景没有完整GT；人工／subagent核验不当作完美GT。M1/M2称为真值辅助参考，不能称为数学上界。AutoDL保持开机。</p></section>
<script>function update(){const f=document.getElementById('frame').value,e=document.getElementById('edit').value;document.getElementById('reference').src='evidence/dataset_r2/privileged/references/'+e+'/shadow_on/rgb/'+f+'.png';for(const k of ['final','white','C','alpha','fill']){document.getElementById(k).src='evidence/edit_r3_10k/'+e+'/'+f+'_'+k+'.png';document.getElementById(k).alt=e+' '+f+' '+k}}for(const id of ['frame','edit'])document.getElementById(id).onchange=update;update();</script></html>'''
(OUT/'index.html').write_text(report,encoding='utf8')
print(json.dumps({'report':str(OUT/'index.html'),'views':len(render['frames']),'status':'actual_receipts_rendered'}))
