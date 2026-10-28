from pathlib import Path
import json,tarfile,re,html
B=Path(__file__).resolve().parents[2];W=Path(__file__).parent
O=B/'outputs/V43_Ownership_20261027/gpu_v43/temporal_multiscene_20261029'
A=W/'light_delivery/audit';audit=json.loads((A/'delivery_summary.json').read_text(encoding='utf-8'))
viewer=(O/'index.html').read_text(encoding='utf-8');diagram=re.search(r'<svg.*?</svg>',viewer,re.S).group()
records=[]
for s in json.loads((O/'manifest.json').read_text(encoding='utf-8')):
 records.append(dict(context=s['context'],frames=len(s['frames']),wall_seconds_including_queue=s['wall_seconds'],peak_reserved_GiB=s['cuda_peak_reserved']/2**30,new_rows_min=min(r['final_removed_rows'] for r in s['frames']),new_rows_max=max(r['final_removed_rows'] for r in s['frames'])))
tr=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [i+1,r['context'],r['frames'],f"{r['new_rows_min']}–{r['new_rows_max']}",round(r['peak_reserved_GiB'],3)])+'</tr>' for i,r in enumerate(records))
body='''<!doctype html><meta charset="utf-8"><title>V43 三场景逐帧诊断记录</title><style>body{font:16px/1.7 system-ui;background:#0b1420;color:#d8e8f6;max-width:1200px;margin:32px auto;padding:20px}td,th{padding:10px;border:1px solid #587085}table{border-collapse:collapse}a{color:#80c9ff}svg{width:100%}</style>
<h1>WS-V43-TEMPORAL-MULTISCENE-20261029</h1>
<p>完成三场景各49帧独立冻结unknown剥离，以及场景二、三原有10K编辑结果的共享工作台接入。训练更新0；复用此前已完成checkpoint，不把历史训练计成本轮成果。</p>'''+diagram+'''
<h2>方法与控制</h2><table><tr><th>项</th><th>本轮执行</th></tr>
<tr><td>范围</td><td>原始body bbox搬运同一归一化用户蓝框和车底query；不由删后RGB生成标签。</td></tr>
<tr><td>候选</td><td>每帧从空集合起步，目标Actor独立省略；原static分支投影中心必须落在蓝框内，query alpha×T质量&gt;1e-7。</td></tr>
<tr><td>剩余贡献</td><td>删除后重新归因，直到没有新eligible候选；最多8次集合更新。本轮147帧均收敛。</td></tr>
<tr><td>白底</td><td>局部G_after+(1-alpha_after)×白底；外部取同帧Actor-omitted基线。并非将非透明残影直接涂白。</td></tr>
<tr><td>三维</td><td>真实static持久行ID筛除，Actor为空；由于跨界支撑，二维局部框外保护不能冒充全局三维原生删除。</td></tr>
<tr><td>消融</td><td>三个anchor对比source_body_contact与更宽source_local_band；后者多删背景而contact alpha不降，故不扩跑。</td></tr>
<tr><td>与旧R2的差别</td><td>不继承场景一专属旧种子，采用empty-init和源观测query；三个anchor目视接近但contact alpha更高，不称完全复现。</td></tr>
</table><h2>实际交付</h2><table><tr><th>场景</th><th>context</th><th>新诊断帧</th><th>每帧候选行数</th><th>峰值reserved GiB</th></tr>'''+tr+'''
</table><p>另接回场景二、三各49帧10K结构，以及4×31帧无编辑/删车/横移/左转原生编辑。旧ProPainter结果只是保留对照；本轮没有调用生成补景。</p>
<h2>验收与保留问题</h2><p>147帧完整远端浮点审核全部通过：独立reset、同参数、框外新增中心0、eligible剩余贡献0、局部外像素差0、白底方程误差0。该结论仅是计算合同通过。相邻ID集合Jaccard最小/中位依次为0.7880/0.8649、0.7391/0.8172、0.6297/0.7660。不能将ID切换直接当RGB闪烁或反过来宣称稳定。</p>
<p>AI目视坏例：场景3 raw13下沿紫褐暗带，raw37/49/61局部白区贴近邻车边缘。三场景完整视觉等待用户验收。优先播放场景1 raw101–105、108–110、112–114；场景2 raw149–153、154–158、170–173；场景3 raw13–19、21–25、51–55。既有static分支含时间变化参数，不等于物理静止；跨边界高斯支撑可仍进入蓝框query。</p>
<p>这是可迁移的受限查询诊断，仍依赖人工给定参考蓝框，不是已学成的unknown语义分类器；没有独立GT，P/R未验证。下一步围绕混合支撑、车底和邻车边界，不扩大框外标签，不把白洞当补景成功。</p>
<h2>现场与成本</h2><p>远端完整浮点：/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/unknown_detection_v3/temporal_multiscene_20261029_r1。原1.4GB tar保留远端；本地下载完整轻量包245238276字节，含1176PNG、147份UI行ID/范围证据及全浮点审核报告。它明确不含全浮点，不能冒充本地重算。旧三帧full pilot在本地可复核。</p>
<p>三场景排队包含等待的墙钟约78.4/162.4/257.0秒；六结果pilot约10.4秒；两次接口名失配各额外重放raw98单帧后停止，失败现场保留。训练更新始终0。远端CPU完整审核读1.25GB浮点，约24.4秒。Git提交后推送与结果取回再核对；其他用户改动不纳入本轮提交。</p>
<p>共享入口：outputs/GS_Scene_Diagnostics_20261027/index.html；独立逐帧报告：outputs/V43_Ownership_20261027/gpu_v43/temporal_multiscene_20261029/index.html。failure_ledger_refs=[V82-F01]，failure_ledger_delta=none，new_sequence_human_verdict=null。</p>'''
(W/'technical_report.html').write_text(body,encoding='utf-8')
code='scripts/worldsim_v82/v43_ownership_20261028/unknown_detection_v3/temporal_multiscene_20261029/'
doc='docs/autoresearch/v43_ownership_20261028/'
with tarfile.open(W/'temporal_publication.tar.gz','w:gz') as tar:
 for name in ['render_temporal.py','scope.py','temporal_audit.py','audit_delivery.py','install_temporal_ui.py','build_report.py','resume_exports.py','make_publication.py','publish_temporal.py']:
  tar.add(W/name,arcname=code+name)
 for name in ['patch_shared_scene_entries.py','test_scene_entries.cjs','test_scene_links.py']:
  tar.add(B/'work/v43_scene_entry_repair_20261029'/name,arcname=code+name)
 tar.add(W/'technical_report.html',arcname=doc+'TEMPORAL_MULTISCENE_20261029.html')
 for name in ['ui_install_validation.json','ui_installed_validation.json','ui_canvas_validation.json','ui_link_validation.json','ai_visual_review.json','ai_visual_review_all49.json']:
  if (W/name).exists():tar.add(W/name,arcname=doc+'temporal_multiscene_20261029/'+name)
 for name in ['temporal_audit.json','delivery_summary.json']:
  tar.add(A/name,arcname=doc+'temporal_multiscene_20261029/'+name)
(W/'publication_ready.json').write_text(json.dumps(dict(training_updates=0,frames=147,PNG=1176,source_checkpoint_unchanged=True,local_full_float=False,records=records),indent=2),encoding='utf-8')
print(W/'temporal_publication.tar.gz')
