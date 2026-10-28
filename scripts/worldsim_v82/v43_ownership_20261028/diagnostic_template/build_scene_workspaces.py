"""Build one offline HTML workspace per physical scene from verified exports."""
from pathlib import Path
import base64,hashlib,html,json,shutil,re,time
BASE=Path(__file__).resolve().parents[2];WORK=Path(__file__).parent;OUT=BASE/'outputs/GS_Scene_Diagnostics_20261027'
TEMPLATE=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-20/file-c-users-dengyunlong-documents-codex-2/outputs/GS_Reconstruction_Diagnostics_20261021_Evidence_r1')
def write(p,s):p.write_text(s,encoding='utf-8')
def js(v):return json.dumps(v,ensure_ascii=False,separators=(',',':'))
scenes=json.loads((OUT/'scenes.json').read_text(encoding='utf-8'))
alignment={x['context']:x for x in json.loads((OUT/'reconstruction_map_alignment.json').read_text(encoding='utf-8-sig'))}
shutil.copy2(TEMPLATE/'style.css',OUT/'style.css')
shutil.copy2(WORK/'scene_workspace.css',OUT/'workspace.css')
shutil.copy2(WORK/'scene_target_detection.js',OUT/'scene_target_detection.js')
write(OUT/'viewer.js',(WORK/'scene_viewer.js').read_text(encoding='utf8').replace('/* EDIT_GEOMETRY_RUNTIME */',(WORK/'scene_edit_geometry.js').read_text(encoding='utf8')).replace('/* SAMPLING_RUNTIME */',(WORK/'scene_sampling.js').read_text(encoding='utf8')).replace('/* SEMANTIC_RUNTIME */',(WORK/'scene_semantics.js').read_text(encoding='utf8')))
architecture='''<svg xmlns="http://www.w3.org/2000/svg" width="1160" height="152" viewBox="0 0 1160 152"><rect width="1160" height="152" rx="10" fill="#101c2d"/><defs><marker id="a" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0 0L6 3L0 6" fill="#87b7dd"/></marker></defs><g fill="#1b2f44" stroke="#496782"><rect x="20" y="33" width="245" height="84" rx="8"/><rect x="305" y="33" width="245" height="84" rx="8"/><rect x="590" y="33" width="245" height="84" rx="8"/><rect x="875" y="33" width="265" height="84" rx="8"/></g><g font-family="Microsoft YaHei, sans-serif" font-size="17" fill="#e4edf6" text-anchor="middle"><text x="142" y="65">冻结 30K / 六档评价</text><text x="142" y="94">原生、编辑与后端输出</text><text x="427" y="65">只读 CPU 导出</text><text x="427" y="94">PNG / 地图 / 真实几何</text><text x="712" y="65">每场景一个 HTML</text><text x="712" y="94">实验切换 / 共享帧与像素</text><text x="1007" y="65">同帧对照与人工诊断</text><text x="1007" y="94">未知留空 / 保存记录</text></g><path d="M265 75H297 M550 75H582 M835 75H867" stroke="#87b7dd" stroke-width="2" marker-end="url(#a)"/></svg>'''
write(OUT/'architecture.svg',architecture)
cards=[];build=[]
for scene in scenes:
    folder=OUT/scene['context'];d=json.loads((folder/'scene.json').read_text(encoding='utf-8'));groups={g['id']:g for g in d['groups']}
    assert alignment[scene['context']]['frames']==31 and alignment[scene['context']]['max_rgb_difference']==0
    if 'continuous_camera_revisit' in groups:groups['continuous_camera_revisit']['title']='修订 5 · 侧移相机重访'
    continuous_order=['delete_original','insert_reused','move_inserted','turn_inserted','camera_revisit','revisit_original_camera']
    order={cat:i for i,cat in enumerate(['reconstruction','benchmark','motion','continuous','control'])}
    def group_order(g):
        if g['category']=='continuous':rank=continuous_order.index(g['id'].removeprefix('continuous_'))
        elif g['id']=='motion_no_edit':rank=-1
        else:rank=d['groups'].index(g)
        return order[g['category']],rank
    d['groups']=sorted(d['groups'],key=group_order)
    # Share actual 30K no-edit mixed-render maps at its central 31 observed views.
    # This is the same frozen checkpoint/camera and native adapter equality was checked.
    for raw,frame in groups['motion_no_edit']['frames'].items():
        if frame.get('maps'):
            groups['reconstruction_30000']['frames'][raw]['maps']=frame['maps']
            groups['reconstruction_30000']['frames'][raw]['maps_source']='same-checkpoint native no-edit adapter; exact GPU render agreement recorded'
    # Export script records per-source hash; preserve it before local index enrichment.
    if not (folder/'scene_export_original.json').exists():shutil.copy2(folder/'scene.json',folder/'scene_export_original.json')
    if not (folder/'scene_export_original.js').exists():shutil.copy2(folder/'scene.js',folder/'scene_export_original.js')
    write(folder/'scene.json',js(d));write(folder/'scene.js','window.SCENE_DATA='+js(d)+';')
    page=(WORK/'scene_viewer.html').read_text(encoding='utf-8')
    if (folder/'target_detection_data.js').exists():
        page=page.replace('<!-- target-detection-data-start --><!-- target-detection-data-end -->','<!-- target-detection-data-start --><script src="target_detection_data.js?v=target-detection-2"></script><!-- target-detection-data-end -->')
    write(folder/'index.html',page)
    if not (folder/'identity.js').exists():write(folder/'identity.js','window.IDENTITY_AUDIT=null;')
    # file:// images can taint Canvas in Chromium. Load exact PNG bytes via
    # local script data URLs so pixel inspection works without a web server.
    for p in (folder/'assets').rglob('*.png'):
        wrapper=p.with_name(p.name+'.js')
        if not wrapper.exists():write(wrapper,'window.IMAGE_DATA['+js(p.relative_to(folder).as_posix())+']='+js('data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode())+';')
    raw=str(sorted(map(int,d['frames']))[32]);im=groups['reconstruction_30000']['frames'][raw]['channels']['rgb'];cp=d['checkpoints'][-1]['metrics'];ctx=scene['context'];n=scene['scene_number']
    totals={cat:sum(g['category']==cat for g in d['groups']) for cat in ['reconstruction','benchmark','motion','continuous','control']}
    cards.append(f'''<article class="panel"><h2>场景 {n} · 30K 完整</h2><p class="note">{html.escape(ctx)}</p><a href="{ctx}/index.html#workspace"><img src="{ctx}/{im}" alt="场景{n}原生30K raw{raw}"></a><p>六档 49 帧重建 · 八类 31 帧编辑<br>同场景共用一个工作台</p><p class="note">完整图重建 PSNR {cp['psnr']:.3f} dB / SSIM {cp['ssim']:.5f}<br>阶段二开发诊断，质量未验收</p><a class="primary" href="{ctx}/index.html#workspace">打开全部实验组 →</a></article>''')
    build.append(dict(context=ctx,html=str(folder/'index.html'),groups=totals,frames=49,index_enrichment='Share verified no-edit maps with 30K reconstruction; Chinese labels and chronological ordering. Export originals preserved.',index_sha256=hashlib.sha256((folder/'scene.json').read_bytes()).hexdigest(),export_original_sha256=hashlib.sha256((folder/'scene_export_original.json').read_bytes()).hexdigest()))
page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三场景 · Gaussian Evidence Lab</title><link rel="stylesheet" href="style.css"><link rel="stylesheet" href="workspace.css"></head><body><header><div><span class="eyebrow">GAUSSIAN EVIDENCE LAB / SCENE WORKSPACES</span><h1>三条新路段 · 共享实验工作台</h1><p>每个场景一个 HTML，原生重建、八类编辑、固定视频修补和诊断控制在页内切换。</p></div><div class="badges"><span class="badge good">3 个完整 30K</span><span class="badge">离线可用</span><span class="badge warn">研究任务暂停 · 质量待人审</span></div></header><nav><a href="#workspace">场景入口</a><a href="../AutoDL_Disk_Incident_Inventory_20261027/index.html">停机后盘点</a><a href="export_manifest.json">导出来源与 SHA256</a><a href="validation.json">文件与交互逻辑检查</a></nav><main id="workspace"><div class="scene-cards">'''+''.join(cards)+'''</div><section class="panel" style="margin-top:24px"><h2>与原诊断系统一致的用法</h2><p>先选实验集合和实验组，再通过共享帧轴或播放按钮检查时序；点击二维图选像素，查看放大裁片与数值。底部一键切换 5K—30K 对照，也可以为六个卡片各选一个实验。可导出带场景、实验、帧号、像素和备注的诊断记录。</p><img src="architecture.svg" class="architecture" alt="诊断工作台组件图"><p class="note">八类编辑 3×8=24 组原生输出、12 组固定 ProPainter；旧运动协议和连续修订不叠加成正式案例。相机新视角与插入没有伪造的视频结果。三维窗口按当前请求复放 30K 高斯采样、相机与资产变换；原始 LiDAR 单独作为观测参照。此次只读导出，不恢复训练或定时任务。</p></section></main><footer>2026-10-27 · 原 PNG / 真实 checkpoint 采样 / 来源可追溯</footer></body></html>'''
page=page.replace('<nav>','<nav><a href="'+TEMPLATE.joinpath('index.html').as_uri()+'#workspace">原道路场景工作台</a>',1)
write(OUT/'index.html',page)
# Existing template gets a durable navigation link, not incompatible scene data.
old=TEMPLATE/'index.html';text=old.read_text(encoding='utf-8');link=OUT.joinpath('index.html').as_uri()
marker='<!-- new-scene-workspaces-20261027 -->'
if marker not in text:
    backup=WORK/'viewer_template_before_new_scenes.html'
    if not backup.exists():shutil.copy2(old,backup)
    text=text.replace('<nav>','<nav>'+marker+f'<a href="{link}">新增三场景 · 全部实验 ↗</a>',1);write(old,text)
# Provide links from the existing evaluation snapshot without rewriting its evidence.
old_report=BASE/'outputs/4D_Editing_Evaluation_20261027/index.html';text=old_report.read_text(encoding='utf-8')
if marker not in text:
    backup=WORK/'eval_report_before_scene_workspaces.html'
    if not backup.exists():shutil.copy2(old_report,backup)
    notice=marker+'<div style="padding:16px;background:#19344c;color:#e7f1ff;border:1px solid #72a7d5"><strong>停机后更新：</strong>目前 3 场景已完整 30K；研究任务已暂停。<a style="color:#9bd6ff" href="../GS_Scene_Diagnostics_20261027/index.html">打开每场景共用的诊断工作台 →</a>　以下保留此前阶段快照。</div>'
    # The legacy report uses an implicit body; insert inside its main element.
    anchor=r'(<body[^>]*>)' if re.search(r'<body\b',text) else r'(<main[^>]*>)'
    text,count=re.subn(anchor,lambda m:m[0]+notice,text,count=1)
    assert count==1 and marker in text, 'Report entry insertion failed'
    write(old_report,text)
write(OUT/'build_receipt.json',json.dumps(dict(scenes=build,template=str(TEMPLATE),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [WORK/'scene_viewer.js',WORK/'scene_viewer.html',WORK/'scene_workspace.css',WORK/'scene_edit_geometry.js',WORK/'scene_sampling.js',WORK/'build_edit_geometry.py']},updated_existing_links=[str(old),str(old_report)],new_training_updates=0,automation_resumed=False),ensure_ascii=False,indent=2))
from build_edit_geometry import main as build_edit_geometry
build_edit_geometry()
print(json.dumps(build,ensure_ascii=False))

# Integrate the existing old-scene HTML after rebuilding the shared index.
if (OUT/"legacy_scene0008/coverage_audit.json").exists():
    import runpy
    runpy.run_path(str(BASE/"work/v43_ownership_20261027/integrate_legacy.py"))

# Restore R2 reference and legacy-study separation only when complete input assets exist.
import runpy
for repair in (Path(__file__).with_name('patch_ui.py'), BASE/'work/v43_r2_regression_20261029/patch_ui.py'):
    if repair.is_file():
        namespace=runpy.run_path(str(repair))
        if namespace['can_repair'](BASE):namespace['main'](BASE)
        break
