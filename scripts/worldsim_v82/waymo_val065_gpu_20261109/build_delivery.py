"""从已取回的实际输出构建独立报告和共享平台子页。"""
from pathlib import Path
import json,tarfile,hashlib,shutil,datetime,base64,gzip,html
import numpy as np
from PIL import Image,ImageDraw,ImageFont
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]
OUT=BASE/'outputs/waymo_val065_experiment'
PLATFORM=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-22/ni-x-a/outputs/GS_Scene_Diagnostics_20261027')
CONTEXT='15021599536622641101_556_150_576_150';KEYS=[30,60,150,198]
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def blob(a):return base64.b64encode(gzip.compress(np.ascontiguousarray(a,dtype='<f4').tobytes(),compresslevel=5,mtime=0)).decode()
def main():
    manifest=read(HERE/'retrieval_manifest.json');archive=HERE/'scene065_results.tar'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==manifest['sha256']
    OUT.mkdir(exist_ok=True);receipt=[]
    with tarfile.open(archive) as tar:
        for row in manifest['files']:
            p=OUT/row['member'];assert p.resolve().is_relative_to(OUT.resolve());data=tar.extractfile(row['member']).read();assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);receipt.append(row['member'])
    write(HERE/'retrieval_verification.json',dict(status='PASS_COMPLETE_RETRIEVAL',files=len(receipt),bytes=archive.stat().st_size,sha256=manifest['sha256']))
    viewer=OUT/'viewer';(viewer/'layers').mkdir(exist_ok=True)
    for name,src in [('viewer.css',BASE/'work/val029_visualization/viewer.css'),('viewer.js',HERE/'viewer.js'),('architecture.svg',HERE/'architecture.svg'),('segmentation_four_frames.png',HERE/'mask_qa_r2.png')]:shutil.copy2(src,viewer/name)
    with (viewer/'viewer.css').open('a',encoding='utf8') as stream:stream.write('\n.contribution-list .metric{display:flex;justify-content:space-between;gap:12px;margin:8px 0}.metric-grid .metric>span{display:block;font-size:11px;color:#94abc4}@media(min-width:851px){.stage-compare{grid-template-columns:repeat(3,minmax(0,1fr))}}\n')
    shutil.copy2(HERE/'surface_evidence.png',OUT/'surface_evidence.png')
    shutil.copy2(HERE/'architecture.svg',OUT/'architecture.svg')
    shutil.copy2(HERE/'mask_qa_r2.png',OUT/'segmentation_four_frames.png')
    metrics=read(OUT/'metrics.json');media=read(BASE/'outputs/waymo_val065_preparation/media.json');frames={f['raw']:f for f in media['frames']}
    roles=[dict(label='待删除的正前方白车',alias='a9',id=10,track='NTjq-OhZ_97OMJ4xwJAxXg',colour='#ff737b'),dict(label='保持的右侧白色皮卡',alias='a7',id=8,track='KOsYCOB6W_n0DzZsyDdJ-w',colour='#55d6ed'),dict(label='前方深色皮卡',alias='a20',id=21,track='bzFbWRrSGOgEaXxnuGV0tw',colour='#e9bd68')]
    by={v['raw']:v for v in read(OUT/'evidence/final_all_eval_summary.json')['frames']}
    rows=[]
    for raw in range(199):
        boxes=[]
        for role in roles:
            b=next((b for b in frames[raw]['boxes'] if b['track']==role['track']),None)
            if b:boxes.append(dict(id=role['id'],colour=role['colour'],corners=[[p[0]/2,p[1]/2] if p else None for p in b['corners2D']]))
        rows.append(dict(raw=raw,heldout=79<=raw<=118,boxes=boxes,metrics=by.get(raw)))
    # 世界→模型缩放来自原生训练变换，不从相机旋转推断。
    scale=read(HERE/'display_contract.json')['meter_to_model']
    data=dict(context=CONTEXT,keys=KEYS,meter_to_model=scale,roles=roles,edges=media['box_edges_corner_indices'],frames=rows)
    (viewer/'data.js').write_text('window.VAL065_DATA='+json.dumps(data,ensure_ascii=False,separators=(',',':'))+';',encoding='utf8')
    for raw in KEYS:
        packed={}
        for state in ['keep','delete']:
            z=np.load(OUT/'evidence/key_float'/str(raw)/f'{state}.npz');a=np.concatenate([z['group_alpha'],z['alpha'],z['depth_sum'],z['rgb']],axis=0);assert a.shape==(9,320,480) and np.isfinite(a).all();packed[state]=blob(a)
        for state in ['keep','delete']:
            z=np.load(OUT/'evidence/native_float'/str(raw)/f'{state}.npz');a=np.concatenate([z['group_alpha'],z['alpha'],z['depth_sum'],z['rgb']],axis=0);packed['native_'+state]=blob(a)
        (viewer/'layers'/f'{raw}.js').write_text(f'window.VAL065_LAYERS=window.VAL065_LAYERS||{{}};window.VAL065_LAYERS[{raw}]='+json.dumps(packed,separators=(',',':'))+';',encoding='utf8')
    page=(HERE/'viewer.html').read_text(encoding='utf8')
    (OUT/'index.html').write_text(page.replace('@@STYLE@@',PLATFORM.joinpath('style.css').as_uri()).replace('@@WORKSPACE@@',PLATFORM.joinpath('workspace.css').as_uri()).replace('@@ROOT@@','viewer/').replace('@@NAV@@',f'<a href="{PLATFORM.joinpath("index.html").as_uri()}">全部场景</a>'),encoding='utf8')
    backup=HERE/'backups'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S');backup.mkdir(parents=True)
    child=PLATFORM/CONTEXT
    for p in [PLATFORM/'index.html',PLATFORM/'scenes.json']:
        shutil.copy2(p,backup/p.name)
    if child.exists():shutil.copytree(child,backup/'previous_child')
    child.mkdir(exist_ok=True);shutil.copytree(viewer,child/'viewer',dirs_exist_ok=True)
    (child/'index.html').write_text(page.replace('@@STYLE@@','../style.css').replace('@@WORKSPACE@@','../workspace.css').replace('@@ROOT@@','viewer/').replace('@@NAV@@','<a href="../index.html">全部场景</a>'),encoding='utf8')
    p=PLATFORM/'index.html';s=p.read_text(encoding='utf8');card=f'<article class="panel" data-scene="{CONTEXT}"><h2>Waymo · scene-065</h2><details class="dataset-origin"><summary>验证集 · 原始路段编号</summary><code>{CONTEXT}</code></details><img src="{CONTEXT}/viewer/current/60/delete.png" alt="正前方白车删除预览"><p>逐帧对照完整重建、白车删除与右侧皮卡保持，查看静态和动态贡献。</p><a class="primary" href="{CONTEXT}/index.html">打开诊断 →</a></article>'
    if f'{CONTEXT}/index.html' not in s:
        assert '<div class="scene-cards">' in s;s=s.replace('<div class="scene-cards">','<div class="scene-cards">'+card,1);p.write_text(s,encoding='utf8')
    scenes=read(PLATFORM/'scenes.json');scenes=[v for v in scenes if v['context']!=CONTEXT];scenes.append(dict(context=CONTEXT,scene_key='waymo-val-065',display_name='Waymo · scene-065',dataset_split='validation',registered_scene_index=65,raws=list(range(199)),groups=5,entry=f'{CONTEXT}/index.html',note='沿用候选库零起始登记，不是官方全局ID'))
    for v in scenes:
        if v['context']=='12820461091157089924_5202_916_5222_916':v['groups']=5
    write(PLATFORM/'scenes.json',scenes)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21);figure=Image.new('RGB',(2400,720),'#102131');draw=ImageDraw.Draw(figure)
    columns=[('原始相机观测','observed/{raw}.png'),('原生完整重建 · 无编辑','native/{raw}/keep.png'),('仅按来源删除白车','native/{raw}/delete.png'),('归属＋道路 · KEEP','current/{raw}/keep.png'),('归属＋道路 · DELETE','current/{raw}/delete.png')]
    for i,raw in enumerate([60,150]):
        pts=np.array([p for b in rows[raw]['boxes'] for p in b['corners'] if p is not None]);lo=pts.min(0)-[18,16];hi=pts.max(0)+[18,16];size=np.maximum(hi-lo,[150,100]);size[0]=max(size[0],size[1]*1.5);size[1]=size[0]/1.5;size=np.minimum(size,[480,320]);origin=np.clip((lo+hi-size)/2,[0,0],np.array([480,320])-size);crop=tuple(np.r_[origin,origin+size].round().astype(int))
        for j,(label,path) in enumerate(columns):
            draw.text((j*480+12,i*360+7),f'帧 {raw} · {label}',font=font,fill='white');figure.paste(Image.open(viewer/path.format(raw=raw)).crop(crop).resize((480,320),Image.Resampling.LANCZOS),(j*480,i*360+40))
    figure.save(OUT/'comparison.png')
    road_receipt=read(OUT/'evidence/world_road__receipt.json')
    native150=next(v['mean_groups_on_target_mask'] for v in metrics['native_source_groups']['frames'] if v['raw']==150 and v['state']=='keep')
    native_table=''.join(f'<tr><td>{v["raw"]}</td>'+''.join(f'<td>{x:.4f}</td>' for x in v['mean_groups_on_target_mask'])+'</tr>' for v in metrics['native_source_groups']['frames'] if v['state']=='keep')
    # 报告单独交付；不向共享平台插入报告、状态或仓库链接。
    table=''.join(f'<tr><td>{name}</td><td>{r["frames"]}</td><td>{r["pixels"]:,}</td><td>{r["native_mae"]:.5f}</td><td>{r["current_mae"]:.5f}</td><td>{r["delete_keep_mae"]:.5f}</td><td>{r["frames_keep_error_increased"]}</td></tr>' for key,name in [('target_white','目标白车'),('preserved_pickup','右侧保持皮卡'),('front_pickup','前方深色皮卡'),('all_other_vehicles','全部其他车辆')] for r in [metrics['regions'][key]])
    full=''.join(f'<tr><td>{label}</td><td>{r["frames"]}</td><td>{r["native_mae"]:.5f}</td><td>{r["current_mae"]:.5f}</td><td>{r["native_psnr"]:.2f}</td><td>{r["current_psnr"]:.2f}</td></tr>' for key,label in [('build','构建'),('heldout','连续留出')] for r in [metrics['full_view'][key]])
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>scene-065 · 原生重建与白车删除</title><style>body{{max-width:1200px;margin:40px auto;padding:0 24px;font:17px/1.8 system-ui;color:#243e4b;background:#f7f9fc}}h1{{font-size:33px;line-height:1.4}}h2{{margin-top:42px}}img{{width:100%;height:auto;border-radius:10px}}table{{border-collapse:collapse;width:100%;font-size:15px;background:white}}td,th{{border-bottom:1px solid #d7e1ea;padding:10px;text-align:right}}td:first-child,th:first-child{{text-align:left}}a{{color:#126f9d}}.note{{background:#e9f1f7;padding:18px;border-radius:10px}}code{{font-size:14px;overflow-wrap:anywhere}}</style><h1>独立完整重建与白车删除：Waymo · scene-065</h1><p><strong>本例尚未删干净。</strong>第150帧最终图仍有明显车形残留；原生完整重建较完整。下文保留两种删除、两个KEEP以及失败的中间阶段，方便直接复核。</p><p>前方同车道白车为删除目标，右侧白色皮卡用于保持对照。原生完整场景已从头训练10,000步，另用共享跨帧归属和世界道路执行删除。两条通路均使用同一相机与时钟。</p><p><a href="index.html">打开199帧可拖动对照与像素贡献</a> · <a href="metrics.json">完整数值</a></p><p class="note">原生来源删除在 {metrics['target_native_source_delete_unchanged_frames']}/{metrics['native_frames']} 个构建时刻没有超过1/255的像素变化。共享归属的KEEP与原生RGB最大差 {metrics['semantic_KEEP_native_maxabs']:.3g}，纯V1与原生评价入口最大差 {metrics['V1_maxabs']:.3g}。目标贡献归零是内部一致性检查，不能单独证明车辆删净或背景正确。</p><p>第150帧白车轮廓内，静态通路平均贡献约{native150[1]*100:.1f}%，其他来源动态通路约{native150[2]*100:.1f}%，自身来源动态通路约{native150[0]*100:.3f}%。因此，重建看起来正确并不意味着来源身份足以执行删除。</p><img src="comparison.png" alt="两个时刻的五路真实输出"><p>上图在每个时刻使用相同的车辆局部裁图；完整视野与原始PNG见逐帧工作台。</p><h2>从输入到输出</h2><img src="architecture.svg" alt="简单组件图"><p>159帧构建，79–118连续40帧留出。完整静态、动态和环境从头训练；V1独立入口不导入编辑包，也不接入道路和车辆恢复资产。编辑通路用冻结的跨帧可见质量决定归属，再融合世界坐标道路和静态不透明度交接，统一计算遮挡。没有二维图像拼接。</p><p>本轮接通的是归属与道路子链，没有新建前方皮卡的专用补全资产，也没有使用scene-029的SUV或道路。新增实测正提示连通域输入清理单列为预处理适配，不能称作完全冻结预处理的跨场景成功。</p><h2>可见区域保持</h2><p>同帧实拍与KEEP的RGB绝对误差，按像素汇总；删除前后变化另列。分割用于定位，不是人工逐像素真值。误差上升帧的分母是该行有可见像素的帧数。</p><table><tr><th>区域</th><th>可见帧</th><th>像素</th><th>原生KEEP</th><th>当前KEEP</th><th>DELETE–KEEP</th><th>误差上升帧</th></tr>{table}</table><h2>完整画面与留出</h2><p>均对同一实拍缩小图计算。PSNR按帧求值后平均。留出帧未提供实例轮廓监督，因此只报告整幅图像，不补造车辆区域分母。</p><table><tr><th>时间窗</th><th>帧</th><th>原生MAE</th><th>当前MAE</th><th>原生PSNR</th><th>当前PSNR</th></tr>{full}</table><h2>白车由哪条原生通路成像</h2><p>在白车预测轮廓内，按统一排序的可见质量 α×透射率求均值。此只读诊断的RGB与原生入口的RGB最大差低于2×10⁻⁷。它区分静态与动态来源，不把通路名称当成语义真值。</p><table><tr><th>原始帧</th><th>目标来源动态</th><th>全部静态</th><th>其他来源动态</th><th>道路</th></tr>{native_table}</table><p>道路图共{road_receipt["cells"]:,}格，{road_receipt["real_cells"]:,}格具有多帧观测支持、{road_receipt["inferred_cells"]:,}格使用邻域插值；最终激活{road_receipt["active_cells"]:,}格，{road_receipt["unknown_cells"]:,}格未接管。插值不是实测，{road_receipt["marking_cells"]}格标线保持不替换。</p><p>第150帧像素(250,300)呈现亮色车形残留：目标贡献为0，静态贡献约0.8953，其他动态约0.0006，道路为0。工作台的“定位亮色残留”按钮可直接检查这一位置。它定位了该像素的成像通路，尚不等于定位唯一的物理对象或根因。</p><h2>哪些显露可以核验</h2><p>前方深色皮卡尚无通过本轮规则的隐藏表面对应。道路最多只有11个2像素网格的稀疏提名，并集中在白车轮廓边缘；相机、滚动快门与预测轮廓误差仍影响对应。下面是参考帧50到查询帧150的实际世界点，不能当作密集删除真值。道路图的多视点观测域、插值域和未知域分开保存；拟合代理域误差也不是独立编辑指标。</p><img src="surface_evidence.png" alt="真实跨时刻稀疏道路对应"><h2>分割与运行验收</h2><img src="segmentation_four_frames.png" alt="四帧实例分割核验"><p>四帧30/60/150/198经代理视觉核验；使用同一规则清理实测提示未支持的离散连通域。新原生输入为16个有实测支持的车辆、574,013静态初始点和620,145动态初始点。</p><p>真实smoke用时{metrics['smoke']['elapsed_seconds']:.1f}秒、{metrics['smoke']['engineering_updates']:,}次额外更新，覆盖完整保存、评价、Adam／随机状态／相机采样／SGCM／EMA精确恢复，以及恢复后两次更新和下一入口。正式训练另从头执行10,000步，保持30,000步学习率时钟；耗时{metrics['formal']['elapsed_seconds']:.1f}秒。道路交接额外预算单列于证据。四组可见质量互斥且守恒，非目标静态与动态通路分别保存；通路分组不等于语义真值。</p><p>代理图像观察：第150帧仍有明显车形残留；执行的归属与道路通路未通过删除清洁度检查。白车KEEP的像素汇总误差也高于原生重建。此判断不代填用户的人工结论。运行完成与质量验收分开，人工结论留空；本轮没有AutoDL电源操作。模型与原始测距留存远端，逐帧PNG、四帧float32、指标、配置和执行源代码已取回。</p><details><summary>稳定身份与证据文件</summary><p>原始路段：<code>{CONTEXT}</code>。目标白车a9：<code>NTjq-OhZ_97OMJ4xwJAxXg</code>；右侧皮卡a7：<code>KOsYCOB6W_n0DzZsyDdJ-w</code>；前方皮卡a20：<code>bzFbWRrSGOgEaXxnuGV0tw</code>。</p><p><a href="evidence/smoke_gate.json">smoke门槛</a> · <a href="evidence/smoke_restore__restore_audit.json">精确恢复</a> · <a href="evidence/formal10k__effective_config.json">正式配置</a> · <a href="evidence/input__verified_surface_correspondences.json">稀疏对应</a></p></details></html>'''
    (OUT/'report.html').write_text(report,encoding='utf8')
    write(HERE/'build_receipt.json',dict(status='BUILT_ACTUAL_SCENE065_RESULTS',local=str(OUT/'index.html'),platform=str(child/'index.html'),frames=199,panels=5,float_frames=KEYS,backup=str(backup),no_report_link_in_platform=True))
    print('BUILT',OUT)
if __name__=='__main__':main()
