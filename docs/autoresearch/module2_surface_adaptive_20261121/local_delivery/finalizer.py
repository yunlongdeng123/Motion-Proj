"""Resumable artifact delivery for actual surface Gaussian insertion."""
from pathlib import Path
import argparse,importlib.util,json,os,shutil

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]
spec=importlib.util.spec_from_file_location('transport',ROOT/'work/road_common1152_20261121/finalizer.py')
F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)
F.REMOTE='/root/autodl-tmp/drivingeditor_repro/campaign/road_surface_adaptive_20261121_r1'
F.OUT=ROOT/'outputs/road_material_three_rounds_20261121'
F.DEST=F.OUT/'surface_adaptive';F.DEST.mkdir(parents=True,exist_ok=True)

def install(archive):
    release=F.DEST/'completed_r1';F.safe_unpack(archive,release)
    candidates=list(release.rglob('latest_manifest.json'))
    if len(candidates)!=1:raise ValueError('unique latest_manifest required')
    m=F.read(candidates[0]);metrics=F.read(release/'bench/metrics.json')
    prefix=release.relative_to(F.OUT).as_posix()+'/'
    def imagepath(p,full):
        q=(release/p).resolve()
        if not q.is_relative_to(release.resolve())or not q.is_file():raise ValueError('bad image '+p)
        with F.Image.open(q)as im:
            if im.width>1152 or im.height>768 or full and im.size!=(1152,768):raise ValueError('bad image size '+p)
        return prefix+p
    models=m['models']
    for model in models:
        if set(model['frames'])!={'86','110','134','179','198'}:raise ValueError('five-view export incomplete '+model['id'])
        if isinstance(model.get('budget'),dict):model['budget']=' / '.join(str(k)+' '+str(v)for k,v in model['budget'].items())
        for fr in model['frames'].values():
            for k,v in list(fr.items()):
                if isinstance(v,str)and Path(v).suffix.lower()in ['.png','.webp','.jpg']:
                    fr[k]=imagepath(v,k in ['road','joint','alpha'])
    teacher=m['teacher']
    if list(teacher['resolution'])!=[1152,768]:raise ValueError('teacher grid mismatch')
    for fr in teacher['frames'].values():
        for k,v in list(fr.items()):
            if isinstance(v,str)and Path(v).suffix.lower()in ['.png','.webp','.jpg']:fr[k]=imagepath(v,k in ['A','C','S'])
    stages={str(x.get('stage',x['id'])).lower():x['id']for x in models}
    expected=['parent','activated','dense_init','coverage','appearance']
    for stage in expected:
        if stage not in stages:
            matches=[x['id']for x in models if str(x['id']).lower().endswith(stage)]
            if len(matches)!=1:raise ValueError('missing stage '+stage)
            stages[stage]=matches[0]
    def counts(stage):
        model=next(x for x in models if x['id']==stages[stage])
        p=model.get('point_counts',m.get('point_counts',{}).get(model['id'],{}))
        return {**p,'total':p.get('total',p.get('total_rows')),'active':p.get('active',p.get('active_rows'))}
    pc={stage:counts(stage)for stage in expected}
    if pc['parent']['total']!=588861 or pc['parent']['active']!=151988:raise ValueError('unexpected parent road count')
    for stage,p in pc.items():
        if not isinstance(p['total'],int)or not isinstance(p.get('appended_new_geometry'),int):raise ValueError('missing actual geometry counts '+stage)
        if p['total']-pc['parent']['total']!=p['appended_new_geometry']:raise ValueError('append count does not match total rows '+stage)
        if stage in ['parent','activated']and p['appended_new_geometry']!=0:raise ValueError('activation counted as new geometry')
    evidence=[dict(name=p.name,path=prefix+p.relative_to(release).as_posix())for p in sorted(release.rglob('*'))if p.is_file()and p.suffix in ['.json','.md']and len(p.relative_to(release).parts)<=3]
    for name in ['report.md','firstwave_four_rows.json','residual_stratification.json','max_allowed_receipt.json','max_allowed_summary.json']:
        source=WORK/name
        if source.exists():
            shutil.copy2(source,F.DEST/name)
            evidence.append(dict(name=name,path=F.DEST.relative_to(F.OUT).as_posix()+'/'+name))
    shutil.copy2(WORK/'architecture_surface.svg',F.DEST/'architecture_surface.svg')
    data=dict(models=models,teacher=teacher,stages=stages,rows=metrics.get('rows',[]),gaps=metrics.get('blackgap_rows',[]),evidence=evidence,
        summary=m.get('summary',metrics.get('summary',{})),point_counts=m.get('point_counts',{}),source_manifest=str(candidates[0]))
    data['newly_covered_rows']=metrics.get('newly_covered_rows',[])
    data['geometry_support']={str(raw):imagepath(path if isinstance(path,str)else path['path'],True)for raw,path in m.get('geometry_support',{}).items()if str(raw)in ['86','110','134','179','198']}
    history=F.OUT/'history_common1152.html'
    if not history.exists():shutil.copy2(F.OUT/'index.html',history)
    # Reuse the verified image/metric UI styling, with a stage-specific controller.
    base=(ROOT/'work/road_common1152_20261121/report_template.html').read_text('utf-8')
    base=base.split('<script>',1)[0]
    base=base.replace('道路材质 · 1152教师统一重训','道路覆盖与材质修复')
    base=base.replace('Scene065 · 新教师 · 三轮方法统一重训','Scene065 · 固定道路表面 · 覆盖驱动局部增密')
    base=base.replace('1152×768教师与道路学生','道路覆盖与材质修复')
    base=base.replace('<h1>道路覆盖与材质修复</h1>','<h1>道路覆盖与材质修复</h1><p id="geometry-counts" class="note"></p>')
    base=base.replace('固定新版二维教师，重新训练三轮正式方法。所有新训练与渲染均为1152×768。','优先启用闲置锚点，再沿固定道路表面局部补点；先训练覆盖，再拟合冻结教师的颜色与细节。')
    base=base.replace('history_before1152.html','history_common1152.html').replace('查看原三轮历史报告','查看1152教师三轮训练报告')
    base=base.replace('common1152/run_status.json','surface_adaptive/run_status.json')
    a=base.index('<nav>');b=base.index('</nav>',a)+6
    base=base[:a]+'<nav><button data-preset="final" class="active">父资产 → 最终</button><button data-preset="geometry">锚点 → 补点覆盖</button><button data-preset="material">覆盖 → 外观</button></nav>'+base[b:]
    base=base.replace('<option value="joint">场景联合渲染</option>','<option value="joint">场景联合渲染</option><option value="alpha">道路透明度</option>')
    base=base.replace('新增训练 / 推理 / 科研指标计算为 0','固定预算训练与分阶段验收')
    base=base.replace('全部合格新教师标签用于拟合；本页不声称隐藏道路真值或严格盲测。','覆盖200步用于新启用锚点；按剩余缺口补点后再训练覆盖400步；外观1200步。原有活动高斯的几何与覆盖保持冻结。')
    base=base.replace('architecture_1152.svg','architecture_surface.svg').replace('组件与实验组','点数、阶段与持久化')
    base=base.replace('各组训练配方和预算不同；表内统一评价道路层。','同一资产按阶段评价道路层。')
    base=base.replace('分辨率变化、教师变化和重新优化同时发生时，不将总改善全部归因于分辨率。','新增表达空间、覆盖训练与外观拟合分别验收。')
    base=base.replace('<table id="metrics"></table></div>','<table id="metrics"></table></div><details><summary>本轮新覆盖区域的最终外观</summary><div class="scroll"><table id="newly-covered"></table></div></details>')
    base=base.replace('<p id="gap-note" class="muted"></p>','<p id="gap-note" class="muted"></p><details><summary>二维编辑域与认证道路表面</summary><div id="geometry-view"></div><p class="muted">颜色含义与认证门见下方几何图例。未认证编辑域不等同于已确认的道路缺口。</p></details>')
    base=base.replace('各方法沿用预先固定配方，不搜索步数、学习率或损失权重。历史阶段快照、合成控制与TRELLIS附加参考不重复训练。像素读取仍依赖世界场与读取算子；普通高斯使用足迹平均RGB，两者明确标记。','本轮新增普通道路高斯，位置与法向来自固定道路表面。覆盖阶段只训练新启用与新建点的不透明度和切向尺度；外观阶段冻结覆盖，用实际高斯渲染反传颜色。未增加扩散模型或生成先验训练。')
    base=base.replace('几何、覆盖参数及其他场景分支按协议冻结。训练域覆盖范围、保留车辆与持久化重载独立核验；整体画质没有自动晋升。','892个历史新增活动点是原有锚点激活，不计入本轮新建几何。启用旧锚点与追加新行分别计数；保留车辆保护仅对已核验的五帧作结论。世界表面与持久化重载独立核验，整体画质不自动晋升。')
    js=(WORK/'viewer.js').read_text('utf-8').replace('__DATA__',json.dumps(data,ensure_ascii=False,allow_nan=False).replace('</',r'<\/'))
    F.write(F.DEST/'report_data.json',data)
    shutil.copy2(WORK/'architecture_surface.svg',F.OUT/'architecture_surface.svg')
    tmp=F.OUT/'surface.tmp.html';tmp.write_text(base+'<script>'+js+'</script></html>','utf-8');os.replace(tmp,F.OUT/'index.html')
    return dict(models=len(models),metric_rows=len(data['rows']),gap_rows=len(data['gaps']),stages=stages,index_sha256=F.sha(F.OUT/'index.html')if hasattr(F,'sha')else F.hashlib.sha256((F.OUT/'index.html').read_bytes()).hexdigest())

F.install=install
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');p.add_argument('--mark-pending',action='store_true');a=p.parse_args()
    if a.mark_pending:
        history=F.OUT/'history_common1152.html'
        if not history.exists():shutil.copy2(F.OUT/'index.html',history)
        banner='<section style="border:1px solid #77d6be;background:#17352f;padding:16px;border-radius:10px"><strong>覆盖驱动的道路局部增密已安排</strong><p>先启用旧锚点，再沿固定路面补点；覆盖训练后拟合冻结教师。下方保留已完成的1152教师训练结果。</p><a href="surface_adaptive/run_status.json">新队列状态</a></section>'
        (F.OUT/'index.html').write_text(history.read_text('utf-8').replace('<main>','<main>'+banner,1),'utf-8')
    else:F.main(a.once)
