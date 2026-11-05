"""Vehicle material validation report, replay assets and clean model inputs."""
import json,hashlib,shutil,tarfile,gzip,csv,html
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
RUN=Path(__file__).resolve().parent.parent;DEST=RUN/'delivery'
METHODS=('median','child_nearest','child_graph','parent_nearest','parent_graph')
LABELS={'median':'默认中位色','child_nearest':'细点最近邻','child_graph':'细点调和','parent_nearest':'父面最近邻','parent_graph':'父面调和'}
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def copy(s,d):Path(d).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(s,d)

def main():
    for name in ('input200_eligibility_audit.json','input_admission_contract.json','evaluation_review.json','review_final_implementation.json','appearance_input_QA.json'):
        assert (RUN/'reviews'/name).is_file(),name
    manifests=read(RUN/'appearance_packages/manifest.json')
    for case,record in manifests.items():assert sha(RUN/'appearance_packages'/f'appearance_inputs_{case}.tar.gz')==record['sha256']
    DEST.mkdir(exist_ok=False);decision=read(RUN/'decision.json');pseudo=read(RUN/'pseudo_evaluation/summary.json');real=read(RUN/'real_evaluation/summary.json')
    assert decision['current_M2_retained'] and not decision['new_scene_adopted']
    for name in ('decision.json','evaluation_contract.json'):copy(RUN/name,DEST/name)
    for source,name in ((RUN/'pseudo_evaluation/summary.json','pseudo_summary.json'),(RUN/'real_evaluation/summary.json','real_summary.json'),
       (RUN/'real/receipt.json','extension_receipt.json'),(RUN/'formal/receipt.json','formal_receipt.json'),(RUN/'smoke/receipt.json','smoke_receipt.json'),
       (RUN/'cases/contract.json','case_contract.json'),(RUN/'cases/preparation.json','case_preparation.json'),(RUN/'data/collection.json','operator_collection.json')):
        copy(source,DEST/'evidence'/name)
    copy(RUN/'real_evaluation/metrics.json.gz',DEST/'evidence/real_metrics.json.gz')
    for p in (RUN/'reviews').glob('*'):
        if p.is_file():copy(p,DEST/'evidence'/p.name)
    for p in (RUN/'source').glob('*.py'):copy(p,DEST/'source'/p.name)
    OLD=RUN.parent/'scene008_a6_asset_controls_20261105'
    copy(OLD/'source/sh_asset.py',DEST/'source/sh_asset.py')
    copy(RUN.parent/'scene008_a6_complete_asset_20261105/source/full_pvd.py',DEST/'source/full_pvd.py')
    for name in METHODS:copy(RUN/'real'/f'{name}_asset.npz',DEST/'candidates'/f'{name}_asset.npz')
    for p in (RUN/'pseudo_evaluation').glob('*.jpg'):copy(p,DEST/'images/pseudo'/p.name)
    for p in (RUN/'real_evaluation/frames').glob('*.jpg'):copy(p,DEST/'images/real'/p.name)
    for p in (RUN/'appearance_packages').iterdir():
        if p.is_file():copy(p,DEST/p.name)
    copy(RUN/'appearance_inputs_v2/contract.json',DEST/'appearance_input_contract.json')
    with tarfile.open(DEST/'pseudo_replay.tar.gz','w:gz') as t:
        for case in ('left','right','top'):
            for p in (RUN/'formal'/case/'extensions').glob('*'):
                t.add(p,arcname=f'{case}/extensions/{p.name}')
            for p in (RUN/'cases'/case).glob('*.npz'):
                if p.name!='train_design.npz':t.add(p,arcname=f'{case}/inputs/{p.name}')
        # Eval-only rows are kept in a separate archive entry, not model inputs.
        t.add(RUN/'data/sealed_targets.npz',arcname='evaluation_only/sealed_targets.npz')
    with tarfile.open(DEST/'evidence/executed_source_and_logs.tar.gz','w:gz') as t:
        for folder in ('source','reviews'):
            for p in (RUN/folder).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts:t.add(p,arcname=str(p.relative_to(RUN)))
        for p in RUN.glob('*.log'):t.add(p,arcname='logs/'+p.name)
    # Archive raw review bytes first; normalize published text only.
    lf=[]
    for p in DEST.rglob('*'):
        if p.is_file() and p.suffix in ('.json','.py','.html','.svg','.md'):
            raw=p.read_bytes();fixed=raw.replace(b'\r\n',b'\n')
            if raw!=fixed:
                lf.append(dict(path=str(p.relative_to(DEST)),original_sha256=hashlib.sha256(raw).hexdigest(),published_sha256=hashlib.sha256(fixed).hexdigest()));p.write_bytes(fixed)
    write(DEST/'evidence/text_publication.json',dict(LF_only=lf,original_reviews_retained_in_executed_archive=True))
    x=np.arange(3);width=.15;fig,ax=plt.subplots(figsize=(9.5,4.5),facecolor='#f4f7f9');ax.set_facecolor('#f4f7f9')
    for j,name in enumerate(METHODS):
        values=[next(m for m in c['methods'] if m['method']==name)['HELD_spatial_withheld']['mean'] for c in pseudo['cases']]
        ax.bar(x+(j-2)*width,values,width,label=name)
    ax.set_xticks(x,[c['case'] for c in pseudo['cases']]);ax.set_ylabel('Withheld observed RGB MAE (lower is better)');ax.legend(fontsize=8,ncol=3);ax.set_ylim(0,.30);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(DEST/'images/pseudo_errors.png',dpi=160);plt.close(fig)
    architecture='''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="310" viewBox="0 0 1200 310"><defs><marker id="a" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" fill="#3a6178"/></marker></defs><rect width="1200" height="310" fill="#f3f7f9"/><g fill="none" stroke="#3a6178" stroke-width="2" marker-end="url(#a)"><path d="M190 70H245"/><path d="M440 70H500"/><path d="M690 70H745"/><path d="M940 70H990"/><path d="M590 110V210H745"/><path d="M940 245H990V110"/></g><g fill="white" stroke="#6c8ea2"><rect x="15" y="28" width="175" height="85" rx="8"/><rect x="245" y="28" width="195" height="85" rx="8"/><rect x="500" y="28" width="190" height="85" rx="8"/><rect x="745" y="28" width="195" height="85" rx="8"/><rect x="990" y="28" width="195" height="85" rx="8"/><rect x="745" y="200" width="195" height="85" rx="8"/></g><g font-family="Microsoft YaHei,Arial" font-size="16" text-anchor="middle" fill="#173a51"><text x="102" y="61">固定 M2 几何<tspan x="102" dy="25">真实可见 RGB</tspan></text><text x="342" y="61">左 / 右 / 顶伪遮挡<tspan x="342" dy="25">跨帧严格删除监督</tspan></text><text x="595" y="61">保留区共同外观拟合<tspan x="595" dy="25">冻结有监督颜色</tspan></text><text x="842" y="61">默认色 / 最近邻 / 调和<tspan x="842" dy="25">细点图与父面图</tspan></text><text x="1087" y="61">密封真值验证<tspan x="1087" dy="25">49 帧保留 / 删除检查</tspan></text><text x="842" y="232">明确未知掩码<tspan x="842" dy="24">外观模型输入包</tspan></text><text x="313" y="187" font-size="14">原始 200 点资格审计：不改 PVD，不重训，不改道路</text><text x="1087" y="171" font-size="14">未通过质量门：保留 M2</text><text x="1055" y="260" font-size="14">下一阶段：显式外观先验</text></g></svg>'''
    (DEST/'architecture.svg').write_text(architecture,encoding='utf-8')
    table=[];csvrows=[]
    for name in METHODS:
        r=decision['methods'][name];reaches=read(RUN/'real/receipt.json')['methods'][METHODS.index(name)]['extension']['propagated']
        table.append(f"<tr><td>{LABELS[name]}</td><td>{r['HELD_macro_MAE']:.5f}</td><td>{r['absolute_gain']:+.5f}</td><td>{reaches:,}/10,557</td><td>{'参照' if name=='median' else '未通过'}</td><td>{'通过' if r['preservation_pass'] else '未通过'}</td></tr>")
        csvrows.append(dict(method=name,HELD_macro_MAE=r['HELD_macro_MAE'],gain_vs_median=r['absolute_gain'],propagated=reaches,quality_pass=r['quality_pass'],preservation_pass=r['preservation_pass']))
    with (DEST/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    case_rows=[]
    for c in read(RUN/'evaluation_contract.json')['cases']:
        case_rows.append(f"<tr><td>{c['case']}</td><td>{c['train_pixels']:,}</td><td>{c['BUILD_test_pixels']:,}</td><td>{c['HELD_test_pixels']:,}</td><td>{c['status']}</td></tr>")
    eligibility=read(RUN/'reviews/input200_eligibility_audit.json')
    # Numeric audit is linked verbatim; plain prose stays within verified results.
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scene008 车辆外观恢复：严格伪遮挡验证</title>
<style>body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;margin:0;background:#f4f7f9;color:#173247}}main{{max-width:1280px;padding:28px 24px 70px;margin:auto}}h1{{font-size:29px}}h2{{font-size:22px;margin-top:34px}}p,li{{line-height:1.75}}.card{{padding:18px 22px;background:white;border:1px solid #d7e2e8;border-radius:9px;margin:17px 0}}.accent{{border-left:5px solid #087c85}}table{{width:100%;border-collapse:collapse;background:white}}td,th{{padding:11px;border:1px solid #dce5eb;text-align:left;font-size:14px}}th{{background:#eaf1f5}}img{{max-width:100%;height:auto}}.scroll{{overflow-x:auto}}.wide{{min-width:1500px}}a{{color:#086b81}}button,select{{font:inherit;border:1px solid #b4c9d6;border-radius:5px;padding:6px 12px;background:white;margin-right:6px}}.muted{{color:#567080}}.tag{{font-size:13px;background:#ddefef;padding:4px 9px;border-radius:5px}}</style>
<main><span class="tag">2026-11-05 · 车辆分支 · 几何固定 · 未训练 PVD</span><h1>颜色能沿 PVD 车身传播好吗？</h1>
<div class="card accent"><b>本轮结论：四种颜色延拓候选均未通过质量门，当前 M2 保留。</b><p>把原本可见的车身分别从左、右、顶部遮掉，所有方法使用同一份剩余观测拟合。外观误差没有稳定优于统一中位色。增加表面连通性扩大了能传播颜色的范围，却没有提高恢复正确性。</p><p>49 帧成像检查中，几何、透明度、深度、分组贡献均逐位不变，已有车身、道路与其他车辆的均值保护门全部通过。这说明可以做到不破坏现有监督，但“能安全改色”仍不等于“补出了正确材质”。没有把这些候选接管正式场景。</p></div>
<img src="architecture.svg" alt="固定几何、严格伪遮挡、共同外观拟合、五种延拓、密封评价和外观输入包架构">
<h2>统一对照结果</h2><p>主指标是在 9 个内部留出时刻，对被人为遮去且原本真实可见的区域计算 RGB MAE；left/right/top 三种情况等权平均，越低越好。所有不可达区域仍保留在分母中。</p>
<table><tr><th>规则</th><th>留出区 MAE ↓</th><th>相对默认色的改善 ↑</th><th>真实缺测点可延拓数</th><th>颜色质量门</th><th>49 帧保护门</th></tr>{''.join(table)}</table>
<p class="muted">“最近邻”是同一表面图上的测地最近邻。调和与最近邻比较的是传播规则；父面与细点比较的是离散尺度及邻接构造。父面规则使用更粗的几何间距，不能称其物理邻域或拓扑保证与细点完全相同。默认色是参照，不参与“优于自身”的通过判定。</p>
<img src="images/pseudo_errors.png" alt="左、右、顶三种伪遮挡的五规则留出RGB误差">
<h2>查看伪遮挡后的完整车身</h2><p><select id="case" onchange="showPseudo()"><option value="left">遮左侧</option><option value="right">遮右侧</option><option value="top">遮顶部</option></select><select id="raw" onchange="showPseudo()"><option>105</option><option selected>113</option><option>130</option></select><span class="muted">第一列橙色轮廓标出用于评价的已知区域。</span></p><div class="scroll"><img class="wide" id="pseudoImage" src="images/pseudo/left_113.jpg" alt="真实可见车身与五种伪遮挡颜色恢复"></div>
<h2>为什么这项验证没有偷看答案</h2><ul><li>遮挡区域只由车体坐标和 BUILD 中的几何投影贡献决定，不用 RGB 定义区域。</li><li>任何像素只要含有隐藏点的一丁点非零贡献，就从全部训练帧中移除；没有贡献阈值豁免。</li><li>每种遮挡重新用剩余原始 RGB 计算中位色与初值，只拟合一份共同颜色。没有使用已拟合 M2、原有 prior 或 native 预测填入被遮答案。</li><li>延拓只接收有训练贡献点的颜色，只能修改严格零列。导出重载后训练预测仍逐位一致。</li><li>评价使用同一批固定像素，混合边界另报，不删掉图上不可达的难例。</li></ul>
<table><tr><th>情况</th><th>保留训练像素</th><th>BUILD 遮去评价像素</th><th>HELD 遮去评价像素</th><th>样本数量门</th></tr>{''.join(case_rows)}</table>
<p>数值验收门来自独立审查，在正式 2000 步拟合及任何质量分数读取之前冻结；其时已完成仅检查工程接线的 32 步 smoke，时间边界如实记入 <a href="evaluation_contract.json">评价合同</a>。本轮是同场景、固定形状的材质控制，不是隐藏几何或跨场景泛化实验。</p>
<h2>真实缺测区与街景接入</h2><p><input id="frame" type="range" min="89" max="137" value="113" oninput="showReal()"><span id="frameLabel">raw113</span><button onclick="setReal(105)">105</button><button onclick="setReal(113)">113</button><button onclick="setReal(130)">130</button></p><div class="scroll"><img class="wide" id="realImage" src="images/real/113_objects.jpg" alt="49帧真实缺测外观延拓对照"></div>
<p class="muted">这里保留原 M2 的所有有监督颜色，只修改其 10,557 个零 RGB 列。该图没有隐藏车身真值，不能用主观显得更丰富或传播点更多来认定恢复准确。</p>
<p><select id="streetMethod" onchange="showStreet()"><option value="child_graph">细点调和</option><option value="parent_graph" selected>父面调和</option><option value="child_nearest">细点最近邻</option><option value="parent_nearest">父面最近邻</option></select><select id="streetRaw" onchange="showStreet()"><option>105</option><option selected>113</option><option>130</option></select></p><div class="scroll"><img class="wide" id="streetImage" src="images/real/113_parent_graph_street.jpg" alt="同一资产保留a37与删除a37的街景对照"></div>
<h2>输入点资格：未发现大规模错归属</h2><p>原 200 个点全部能精确对应真实 LiDAR 来源行。逐点检查未发现落入其他标注车辆实体盒的点；按固定 5cm 邻域、至少 3 个不同时刻和至少 0.5 秒跨度的几何自洽规则，199 个通过，1 个仍不确定。排除原有 14 个 HELD 来源点后，未来干净准入为 185 个。新建的 5,399 个候选点保留来源与拒绝理由，全部仍标为语义未知；几何自洽不是逐点语义真值。</p><p>这轮没有把它们重新送进 PVD，避免把几何输入变化混入颜色实验。<a href="evidence/input200_eligibility_audit.json">完整审计</a> · <a href="evidence/input_admission_contract.json">统一准入合同</a>。</p>
<h2>下一阶段的外观输入接口已经整理</h2><p>独立条件包：<a href="appearance_inputs_full.tar.gz">真实缺测 full</a> · <a href="appearance_inputs_left.tar.gz">伪遮挡 left</a> · <a href="appearance_inputs_right.tar.gz">伪遮挡 right</a> · <a href="appearance_inputs_top.tar.gz">伪遮挡 top</a>。每个包仅含本条件的 40 个 BUILD 视图、自己的合同与公共几何，不含其他条件的 RGB。这里是同一批 40 个 FRONT 时刻在四种掩码下的输入，不是 160 个独立观察角度。</p><p>包内提供真实已知 RGB、显式未知掩码、相机、车姿及固定几何提案，没有 HELD RGB、被遮答案或 M2 已拟合纹理。RGBA 透明处表示未知；NPZ float32 RGB 为权威数据，PNG 仅为量化预览。另存裁剪后内参、米制 object-to-camera 和 meter_to_model；形状覆盖与深度明确标为先验预测，不能当作实测真值。</p><p>这使下一步可以在同一份固定基准上接入显式外观先验，先检验预训练推理是否改善，再判断是否需要后训练。当前结果不支持继续调图邻域来声称补出了尾灯、车牌或轮胎纹理，也不支持把颜色问题归因于只输出 XYZ 的 PVD 权重。</p>
<p><a href="decision.json">统一验收判定</a> · <a href="comparison.csv">比较表</a> · <a href="evidence/pseudo_summary.json">伪遮挡指标</a> · <a href="evidence/real_summary.json">49 帧固定域指标</a> · <a href="appearance_input_contract.json">外观条件合同</a> · <a href="pseudo_replay.tar.gz">伪遮挡重放资产与独立评价标签</a></p>
<p>独立审查：<a href="evidence/evaluation_review.json">评价与泄漏</a> · <a href="evidence/review_final_implementation.json">最终实现</a> · <a href="evidence/appearance_input_QA.json">外观输入包</a>。候选资产均仅供复现：{''.join(f'<a href="candidates/{name}_asset.npz">{name}</a> · ' for name in METHODS)}</p>
<small class="muted">6000 个正式数值颜色优化步骤；额外 96 步 smoke 与 6 次 next-step 重放单列。未训练神经网络，未更改 PVD、几何、道路或原始检查点；AutoDL 保持开机。</small>
</main><script>function showPseudo(){{document.getElementById('pseudoImage').src='images/pseudo/'+document.getElementById('case').value+'_'+document.getElementById('raw').value+'.jpg';}}function showReal(){{let r=document.getElementById('frame').value;document.getElementById('frameLabel').textContent='raw'+r;document.getElementById('realImage').src='images/real/'+String(r).padStart(3,'0')+'_objects.jpg';}}function setReal(r){{document.getElementById('frame').value=r;showReal();}}function showStreet(){{document.getElementById('streetImage').src='images/real/'+document.getElementById('streetRaw').value+'_'+document.getElementById('streetMethod').value+'_street.jpg';}}</script></html>'''
    (DEST/'index.html').write_text(doc,encoding='utf-8')
    qa=dict(status='PASS_BUILD',real_frames=49,pseudo_cases=3,methods=5,formal_colour_steps=6000,extra_smoke_steps=96,extra_nextstep_replays=6,
        neural_training_steps=0,new_scene_adopted=False,current_M2_retained=True,
        full49_geometry_alpha_depth_groups_bitwise_unchanged=True,original_scene_road_environment_unchanged=True,
        case_frame_inputs=160,distinct_camera_views=40,independent_input_archives=4,RGBA_unknown_is_not_black_background=True,
        published_text_LF=True,raw_source_reviews_archived=True,AutoDL='ON')
    for p in DEST.rglob('*'):
        if p.suffix.lower() in ('.png','.jpg'):
            with Image.open(p) as im:im.verify()
    write(DEST/'evidence/delivery_QA.json',qa)
    manifest={str(p.relative_to(DEST)).replace('\\','/'):sha(p) for p in DEST.rglob('*') if p.is_file()}
    write(RUN/'publication_manifest.json',manifest)
    with tarfile.open(RUN/'delivery.tar.gz','w:gz') as t:t.add(DEST,arcname='delivery')
    (RUN/'delivery.tar.gz.sha256').write_text(sha(RUN/'delivery.tar.gz')+'  delivery.tar.gz\n')
    print(json.dumps(dict(files=len(manifest),archive_bytes=(RUN/'delivery.tar.gz').stat().st_size,archive_sha256=sha(RUN/'delivery.tar.gz'))))

if __name__=='__main__':main()
