"""Publish the vehicle-only I/O audit, with frozen source provenance."""
import json,hashlib,shutil,tarfile,html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

RUN=Path(__file__).resolve().parent.parent
CAM=RUN.parent;OLD=CAM/'scene008_a6_asset_controls_20261105'
PVD=CAM/'scene008_a37_C2_20261103_r1/source/prior/proposals/pvd_seed000'
DEST=RUN/'delivery';PROBE=RUN/'geometry_probes_v3'
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def copy(src,dst):Path(dst).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)

def main():
    DEST.mkdir(exist_ok=False)
    audit=read(RUN/'audit/current_audit.json');probe=read(PROBE/'summary.json');review=RUN/'reviews'
    smoke=read(PROBE/'smoke_receipt.json');oldsmoke=read(PROBE/'smoke_receipt_before_gate_fix.json')
    for name in ('geometry_io_review.json','appearance_io_review.json','observation_io_review.json','root_probe_review.json','review_gate_fix.json'):
        copy(review/name,DEST/'evidence'/name)
    for name in ('current_audit.json','appearance_provenance.npz'):
        copy(RUN/'audit'/name,DEST/'evidence'/name)
    for name in ('contract.json','smoke_receipt.json','summary.json'):
        copy(PROBE/name,DEST/'evidence'/('geometry_'+name))
    copy(PROBE/'smoke_receipt_before_gate_fix.json',DEST/'evidence/geometry_smoke_receipt_formal.json')
    for name in ('io_contract.json','receipt.json','m2_with_io_contract.npz'):
        copy(RUN/'contract_asset'/name,DEST/'asset'/name)
    for raw in (105,113,130):copy(RUN/'audit/frames'/f'{raw:03d}_provenance.png',DEST/'images'/f'{raw:03d}_provenance.png')
    for arm in ('R0','R0_repeat','RY','RV'):
        for name in ('input.npz','output.npz','sampling_snapshots.npz','receipt.json'):
            copy(PROBE/arm/name,DEST/'geometry'/arm/name)
    # Original uncoloured geometry is included so the point model output is
    # inspectable before any material assignment or Gaussianization.
    copy(PVD/'candidate_points_actor_native.npz',DEST/'geometry/original_PVD_xyz.npz')
    copy(PVD/'fixed_partial_input.npz',DEST/'geometry/original_200_input.npz')
    copy(PVD/'transform_contract.json',DEST/'geometry/original_transform_contract.json')
    full=np.load(CAM/'scene008_a37_C2_20261103_r1/inputs/a6_geometry_minimal.npz')['canonical_xyz']
    fig,axes=plt.subplots(3,4,figsize=(16,8.3),facecolor='#f6f8fa')
    view=((1,2,'Rear plane: y / z'),(0,2,'Side plane: x / z'),(0,1,'Top plane: x / y'))
    for c,arm in enumerate(('R0','R0_repeat','RY','RV')):
        data=dict(np.load(PROBE/arm/'output.npz'));points=data['points_actor_native_m']
        for r,(x,y,label) in enumerate(view):
            ax=axes[r,c];ax.set_facecolor('#f6f8fa')
            ax.scatter(full[::5,x],full[::5,y],s=.4,c='#9aa2ac',alpha=.22,rasterized=True)
            ax.scatter(points[200:,x],points[200:,y],s=1.7,c='#d56a24',alpha=.7,rasterized=True)
            ax.scatter(points[:200,x],points[:200,y],s=4,c='#008f93',alpha=.9,rasterized=True)
            ax.set_aspect('equal');ax.set_xlim(((-1.25,1.25) if x==1 else (-2.7,2.7)))
            ax.set_ylim(((-1.05,1.05) if y==2 else (-1.3,1.3)))
            if r==0:ax.set_title(arm,fontsize=12)
            ax.set_xlabel(label+' [m]',fontsize=8);ax.tick_params(labelsize=7);ax.grid(alpha=.18)
    fig.suptitle('Frozen PVD geometry only: teal = 200 inputs, orange = 1848 proposals, grey = partial real observations',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(DEST/'images/geometry_io_probes.png',dpi=140);plt.close(fig)
    architecture='''<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="300" viewBox="0 0 1100 300"><defs><marker id="a" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" fill="#41647b"/></marker></defs><rect width="1100" height="300" fill="#f4f8fa"/><g stroke="#41647b" stroke-width="2" fill="none" marker-end="url(#a)"><path d="M175 75 H225"/><path d="M410 75 H455"/><path d="M635 75 H680"/><path d="M870 75 H920 V175 H900"/><path d="M175 230 H455"/><path d="M635 230 H680"/><path d="M870 230 H920 V175"/><path d="M790 110 V195"/></g><g fill="#fff" stroke="#7190a2"><rect x="15" y="38" width="160" height="74" rx="8"/><rect x="225" y="38" width="185" height="74" rx="8"/><rect x="455" y="38" width="180" height="74" rx="8"/><rect x="680" y="38" width="190" height="74" rx="8"/><rect x="15" y="194" width="160" height="74" rx="8"/><rect x="455" y="194" width="180" height="74" rx="8"/><rect x="680" y="194" width="190" height="74" rx="8"/><rect x="920" y="133" width="165" height="84" rx="8"/></g><g font-family="Microsoft YaHei,Arial" font-size="16" text-anchor="middle" fill="#17354b"><text x="95" y="68">实测 LiDAR<tspan x="95" dy="24">200 个 XYZ</tspan></text><text x="317" y="68">固定坐标与归一化<tspan x="317" dy="24">完整车框 / 来源审计</tspan></text><text x="545" y="68">冻结 PVD<tspan x="545" dy="24">只输出 2048 XYZ</tspan></text><text x="775" y="68">几何转为高斯<tspan x="775" dy="24">法向 / 尺度 / 覆盖</tspan></text><text x="95" y="223">真实可见 RGB<tspan x="95" dy="24">native 仅作参考</tspan></text><text x="545" y="223">已观测外观拟合<tspan x="545" dy="24">颜色通道与可见性审计</tspan></text><text x="775" y="223">同一份完整 a6<tspan x="775" dy="24">未知色 = 标记的回退值</tspan></text><text x="1002" y="164">统一排序与成像<tspan x="1002" dy="25">保留 / 删除 a37</tspan></text><text x="323" y="216" font-size="13">PVD 没有接收图像或生成纹理</text></g></svg>'''
    architecture=architecture.replace('<path d="M870 75 H920 V175 H900"/>','').replace('<path d="M870 230 H920 V175"/>','<path d="M870 230 H900 V175 H920"/>')
    (DEST/'architecture.svg').write_text(architecture,encoding='utf-8')
    readiness=dict(status='NOT_READY_TO_ATTRIBUTE_ALL_REMAINING_ERRORS_TO_MODEL',model_training_started=False,
        frozen_reference_M2_sha256=audit['asset_sha256'],road_branch_started=False,
        established=['Mechanical coordinate/unit/normalization roundtrip and output row preservation pass.',
          'Raw RGB/PNG/observation values and BGR/unit/camera/time/material decoding checks pass.',
          '10557 zero RGB columns remain exactly at global fallback; null-space intervention leaves BUILD predictions exact.',
          'PVD is geometry-only; missing texture completion is a missing material capability, not a colour-channel failure.'],
        unresolved=['Semantic/temporal qualification of raw multi-frame OBB conditioning versus a single visible partial scan.',
          'Car-front convention and full-box statistical adaptation, without choosing by incomplete surface score.',
          'Point-to-surface Gaussianization can change visible support and blur; shape-only output is not a textured surface.',
          'Hidden material/geometry lack independent truth; known-surface pseudo-occlusion tests are needed before model-training attribution.'],
        next_order=['Keep current M2 geometry/material and physical scene replacement as fixed reference.',
          'Add or audit object-specific RGB masks/measurement visibility; unreviewed box hints stay candidates.',
          'Run whole-object known-surface pseudo-occlusion checks across directions, separating geometry and appearance targets.',
          'Test an explicit appearance completion prior while preserving measured material; PVD XYZ post-training alone cannot add an RGB output.',
          'Only then design model post-training; pre-training requires a separate data/compute contract.'],
        no_adoption_from_geometry_probes=True)
    write(DEST/'training_readiness.json',readiness)
    rows=[]
    for row in probe['arms']:
        d=row['partial_only_distance_NOT_hidden_accuracy'];st=row.get('symmetric_generated_set_distance_vs_R0_m')
        rows.append(f"<tr><td>{row['arm']}</td><td>{d['mean_m']:.6f}</td><td>{d['p90_m']:.6f}</td><td>{'—' if st is None else f'{st:.6f}'}</td><td>{row['below_real_lowest_z']}</td></tr>")
    zero=audit['zero_column_points'];lower=audit['proposal_below_real199_lowest_z'];lz=audit['proposal_below_real199_lowest_z_zero_columns']
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scene008 车辆先验：输入输出归因</title>
<style>body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;background:#f4f7f9;color:#183247;margin:0}}main{{max-width:1280px;margin:auto;padding:30px 28px 60px}}h1{{font-size:29px;margin-bottom:12px}}h2{{font-size:22px;margin-top:34px}}p,li{{line-height:1.75}}.card{{background:white;border:1px solid #d7e2e8;border-radius:10px;padding:18px 22px;margin:16px 0}}.accent{{border-left:5px solid #07878c}}table{{border-collapse:collapse;width:100%;background:#fff}}th,td{{border:1px solid #dbe4ea;text-align:left;padding:10px;font-size:14px}}th{{background:#eaf1f5}}img{{max-width:100%;height:auto}}.scroll{{overflow-x:auto;background:white}}#provenance{{min-width:1350px}}button{{font:inherit;padding:7px 18px;margin-right:8px;border:1px solid #b7cbd8;background:white;border-radius:5px;cursor:pointer}}a{{color:#07667e}}small,.muted{{color:#526c7e}}code{{font-size:13px}}.tag{{font-size:13px;background:#def0ef;padding:4px 9px;border-radius:6px}}</style>
<main><span class="tag">2026-11-05 · 车辆分支 · M2 固定 · 未训练模型</span><h1>下半车身为什么呈现整块灰白？</h1>
<div class="card accent"><b>主因已经定位：当前没有隐藏车身纹理补全模型。</b><p>PVD 只输出 XYZ。后续颜色拟合没有观测到的点，被赋予同一个蓝灰回退色。当前 M2 的 {zero:,}/18,432 个基元在 RGB 设计矩阵中是严格零列，最终颜色全部等于回退值 <code>[0.3392, 0.4157, 0.4892]</code>。这解释了大块平色；它不是 PVD 生成的材质。</p><p>这轮核对了输入输出链，保留当前车辆资产。没有用提亮、前车颜色或未经确认的道路像素修饰未知车身，也没有启动后训练。</p></div>
<img src="architecture.svg" alt="从点云几何先验和真实可见RGB到统一车辆资产的数据流">
<h2>直接看颜色来自哪里</h2><p>品红是临时诊断色。仅改动没有 RGB 拟合支持的点，27,429 个 BUILD 拟合像素的预测逐位不变。灰白下半部因此可以大幅改色，而现有数据项完全无法区分。前三类贡献来自同一次深度排序，守恒误差 ≤{audit['group_conservation_maxabs']:.2g}。</p>
<p><button onclick="showFrame(105)">raw105</button><button onclick="showFrame(113)">raw113</button><button onclick="showFrame(130)">raw130</button><span id="frameLabel">raw113</span></p><div class="scroll"><img id="provenance" src="images/113_provenance.png" alt="真实观测、未改M2、品红诊断、颜色监督来源、低于实测最低点的贡献"></div>
<p class="muted">来源图：绿色 = 多帧且足够像素贡献的 RGB 拟合支持；黄色 = 弱支持；品红 = 完全无 RGB 拟合支持。它们说明损失来源，不等于逐点真实材质认证。最后一列白色只标出低于全部 199 帧关联测点最低高度的几何贡献；不是显露率或补全准确率。</p>
<div class="card">实测最低高度为车体坐标 z={audit['original_real199_lowest_point_z_m']:.5f}m。其下共有 {lower:,} 个提案基元，其中 {lz:,} 个（{100*lz/lower:.2f}%）没有 RGB 拟合支持。已有底部真值没有被这项统计假定出来。</div>
<h2>输入输出排查结果</h2><table><tr><th>环节</th><th>证据与结果</th><th>结论边界</th></tr>
<tr><td>原图 → 观测 RGB</td><td>独立解码 3 帧 TFRecord JPEG，与 prepared PNG 的最大 uint8 差为 0；观测 RGB 同样一致。40 帧拟合 target 重建最大差为 0。</td><td>排除已检查路径上的 BGR、重复 /255 与导出色偏。</td></tr>
<tr><td>坐标、单位、相机和时间</td><td>完整仿射互逆误差约 1e−16；原 200 点与输出固定前缀一致；2048 点反变换与发布 XYZ 一致。BUILD40 相机矩阵误差为 0。</td><td>机械链通过。车头方向语义与整框统计适配仍是需要验证的约定。</td></tr>
<tr><td>PVD 条件与输出</td><td>冻结官方 car_completion_3199，仅 XYZ；RGBA 图像与 confidence 字段未被当前 worker 消费。</td><td>图像没有进入 PVD 不是丢失 RGB 输出通道；模型本来就没有该接口。不能称它已具备纹理先验。</td></tr>
<tr><td>原始 200 点</td><td>从 54,715 条 OBB 关联返回中随机取点，横跨 123 帧；仅 7 点低于车体中心。与官方单视点合成深度的输入分布不同。</td><td>多帧关联并非精确表面真值；需区分输入资格与模型适配。</td></tr>
<tr><td>2048 点 → 高斯表面</td><td>法向、尺度、0.995 透明度及 0.003m 厚度均为下游适配约定；M2 再细化为 18,432 点。</td><td>高斯覆盖、连续性和材质缺失不能全部归因于扩散输出。</td></tr>
<tr><td>颜色优化与渲染</td><td>SH0 解码与保存 RGB 一致；设计矩阵行和≈1，目标函数复算一致，收敛残差≈1e−6。</td><td>无数据列继续迭代不会获得纹理。少量弱支持饱和白点是另一种局部问题。</td></tr></table>
<h2>199 帧是否漏掉了能给底部上色的观测？</h2><p>当前 49 帧之外，150 帧有 39,694 条关联测点。旧流程只留下 445 条稀疏颜色候选，均位于 z≈0.577–0.720m 的车顶；所有 199 帧关联点都没有进入 z&lt;−0.21045m 的缺失带。相机也几乎一直从车后同侧观察。增加这些候选不能直接恢复下半车身颜色。</p><p>外 49 帧的包围盒相交检查过于保守，可能压掉真实上部观测；但附近其他时间窗现有 body masks 属于其他车辆，没有漏接一套现成的、已审核的 a6 掩码。下一步应审核新的对象身份与可见性，再重读原始图像收集颜色。候选缓存里的零 RGB 代表无标签，不能用作黑色材质真值。</p>
<h2>几何输入的短对照</h2><p>R0 使用原 200 点；R0_repeat 同输入、同初始噪声和 RNG 重复；RY 只改绕车体中心的正规 yaw180 约定；RV 用 raw106 的 255 个确认可见实测点中固定随机选 200 点。每组冻结同一权重，1000 步推理，无颜色拟合、无模型训练、无资产采用。</p>
<table><tr><th>对照</th><th>实测部分 → 提案平均距离 /m</th><th>实测部分距离 P90 /m</th><th>与 R0 生成点集的对称距离 /m</th><th>低于实测最低点的提案数</th></tr>{''.join(rows)}</table>
<p class="muted">上述距离只检查已测部分的自洽性。真实点不覆盖下部，不能据此评价隐藏车身准确率，也不能按较好分数选择真实车头方向。RV 同时改变点源资格、时段、视点与采样覆盖，不能把差值全部解释为输入去噪。</p>
<img src="images/geometry_io_probes.png" alt="四组PVD输入输出在车体三个坐标平面上的点云对照">
<p>同一状态和 RNG 恢复的单步输出仍有 {oldsmoke['next_step_maxabs_normalized']:.2g}–{smoke['next_step_maxabs_normalized']:.2g} 的 normalized 差异；原始 PVCNN 含浮点 atomicAdd，未改其后端。完整 R0_repeat 与 R0 的生成点集对称距离为 0.00551m，RY、RV 分别为 0.13645m、0.09170m，说明输入约定会产生可见的几何敏感性，但不是正确性证明。精确采样重放未通过，所有试验只标为有限精度诊断；不能把小幅差值当成方法改进。两次重复也不足以估计统计分布。</p>
<h2>已落实的接口修正与下一步</h2><p>交付一份 <a href="asset/m2_with_io_contract.npz">附显式 I/O 合同的 M2</a>：XYZ、法向、尺度、透明度、颜色与原 M2 全部保持，增加逐基元 RGB 损失支持状态、支持帧数与来源。raw105/113/130 的黑底、保留、删车渲染与原资产逐位相同。它修正了来源表达，不冒充画质提升。</p><ol><li>先整理对象专属的可靠观测，继续检查几何输入的语义适配；不靠包围盒直接给隐藏面取色。</li><li>用本来可见的车身做方向无关的伪遮挡实验，分别验收几何恢复、颜色保持和未知外观延拓。</li><li>颜色需要显式外观补全能力。单纯继续训练只输出 XYZ 的 PVD，不会凭空出现尾灯、车牌或轮胎材质输出。先把外观接口与评价补齐，再决定后训练；预训练另立数据与算力合同。</li></ol>
<p><a href="training_readiness.json">训练前置门与未决问题</a> · <a href="asset/io_contract.json">资产 I/O 合同</a> · <a href="asset/receipt.json">渲染不变验收</a> · <a href="evidence/current_audit.json">49 帧贡献审计</a> · <a href="evidence/geometry_summary.json">四组几何探针</a> · <a href="evidence/geometry_contract.json">预注册合同</a></p>
<p>独立审查：<a href="evidence/appearance_io_review.json">外观链</a> · <a href="evidence/geometry_io_review.json">几何链</a> · <a href="evidence/observation_io_review.json">观测覆盖</a> · <a href="evidence/root_probe_review.json">探针实现</a> · <a href="evidence/review_gate_fix.json">状态门修复复核</a>。模型范围参见 <a href="https://github.com/alexzhou907/PVD">PVD 官方实现</a>与<a href="https://openaccess.thecvf.com/content/ICCV2021/html/Zhou_3D_Shape_Generation_and_Completion_Through_Point-Voxel_Diffusion_ICCV_2021_paper.html">原论文</a>。</p><small>当前 M2 继续作为研究参照；道路分支未启动。AutoDL 保持开机。所有旧现场、失败 smoke 和重放记录保留。</small>
</main><script>function showFrame(raw){{document.getElementById('provenance').src='images/'+String(raw).padStart(3,'0')+'_provenance.png';document.getElementById('frameLabel').textContent='raw'+raw;}}</script></html>'''
    (DEST/'index.html').write_text(doc,encoding='utf-8')
    # Retain byte-exact executed sources and failed attempts separately.
    with tarfile.open(DEST/'evidence/executed_source_and_logs.tar.gz','w:gz') as out:
        for folder in (RUN/'source',):
            for p in folder.rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts:out.add(p,arcname=str(p.relative_to(RUN)))
        for p in RUN.glob('*.log'):out.add(p,arcname='logs/'+p.name)
    for p in (RUN/'source').glob('*.py'):copy(p,DEST/'source'/p.name)
    copy(OLD/'source/sh_asset.py',DEST/'source/sh_asset.py')
    copy(CAM/'scene008_a6_complete_asset_20261105/source/full_pvd.py',DEST/'source/full_pvd.py')
    for p in DEST.rglob('*'):
        if p.suffix.lower() in ('.png','.jpg'):
            with Image.open(p) as im:im.verify()
    qa=dict(status='PASS_DELIVERY_BUILD',M2_unchanged=True,model_training_steps=0,formal_inference_steps=4000,
       successful_smoke_steps_extra=80,failed_smoke_steps_extra=34,
       before_inference_engineering_failures=['Missing ninja PATH in general env','Missing ninja PATH in PVD env before base env PATH added'],
       precision_smoke_failure='Earlier exact next-step equality gate failed after32+2 inference calls; preserved, not silently called exact.',
       geometry_sampling_exact_replay=False,assets_adopted_from_probes=0,AutoDL_power='ON')
    write(DEST/'evidence/delivery_QA.json',qa)
    manifest={str(p.relative_to(DEST)).replace('\\','/'):sha(p) for p in DEST.rglob('*') if p.is_file()}
    write(RUN/'publication_manifest.json',manifest)
    with tarfile.open(RUN/'delivery.tar.gz','w:gz') as t:t.add(DEST,arcname='delivery')
    (RUN/'delivery.tar.gz.sha256').write_text(sha(RUN/'delivery.tar.gz')+'  delivery.tar.gz\n')
    print(json.dumps(dict(files=len(manifest),archive_sha256=sha(RUN/'delivery.tar.gz'),path=str(DEST))))

if __name__=='__main__':main()
