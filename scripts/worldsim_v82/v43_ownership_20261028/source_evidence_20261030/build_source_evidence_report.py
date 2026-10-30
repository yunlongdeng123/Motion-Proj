"""来源/证据控制的实际运行报告；只复制原生输出，不修改实验图像。"""
from pathlib import Path
import csv
import json
import shutil
import html

BASE = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
OUT = BASE / 'outputs/V43_Ownership_20261027/gpu_v43/source_evidence_20261030'
SRC = WORK / 'source_control_r1/evidence'
VIS = WORK / 'visibility_control_r1'
NATIVE = WORK / 'evidence_reconstruction_r1/evidence'
M0 = BASE / 'outputs/V43_Ownership_20261027/gpu_v43/sim_mechanism_20261030/evidence'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p, d):
    Path(p).write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def copy(p, dest):
    target = OUT / dest
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, target)
    return dest


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = read(SRC/'completion_receipt.json')
    trace = read(SRC/'observed16_source_trace.json')
    visibility = read(VIS/'visibility_summary.json')
    copy(WORK/'SOURCE_EVIDENCE_CONTRACT.json', 'contract.json')
    for label, root in [('source', SRC), ('visibility', VIS), ('rich', NATIVE)]:
        if not root.exists():
            continue
        for path in root.rglob('*'):
            if path.is_file() and path.suffix in {'.json', '.csv', '.md'} and 'scene_loader' not in path.parts:
                copy(path, 'evidence/'+label+'/'+path.relative_to(root).as_posix())
    copy(WORK/'source_control_r1/LOSS_PATH_AUDIT.md', 'evidence/source/LOSS_PATH_AUDIT.md')
    for path in (SRC/'source').rglob('*.png'):
        copy(path, 'evidence/source/source/'+path.relative_to(SRC/'source').as_posix())
    copy(VIS/'visibility_control_overview.png', 'visibility.png')
    assets = {}
    frames = ['000000', '000008', '000015']
    cases = {'baseline_delete': '原 M1 删除后', 'remove_top_error_32': '移除误差加权贡献前32行',
             'remove_top_error_512': '移除同排序前512行', 'baseline_delete_env_white': '仅换统一白环境',
             'top128_color_only_reference_moment': '固定几何/alpha，仅改128行颜色（特权诊断）'}
    for frame in frames:
        ref = M0/'dataset_r2/privileged/references/delete/shadow_on/rgb'/(frame+'.png')
        assets['ref/'+frame] = copy(ref, 'images/ref/'+frame+'.png')
        for case in cases:
            for channel in ['final', 'white', 'C', 'fill']:
                p = SRC/'run_r2_color_maps'/('source_'+frame)/(case+'_'+channel+'.png')
                assets[frame+'/'+case+'/'+channel] = copy(p, 'images/'+frame+'/'+p.name)
    top_images = []
    for p in sorted((SRC/'run_r2_color_maps/source_000008').glob('top*_alphaT.png')):
        top_images.append(dict(src=copy(p, 'images/row_maps/'+p.name), title=p.stem))
    history_images = []
    for p in sorted((SRC/'training_history_r2/history').glob('*.png')):
        history_images.append(dict(src=copy(p, 'images/history/'+p.name), title=p.stem))
    write(OUT/'assets.json', assets)
    source_table = ''
    for row in receipt['frames']:
        values = {x['case']: x for x in row['cases']}
        source_table += '<tr><td>'+row['frame']+'</td><td>%.4f</td><td>%.2f</td>' % (row['retained_pre'], row['retained_post'])
        for key in ['baseline_delete', 'remove_top_error_32', 'remove_top_error_512', 'top128_color_only_reference_moment']:
            source_table += '<td>%.5f</td>' % values[key]['body_clamped_mae']
        source_table += '</tr>'
    vis_table = ''
    for frame, arm in visibility['anchors'].items():
        old, new = arm['original'], arm['hybrid8_z3.5']
        vis_table += '<tr><td>'+frame+'</td><td>%d</td><td>%d</td><td>%.2f%%</td><td>%.2f°</td><td>%.2f%%</td><td>%.2f°</td></tr>' % (
            old['pixels'], old['observed_0_frames'], old['observed_1plus_fraction']*100,
            old['baseline_angle_2plus_deg_quantiles_10_50_90'][1], new['observed_1plus_fraction']*100,
            new['baseline_angle_2plus_deg_quantiles_10_50_90'][1])
    row_table = ''
    for row in trace['top8']:
        row_table += '<tr><td>%d</td><td>%.6f</td><td>%.6f</td><td>%.6f</td><td>%d</td></tr>' % (
            row['row_id'], row['sum_total'], row['sum_vehicle'], row['sum_road'], row['frames_total_gt1e4'])
    allrow = read(SRC/'allrow_observed16_r1/weighted_visibility_bins.json')
    bin_table = ''.join('<tr><td>'+html.escape(r['bin'])+'</td><td>%.2f%%</td><td>%.2f%%</td></tr>' %
                       (100*r['postdelete_body_alphaT_fraction'], 100*r['shared_pixel_error_weighted_alphaT_fraction'])
                       for r in allrow['bins'])
    rich_section = '<p>更充分视角已通过世界道路可见性预检；新的同预算原生重建执行记录将单列，当前不凭预检宣称编辑改善。</p>'
    extra = OUT/'rich_section.html'
    if extra.exists():
        rich_section = extra.read_text(encoding='utf8')
    architecture = '''<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="480" viewBox="0 0 1400 480"><defs><marker id="a" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0L9,3L0,6" fill="#54788b"/></marker></defs><style>text{font-family:Microsoft YaHei,Arial;font-size:19px;fill:#18384a}.title{font-size:22px;font-weight:bold}.box{fill:#e9f2f8;stroke:#a2bbca;stroke-width:1.5}.green{fill:#e2f1e9;stroke:#a4c4b2}.warn{fill:#fff1da;stroke:#d0bc91}.line{stroke:#54788b;stroke-width:2;fill:none;marker-end:url(#a)}</style>
<rect class="box" x="16" y="22" width="320" height="431" rx="10"/><text class="title" x="40" y="60">同一世界与冻结旧现场</text>
<rect class="box" x="36" y="91" width="280" height="98" rx="7"/><text x="51" y="125">旧16帧 → 原生10K高斯</text><text x="51" y="159">保留既有M1/M2失败结果</text>
<path class="line" d="M176 190V211"/><rect class="green" x="36" y="218" width="280" height="92" rx="7"/><text x="51" y="250">独立世界表面 / 编辑参考</text><text x="51" y="285">定义残影ROI，不赋高斯GT</text>
<rect class="warn" x="36" y="335" width="280" height="95" rx="7"/><text x="51" y="367">参考只用于特权诊断 / 评价</text><text x="51" y="402">不进入原生重建训练</text>
<rect class="green" x="376" y="22" width="607" height="431" rx="10"/><text class="title" x="398" y="60">两个互相补充、分别验收的控制</text>
<rect class="box" x="397" y="91" width="565" height="151" rx="7"/><text x="415" y="122">来源控制：删除前 / 后实际逐行 αT</text><path class="line" d="M680 134V152"/><text x="415" y="179">行ID → 16输入视图回放 / 测量支持</text><text x="415" y="217">集合移除、颜色、环境分开干预</text>
<rect class="box" x="397" y="266" width="565" height="164" rx="7"/><text x="415" y="298">证据控制：世界道路点 → 16帧遮挡查询</text><path class="line" d="M680 311V331"/><text x="415" y="355">补足观测：保留原三个验收相机</text><text x="415" y="393">同坐标 / 同方法 / 同10K重新重建</text>
<rect class="box" x="1023" y="22" width="361" height="431" rx="10"/><text class="title" x="1045" y="60">原相机上的配对验收</text>
<rect class="box" x="1043" y="91" width="321" height="103" rx="7"/><text x="1059" y="124">C_G / alpha / 环境实际回填</text><text x="1059" y="160">原环境与统一诊断白底</text>
<path class="line" d="M1204 195V218"/><rect class="green" x="1043" y="225" width="321" height="98" rx="7"/><text x="1059" y="257">none / delete / shift / rotate</text><text x="1059" y="292">已见 / 未见道路分区记录</text>
<path class="line" d="M1204 324V347"/><rect class="warn" x="1043" y="354" width="321" height="76" rx="7"/><text x="1059" y="386">定位选点、表示或证据缺口</text><text x="1059" y="414">不将一次失败推广为方向否定</text>
<path class="line" d="M336 141H396"/><path class="line" d="M983 141H1042"/><path class="line" d="M983 348H1005V270H1042"/></svg>'''
    (OUT/'architecture.svg').write_text(architecture, encoding='utf8')
    report = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V43：残影来源与观测证据控制</title><style>body{max-width:1500px;margin:25px auto;padding:0 22px;background:#f4f8fb;color:#193b4e;font:16px/1.65 "Microsoft YaHei",sans-serif}section{background:white;border:1px solid #c8dbe5;padding:20px;margin:20px 0;border-radius:10px}h1{font-size:29px}h2{font-size:21px}img{display:block;width:100%}a{color:#176b9e}code{overflow-wrap:anywhere}figure{margin:0}figcaption{font-size:14px;padding:6px 0}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.notice{background:#fff1dc;padding:14px}.controls{display:flex;gap:20px;flex-wrap:wrap;margin:14px 0}select{padding:7px;font:inherit}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #c8dbe5;padding:8px;text-align:left}th{background:#e9f2f8}.scroll{overflow:auto}.small{font-size:14px;color:#566f7d}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style>
<h1>V43：删后残影由谁贡献，原输入能否约束它？</h1><p><a href="../sim_m1m2_20261030/index.html">保留M1/M2原实验</a> · <a href="../../index.html">V43设计</a> · <a href="contract.json">本轮控制合同</a></p>
<section><h2>修正研究问题</h2><p><b>此前M1没有提供准确归属，M2也没有提供充分表示修复；它们的失败不能回答归属分类或联合修复是否有用。</b> 本轮沿具体残影追到行ID、实际可见贡献和输入证据，再改变同一世界的观测条件做成对重建。</p><p class="notice">99.9%仅为删除前原车像素贡献覆盖。删后可见性会改变，原来被遮住的错误表示可能成为主要来源。unknown应记录这种不确定性，不能自动等于car或删除集合。当前没有逐高斯唯一身份GT。</p><img src="architecture.svg" alt="同世界冻结现场，来源与观测两个控制，原相机分层验收"></section>
<section><h2>来源已经追到具体行</h2><p>ROI固定为“独立原图first-hit车体 ∩ 独立删车first-hit道路”，不把所有删后alpha称为残影。逐行原生αT与完整渲染alpha闭合；排序后另做真实移除验证，排序本身不是单点因果效应。</p>
<div class="controls"><label>帧 <select id="frame"><option>000000</option><option selected>000008</option><option>000015</option></select></label><label>分量 <select id="channel"><option value="final">最终RGB</option><option value="white">白底诊断</option><option value="C">高斯颜色贡献</option><option value="fill">实际环境回填</option></select></label></div><div class="grid" id="source"></div>
<p class="small">颜色控制读取独立删车参考，只用于敏感性分析，不能作为部署方法或完成修复。删除/颜色干预均未训练；白底不是补景。</p><div class="scroll"><table><tr><th>帧</th><th>保留行删前ROI αT</th><th>同批行删后ROI αT</th><th>原M1删车误差</th><th>移除32行</th><th>移除512行</th><th>仅校正128行颜色</th></tr>__SOURCE_TABLE__</table></div>
<p>误差为ROI clamped RGB MAE。删得更多没有单调变好；仅替换环境也未消除残影。颜色控制改善说明这些保留行的外观确实参与错误，但不证明它们具有车辆身份，也不解决几何问题。</p><p><a href="evidence/source/completion_receipt.json">3锚点完整干预记录</a> · <a href="evidence/source/run_r2_color_maps/source_000008/top_rows.csv">中帧top2000逐行表</a></p></section>
<section><h2>全部保留行：约78%的删后贡献来自原观测极弱可见的部分</h2><p>复查全部667,559行静态高斯，以中帧删除后654,881个保留行的实际ROI贡献为分母：原16视图累计可见贡献&lt;0.1像素权重的行，占删后αT的78.08%，占共享像素误差加权αT的79.77%。误差加权值不是“这些行造成了79.77%的误差”的因果分解。</p><table><tr><th>每行原16视图累计αT分箱</th><th>占中帧删后ROI αT</th><th>占共享误差加权αT</th></tr>__BIN_TABLE__</table><p><a href="evidence/source/allrow_observed16_r1/weighted_visibility_bins.json">全行分母、类别曝光混合与分箱原值</a></p>
<p>中帧后删top128行，在全部16原视角中累计可见贡献仅约28.15像素权重；102行累计小于0.1、123行小于1。同128行在中帧删前ROI贡献0.00253，删后为865.11。top128仅占全部删后ROI αT的25.40%、误差加权量31.60%；top8仅占6.74%/9.38%，不能代表全部。这是<b>冻结终模回放</b>，不是每一步训练历史，不能据此说这些行从未接受过监督。</p><table><tr><th>静态行ID</th><th>16帧总贡献</th><th>落入原车像素</th><th>落入原道路像素</th><th>贡献&gt;1e-4的帧数</th></tr>__ROW_TABLE__</table>
<p>例如461782行：原M1未选中；中帧删后ROI贡献61.36，视向RGB约[0,2.04,2.38]；中心深度11.56m，而原车first-hit约9.23m。它在16个视图的总贡献仅0.01176，且没有输入LiDAR点落入其3σ范围。其他top行存在部分测量支持，不能统一认定无约束。中心深度与投影足迹参考深度的差异也不是唯一物理身份判断。</p>
<details><summary>展开top8在完整删后场景中的真实αT贡献图</summary><div class="grid" id="rows"></div></details><details><summary>展开原输入投影与独立first-hit类别核验</summary><div class="grid" id="history"></div></details>
<p><a href="evidence/source/observed16_source_trace.json">16视图逐行记录</a> · <a href="evidence/source/observed16_top128.csv">top128汇总</a> · <a href="evidence/source/constraint_audit_r2/constraint_receipt.json">测量与尺度支撑</a> · <a href="evidence/source/LOSS_PATH_AUDIT.md">实际损失/渲染源码审计</a></p>
<p>静态RGB项排除车辆mask内部；全场RGB、稀疏合成深度、静态寿命/速度及其他正则仍可能约束。原生SH视向颜色仅下界截断，环境直接学习纹理，均无上界1。这解释了HDR数值如何进入渲染，但未证明它们的完整训练成因。</p></section>
<section><h2>证据控制：同一块道路是否真的被输入看见</h2><p>按世界道路点跨帧投影，并做真实BVH first-hit遮挡查询。统计对象是三个旧锚点的10995个道路像素采样，跨锚点可能重复，不是世界面积比例。</p><img src="visibility.png" alt="旧观测与补足视角的世界道路可见性对照"><div class="scroll"><table><tr><th>锚点</th><th>道路像素样本</th><th>旧输入从未可见</th><th>旧至少1帧</th><th>旧多视角夹角中位</th><th>新至少1帧</th><th>新多视角夹角中位</th></tr>__VIS_TABLE__</table></div>
<p>旧输入合计仅50.54%至少一帧可见，2+帧47.58%。hybrid8_z3.5保留0/8/15原相机，仅改变其余13个观测视角；几何可见性达100%，2+帧98.954%。这是充分证据的机制控制，环绕采样并非真实车载连续轨迹。相机、稀疏LiDAR覆盖和初始化随之改变，不能称纯RGB效应。</p><p>连续射线可见还不等于有限像素采样或未受阴影污染的辐射已知。阴影开关差分仍只作为有限采样proxy，当前不宣称精确阴影标签。</p><p><a href="evidence/visibility/visibility_summary.json">完整可见性/角度/LiDAR证据</a> · <a href="evidence/visibility/sensor_validation.json">传感器核验</a></p></section>
<section><h2>同世界、同10K的实际重建对照</h2>__RICH_SECTION__</section>
<section><h2>如何据此推进V43</h2><ol><li>将“身份不确定”和“编辑后显露的几何/外观证据不足”分别记录。隐藏错误道路表示不应被强行归类为car；归属正确也不保证其颜色和形状可信。</li><li>候选审计包含编辑后的实际可见贡献，并回查所有原观测和稀疏测量；不再只用删前权重封闭候选。</li><li>未来表示修复须依据单一表面支持拟合位置/协方差，并重新解释外观；跨帧保持与编辑监督分别核验。均值移位、统一缩小、继承父色只保留为失败控制。</li><li>在证据充分但仍失败的部分，才比较只归属与可学习联合修复；原观测从未提供的信息，单列为先验/补全问题。环境与车关联阴影仍独立登记。</li></ol><p>以上为一个机制场景的来源和证据诊断；不代表真实场景泛化，不训练新的分类器，也不将特权颜色诊断包装为算法。AutoDL保持开机。</p></section>
<script>const A=__ASSETS__,N=__CASES__,R=__ROWS__,H=__HISTORY__;function card(src,title){return '<figure><img src="'+src+'" alt="'+title+'"><figcaption>'+title+'</figcaption></figure>'}function show(){const f=document.querySelector('#frame').value,c=document.querySelector('#channel').value;document.querySelector('#source').innerHTML=card(A['ref/'+f],'独立删车参考')+Object.keys(N).map(k=>card(A[f+'/'+k+'/'+c],N[k]+' · '+c)).join('')}for(const id of ['frame','channel'])document.getElementById(id).onchange=show;document.querySelector('#rows').innerHTML=R.map(r=>card(r.src,r.title)).join('');document.querySelector('#history').innerHTML=H.map(r=>card(r.src,r.title)).join('');show();</script></html>'''
    for key, value in {'SOURCE_TABLE': source_table, 'ROW_TABLE': row_table, 'BIN_TABLE': bin_table, 'VIS_TABLE': vis_table, 'RICH_SECTION': rich_section,
                       'ASSETS': json.dumps(assets), 'CASES': json.dumps(cases, ensure_ascii=False),
                       'ROWS': json.dumps(top_images), 'HISTORY': json.dumps(history_images)}.items():
        report = report.replace('__'+key+'__', value)
    report = report.replace('此前M1没有提供准确归属', '此前M1未建立经验证的准确归属参考')
    report = report.replace('这是充分证据的机制控制', '这是补足目标道路几何可见性的机制控制')
    bridge = '''<section><h2>桥接核验：新观测是否也能看见旧错误高斯？</h2>
<p>仅冻结旧10K权重、换成新16个查询视图，原0/8/15相机与时刻保留。旧中帧残影top128的累计实际αT由28.15增至539.53，其中新道路像素贡献223.58；103/128行累计达到0.1，45行达到1。这既包含道路像素也包含车辆像素曝光，不是身份判定。</p>
<p>以旧中帧全部删后ROI贡献为固定分母，原累计可见量&lt;0.1的行占78.08%；换新查询后仍低于0.1的占39.45%。41.53个百分点从弱变强，另有2.90个百分点从强变弱，净下降38.63个百分点。<b>更多世界道路可见，确实使部分错误表示受到更多观测，但没有覆盖全部隐藏高斯。</b></p>
<p>例如535849行的累计贡献0.142→1.329，输入LiDAR点落入其3σ范围的帧数1→9；461782行则0.0118→0.3062，但仍无LiDAR点进入3σ。新视图对该行颜色的不一致只是贡献加权诊断，不等于训练梯度或可恢复性证明。</p>
<p>这是固定旧模型的查询探针，不是新10K训练结果。原活动Scene的transform文件与mtime均未改变；逐行贡献与原生alpha闭合最大相对误差1.05e-6。<a href="evidence/source/bridge_oldmodel_newviews_r1/bridge_receipt.json">完整转移矩阵、128行与测量收据</a></p></section>'''
    report = report.replace('<section><h2>同世界、同10K的实际重建对照</h2>', bridge+'<section><h2>同世界、同10K的实际重建对照</h2>')
    report = report.replace('AutoDL保持开机。</p></section>', 'AutoDL保持开机。</p><p><a href="evidence/visibility/PROTOCOL_NEXT.md">下一步最小可辨识协议及停止条件</a>：固定表示的统一路由只称真值辅助可行性参考；有限穷举只能证明其受限子问题，不能证明全局不可能。</p></section>')
    (OUT/'index.html').write_text(report, encoding='utf8')
    write(OUT/'report_state.json', dict(task_id='WS-V43-SIM-SOURCE-EVIDENCE-20261030',
        source_frames=3, frozen_replay_views=16, source_training_updates=0, unique_gaussian_gt=False,
        accurate_ownership_control_completed=False, ideal_repair_completed=False,
        evidence_reconstruction_report_attached=extra.exists(), fresh_same_rule_source_completed=(SRC/'source/comparison_summary.json').exists(),
        same_world_fresh_formal_updates=10000 if (NATIVE/'completion_receipt.json').exists() else None,
        source_replay_original_and_new_views=16, native_common_queries=12, native_heldout_queries=4,
        new_M1_edit_scope='delete only, 3 anchors; no new M1 shift/rotate verification',
        editing_quality_passed=False, human_verdict=None, automatic_shutdown=False))
    print(json.dumps(dict(report=str(OUT/'index.html'), images=len(assets)+len(top_images)+len(history_images))))


if __name__ == '__main__':
    main()
