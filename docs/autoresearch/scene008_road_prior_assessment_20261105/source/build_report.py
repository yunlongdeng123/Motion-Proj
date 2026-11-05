"""Read-only diagnosis deliverable; future road experiments explicitly not run."""
from pathlib import Path
import gzip,hashlib,html,json,shutil,tarfile
from html.parser import HTMLParser

WORK=Path(__file__).resolve().parent
ROOT=WORK.parent.parent
OUT=ROOT/'outputs/scene008_road_prior_assessment/delivery'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    for name in ('coverage','observations'):
        shutil.copytree(WORK/name,OUT/name,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('road_plan.json','literature_review.json','literature_source_checks.json','architecture.svg','latest_vehicle_IO_check.json','vehicle_io_contract.json'):
        shutil.copyfile(WORK/name,OUT/name)
    coverage=read(OUT/'coverage/coverage_summary.json');case=coverage['decisive_raw137']
    ledger=read(OUT/'observations/patch137_observation_ledger.json')['groups']['fixed_roi']
    colour=read(OUT/'observations/patch137_colour_check.json')['groups']['fixed_roi']
    plan=read(OUT/'road_plan.json');literature=read(OUT/'literature_review.json')
    assert case['count']==720 and case['isolated_road_alpha']==0 and case['joint_road_alpha_T']==0
    assert ledger['inherited_geometry_valid']==295 and ledger['points_with_geometric_RGB_witness']==294
    assert colour['stable_upper_cluster_ge3']==286
    (OUT/'source').mkdir();shutil.copyfile(__file__,OUT/'source/build_report.py')
    # Source/evidence byte snapshots remain exact even though published text is normalized for Git.
    with tarfile.open(OUT/'evidence_raw_snapshots.tar.gz','w:gz') as archive:
        for folder in ('coverage','observations'):
            archive.add(WORK/folder,arcname=folder,filter=lambda info:None if '__pycache__' in info.name else info)
        archive.add(WORK/'literature_review.json',arcname='literature_review.json')
        archive.add(WORK/'literature_source_checks.json',arcname='literature_source_checks.json')
    references=[
      ('RoGs','2024预印本','世界网格薄道路、pose/LiDAR初始化；可作结构化重建基线。','不是生成式补全，Waymo适配与既有引擎对接需做；2026新稿模块未核实发布。','https://arxiv.org/abs/2405.14342','https://github.com/fzhiheng/RoGS'),
      ('2DGS','SIGGRAPH 2024','薄表面、深度/法向约束；区分表面几何与体高斯读出的误差。','仍有alpha合成；不能据此宣称MOG-2不透明首交已实现。','https://surfsplatting.github.io/','https://github.com/hbb1/2d-gaussian-splatting'),
      ('StreetUnveiler','ICLR 2025','最直接的背景补全基线：跨时间证据、补全候选，再固化到持久3D。','原反时间策略偏前向相机；alpha低不等于传感器未观测，补图伪标签不是真值。','https://streetunveiler.github.io/','https://github.com/DavidXu-JJ/StreetUnveiler'),
      ('StreetCrafter','CVPR 2025','借pose/LiDAR条件生成候选、再蒸馏到同一3D表示的接口。','整套联合训练会混入额外变量；作者80GB配置不是本机最低显存结论，当前32GB需先小样验证。','https://zju3dv.github.io/street_crafter/','https://github.com/zju3dv/street_crafter'),
      ('InfiniCube','ICCV 2025','借道路/语义/坐标buffer作为世界结构条件。','生成新世界与忠实补全实录场景不同；全栈过重，当前代码的视频骨干也与论文不同。','https://research.nvidia.com/labs/toronto-ai/infinicube/','https://github.com/nv-tlabs/InfiniCube')]
    paper_rows=''.join(f'<tr><td><a href="{paper}">{name}</a><br>{venue}<br><a href="{code}">作者代码</a></td><td>{borrow}</td><td>{limit}</td></tr>' for name,venue,borrow,limit,paper,code in references)
    phases=[
      ('R0：先补实测道路','冻结M2，补齐所需世界范围；优先用raw154–175恢复已核对部分，并为425个无旧交点像素重新寻找几何。','当前场景 / 实测扩域 / 扩域＋持久旧层接管','孤立道路覆盖、环境透射、实测高度与深度、接缝、车辆保护；工程收益单列'),
      ('R1：世界patch虚拟遮挡','把已知道路的物理patch及halo从所有合法观测和派生状态中撤去，建立封存真值。','仅纹理遮挡与几何＋纹理同时遮挡分开；不同worldpatch分train/dev/test','坡度与法向、标线位置/宽度/连通性、颜色；不能只遮一帧或一批RGB'),
      ('R2：道路先验对照','先平面/稳健高度、样条/RBF与标线延续，再共享条件模式，最后可选冻结生成教师。','相同合法输入与预算；均值 / 共享latent / 独立隐藏系数；同先验的直接网格 / 高斯','看先验相对强插值基线增加什么；K个候选只由可见上下文选，不用隐藏真值选'),
      ('R3：放回同一场景','持久道路资产与M2共存；覆盖补好后再退役冲突旧层，不在delete时临时换路。','keep/delete只改a37存在状态；M2/道路系数跨时间、相机和编辑共享','raw105/113/130/137→全49；随后有效侧相机；车外a37贡献、旧静态混合支撑另查')]
    phase_rows=''.join('<tr>'+''.join('<td>'+v+'</td>' for v in row)+'</tr>' for row in phases)
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scene008 道路先验：覆盖诊断与推进计划</title>
<style>body{font:17px/1.75 system-ui,"Microsoft YaHei",sans-serif;background:#f4f7fa;color:#173b4c;margin:0}main{max-width:1320px;margin:auto;padding:28px}h1{font-size:30px}h2{font-size:24px;margin:26px 0 12px}h3{font-size:19px}section{background:white;padding:23px;margin:20px 0;border-radius:10px}img{display:block;width:100%;margin:12px 0}.note{padding:14px;background:#fff3dd;border-left:4px solid #cf9634}.ok{background:#eaf6ef;padding:14px;border-left:4px solid #488e68}table{width:100%;border-collapse:collapse;font-size:15px}th,td{text-align:left;padding:10px;border-bottom:1px solid #d9e3eb;vertical-align:top}th{background:#edf4f8}a{color:#006d87}code{background:#edf3f6;padding:2px 4px;overflow-wrap:anywhere}.small{font-size:14px;color:#4a6474}li{margin:7px 0}summary{cursor:pointer;font-weight:bold}button{padding:7px 18px;margin-right:5px}</style>
<main><h1>Scene008 道路先验：先补足真实表面，再验证条件补全</h1><p>日期：2026-11-05。保留当前完整车辆资产M2。本次完成只读诊断、来源核查与实验设计；没有补建道路、训练先验、修改检查点或执行关机。</p>
<p class="ok"><strong>结论：</strong>raw137车尾下的主要缺口来自当前道路资产的有限世界范围。已核对部分在后续真实帧有道路见证，优先归跨时刻恢复。它不能作为“所有199帧始终不可见、必须生成”的证据。道路接缝与蓝紫色也需区分：有些射线连有效道路表面都没有，先解决覆盖，再处理已有表面的材质衔接。</p>
<section><h2>1．复核了什么</h2><p>用户给出的570像素区域没有像素坐标。本报告登记另一块独立固定ROI：raw137、半开区间<code>x∈[230,270), y∈[204,222)</code>，共720像素；没有冒充精确重现570。a6候选车身底部到y=202，且ROI的新车贡献实际接近零。紧贴车尾、随车移动的灰色部分仍属车辆资产。</p>
<img src="coverage/137_roi_annotation.png" alt="raw137独立ROI与车尾位置"><table><tr><th>独立720px区域</th><th>真实前向结果</th><th>解释</th></tr>
<tr><td>孤立道路alpha</td><td>全720像素严格为0</td><td>道路自己就没有覆盖，不能归因为被旧静态层挡住</td></tr><tr><td>联合道路贡献</td><td>严格为0</td><td>不是只有颜色不合适</td></tr><tr><td>旧静态 / 环境</td><td>15.0586% / 84.9389%</td><td>主要透给方向背景；不是真正的地面</td></tr><tr><td>当前道路资产</td><td>141888行全部active</td><td>没有按帧语义mask临时挖洞</td></tr><tr><td>场景局部世界x范围</td><td>道路8.17–19.75m；可定位目标21.24–28.27m</td><td>目标超出既有envelope；没有道路splat足迹与ROI相交</td></tr></table>
<img src="coverage/137_fixed_fresh.png" alt="孤立道路、联合道路、车身、旧静态和环境贡献"><img src="coverage/world_xy_footprint.png" alt="道路当前范围与缺口查询的世界位置"><p class="small">ROI新a6贡献均值约2.46×10⁻⁵。共8次no_grad渲染、约6.1秒，0优化更新；模型、源文件、道路检查点、颜色场和冻结状态校验均未变。环境项是(1−alpha)×方向背景，不等同于真实道路反射率或照明。</p>
<h3>后段为何更明显</h3><img src="coverage/all49_gap_overview.jpg" alt="49帧车尾候选带缺口增长"><p>几何定义的车尾候选带共57560像素，诊断条件“低车辆贡献、低道路贡献、环境&gt;0.5”标出8439像素；这是指定候选带的排查统计，不是全图道路准确率。前段道路覆盖较好，后段随所需世界范围越出资产而下降。诊断用带区不是最终渲染的抹图掩码。</p></section>
<section><h2>2．这块地面在其他时刻能否找到</h2><table><tr><th>同一固定ROI账本</th><th>数量</th><th>能说明什么</th></tr><tr><td>可信旧路面几何查询</td><td>295 / 720</td><td>其余425没有可信旧交点，不等于现实中没有道路</td></tr><tr><td>后续真实FRONT RGB/几何见证</td><td>294 / 295，raw154–175</td><td>这些查询优先实测恢复，不是prior-only</td></tr><tr><td>至少3帧几何RGB见证</td><td>286 / 295</td><td>可做多帧稳健关联</td></tr><tr><td>既有稳定高亮簇规则</td><td>286；其中274跨度≥6帧</td><td>是候选干净道路proxy，不是独立无影材质真值</td></tr></table>
<img src="observations/patch137_fixed295_rgb_witnesses.png" alt="相同295路面查询在raw137被遮挡、raw162163165露出的真实图像见证"><p>图中使用同一组295查询。raw137本来落在前车投影上，后续raw162/163/165可见实际道路。来源资格使用已有标定、物体排除、近邻回波与小三角形门槛；不是把生成图当作实测。来源与阴影/曝光的一致性仍应在重建前逐项检查。</p>
<p class="note"><strong>“199帧”范围必须说清：</strong>现有道路RGB缓存是199×FRONT，未逐帧检索其他相机。四个原始record样本各含5相机、5LiDAR×2return，但这不是全输入可见性审计。因此“零合格来源”“road alpha=0”“旧uid&lt;0”都不足以宣布全199、全相机从未看到。无效标定、视野外、候选高度不确定时保留unknown。</p>
<details><summary>当前道路资产与观测资格的区别</summary><p>原20cm候选26088格中9547有合格来源，7848进入旧Road；另1699有来源但被资格门限拒绝。当前141888个fine格中137171为real、4717为inferred；inferred内2780已有1–2个来源，1937零合格来源。统计均针对已选域，不代表全场景未知为零。</p><p>完整账本与接口：<a href="observations/observation_audit.json">来源审计</a> · <a href="observations/patch137_observation_ledger.json">295/331查询逐帧证据</a> · <a href="observations/patch137_colour_check.json">实际颜色proxy</a> · <a href="observations/projection_ledger_interface.json">五相机候选解码/投影接口</a>。</p></details></section>
<section><h2>3．道路分支应恢复一份什么表示</h2><img src="architecture.svg" alt="真实道路恢复与条件道路先验的架构组件"><p>道路放在世界坐标，用一份局部连续高度表面h(u)、标线结构与材质m(u)表示，保存覆盖范围、来源和推断置信。首版只处理局部单层道路，不把桥上与桥下压进同一个高度场。新道路在保留/删除a37时始终存在，只由几何遮挡决定是否成像。</p>
<p><strong>R4提供问题边界：</strong>固定几何、颜色、透明度、数量，仅做keep/rebind/defer，不能创造这片缺失路面；它仍适合研究归属决策的独立效果。<strong>MOG-2提供主线：</strong>从真实观测与条件先验恢复表面和材质，持久接管冲突旧表示，再检验编辑后的可见性。</p>
<p>MOG-2 §3.3的具体可测版本为<code>c=[c_h,c_m]=μ(q)+U(q)z</code>：可见与隐藏道路共享坡度、曲率、标线走向/宽度及材质模式，z只由合法观测拟合。隐藏区域不另开自由残差通道。先用显式低容量结构基线检验这种耦合，再考虑预训练生成模型提供候选。该理论有条件，不能保证任意隐藏纹理唯一恢复。</p>
<p>先沿用当前统一Gaussian排序，约束路面薄支撑并检查孤立覆盖和透射残余；另用解析高度表面/网格作为几何读出参照。完整MOG-2不透明首交仍是独立实现项。关掉环境或把低alpha像素涂成灰色不等于补建路面。</p></section>
<section><h2>4．建议按这四步执行</h2><table><tr><th>阶段</th><th>具体工作</th><th>对照</th><th>验收与允许主张</th></tr>'''+phase_rows+'''</table>
<p><strong>最近一次可交付实验应是R0：</strong>补足raw137这块已见道路，并验证整个49帧所需world范围，而不是只增大当前图上的修补框。先显示“加表面”带来的环境透射下降，再单测旧层接管的颜色与接缝变化，避免把两种收益混在一起。</p>
<p>建议预先锁定：在确认应命中且未被车辆挡住的道路射线上，孤立道路alpha≥0.98、环境残余≤0.02；保护域单独报告RGB/深度、逐帧变化和大误差尾部。M2参数不变也须重新检查成像，因为背景可能透过车缘。车辆损伤不能被道路分数改善抵消。正式执行前冻结阈值与参考角色，不看结果放宽。</p></section>
<section><h2>5．“遮起来再补”的实验必须防止什么泄漏</h2><p>以世界道路patch为单位撤源，不是只遮raw137的一块RGB。所有相关时刻、声明可用相机、LiDAR、深度/标线、融合初始化、纹理图、特征与条件缓存均需移除patch及guard。已经看过目标的native/environment/road模型本身也会带信息；最后屏蔽它们的输出并不能擦除已编码内容。</p>
<p>因此严格先验benchmark应独立从blinded上下文重建道路，不以现有完整场景或M2渲染作为先验输入。M2留给后续固定前景集成，不参与隐藏目标校色。若无法排除同源父模型的信息影响，结果必须单列非严格控制。</p>
<ul><li><strong>只遮外观：</strong>可保留实测几何，检验材质/标线补全；不声称坡度生成成功。</li><li><strong>几何与外观都遮：</strong>同时撤去target LiDAR/深度与派生表面，检验坡度、标线和材质。真值封存，只在最终评分使用。</li><li>先注册不同路面类型、0.5/1/2m物理尺度及patch划分。近邻重叠halo不得跨train/dev/test；候选选择仅看可见上下文，不用隐藏GT挑最好seed。</li><li>无真实隐藏参考的区域只报告结构可行性、时序/跨视角一致性和不确定性；不写隐藏纹理准确率。</li></ul>
<p>指标需分开：世界高度/法向/坡度、标线中心线与宽度/连通性、真实视图颜色与接缝、道路coverage/环境透射，以及车辆/其他表面保护。观测恢复与生成收益分账，Scene008仍是同场景机制验证。</p><p><a href="observations/benchmark_contract.json">完整blinding与撤源合同</a> · <a href="road_plan.json">阶段、冻结项、对照与门槛建议</a></p></section>
<section><h2>6．相关开源工作，具体借什么</h2><p>下表核对了作者论文、项目及实际源码树。均未在当前栈安装/执行；公开权重目录或链接不等于已验证二进制可用。已有重建代码也不等于已有可直接调用的预训练道路生成先验。</p><table><tr><th>工作/来源</th><th>本分支可借鉴</th><th>边界和接入判断</th></tr>'''+paper_rows+'''</table>
<p>优先顺序：RoGs式结构道路和强插值基线 → StreetUnveiler式补全候选固化到3D → 必要时StreetCrafter式带pose/LiDAR条件的教师。2DGS提供表面表示对照，InfiniCube提供世界结构buffer思路；不因此切换整套训练系统。2026版RoGS仍为预印本，新增模块未核实公开，不拿它作为即用能力。</p><p>当前GPU约32GB。作者80GB测试配置不能直接解释为最低显存要求；先做冻结推理/局部适配smoke再决定是否接入。未来约1小时以上的正式实验须按既有规则先做约10分钟真实smoke，覆盖保存、评价、完整恢复与下一阶段入口。</p>
<p><a href="literature_review.json">五项详细方法映射与可证伪假设</a> · <a href="literature_source_checks.json">源码树/权重目录核验收据</a></p></section>
<section><h2>状态与可复核记录</h2><p>本次只评估和设计，M2车辆、原道路、旧静态、a37和环境均未修改；没有声称缺口已修好、道路先验已训练或隐藏路面已验证。AutoDL保持开机。下一执行建议为R0实测覆盖恢复，同时建R1严格worldpatch基准；车辆后续优化另开对照，保持道路基线中的M2版本固定。</p><p><a href="coverage/fresh_render_audit.json">真实渲染与状态不变</a> · <a href="coverage/coverage_summary.json">覆盖诊断总表</a> · <a href="observations/source_evidence/source_index.json">原代码证据与行号</a> · <a href="evidence_raw_snapshots.tar.gz">原始审计字节快照</a> · <a href="../../scene008_asset_controls/delivery/index.html">当前M2车辆对照</a></p></section></main></html>'''
    text=text.replace('<h2>3．道路分支应恢复一份什么表示</h2>', '<h2>3．道路分支应恢复一份什么表示</h2><p>已参考最新车辆I/O审计：PVD本身只输入/输出XYZ，隐藏车身的灰蓝色主要是后续材质fallback，不能把它当成PVD生成的真实纹理。新I/O封装与本次诊断M2的xyz、normal、scale、opacity、SH颜色和inferred标记逐值相同，已做CPU核对；后续可固定这一带来源字段的同数值版本。<a href="latest_vehicle_IO_check.json">数值等价核对</a> · <a href="vehicle_io_contract.json">车辆输入输出能力合同</a></p>')
    text=text.replace('先用显式低容量结构基线检验这种耦合，再考虑预训练生成模型提供候选。','先用显式低容量结构基线检验这种耦合，再考虑预训练生成模型提供候选。道路也要分清几何与外观接口：RGB补图不自动给出高度，网格重建代码也不自动具备预训练补全能力；模型输出、缺失通道、单位和下游Gaussian化分别登记。')
    text=text.replace('<a href="../../scene008_asset_controls/delivery/index.html">当前M2车辆对照</a>','<a href="latest_vehicle_IO_check.json">当前M2数值版本</a>')
    text=text.replace('原反时间策略偏前向相机；alpha低不等于传感器未观测，补图伪标签不是真值。','原反时间策略偏前向相机；alpha低不等于传感器未观测，补图伪标签不是真值。借空街背景模块，不能把其删除全部临停对象的最终图直接替代需要保留a6的场景。')
    (OUT/'index.html').write_text(text,encoding='utf-8')
    local_docs={}
    for name in ('MOG2_Method_Probes_CN.md','V43_METHOD_PROPOSAL_R4.md'):
        path=Path('C:/Users/dengyunlong/Downloads')/name
        local_docs[name]=dict(path=str(path),sha256=sha(path))
    write(OUT/'assessment_receipt.json',dict(status='DIAGNOSIS_COMPLETE_PLAN_ONLY',road_repair_applied=False,prior_trained=False,
      user570_exactly_reproduced=False,independent_ROI_pixels=720,raw137_known_queries=295,FRONT_witnessed=294,
      stable_colour_proxy=286,unknown_geometry_pixels=425,current199_camera_scope='FRONT only',
      no_all_camera_never_visible_claim=True,frozen_M2_sha256=plan['frozen_vehicle']['sha256'],local_method_docs=local_docs,
      GPU_forward_calls=8,optimizer_updates=0,autodl='KEEP_ON',shutdown_performed=False,
      evidence_byte_snapshot_sha256=sha(OUT/'evidence_raw_snapshots.tar.gz')))
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('src','href') and v)
    parser=Links();parser.feed(text)
    for link in parser.links:
        if not link.startswith(('https:','http:','#','data:')):assert (OUT/link).is_file(),link
    for path in OUT.rglob('*'):
        if path.is_file() and path.suffix in ('.py','.json','.md','.html','.svg','.txt'):
            raw=path.read_bytes()
            if b'\r\n' in raw:path.write_bytes(raw.replace(b'\r\n',b'\n'))
    write(OUT/'report_QA.json',dict(status='PASS_LINKS_AND_SOURCE',relative_and_external_links_total=len(parser.links),
      critical_source_counts_checked=True,byte_exact_evidence_preserved=True,scientific_claim='diagnosis and future plan only'))
    print(json.dumps(dict(output=str(OUT),files=sum(p.is_file() for p in OUT.rglob('*')),source_counts=[720,295,294,286]),ensure_ascii=False))

if __name__=='__main__':main()
