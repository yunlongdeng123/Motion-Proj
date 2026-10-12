"""Append witnessed initialization data gaps without changing any training input."""
import hashlib
import json
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_initialisation_gap_evidence(here, out, page):
    root = here / 'loss_audit/next_diagnostics_cpu/static_initialisation_actor_cpu'
    expected = {
        'static_initialisation_actor_audit.json': 'df97e5d48ee2be553b54e127dbeff2be12dafa8561c93d6bc87bc630059b8ecd',
        'front_visibility_followup_cpu.json': '01fb4c66661435e69851697c22bbcd09a2a5cf56d3e75e82e6f9cdab7d603f34',
        'dynamic_actor_native_semantics_followup_cpu.json': '9cc9bf4cc0caeecd1659127410b9c116d3a6c1155e8f1cfe05bf96d0be9175da',
        'raw49_labels_layout.json': 'fa0448b99abc28c22f40fc7722ecb1d68dbcc68dbe88ae44b5aed0357b894806'
    }
    for name, digest in expected.items():
        if sha(root / name) != digest:
            raise ValueError('Initialization source evidence differs: ' + name)
    audit = json.loads((root / 'static_initialisation_actor_audit.json').read_bytes())
    front = json.loads((root / 'front_visibility_followup_cpu.json').read_bytes())
    dynamic = json.loads((root / 'dynamic_actor_native_semantics_followup_cpu.json').read_bytes())
    assert audit['all600000_source_draw_pose_worldXYZ_modeltime_fields_exact'] and audit['all36406_actor_source_return_replay_exact']
    assert front['canonical_audit_SHA256'] == dynamic['canonical_audit_SHA256'] == expected['static_initialisation_actor_audit.json']
    target = out / 'evidence/initialisation_data_gap'
    target.mkdir(parents=True, exist_ok=True)
    for name in expected:
        shutil.copy2(root / name, target / name)
    for name in ('HANDOFF.md', 'retained_static_target_OBB_witnesses.csv', 'dynamic_actor_noncar_or_undefined_witnesses.csv'):
        if (root / name).exists():
            shutil.copy2(root / name, target / name)
    c, sem = audit['counts'], audit['semantic_identification']
    d = dynamic['counts']
    section = f'''<section class="section" id="initialisation-data-gap"><h2>独立数据缺口：初始化分配的已见证子集</h2><div style="display:flex;align-items:center;gap:10px;overflow-x:auto;padding:12px 0"><div class="metric">原49帧LiDAR＋pose</div><span>→</span><div class="metric">seed0抽样／原actor adapter</div><span>→</span><div class="metric">原0.1m排除<br>预计static／dynamic输入</div><span>→</span><div class="metric">稀疏原生语义＋实例<br>独立lineage审计</div><span>→</span><div class="metric">见证return IDs<br>覆盖未知与后续设计</div></div><p>CPU按原recipe完整回放600,000条static PLY和36,406条actor PLY的原return、ego pose、世界XYZ与model time，逐字节一致。原0.1m动态点近邻排除后，预计static输入为595,894行；已进入actor cloud的抽样2,568行全部排除，exact selected-return残留为0。历史step0 Gaussian bundle与原抽样ID清单未保存，所以这是一致的初始化输入回放，未冒充历史参数快照。</p><table><tr><th>独立原生点标签见证</th><th>实际子集</th><th>限制</th></tr><tr><td>static保留的目标车辆关联return</td><td>{c['native_linked_target_retained_rows']}行／{c['native_linked_target_retained_unique_raw_returns']}个独立return</td><td>只在raw25／30／35／40／45的TOP两return提供标签；native instance→track是同帧交集与已关联目标导出的link，原raw未提供该映射。</td></tr><tr><td>static未获原生语义标签</td><td>{c['direct_target_identity_unresolved_rows']:,}行／{100*sem['fraction_without_native_semantic_class']:.4f}%</td><td>169行是已见证子集，0.02836%只是下界；全49污染率未知，不对未标注帧外推。</td></tr><tr><td>dynamic actor PLY的物理ROAD</td><td>{d['native_ROAD_rows']}行；已知CAR {d['native_CAR_rows']}，未标注{d['missing_native_labels']:,}</td><td>24行都在原body mask内、当前目标3D box外，距原拟合ground为0.036–0.140m。ROAD是物理语义，不是运动或shadow标签。</td></tr></table><p>同时间3D box中的884行是几何候选，包含ROAD，不能全部计作车辆污染。169条原生目标见证在自己的采集帧中有163条FRONT外、6条FRONT内但body mask外；在49个原FRONT pose中，143行／137独立return至少一次满足pinhole中心与初始化temporal／alpha上界资格。该资格未计算高斯footprint、遮挡或实际raster贡献，不能据此解释道路彩噪。</p><p class="muted">此处GT box与原生语义仅用于独立审计，没有清洗点、替代M-detector预测或改S／原始输入。原actor中的24 ROAD也不能单独解释训练后大范围动态图道路；body-only与缺shadow输入仍需区分物理面和动态外观。M-detector／shadow的公开源码可行性记录不等于已复现预处理。本附录没有证明上述见证导致confetti，也没有启动从零I实验。</p><div class="links"><a href="evidence/initialisation_data_gap/static_initialisation_actor_audit.json">完整初始化lineage、原生目标子集与未知覆盖</a><a href="evidence/initialisation_data_gap/front_visibility_followup_cpu.json">FRONT pinhole资格及非raster因果限制</a><a href="evidence/initialisation_data_gap/dynamic_actor_native_semantics_followup_cpu.json">dynamic物理ROAD见证及原enum来源</a><a href="evidence/initialisation_data_gap/HANDOFF.md">原审计交接与后续条件</a></div></section>'''
    marker = '<section class="section"><h2>实现来源：哪些是原代码，哪些仍需验证</h2>'
    if marker not in page:
        raise ValueError('Initialization report insertion marker absent')
    return page.replace(marker, section + marker)
