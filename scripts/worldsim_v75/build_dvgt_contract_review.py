"""在同一闭环审阅页保留原生接口与额外时间观测的完整结果。"""
import json
import shutil


def build(source, output):
    dst = output/'native-contract'; dst.mkdir(exist_ok=True)
    for path in source.rglob('*'):
        if path.is_file() and path.suffix in ['.json', '.png', '.svg']:
            target = dst/path.relative_to(source); target.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy2(path, target)
    result = json.loads((source/'result.json').read_text())
    rows = []; figures = []
    for case in result['cases']:
        log = case['log_id'][:8]
        figures.append(f'<h3>{log}：相同最后一帧，补入一秒真实历史</h3><img src="native-contract/{log}-native-contract.png" alt="{log}真实前视输入、原生点方向空间误差和全部视角配对结果">')
        for r in case['projection']:
            t1, t3 = r['T1'], r['T3']
            rows.append(f'<tr><td>{log}</td><td>{r["camera"].replace("ring_","")}</td><td>{r["common_samples"]}</td>'
                        f'<td>{t1["median_angular_error_deg"]:.2f} → {t3["median_angular_error_deg"]:.2f}</td>'
                        f'<td>{100*t1["positive_fraction"]:.1f}% → {100*t3["positive_fraction"]:.1f}%</td>'
                        f'<td>{t1["median_positive_reprojection_px"]:.1f} → {t3["median_positive_reprojection_px"]:.1f}</td></tr>')
    return '''<section id="native-contract"><h2>原生输入范围：真实历史能消除严重错位，但不能由此推断闭环影响</h2>
    <p>本轮检查此前受控距离适配丢掉了什么。旧前向仅输入一个时刻的七张RGB；后续取点到ego的距离，再用已知相机射线放回世界。这会丢掉原生点图的方向误差，因此旧距离结果不能作为完整三维状态重建排名。</p>
    <img src="native-contract/architecture.svg" alt="真实RGB至DVGT原生点图与标定射线评价的模块图">
    <div class="card"><strong>普通时间上下文解释了第一例最严重的方向错误。</strong><p>冻结两个已有任务，分别追加两张历史环视帧，采样为−1、−0.5、0秒，每个时刻七视角。最后一帧逐像素相同，所有图像都在原截止时刻以内；官方固定权重只接RGB。第一例前视角误差97.68°→4.01°，相机前方有效点26.4%→100%；14个相机中13个改善，第二例右后视角5.47°→6.84°变差，完整保留。</p><p>三帧输入增加了真实观测，不能称相同信息预算优势。残余角误差3.21°–11.37°，两例都未通过事前的原生使用筛查（每视角中位角误差≤1°、正深度比例≥95%）；该筛查也不能证明深度/形状正确。没有按结果再延长历史、重排相机、换seed、拟合坐标或追加世界模型生成。</p></div>
    '''+''.join(figures)+'''<details><summary>全部14个视角的配对分母与指标</summary><table><tr><th>日志</th><th>相机</th><th>共同点数</th><th>角误差 °：1→3时刻</th><th>正深度比例</th><th>正深度投影误差 px</th></tr>'''+''.join(rows)+'''</table><p>仅使用两输出共同有限、距截止时刻ego为2–80米且不在padding内的固定8px网格。角误差包括相机后方点；像素误差只在正深度点上有物理意义，两者分母不同。图中色阶截于60°，数值表不截断。</p></details>
    <h3>工程检查及科学边界</h3><p>官方预处理与原保存输入像素完全一致；单时刻调用没有进入点头分块分支。官方0.1单位换算、RDF→FLU与已知射线往返代数一致，往返误差小于4×10⁻¹³px。第一ego姿态接近单位阵；各相机最大时间差约42.5ms，对应ego位移约0.46/0.41m，不能自动把全部残余归为学习问题。上述检查不证明原始rig推断本身正确，也没有把算术自洽冒充几何精度。</p><p>三时刻的1秒ego平移误差为0.880/0.491米，旋转误差0.324°/0.064°。本轮发现的是原生接口的范围限制及普通额外观测的改善，尚不是“重建误差损害生成状态”的新因果证据；下一项闭环实验须先说明具体状态输入、普通修复和任务后果。</p>
    <p>两次DVGT推理合计34.53秒（含加载与保存），PyTorch峰值16.77GiB，无OOM；0次新世界模型生成。两例均为已曝光开发日志，人工verdict为null，无新增failure卡。</p>
    <p><a href="native-contract/protocol.json">事前协议</a> · <a href="native-contract/result.json">完整结果</a> · <a href="native-contract/single_frame_audit/result.json">旧调用CPU审计</a> · <a href="https://github.com/wzzheng/DVGT/blob/51cf3f6d11fdff8bc7e2bbe1a88f71665ccb2236/README.md">官方输入约定</a></p></section>'''
