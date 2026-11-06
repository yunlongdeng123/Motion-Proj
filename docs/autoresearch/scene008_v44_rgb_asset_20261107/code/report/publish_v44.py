"""发布 v44 接线评估；GPU结果缺失时只展示明确标记的历史证据。"""
from pathlib import Path
import html
import json
import shutil

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
DEST = WORKSPACE/'outputs/scene008_v44_rgb_asset_integration'
PRIOR = WORKSPACE.parent/'10k-rgb-v1-v41-b-case/outputs/delete_comparison_review'
CURRENT = WORKSPACE/'outputs/scene008_vehicle_appearance/diagnosis'


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST/'assets').mkdir(exist_ok=True)
    (DEST/'code').mkdir(exist_ok=True)
    (DEST/'evidence').mkdir(exist_ok=True)
    for raw in (113, 137):
        for kind in ('reference', 'candidate'):
            shutil.copy2(PRIOR/f'assets/{kind}_{raw}.png', DEST/f'assets/RCR1_{kind}_{raw}.png')
        shutil.copy2(CURRENT/f'frames/{raw}/delete/surface_lod.png', DEST/f'assets/current_asset_{raw}.png')
    for file in (PRIOR/'evidence').glob('*.json'):
        if file.name in ('rcr1_silver315_contributions.json', 'user_acceptance_20261105.json',
                         'candidate_render_manifest.json', 'reference_render_manifest.json'):
            shutil.copy2(file, DEST/'evidence'/('historical_'+file.name))
    for folder in ('source', 'contract', 'review'):
        if (ROOT/folder).exists():
            shutil.copytree(ROOT/folder, DEST/'code'/folder, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for file in ROOT.glob('*receipt*.json'):
        shutil.copy2(file, DEST/'evidence'/file.name)
    diagram = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 295" role="img" aria-label="v44组件与数据流">
    <defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8Z" fill="#64748b"/></marker></defs>
    <style>text{font:17px system-ui,'Microsoft YaHei',sans-serif;fill:#203245}rect{fill:#eff5f8;stroke:#b9cbd8;rx:8}path.link{stroke:#64748b;fill:none;stroke-width:2;marker-end:url(#a)}</style>
    <rect x="12" y="15" width="244" height="70"/><text x="31" y="43">原始 RGB／LiDAR／轨迹</text><text x="31" y="68">同一组相机与原始观测</text>
    <rect x="310" y="15" width="305" height="70"/><text x="328" y="43">RCR-1 原生 10K（已保存）</text><text x="328" y="68">联合 RGB ×1 ＋实例约束</text>
    <rect x="310" y="114" width="305" height="70"/><text x="328" y="143">匹配的道路 1K／静态 gates</text><text x="328" y="168">固定 11×9 色场迁移</text>
    <rect x="12" y="215" width="603" height="65"/><text x="32" y="243">当前完整 a6：C2 表面 ＋固定 RGBA ＋ SurfaceLOD</text><text x="32" y="266">PVD 与资产参数冻结；原生 a6 退出成像</text>
    <rect x="690" y="100" width="365" height="95"/><text x="711" y="129">v44：所有层统一深度排序</text><text x="711" y="155">保留 a37 ／只删除 a37</text><text x="711" y="179">输出 RGB、深度、分组 RGB×T</text>
    <path class="link" d="M256 50 H310"/><path class="link" d="M462 85 V114"/><path class="link" d="M615 149 H690"/><path class="link" d="M615 248 H650 V178 H690"/>
    </svg>'''
    (DEST/'architecture.svg').write_text(diagram, encoding='utf-8')
    sections = []
    for raw in (113, 137):
        cards = []
        for src, title in [(f'RCR1_reference_{raw}.png', '历史配对：RCR-1 采用前／旧 M2'),
                           (f'RCR1_candidate_{raw}.png', '历史配对：RCR-1 去蓝后／同一旧 M2'),
                           (f'current_asset_{raw}.png', '当前车体／旧背景（尚未接去蓝）')]:
            cards.append(f'<figure><figcaption>{title}</figcaption><a href="assets/{src}"><img src="assets/{src}" alt="raw{raw} {title}"></a></figure>')
        sections.append(f'<h3>raw{raw} · 原始 480×320 PNG</h3><div class="triple">'+''.join(cards)+'</div>')
    cpu_receipt = ROOT/'contract/cpu_preflight_receipt.json'
    if cpu_receipt.is_file():
        cpu = json.loads(cpu_receipt.read_text(encoding='utf-8-sig'))
        cpu_note = ('实际远端 CPU 输入预检通过；相机、轨迹、道路行号与当前资产兼容性已检查。GPU 光栅器和画质仍待验证。'
                    if cpu.get('status') == 'PASS_V44_CPU_INPUT_CONTRACT'
                    else 'CPU 输入预检尚有阻塞，见收据；不能开始 GPU 评价。')
    else:
        cpu_note = 'CPU 输入预检尚未完成；代码与依赖正在复核。'
    report = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>v44 · 当前车体与 RCR-1 去蓝背景接入评估</title><style>
    *{box-sizing:border-box}body{margin:0;color:#203245;background:#fff;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1350px;margin:auto;padding:32px 24px}h1{font-size:28px;line-height:1.35}h2{font-size:21px;margin-top:32px}h3{font-size:17px}.status{background:#fff4d8;border-left:4px solid #b78013;padding:14px 18px}.note{color:#596b7a}.triple{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}figure{margin:0}figcaption{font-size:14px;margin:6px 0}img{display:block;width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:10px;border-bottom:1px solid #d9e3eb;text-align:left;vertical-align:top}a{color:#196389}code{overflow-wrap:anywhere;background:#f2f5f8;padding:2px 4px}.scroll{overflow:auto}li{margin:8px 0}footer{margin-top:32px;padding-top:16px;border-top:1px solid #d9e3eb;font-size:14px;color:#596b7a}@media(max-width:850px){.triple{grid-template-columns:1fr}.scroll img{min-width:780px}main{padding:20px 14px}}
    </style><main>
    <h1>v44：当前车体资产与 RCR-1 去蓝背景接入</h1>
    <p class="status"><strong>当前阶段：代码接线与 CPU 预检；新组合尚未 GPU 渲染。</strong><br>远端 SSH 可用，当前 CUDA 设备数为 0。没有新 v44 画质指标、没有训练和自动电源操作；以下图片是历史证据。</p>
    <p>__CPU_NOTE__</p>
    <p>可以接入，但去蓝方法修改的是原生场景训练与配套道路接管，不能只往车体资产上加一个颜色滤镜。v44 保留当前 C2／RGBA／SurfaceLOD 车体，使用 RCR-1 已保存的联合 RGB×1 原生 10K、匹配道路 1K 与 gates，并明确迁移原有世界坐标颜色场。</p>
    <div class="scroll"><img src="architecture.svg" alt="输入、RCR原生背景、当前车体和统一深度排序的组件图"></div>
    <h2>异常评估</h2><table><tr><th>现象</th><th>已有证据</th><th>v44 的处理与边界</th></tr>
    <tr><td>蓝紫色道路、局部颜色割裂</td><td>RCR-1 在原生 10K 补回联合 RGB×1；历史旧 M2 配对中，raw113 原框绝对蓝色误差 0.084625→0.003897，raw137 右斑 0.074090→0.004888。用户已采用该去蓝结果。</td><td>复用已有原生 10K 与它自己的新道路、行号和 gates。新车体会改变遮挡，历史数字不能冒充 v44 效果。</td></tr>
    <tr><td>车底／车后银灰亮片</td><td>历史 raw137 的 315 亮点中，静态 premultRGB 净亮化约 +0.3311；环境×T 约 −0.19285；a6／a37 为 0，道路接近 0。环境×T 仍含车形结构；这些点在既有道路接管范围外。</td><td>去蓝未解决该项。新组合输出同一次排序下各层真实 RGB 与 alpha×T，先复查静态层与环境；不扩车体或任意删高斯掩盖亮片。</td></tr>
    <tr><td>既有暗团／阴影样斑块</td><td>此前 raw105 暗核主要为近道路的黑色旧静态前缀，raw113 底边暗核主要为道路基底 RGB；没有独立物理投影阴影层。</td><td>保留为独立问题，不把银亮片、暗团和蓝斑合成一个成因，也不宣称换车体已经修复背景。</td></tr></table>
    <p class="note">历史道路 target 是跨时刻重观测 proxy，始终遮挡区域没有独立删车 RGB 真值。315 亮点是旧版本诊断选区，不作为新版本评价分母。</p>
    <h2>历史图像证据：去蓝能力与亮片代价</h2>
    <p>前两列为同一旧 M2 的历史配对；第三列展示当前车体与旧背景。第三列与前两列车体不同，不能据此把差异归因于单项修复。真正 v44 对照将在保持当前车体完全一致后生成。</p>
    __FIGURES__
    <h2>接线规则</h2><ul>
    <li><strong>版本独立：</strong>运行入口、契约、输出和代码统一使用 v44；RCR-1 是去蓝组件来源，v43 及更早现场不覆盖。</li>
    <li><strong>车体保持：</strong>canonical 资产 45,314 行，C2 后验与固定 RGBA 不训练；SurfaceLOD 按相机采样，保留／删除状态共用同一车体数据。</li>
    <li><strong>背景完整成套：</strong>新原生模型必须配它自身重建的道路 1K、contributors 与 gates；预检拒绝旧静态行号串入新模型。</li>
    <li><strong>色场边界：</strong>原 11×9 色场在旧 M2 联合场景校准。首版仅作冻结世界坐标残差迁移，保留源绑定及新绑定；没有伪称重新标定。</li>
    <li><strong>成像一致：</strong>原生 a6 从输入物理移除；所有剩余层进入同一次 native Gaussian 深度排序，删除操作只移除 a37，不切换背景旁路。</li></ul>
    <h2>GPU 恢复后的最小验证</h2>
    <p>先 raw105、113、130、137 的双状态小对照，通过工程检查后再完整 49 帧。两组都使用当前同一车体：当前背景 vs v44 去蓝背景；固定原始可见 a6／a37、LiDAR 和道路候选域，报告误差、低覆盖点与逐层来源。保留天空、其他车辆与非目标区域的变化，不只检查蓝框。</p>
    <p>本轮不重跑 10K，不训练 PVD，不重拟合车体，不启动道路生成模型。色场冻结迁移若出现问题，再单独登记相同方法的重标定对照，不暗中增加训练。</p>
    <p><a href="code/source/evaluate_v44.py">v44 成对渲染入口</a> · <a href="code/source/v44_runtime.py">独立运行时</a> · <a href="code/contract/input_contract.json">输入契约</a> · <a href="code/contract/cpu_preflight_receipt.json">CPU 预检收据</a> · <a href="code/review/REVIEW.md">代码审查</a></p>
    <p><a href="../scene008_vehicle_appearance/diagnosis/index.html#frame-compare">当前车体完整 49 帧报告</a> · <a href="../scene008_vehicle_appearance/diagnosis/data/shadow/qa_receipt_v2.json">既有暗斑诊断收据</a> · <a href="evidence/historical_rcr1_silver315_contributions.json">历史银亮片归因</a></p>
    <footer>2026-11-07（Asia/Singapore）。v44 是当前资产＋RCR-1 背景的接入版本；尚未确认该新组合画质，不继承旧 M2 的完整画质 verdict。AutoDL 保持开机，未启动等待 GPU 的后台控制器。failure_ledger_delta=none。</footer>
    </main></html>'''.replace('__FIGURES__', '\n'.join(sections)).replace('__CPU_NOTE__', html.escape(cpu_note))
    (DEST/'index.html').write_text(report, encoding='utf-8')
    readme = '''# v44：当前车体与 RCR-1 去蓝背景接入

代码接线与 CPU 预检阶段，GPU 当前不可用；新组合未渲染，未确认画质。

![组件](architecture.svg)

RCR-1 的已保存 native10K（补全图 RGB×1）与自身 fresh road1K/gates，配当前 C2/固定RGBA/SurfaceLOD。原 11×9 色场仅作 frozen-transfer，不称为新场景重新标定。保持同一车体，原生 a6 物理移除，保留/删除仅开关 a37；原生合成不等于 MOG-2 不透明首交。

历史 raw137 银亮片净增亮主要是 static RGB，环境×T 含车形；去蓝没有解决银亮片。新组合 GPU 后按固定 ROI 和真实分组 RGB 复查，不用亮度后选子集充当固定分母。HELD9 已参与内部开发，不是独立泛化。

执行：见 `RUN_COMMANDS.txt`。第一步实际 GPU pilot 四帧，工程检查后再 full49。无新训练、无模型权重更改，无自动电源操作。现有 v43/RCR-1/车体报告不覆盖。

failure_ledger_refs=[]；failure_ledger_delta=none；human_verdict=null。
'''
    (DEST/'README.md').write_text(readme, encoding='utf-8')
    commands = '''# 手动在带GPU环境执行；无自动排队、不开关实例。旧模型/结果保留。
cd /root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_rgb_asset_20261107_r1
/root/autodl-tmp/envs/drivingeditor/bin/python contract/v44_contract.py contract/input_contract.json --receipt contract/cpu_preflight_recheck.json
# GPU恢复后：
/root/autodl-tmp/envs/drivingeditor/bin/python source/evaluate_v44.py --contract contract/input_contract.json --mode pilot --output results/pilot_r1
# 人工/工程复核pilot结果后再执行，不是自动已排队：
/root/autodl-tmp/envs/drivingeditor/bin/python source/evaluate_v44.py --contract contract/input_contract.json --mode full49 --output results/full49_r1
'''
    (DEST/'RUN_COMMANDS.txt').write_text(commands, encoding='utf-8')
    print(DEST/'index.html')


if __name__ == '__main__':
    main()
