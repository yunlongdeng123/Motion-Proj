from pathlib import Path
import json,shutil,html
W=Path(__file__).parent;O=W.parents[1]/'outputs/r15_repair_4k_20261018';O.mkdir(parents=True,exist_ok=True)
for name in ['experiment_contract.json','config.json','input_preflight.json','cpu_tests.log','cpu_integration.log','cpu_integration_pristine.log','launch.json','queue_status.json']:
    shutil.copy2(W/name,O/name)
status=dict(code='implemented',targeted_cpu_tests=dict(passed=26,failed=0),legacy_reference_suite=dict(passed=12,preexisting_failures=4,pristine_reproduces=True),
    actual_gpu_smoke='queued',new_training_updates=0,maximum_new_training_updates=4000,queue_controller_pid=20154,
    original_checkpoint_preserved=True,old5_baseline='running',shutdown_performed=False,human_verdict=None)
(O/'status.json').write_text(json.dumps(status,indent=2,ensure_ascii=False),encoding='utf-8')
diagram='''<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="300" viewBox="0 0 1120 300" role="img" aria-label="R15修复组件与数据流">
<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="#5d7193"/></marker></defs>
<rect width="1120" height="300" rx="12" fill="#f3f6fc"/>
<g fill="white" stroke="#7990b8" stroke-width="1.5"><rect x="18" y="52" width="192" height="100" rx="9"/><rect x="260" y="52" width="230" height="100" rx="9"/><rect x="540" y="52" width="230" height="100" rx="9"/><rect x="820" y="52" width="280" height="100" rx="9"/><rect x="260" y="206" width="510" height="65" rx="9"/></g>
<g fill="none" stroke="#5d7193" stroke-width="2" marker-end="url(#a)"><path d="M210 102H260"/><path d="M490 102H540"/><path d="M770 102H820"/><path d="M375 152V206"/><path d="M655 206V152"/></g>
<g fill="#192b49" font-family="Microsoft YaHei,sans-serif" text-anchor="middle" font-size="17"><text x="114" y="83">相同原始测量 / seed 0</text><text x="114" y="112">原 R15 三值掩码</text><text x="114" y="138" font-size="13">不加载已训练几何</text><text x="375" y="83">几何优化 + 有预算增密</text><text x="375" y="112">不可重置锚点 / 空间边界</text><text x="375" y="138" font-size="13">原损失、背景训练与 SGCM</text><text x="655" y="83">共享实际 Gaussian</text><text x="655" y="112">相机光栅 + FP64 阴影</text><text x="655" y="138" font-size="13">5 个太阳方向批量积分</text><text x="960" y="83">4K 检查点 + 49 帧对照</text><text x="960" y="112">独立 / 删除 / 0·1·unknown</text><text x="960" y="138" font-size="13">框外率、保体、a6 碎片与 a37 残影</text><text x="515" y="232">1.5K / 2.5K / 3.5K：当前贡献 + 重复实测 3D</text><text x="515" y="257" font-size="14">更新证据状态；证据充分才改 owner；记录候选与拒绝原因</text></g></svg>'''
(O/'architecture.svg').write_text(diagram,encoding='utf-8')
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>R15 修复与 4K 实验</title>
<style>body{font:16px/1.7 "Microsoft YaHei",sans-serif;color:#24334c;background:#eef2f8;margin:0}main{max-width:1120px;margin:32px auto;background:white;padding:32px;border-radius:16px}h1{font-size:30px;line-height:1.3}h2{font-size:21px;margin-top:32px}p{max-width:1000px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;vertical-align:top;border-bottom:1px solid #d9e1ed;padding:12px}th{background:#edf3fb}a{color:#215cc1}.notice{background:#fff3dc;padding:16px;border-left:4px solid #d49427}.good{background:#e6f6ed;padding:16px}code{background:#eef2f7;padding:2px 5px}img{width:100%}small{color:#64748b}li{margin:6px 0}</style>
<main><small>2026-10-18 · WS-V82-R15-REPAIR-4K-20261018 · R15 内收敛修复</small><h1>限制训练外扩，让当前几何真正进入归属纠错</h1>
<p class="notice"><b>当前：代码已实现；26 项针对性 CPU 测试通过；真实输入预检通过。新 4K 尚未开始。</b><br>远端控制器 PID 20154 已排队：旧五身份 30K → 旧五身份渲染 → 冻结三值投影 → 修复 GPU 自检 → 从头 4K → 49 帧逐身份评估。任何新阶段失败都保留资产并停止后续。</p>
<img src="architecture.svg" alt="R15输入、几何预算、相机和阴影、周期归属审计与49帧评估组件图">
<h2>先把现象与机制说清楚</h2><p>五辆车内部没有 unknown，是因为初始点和其 4K 后代均继承状态 1。全场景仍有 463,520 个 unknown 证据 Gaussian；外部未入场测量池与 Gaussian 是两类对象。原归属状态是 buffer，反传不会直接改它；归属本应由定期审计、显式 owner 转移完成。此前候选入口、reset 时机与证据门控使这条路径没有完成有效转移。</p>
<p>本次仍采用显式证据纠错：<b>几何和外观由梯度优化，证据状态和 owner 由重复观测审计更新</b>。明确反证可以使 1 → unknown；unknown 不会因为没证据而变成道路。内部 raw 状态是 0=unknown、1=Actor、2=static；可视化的二值“0=非目标”需与 raw 编码区别。</p>
<h2>实际代码修复</h2><table><thead><tr><th>问题</th><th>修复与本次启用值</th><th>证据边界</th></tr></thead><tbody>
<tr><td>增密不断重设参考，允许累计漂移</td><td>新增不可被 clone/split 重设的来源坐标。Actor 中心相对来源的累计位移 ≤0.25m；每步与拓扑后都投影。只清零被截断分量的 Adam 动量。</td><td>独立归属变更有实测依据时才允许重设坐标系与锚点。</td></tr>
<tr><td>软框损失不能阻止外扩</td><td>使用真实车框 +0.10m 余量；初始已越框种子保持其原始容许包络。最大主轴 σ 不超过原来源 σ 的 1.5 倍。</td><td>不按高度切车底，不把初始越框点删除。0.25m /0.10m /1.5 是本次明确策略，须检查保体代价。</td></tr>
<tr><td>Actor 点数指数增长</td><td>每次通用增密，每个 Actor 的净增量 ≤当前点数 2%；不超过初始数量 4 倍的增密容量。审计为 unknown 的 Actor 点暂停增密。</td><td>静态 SGCM、背景容量保持原设置。独立有证据的 owner 转入单独计数。</td></tr>
<tr><td>旧来源标签拦住当前错误点；状态一直继承 1</td><td>对当前全场景 alpha×T 计算正域、已知负域、实测道路域贡献。无旧 source tag 也能进入候选；重复反证撤销确定状态，重复身体 3D 可确认状态或改 owner。</td><td>只有具有 BUILD 三维证据库的 7 个身份可获认证。来自真实道路的正证据、过大支撑均可否决整点转车。</td></tr>
<tr><td>审计碰到 opacity reset</td><td>改为 1500、2500、3500 步；reset 仍在 3000 步。配置检查拒绝两者碰撞；逐次保存候选漏斗、拒绝与状态／owner 转移。</td><td>实际改 owner 可以仍为零；必须由数据说明，不能用“脚本完成”替代效果。</td></tr>
<tr><td>阴影数值失稳且过慢</td><td>把已验证的 FP64 因子积分接入模型和恢复入口，五太阳方向批处理；相机与阴影使用同一实际几何。</td><td>旧配置可精确历史重放。修复实验显式选用 factor_fp64；真实 GPU 自检尚在队列。</td></tr></tbody></table>
<h2>4K 实验合同</h2><ul><li>同一 R15 原始测量初始化、原三值掩码、seed=0；771,821 个初始 Gaussian、43 个 Actor。原 GN11／旧3／旧5／原 R15 检查点保留。</li>
<li>新目录从头训练 4,000 次；仍按原 geometry_steps=30,000 计算学习率，防止压缩日程造成混淆。保留原损失、100×完整已知域身份监督、SH、环境贴图、背景联合训练。</li>
<li>新路径审计现有点；外部候选池保留，不重放旧 3K 的 459 点补入。这项协议差异明确登记。</li>
<li>这是组合工程修复，不能从单次结果归因某一个改动。没有新增 R16／R17，也未恢复原 45K 长训。</li>
<li>输出 1K／2K／3K／4K 检查点及三次审计。评估 a5、a6、a18、a26、a37 全 49 帧：独立 RGB、alpha、全场景可见 alpha、删除，以及关闭阴影后的删除。</li></ul>
<h2>怎样判定有用</h2><p>首先检查有限梯度、恢复一致、阴影分块误差、累计漂移和新增越界；然后比较点数增长、精确框外率、带余量框外率、主轴尺度、证据状态变化及实际 owner 变化。必须同时看车体保留与已知正域可见 alpha，不能仅用框外率降低宣布改善。</p>
<p><b>a6：</b>检查独立资产下部碎片是否减少，以及轮胎／车体是否被过度约束。<b>a37：</b>检查删除后静态车形是否减少；若静态道路仅在外观上画了车，三维审计应拒绝整点转车。此项外观残影尚无修复通过证据。</p>
<h2>验证记录</h2><p class="good">针对性 26/26 通过：分裂谱系、累计边界、Adam 分量、旧行为兼容、真实审计函数的 unknown/owner 更新、FP64 薄高斯分块一致与有限梯度，以及既有归属／后端测试。</p>
<p>额外运行旧 CPU reference 集成套件：12/16 通过，4 项失败；在未改动的原 R15 源码上复现相同 4 项失败（3 项缺 reference 静态分支输出，1 项旧外观初始化断言）。这些未计为通过，也未改动无关历史测试。真实训练入口需由队列中的 GPU 自检验收。</p>
<p><a href="config.json">完整配置</a> · <a href="experiment_contract.json">实验合同</a> · <a href="input_preflight.json">真实输入预检</a> · <a href="cpu_tests.log">26 项测试</a> · <a href="cpu_integration.log">旧套件记录</a> · <a href="cpu_integration_pristine.log">原版复现记录</a> · <a href="queue_status.json">排队快照</a></p>
<p><small>页面为本次登记快照，不是实时状态。GPU 自检通过后才启动 4K；新旧全部任务、结果保存和 GitHub v82 推送完成后，按用户授权关闭 AutoDL。human_verdict 未填写。</small></p></main></html>'''
(O/'index.html').write_text(page,encoding='utf-8')
print(O.resolve())
