"""Publish scoped code and rewrite the single current research snapshot."""
from pathlib import Path
import json,shutil,subprocess
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
CODE=REPO/'scripts/worldsim_v82/persistent_world_20261027';DOC=REPO/'docs/autoresearch/persistent_world_20261027'
CODE.mkdir(parents=True,exist_ok=True);DOC.mkdir(parents=True,exist_ok=True)
for p in (ROOT/'code').glob('*.py'):(CODE/p.name).write_text(p.read_text(encoding='utf-8'),encoding='utf-8')
for name in ['PLAN.md','architecture.svg','EVALUATION_PROTOCOL.md','evaluation_architecture.svg','ADMISSION_POLICY.json','admission_architecture.svg','native_training_architecture.svg','motion_diagnostics_architecture.svg','BENCHMARK_SCALE.md','benchmark_expansion_architecture.svg','benchmark_execution_architecture.svg']:
    (DOC/name).write_text((ROOT/'code'/name).read_text(encoding='utf-8'),encoding='utf-8')
quota=json.loads((ROOT/'quota_state.json').read_text())
native={p.parent.name:json.loads(p.read_text()) for p in (ROOT/'native_scene_training_v1').glob('*/status.json')}
current='''# 当前研究状态

更新：2026-10-27。任务WS-V82-PERSISTENT-WORLD-20261027，分支research/worldsim-v8.2-drivingeditor-v42。

**阶段二：多场景、多操作评测与失败归因。** 阶段一最小闭环已有实际输出；阶段三核心算法尚未确定。固定①编辑与资产接口和③预训练ProPainter，区分①实例/轨迹/资产、②背景污染或缺失、③后端修改问题。持久可编辑世界状态是框架目标，尚未证明论文创新。完整重建从冻结原V41-B/GN11直接输出，编辑独立载入副本，不改主副本。

[计划](autoresearch/persistent_world_20261027/PLAN.md) · [评测合同](autoresearch/persistent_world_20261027/EVALUATION_PROTOCOL.md) · [准入规则](autoresearch/persistent_world_20261027/ADMISSION_POLICY.json) · [规模合同](autoresearch/persistent_world_20261027/BENCHMARK_SCALE.md)

![逐场景训练与操作诊断](autoresearch/persistent_world_20261027/motion_diagnostics_architecture.svg)

## 实际完成

首个新增场景147053...的原生fresh30K正常返回：30000条有序有限更新，六档5K完整四件与各49帧评价核对。此前真实smoke为fresh1000+恢复1000额外更新、965.23秒，模型、Adam、Python/NumPy/Torch/CUDA RNG、采样状态完整恢复通过。CUDA整个training调用峰allocated 5.19GiB、reserved 6.59GiB，与每秒NVML采样峰分开记账。数据盘紧张时已按授权验证最终四件CPU可读及49帧评价，保存SHA/bytes清单后仅释放本场景5K–25K模型四件；最终30K和全部评价/事件保留。

首轮实际原生适配差0、主副本权重/张量版本未变；11个单请求变体×31帧、6类固定ProPainter，以及6次连续原生修订×31帧均完成。连续序列维持删除状态，反复操作同一个克隆，不重新出现原车；普通缓存31次命中及原相机回访差0只证明工程状态一致，不是世界记忆算法收益。连续请求的固定后端对照正在排队。

首次GPU入口因Torch2.2的mmap要求字符串路径失败，已修复并保留失败attempt，从新只读attempt接续，没有重训。成功目录由motion_suite_execution_v1/*/status.json的native_suite_path指明，不能读取失败旧目录代替结果。

可视证据：删除后静态SUV残影仍在；平移、复用插入和转向出现失真；固定视频将删除残影变为暗块，不能称已恢复正确背景；相机侧移也暴露其他车辆/背景问题。原生DEVA车身内缩2像素区域，动态合成权重均值0.8410、静态0.1588，但这不是污染率（玻璃/透明度/掩码本身可含背景）。资产问题、SH变换、静态遮挡仍需控制变量定位，不能直接归因道路缺几何。新增只读SH旋转控制CPU等变/逆变误差<2e-14，GPU渲染尚未执行；原11变体仍为原世界系SH基线，不悄悄替换旧结果。

## 其他场景与规模

30个追加上下文全部完成真实RGB、五LiDAR双回波、标定、位姿、物理微秒时间与标注提取，errors=[]；连同原3个新开发场景共33段、1617帧完整传感器输入。八类31时刻物理请求共264个。输入准入前瞻修订将18个停放车辆的变道/转向请求从普通运动层排除，当前178项输入筛查通过（开发53、封存125），正式准入和完成新八类基准仍为0。封存仅准备输入/固定请求，编辑结果不参与方法选择；物理路段重叠尚待审计。

目标至少109个有效编辑，执行目标128、长期240；HorizonForge论文109例，展示32编辑/128视频，不混作独立场景数。帧、种子、方法、checkpoint、生成前后不重复计数。旧216份六类草案不与264请求叠加。同协议效果和成本对比未完成，不宣称超过。

第二场景429544...通过真实smoke及完整恢复，已进入fresh30K。第三473735...等待第二场景首轮诊断。新100966...49帧DEVA62已逐帧核验，17273真实动态点；新255598...49帧DEVA31已核验，46005真实动态点，已分别排第4/5场景训练及诊断。100966停放车辆跳过无意义变速，转向只作静态姿态压力层。135064...目标关联在DEVA63/65切换且原生主连通组件守卫未通过，明确HOLD，不能标注已核验或硬塞训练。GT框仅身份建议与只读审计，不过滤训练点。

当前控制器快照（实际PID/start_ticks和child以campaign文件为权威）：
```json
NATIVE_STATUS
```

每场景只fresh原生30K一次，多编辑复用；每5K保存静态/动态/环境/完整训练状态与49帧评价。按训练→本场景多操作诊断→下一场景交错，CPU准备/分析异步；不让连续长训练挤掉操作诊断。新②训练和③切换仍禁启。健康worker不重启；零更新等待父的边界调整与首轮GPU接线失败均保留收据。当前分割父53383在GPU锁外等待，旧46606已在健康GPU子正常退出后重接门控，不恢复旧父。

## 接下来应解决的链路

1. 在已完成首场景结果上补静态遮挡/资产/SH变换控制，并汇总跨场景①②③失败矩阵；不把所有问题预归②。
2. 真实自车速度/变道/转向相机曲线与不同场景资产插入已接新的只读执行器，GPU entry与完整31帧排队，尚未声称通过。CPU核对49相机差1.11e-13、资产位置/协方差往返及真实尺寸碰撞；首插入请求0框碰撞。米制尺寸保留，底部仅按目标框粗对齐，不代表实测道路支撑。已有0.5m侧移、3°yaw和同资产克隆不冒充这四类已完成。
3. 连续编辑固定ProPainter对照：3次实际推理+2个完全相同输入/掩码的普通缓存控制；未知相机域不伪造生成掩码。复用一致性不等于正确三维背景，需要留出视角、参考域和生成负担共同验证。
4. 继续合格新开发目标关联、原生几何守卫、约10分钟真实smoke和逐场景30K；输入有问题与真实遮挡困难分层，不能按编辑效果删样。

历史scene0017/0029已有整动态分支操作及视频，不冒充精确多实例编辑。道路原始输入撤源v2是两个路段组件测试，参考为稀疏可见LiDAR域，不是全洞GT、完整B撤源重训或三维补全成功。旧保背景repair10K可只读评价，旧geometry正式10K未启动、旧三父调度不恢复。保留用户选定保背景状态，不追求道路贡献最大；199帧参考进入初始化，不能当独立GT或记忆收益。

资源停止线：工具周剩余QUOTA%，至少每30分钟复查；严格低于20%即禁启、最近可恢复边界保存、取回必要结果、提交推送验证并按授权关闭AutoDL，然后暂停同一心跳。此前持续研究，不因阶段结束/负结果/队列暂空关机。每秒NVML监控与历史峰不重置。磁盘紧张只按本场景30K完成后的精确中间四件保留策略处理，不清其他数据资产。保留用户桥接脚本修改，数据/模型/日志不入Git。

本地报告outputs/4D_Editing_Evaluation_20261027/index.html。quality_accepted=false，human_verdict=null。
'''.replace('NATIVE_STATUS',json.dumps({k:{f:v.get(f) for f in ['pid','start_ticks','phase','child','formal_steps','after_motion_context']} for k,v in native.items()},ensure_ascii=False,indent=2)).replace('QUOTA',str(quota['remaining_percent']))
(REPO/'docs/RESEARCH_STATUS.md').write_text(current,encoding='utf-8')
index=REPO/'docs/EXPERIMENTS.md';prefix='- WS-V82-PERSISTENT-WORLD-20261027：'
line=prefix+'[4D编辑评测驱动研究](autoresearch/persistent_world_20261027/PLAN.md)；阶段二按场景交错30K训练与多操作诊断，CPU准备异步，5K保存、多编辑复用；质量未接受，周剩余<20%快速保存关机。'
lines=[line if x.startswith(prefix) else x for x in index.read_text().splitlines()]
if not any(x.startswith(prefix) for x in lines):lines.append(line)
index.write_text('\n'.join(lines)+'\n',encoding='utf-8')
paths=['scripts/worldsim_v82/persistent_world_20261027','docs/autoresearch/persistent_world_20261027','docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md']
for args in [['add','--']+paths,['diff','--cached','--check'],['diff','--cached','--stat'],['status','--short']]:subprocess.run(['git','-C',str(REPO)]+args,check=True)
