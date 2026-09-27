import argparse,json,pathlib
BASE=pathlib.Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,default=BASE/'outputs/v77-delete-full');OUT=p.parse_args().output
def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
data=read('summary.json');obs=read('observations.json');v=read('validation.json');d=read('drive_state.json');o=read('omega_state.json');b=read('build_validation.json');reg=read('registration.json')
lines=['# v77 DELETE：完整工程回放与边界','',obs['headline'],'',obs['overview'],'',
'task/run：`WS-V77-DELETE-FULL-20260927/r1`。源码提交与当前状态见远端v77分支。人工 `human_verdict: null`；以下视觉判断均为助手观察。','',
'''```mermaid
flowchart LR
 X[六相机RGB] --> M[已有SAM2 mask]
 X --> D[DriveEditor deletion / frozen]
 M --> D
 D --> O[六相机VGGT-Ω 512 / frozen]
 G[GT相机 + 框外LiDAR单尺度] --> O
 O --> B[逐时刻背景B_t]
 A[原GLB + GT位姿尺寸] --> F[原位factual]
 B --> F
 B --> Q[DELETE: actor.visible=false]
 F --> R[同步视频 + 纯几何诊断]
 Q --> R
```''','',
'## 实际执行','',
f'- DriveEditor：{len(d["completed_windows"])} / {d["total_windows"]} 窗；10帧/窗、stride9、seed42、25步；本轮总耗时{d["elapsed_s"]/60:.1f}分钟，记录的峰值allocated显存{max(x["peak_gib"] for x in d["completed_windows"]):.2f}GiB。完整官方训练后包初始化，执行deletion，无有效目标appearance/3D条件，不训练。',
f'- Ω：{len(o["completed_frames"])} / {o["expected_frames"]} 个时刻，每次联合六相机；512 balanced实际输入688×384。纯RGB网络预测后，使用GT相机和框外LiDAR拟合一个米制尺度。',
'- 原GLB：复用Hunyuan3D-2.1单图资产；没有重新生成。0230 yaw180°、0255 yaw0°，GT逐轴尺寸；Blender5.2.2、Cycles CPU、16 samples、seed77、固定世界光与太阳光。181个RGBA/camera-Z层。',
'- QUERY：读取B_t与GLB层，仅切换目标visible=False；四项测试覆盖目标唯一性、重复/不存在指令、背景不变、前后遮挡及无效深度。没有再次调用DriveEditor、Ω或3D生成器。',
'- 原位渲染沿用主点校准；已知z=5平面验证Blender Z pass直接为camera z，修正了原草稿中不应再除射线模长的实现。输出尚不包含光照拟合、地面接触阴影或反射。','',
'| 场景 / 目标 | 时间范围 | 处理目标视图数 / RGB视图数 | 几何覆盖率最低 / 平均 | 删除区域覆盖率最低 / 平均 |',
'|---|---|---|---|---|']
for s in data['scenes']:
    r=next(x for x in reg['scenes'] if x['name']==s['scene']);active=sum(len(c['active_frames']) for c in r['streams']);cov=s['geometry_coverage_min_mean'];m=s['deletion_mask_geometry_coverage_min_mean']
    lines.append(f'| {s["scene"]} / {s["actor_id"]} | f0–{s["frame_count"]-1}，{s["duration_s"]:g}s @10Hz | {active} / {s["frame_count"]*6} | {cov[0]:.1%} / {cov[1]:.1%} | {m[0]:.1%} / {m[1]:.1%} |')
lines+=['','覆盖率只表示有投影点，不能说明几何正确。主三栏在几何缺点像素使用保存的DriveEditor RGB补足；另存完全不补足的灰洞纯几何视频，避免以图像掩盖几何缺口。','',
'## 逐场景观察','']
for s in data['scenes']:
    ob=obs[s['scene']];lines+=['### '+s['scene']+' / actor'+s['actor_id'],'',ob['target'],'',ob['verdict'],'']+['- '+t for t in ob['observations']]+['']
    total=s['cross_camera_mask_sample_pairs'];frac=s['cross_camera_relative_depth_over_10pct_fraction'];pct=f'{frac:.1%}' if frac is not None else '无样本'
    lines.append(f'删除区域按8px网格采样重投影，得到{total}个跨相机样本对，其中相对深度差超过10%的比例为{pct}。该诊断混合了真实遮挡、估计误差和生成不一致，不是真值错误率，也不能据此单独断定哪一个模块失败。')
lines+=['','## 时序和mask合同','',
'0230固定SAM可见区域外包矩形：左/右/上8px、下24px。CAM5两个帧存在车外离散SAM污染，只留最大连通块；原mask、原登记和清理记录均保留。0255保留官方scale_bbox方案并物化随机结果，复用旧f15–24的已验证mask，必要时最小扩大覆盖SAM。不是两个scene共享一种mask的消融实验。',
'10帧滑窗用同一源时刻重叠1帧，上一窗composite作为下一窗首帧条件。最终保存旧窗重叠输出，不重复首帧；尾窗输入重复只用于填满模型输入，最终视频不重复尾帧。跨窗条件并不保证道路、栏杆和车辆语义一致。页面提供同时间旧窗末帧/新窗首帧比较。',
f'逐像素核验{b["decoded_rgb_views"]}个RGB视图：SAM均在最终mask内；背景局部合成mask外全部与原RGB一致。局部合成能保护mask外像素，不能保护扩大mask里的邻车和围栏。',
'继承此前SAM的小目标与近裁面门控；无mask的相机直接复制原RGB，并不等同“所有目标观测已清理”。本轮只把目标actor显式化，其他车辆仍作为背景内容保留。','',
'## 指令合法性与结果边界','',
'本次DELETE不产生新的占据体。助手通过源帧/连续抽帧核对目标身份和删除区域；没有实现神经网络或启发式交通规则判定。旧0230 MOVE碰撞和0255近距/静态占据拒绝保持，不运行新MOVE。',
'GT用于mask历史提示/门控、相机内外参、actor原位姿与尺寸，框外LiDAR用于单尺度；不是RGB端到端自动方案。B_t分别存储每个时刻，不称持久静态或4D世界。两个场景已曝光，无隐藏背景真实GT，官方训练重叠未核对。',
'本轮没有重新训练或修改Ω、Hunyuan、DriveEditor权重。失败只记录到具体视角、时段和处理环节，不据此否定GLB或模型家族。','',
'## 可复核产物和验证','',
f'- [审核页](index.html)：22段MP4，共{sum(int(x["nb_read_frames"]) for x in v["videos"])}个视频帧实际解码，均10Hz且匹配50/100帧长度。包含跟随视角三栏、六相机三栏、直接补景、mask、纯几何两栏、取消深度测试的同GLB控制；每scene十个时间点静帧。',
'- [validation.json](validation.json)：媒体实解码、本地文件链接、JavaScript语法；未宣称完成浏览器交互测试。',
'- [summary.json](summary.json)、[registration.json](registration.json)、[build_validation.json](build_validation.json)；每scene独立JSON保存factual/DELETE状态、指令、逐帧可见性和覆盖率。',
'- 远端完整资产根：`/root/autodl-tmp/runs/worldsim_v77/WS-V77-DELETE-FULL-20260927/r1`。`background_world/<frame>/`保留原Ω预测、点云、尺度诊断、投影深度和几何图；`cam*/windows/`保留所有原生DriveEditor输出；`actor_layers/`保留原位RGBA与Z。',
'- 本地审核包只含媒体与清单，不重复下载全部点云/模型。场景JSON中的asset_root_remote明确给出完整资产根。','',
'`failure_ledger_refs: [V77-F02]`；`failure_ledger_delta: updated V77-F02`；`human_verdict: null`。']
(OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(OUT/'report.md')
