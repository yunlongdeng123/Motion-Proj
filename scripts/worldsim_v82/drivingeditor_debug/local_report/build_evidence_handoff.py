"""Build only the independent evidence handoff. No remote/GPU/model/media work."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone, timedelta
from urllib.parse import quote, urlsplit, unquote
from html import escape
from html.parser import HTMLParser
import hashlib
import json
import os
import re

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'outputs/drivingeditor_evidence_handoff'
DEBUG = ROOT / 'outputs/drivingeditor_vgpu_debug'
WORK = ROOT / 'work/vgpu_bugfix_20261012'
BASE = '/root/autodl-tmp/drivingeditor_repro'
D = BASE + '/campaign/vgpu_bugfix_20261012'
FROZEN_SHA = '68a2bba9082da4f0e92abef0354808094eb2d2ae599e2011766e2d24fff54419'
BRANCH = 'research/worldsim-v8.2-drivingeditor-debug'
REPO = 'https://github.com/yunlongdeng123/Motion-Proj'
DEST.mkdir(parents=True, exist_ok=True)
refs: list[dict] = []
inputs: list[dict] = []

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p: Path):
    j = json.loads(p.read_text(encoding='utf-8-sig'))
    inputs.append({'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)})
    return j

def link(p: Path, label: str, cls='') -> str:
    if not p.exists():
        raise FileNotFoundError(str(p))
    rel = Path(os.path.relpath(p, DEST)).as_posix()
    refs.append({'href': rel, 'path': p.relative_to(ROOT).as_posix(), 'label': label})
    class_attr = f' class="{escape(cls, quote=True)}"' if cls else ''
    return f'<a{class_attr} href="{quote(rel, safe="/._-")}">{escape(label)}</a>'

def local(rel: str, label: str) -> str:
    return link(ROOT / rel, label)

def debug(rel: str, label: str) -> str:
    return link(DEBUG / rel, label)

def audit(rel: str, label: str) -> str:
    return link(WORK / rel, label)

def links(items) -> str:
    return '<div class="links">' + ''.join(items) + '</div>'

def table(headers, rows, cls='') -> str:
    return f'<div class="table-wrap"><table class="{cls}"><thead><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'

def code(s: str) -> str:
    return '<code>' + escape(s) + '</code>'

s_handoff = read(DEBUG / 'S_report_handoff.json')
task = read(WORK / 'task_manifest.json')
s = read(DEBUG / 'evidence/S_observed_static_rgb30k_to32000/receipts/S_vs_originals_locked_metrics.json')
v = read(DEBUG / 'evidence/V_opacity_outside30k_to32000/receipts/three_arm_locked_metrics.json')
core = read(WORK / 'sgcm_data_audit/observed_static_rgb_S_review/actual_S_locality_core_cpu.json')
temporal = read(WORK / 'sgcm_data_audit/fixed_pose_time_review/actual_temporal_interpretation_cpu.json')
sgcm = read(WORK / 'sgcm_data_audit/audit_findings.json')
time_contract = read(WORK / 'loss_audit/next_diagnostics_cpu/mdet_time_contract_cpu/time_contract_handoff_receipt.json')
clock = read(WORK / 'loss_audit/next_diagnostics_cpu/mdet_time_contract_cpu/clock_source_audit_cpu.json')
native = read(WORK / 'preprocessing_gap_audit/mdetector_adapter_candidate/build_attempt_01/native_build_receipt.json')
shadow = read(WORK / 'preprocessing_gap_audit/shadow_weight_access_20261012.json')
receipt_path = DEST / 'delivery_receipt.json'
delivery = read(receipt_path) if receipt_path.exists() else {}
assert sha(DEBUG / 'index.html') == FROZEN_SHA, 'Frozen report changed'
assert s_handoff['reference_count'] == 138
assert s_handoff['static_report_QA']['checks'] == 15
assert not native['detector_initialized'] and not native['real_scans_processed']
assert not time_contract['author_equivalence'] and not time_contract['I_readiness']

def find_metric(j, region, array, metric):
    return next(x for x in j['primary_metrics'] + j['context_metrics_all_six_domains'] if x['region']==region and x['array']==array and x['metric']==metric)

def row_values(region, array, metric, kind='float'):
    sm = find_metric(s, region, array, metric)
    vm = find_metric(v, region, array, metric)
    vals = [sm['equal_frame_mean_B1'], sm['equal_frame_mean_B2'], vm['equal_frame_mean_V'], sm['equal_frame_mean_S']]
    if kind=='percent': return [f'{x*100:.3f}%' for x in vals]
    if kind=='mse': return [f'{x:.8f}' for x in vals]
    return [f'{x:.5f}' for x in vals]

metric_rows = [
    ['全图 composite PSNR ↑ · dB', *row_values('full','composite','clamped_rgb_psnr_db')],
    ['全图 composite MSE ↓', *row_values('full','composite','clamped_rgb_mse_vs_gt','mse')],
    ['已观测非天空车体外 static PSNR ↑ · dB', *row_values('non_sky_outside_actor','static','clamped_rgb_psnr_db')],
    ['同域 dynamic α 均值', *row_values('non_sky_outside_actor','alpha_d','mean')],
    ['同域 dynamic RGB 支撑比例', *row_values('non_sky_outside_actor','dynamic','rgb_support_gt2_uint8_pixel_fraction','percent')],
    ['actor composite PSNR · dB（上下文）', *row_values('actor','composite','clamped_rgb_psnr_db')],
    ['actor 像素汇总 MSE ↓（上下文）', *[f'{core["core_regions"]["actor"][a]["metrics"]["composite_clamped_MSE"]["pixel_weighted_mean"]:.9f}' for a in ['B1','B2','V','S']]],
    ['5 px actor core 像素汇总 MSE ↓', *[f'{core["core_regions"]["actor_core"][a]["metrics"]["composite_clamped_MSE"]["pixel_weighted_mean"]:.9f}' for a in ['B1','B2','V','S']]],
]

report_rows = [
    ['当前 vGPU 排错主报告', debug('index.html','打开冻结主报告'), 'B1/B2/V/S、短续训、组件、时间、初始化见证；138引用、S静态QA15项。'],
    ['3090 历史 30k 与本次 30k', local('outputs/drivingeditor_30k_edit_comparison/index.html','打开历史/当前对照'), '原3090为reset前历史21帧；当前49帧。共同窗口21帧，分开显示、分开统计。'],
    ['10k / 20k / 30k checkpoints', local('outputs/drivingeditor_checkpoint_comparison/index.html','打开同步视频'), '真实保存节点与四组件导出；不是B1/B2/V/S32k指标。'],
    ['20k→30k 分支诊断原始数据', local('outputs/drivingeditor_branch_diagnosis/evidence/diagnosis.json','diagnosis.json'), '训练曲线、节点快照、对齐帧；此目录没有HTML入口。'],
    ['论文协议 / 代码与方法缺口', local('outputs/drivingeditor_paper_alignment/index.html','打开论文核对'), '数据、公式、SGCM、完整状态及from-zero计划的历史审计。'],
    ['内存 / renderer / resume 核验', local('outputs/drivingeditor_infra_review/index.html','打开基础设施核验'), '严格恢复与候选CPU证据；compact及renderer cleanup未部署。'],
    ['官网场景与实际 Waymo 原帧', local('outputs/drivingeditor_scene_evidence/index.html','打开场景证据'), '场景对应、搜索范围与原始证据。'],
    ['环境和数据准备', local('outputs/drivingeditor_setup/index.html','打开环境准备'), '历史环境安装、数据及模型位置。'],
    ['49帧输入与actor预处理', local('outputs/drivingeditor_paper49_data/dynamic_actor_validation.json','actor验证')+' · '+local('outputs/drivingeditor_paper49_data/mask_validation.json','mask验证'), '既有DEVA/SAM body输入，不能称M-detector/shadow预处理。'],
    ['其它场景 / 队列与编辑工具', local('outputs/drivingeditor_campaign/index.html','打开历史队列'), 'oncoming、red-scene、mirrors、编辑候选与失败记录；均为历史，取消后不继续。'],
]

fact_rows = [
    ['checkpoint / 导出', '<span class="tag fact">已核验</span> 新30k四件模型、固定相机和196 PNG一致；重渲染与保存评估逐像素相同。', '尚无模型/环境错配证据。旧3090清晰图与当前saved30k处于reset的不同侧。', links([debug('evidence/diagnostic_manifest.json','组件运行回执'), debug('evidence/train_eval_semantics.json','train/eval语义'), debug('evidence/no_modification_sha.json','原件保护')])],
    ['reset 与早期分支混入', '30k所有2,234,298个static base opacity≤0.01；训练PSNR step30000=34.513，30001=8.591；saved30k全49=10.484 dB。', '每3000步static reset在15k增密结束后继续。短续训恢复全图，但20k static道路已彩噪，reset不能解释全部分支退化。', links([audit('loss_audit/events_loss_trends_cpu.json','所有event与窗口'), debug('evidence/checkpoint_geometry.json','模型统计')])],
    ['6组 × 5相机组件替换', '原20k / 原30k为35.33 / 10.85 dB；仅替换30k static→20k为33.55，env→20k为11.78。40基线PNG exact，64保护项不变。', '20k static高斯颜色贡献暗，彩噪直接位于env·(1−αs)。组件混配是诊断，不能当训练模型或全49结果。', links([debug('evidence/component_case_summary.json','组件定量'), debug('assets/static_env_contribution_20k_0017010.png','实际env贡献')])],
    ['道路观测', '原车体投影地面抽样raw0/10/17：93.56% / 99.41% / 100%存在其它帧可见性和真实地面LiDAR支持。', '已验证子集不能解释为“从未观测”；不是dense无遮挡GT，raw48该抽样不确认。', links([audit('sgcm_data_audit/road_observation_coverage.json','完整覆盖回执')])],
    ['损失与梯度', 'Eq7/13解析梯度、Eq9独立loop/gradcheck、Eq16归一化、dynamic xyz LR等检查通过。完整已实现目标存在逐像素分支分配等价构造。', '等价构造不证明有限3D Gaussian可实现或真实优化路径。NPS权重/采样、mask域、harmonic900sky等仍有作者实现缺口。', links([audit('loss_audit/HANDOFF.md','损失完整交接'), audit('loss_audit/loss_formula_gradient_cpu.json','公式与梯度'), audit('loss_audit/candidates/candidates_tests_remote_cpu.json','反例与候选CPU')])],
    ['稀疏深度碰撞', '<span class="tag fact">已确认代码问题</span> duplicate pixel last-writer依赖点顺序；实际3帧8.09–11.73%有效像素写入了比最近点更远的z。反转顺序可翻转Eq16梯度。', 'nearest positive finite z候选CPU通过；尚无隔离GPU因果实验，不能认定为分支退化主因。逆深度由论文跨页定义明确要求。', links([audit('loss_audit/sparse_depth_collisions_cpu.json','真实碰撞数据'), audit('loss_audit/candidates/nearest_depth.patch','默认独立候选'), audit('loss_audit/depth_definition_followup/depth_definition_followup_cpu.json','深度定义补充')])],
    ['SGCM 重复 / 无贡献点', '20k exact xyz额外copies=999,266；其中773,564 clone在全部连续模型时间−0.48..0.48低于实际1/255 alpha cutoff。', '这批解析不可见点不能产生彩噪。visibility guard默认关，CPU7项通过；SGCM在30k→32k不触发，未做早期独立干预。', links([audit('sgcm_data_audit/audit_findings.json','SGCM完整结论'), audit('sgcm_data_audit/visibility_guard_cpu_regression.json','guard回归'), audit('sgcm_data_audit/static20k_dead_clone_index_manifest.json','dead-clone索引')])],
    ['mask / 单位 / 原始点', '49 mask与masked RGB逐一匹配；sky overlap=0；scene scale=0.1；600,000 PLY抽样及36,406 actor点原始回放exact。', '未见确定帧错位、plane/单位/parent/旋转/Adam索引错误。ground proximity或box投影不能直接用作actor身份。', links([audit('sgcm_data_audit/mask_alignment_49frames.json','mask与时间'), audit('sgcm_data_audit/metric_and_normal_invariants.json','单位与normal'), audit('loss_audit/next_diagnostics_cpu/static_initialisation_actor_cpu/input_lineage_contract.json','point lineage约定')])],
]

model_rows=[]
source_rows=[]
for arm, folder in [
    ('B1 原版','B1_original30k_to32000'),('B2 原版重复','B2_original30k_to32000_noise_reference'),
    ('V 0.05 outside α','V_opacity_outside30k_to32000'),('S 1.0 observed static RGB','S_observed_static_rgb30k_to32000')
]:
    manifest_rel = f'evidence/{folder}/terminal_32000/diagnostic_snapshot_manifest.json'
    m=read(DEBUG / manifest_rel)
    chk=m['checkpoints']
    dirs={str(Path(x['path']).parent).replace('\\','/') for x in chk.values()}
    model_rows.append([arm, code(next(iter(dirs))), debug(manifest_rel,'四件状态与SHA'), '<br>'.join(f'{escape(k)} {code(x["sha256"][:12]+"…")}' for k,x in chk.items())])
    source_root=next(x for x in m['source_pins'] if x.endswith('/train.py')).removesuffix('/train.py')
    source_rows.append([arm,code(source_root),debug(manifest_rel,'实际source pins')])

inventory_files = [DEST / 'local_inventory.json', DEST / 'remote_inventory.json']
inv_links=[]
for p in inventory_files:
    if p.exists(): inv_links.append(link(p,p.name))
inv_links.extend([link(DEST/'manifest.json','本页引用与输入pin清单')]) if (DEST/'manifest.json').exists() else None
package_links=[]
for p in sorted((ROOT/'outputs').glob('*evidence*')):
    if p.is_file() and p.suffix.lower() in ('.zip','.gz','.7z'):
        package_links.append(link(p,p.name))
for p in sorted(DEST.glob('*')):
    if p.is_file() and p.suffix.lower() in ('.zip','.gz','.7z'):
        package_links.append(link(p,p.name))

receipt_link = link(receipt_path,'取消与Git实际交付回执') if receipt_path.exists() else '<span class="muted">取消与Git实际回执：由root完成后绑定</span>'
commit = delivery.get('commit') or delivery.get('commit_sha') or delivery.get('pushed_commit')
repo = delivery.get('repo', REPO)
branch = delivery.get('branch',BRANCH)
git_url = f'{repo}/tree/{quote(branch,safe="/")}'
git_state = '<b>远端分支已推送并验证</b>' if delivery.get('push_verified') is True else '<b>待root最终绑定push回执</b>'
git_box = f'<p>{git_state} · <a href="{escape(git_url,quote=True)}">{escape(branch)}</a> · {code(str(commit)) if commit else "commit尚未绑定"}</p><p class="muted">v8.2按现有research版本格式，从v8.1起点 {code("c21547193d765e8f8123b27b73e31e033295a316")} 创建。代码版本与实验来源由交付回执记录；此页面不重新运行研究。</p>'
stamp=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')

css='''
:root{color-scheme:dark;--bg:#0b1018;--card:#111b29;--card2:#152235;--line:#2b3b50;--ink:#e6eef7;--muted:#a4b4c6;--accent:#69dbcf;--amber:#ffcf86;--rose:#ffa1a6}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 "Segoe UI","Microsoft YaHei",system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:32px 24px 64px}h1{font-size:32px;line-height:1.25;margin:10px 0 16px}h2{font-size:22px;line-height:1.35;margin:0 0 14px}h3{font-size:17px;margin:18px 0 8px}p{margin:10px 0}a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}small,.muted{color:var(--muted)}code{font-family:Consolas,monospace;font-size:12px;color:#c6d8ed;overflow-wrap:anywhere}section{margin:24px 0;padding:22px;background:var(--card);border:1px solid var(--line);border-radius:12px}header{padding:0 0 8px}.badge,.tag{display:inline-block;border:1px solid #4a4a3d;color:var(--amber);background:#29251b;border-radius:5px;padding:2px 8px;font-size:12px}.tag.fact{background:#14302b;color:var(--accent);border-color:#245b51}.tag.hyp{background:#292344;color:#c6b7ff;border-color:#514475}.callout{border-left:4px solid var(--amber);padding:12px 16px;background:#22251f;border-radius:5px}.links{display:flex;flex-wrap:wrap;gap:7px 10px;margin:10px 0}.links a{display:inline-block;background:#182c3a;border:1px solid #2c4b5b;border-radius:5px;padding:4px 9px;font-size:13px}.toc{display:flex;gap:8px 15px;flex-wrap:wrap;font-size:14px}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:10px 11px;vertical-align:top;border:1px solid var(--line);text-align:left}th{background:#1b2a3e;color:#d1e0f1}td:nth-child(1){font-weight:600}tr:nth-child(even){background:#132032}.numeric td:not(:first-child),.numeric th:not(:first-child){text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}.numeric td:first-child{min-width:215px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.grid>div{padding:15px;background:var(--card2);border-radius:8px}.limit{color:var(--amber)}.architecture{display:block;width:100%;height:auto;min-width:850px}.diagram-wrap{overflow:auto}.architecture text{font-family:"Segoe UI","Microsoft YaHei",sans-serif}.architecture .box{fill:#183040;stroke:#4b7185;stroke-width:1.4}.architecture .missing{fill:#312a1c;stroke:#a38a57;stroke-dasharray:5 4}.architecture .t{fill:#e6eef7;font-size:16px;text-anchor:middle}.architecture .s{fill:#b7cadb;font-size:12px;text-anchor:middle}.architecture .arrow{fill:none;stroke:#74c4ce;stroke-width:2;marker-end:url(#arrow)}details{margin:14px 0;border-top:1px solid var(--line);padding:12px 0 0}summary{cursor:pointer;color:var(--accent);font-weight:600}figure{margin:12px 0}img{max-width:100%;height:auto;border-radius:6px;border:1px solid var(--line)}figcaption{color:var(--muted);font-size:12px;margin-top:6px}ul{padding-left:22px}li{margin:7px 0}footer{color:var(--muted);font-size:12px;padding-top:20px}@media(max-width:800px){main{padding:22px 14px 40px}h1{font-size:27px}section{padding:16px}.grid{grid-template-columns:1fr}}
'''
svg='''<svg class="architecture" viewBox="0 0 1160 290" role="img" aria-labelledby="arch-title arch-desc" xmlns="http://www.w3.org/2000/svg"><title id="arch-title">DrivingEditor 当前实现的组件与证据流</title><desc id="arch-desc">Waymo输入经过现有初始化和body masks进入静态与动态高斯，共享环境组成渲染和损失，输出实际49帧与编辑诊断；M-detector和shadow尚缺。</desc><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#74c4ce"/></marker></defs>
<rect class="box" x="10" y="95" width="150" height="86" rx="8"/><text class="t" x="85" y="123">Waymo 49帧</text><text class="s" x="85" y="147">RGB · LiDAR · pose</text><text class="s" x="85" y="165">FRONT raw0–48</text>
<rect class="box" x="205" y="95" width="165" height="86" rx="8"/><text class="t" x="287" y="123">现有初始化 / mask</text><text class="s" x="287" y="147">PCA scale=0.1</text><text class="s" x="287" y="165">DEVA/SAM body + sky</text>
<rect class="box" x="420" y="20" width="176" height="76" rx="8"/><text class="t" x="508" y="47">Static Gaussian</text><text class="s" x="508" y="70">SGCM · NPS · 时间字段</text>
<rect class="box" x="420" y="175" width="176" height="76" rx="8"/><text class="t" x="508" y="202">Dynamic Gaussian</text><text class="s" x="508" y="225">foreground RGB · α</text>
<rect class="box" x="670" y="14" width="170" height="64" rx="8"/><text class="t" x="755" y="40">共享 EnvMap</text><text class="s" x="755" y="60">两分支训练共享残差</text>
<rect class="box" x="670" y="111" width="170" height="100" rx="8"/><text class="t" x="755" y="139">渲染 / 损失 / Adam</text><text class="s" x="755" y="164">RGB · sky · depth</text><text class="s" x="755" y="187">B1 / B2 / V / S 诊断</text>
<rect class="box" x="914" y="111" width="231" height="100" rx="8"/><text class="t" x="1029" y="139">实际结果与证据</text><text class="s" x="1029" y="163">49帧 composite / static / dynamic</text><text class="s" x="1029" y="185">指标 · SHA · 移除 / 平移诊断</text>
<rect class="missing" x="205" y="222" width="165" height="54" rx="8"/><text class="s" x="287" y="245">缺口：M-detector / shadow</text><text class="s" x="287" y="263">尚无实际输入补齐</text>
<path class="arrow" d="M160 138H205"/><path class="arrow" d="M370 138H394V58H420"/><path class="arrow" d="M394 138V213H420"/><path class="arrow" d="M596 58H630V135H670"/><path class="arrow" d="M596 213H630V185H670"/><path class="arrow" d="M755 78V111"/><path class="arrow" d="M840 161H914"/>
<path class="arrow" d="M85 181V280H650V204H670"/><text class="s" x="510" y="277">原 RGB / mask / LiDAR 监督</text></svg>'''

html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DrivingEditor · 现有全部证据交接</title><style>{css}</style></head><body><main>
<header><span class="badge">用户已取消继续调试 · 定时任务已暂停</span><h1>DrivingEditor：现有证据交接</h1><p>当前结论是<b>分支分配改善，但算法、移除与完整论文复现尚未验收</b>。本页将已取得的报告、运行回执、CPU审计、源码与远端模型位置汇总为可追溯入口。</p><div class="callout">停止新实验、训练、native初始化/扫描与关机操作。此前报告和HANDOFF中的“下一步”只代表当时的研究建议。原始数据、模型、冻结源码与既有结果继续保留在 <b>wm-vgpu-1011</b>。</div><p class="muted">交接生成：{escape(stamp)} · 主场景silver-car / Waymo1.4.3 / FRONT raw0–48 · 历史task_manifest状态不代替此次取消指令。</p><nav class="toc"><a href="#outcome">量化折中</a><a href="#facts">已确认事实</a><a href="#time">静态时间实验</a><a href="#init">初始化见证</a><a href="#preprocess">预处理缺口</a><a href="#reports">全部报告入口</a><a href="#inventory">文件清单与远端位置</a><a href="#unfinished">未完成事项</a></nav></header>

<section id="architecture"><h2>组件与数据流</h2><div class="diagram-wrap">{svg}</div><p class="muted">图中展示实际实现与诊断路径。Static也具有SHM位置、时间marginal opacity与可训练时间字段；这项语义差异已经检查。M-detector和shadow框是缺失输入，未参与这些训练。</p></section>

<section id="outcome"><h2>B1 / B2 / V / S：改善与代价同时保留</h2><p>四臂都从同一<b>已退化的30k完整状态</b>恢复，执行2000步到32k，学习率horizon保留50k；完整参数、conditional Adam、placement、RNG、sampler、scale、EMA、SGCM和首步10标量恢复已核验。全2000步相机/时间扰动/LR配对，额外评估恢复RNG，终点实际49帧及四件状态均完成。</p><p><b>B1</b>原版；<b>B2</b>原版重复；<b>V</b>额外0.05掩码外dynamic opacity诊断项；<b>S</b>额外1.0已观测static RGB诊断项（0.8L1+0.2DSSIM，body mask外），没有V项。新增项都超出当前确认的作者公式。</p>{table(['32k实际指标','B1','B2','V','S'], metric_rows, 'numeric')}<p class="muted">主要指标按每帧等权且保留全部49帧。static含共享环境；dynamic是导出actor通道，不含训练共享环境。actor / core像素汇总是上下文，不与均帧PSNR或原3090历史21帧混池。动态RGB支撑、α与编辑影响范围是不同测量。</p>
<div class="grid"><div><span class="tag fact">已观测改善</span><p>V和S的static MSE相对两份原版均49/49改善；S的道路与横道线明显恢复。comp↔static的车体外RGB差异支撑由双原版约99.34%降到V约59.28%、S约31.20%。</p></div><div><span class="tag hyp">未通过效果验收</span><p>V和S的全图composite MSE相对两份原版均49/49变差；V actor MSE均49/49变差，S对B1为46/49、对B2为47/49变差。S的actor pooled/core误差也更高，native raw48仍有车辆形彩片。</p></div></div><p>一次原版重复只描述已观测差异，不是置信区间、通用噪声上限或显著性检验。S与V附加目标不同，其对比只能描述折中。车辆mask共131,747像素，每帧226–6,125，raw45仅307；小区域等权读数不能概括整车视觉崩坏，也不能删掉不利帧。</p>
{links([debug('evidence/V_opacity_outside30k_to32000/receipts/three_arm_locked_metrics.json','V全部锁定指标'), debug('evidence/S_observed_static_rgb30k_to32000/receipts/S_vs_originals_locked_metrics.json','S全部锁定指标'), audit('sgcm_data_audit/observed_static_rgb_S_review/actual_S_locality_core_cpu.json','pooled / core / locality'), audit('sgcm_data_audit/long_pair_review/actual_three_arm_interpretation.json','V解释'), audit('sgcm_data_audit/observed_static_rgb_S_review/actual_S_interpretation.json','S解释'), debug('assets/B1_B2_S32000_synchronous_actual49.mp4','实际同步49帧视频')])}
<details><summary>实际图像与不利帧（复用冻结资产）</summary><figure><img loading="lazy" src="../drivingeditor_vgpu_debug/assets/S32000_paired_raw10.png" alt="实际raw10，B1 B2 S三排，每排GT composite static dynamic与两种alpha"><figcaption>原始raw10：B1 / B2 / S，每排GT、composite、static、dynamic、αs、αd；完整49结果保留在指标和视频中。</figcaption></figure>{links([debug('assets/S32000_fixed8_part1.png','预选0/10/17/20'), debug('assets/S32000_fixed8_part2.png','预选24/30/40/48'), debug('assets/S32000_adverse_raw44_48.png','终点后明确披露的不利44/48'), debug('assets/S32000_static_raw48_native.png','原生S static raw48'), debug('long_S_video_validation.json','49帧解码验证')])}</details></section>

<section id="facts"><h2>已确认事实、来源及结论边界</h2>{table(['问题','实际证据','结论边界','可读回执'],fact_rows)}<p class="muted">短30k→30300原版全图33.2016 dB、static10.3367 dB；outside诊断全图32.7753、static22.1225。它们是300步短实验，分别保留，不与32k终点混用。首次outside启动在训练前observer导入失败；原失败已归档，3处观察绑定修复经独立复核后才重试。</p>{links([debug('evidence/original30k_to30300/receipts/full49_summary.json','短原版'), debug('evidence/opacity_outside30k_to30300/receipts/full49_summary.json','短outside'), debug('evidence/diagnostic_observer_import_failure.json','失败与修复')])}<details><summary>仍作为假说保留的因果解释</summary>{table(['假说','支持证据','未解决'],[
['监督/目标域允许错误分配','实际目标有逐像素等价构造；V/S单独增加监督后static道路可恢复。','不能认证作者公式、有限3D可实现性或证明唯一原因；两种诊断均有代价。'],
['static时间字段影响分离','静态xyz/时间opacity实际可变，固定pose控制已测。','未隔离训练干预、未证明hard-freeze作者意图或最终编辑修复。'],
['初始化与body-only预处理造成残留','目标native回波留static、ROAD回波进actor；M-detector/shadow输入缺口确实存在。','稀疏标签不足以估全率；没有genuine检测标签和初始化operator因果实验。'],
['SGCM浪费影响资源或学习','存在大量重复与解析不可见clone。','这批不可见点不能直接产生道路RGB彩噪；早期单变量干预未做。'],
])}</details></section>

<section id="time"><h2>静态时间实验：已渲染，未训练修复</h2><p>在固定raw10相机pose/rays下，B1/V32k各渲染49个既有时间（98实际渲染），并在raw0/raw48做12个单独位置、时间opacity/support及二者同时冻结的只读控制。非天空车体外static FP32 RGB对anchor最大MAE：B1 <b>0.10000</b>、V <b>0.05665</b>。</p>{table(['raw48固定pose控制','B1 static RGB MAE对anchor','V static RGB MAE对anchor'],[
['原实际时间','0.096696','0.055696'],['只冻结SHM位置','0.092413','0.053628'],['只冻结时间opacity/support','0.017569','0.009566'],['同时冻结','0（exact）','0（exact）']],'numeric')}<p>两个端点上的opacity/support控制作用更大；控制影响不是可相加的占比。raw0与raw10恰好差一个SHM周期0.2，不能用它证明其它时间的位置变化无关。实际图像覆盖49时间，单独控制只覆盖端点。</p><p class="limit">固定pose加其它时间是反事实组合，没有non-anchor GT质量或PSNR结论。模型距离按已pin的PCA scale0.1换回米；模型时间不称为物理秒。该实验证明现有static的时间依赖，没有完成hard-static训练或证明作者应硬冻结。</p>{links([debug('evidence/fixed_pose_temporal32k/temporal_summary.json','实际110记录'), audit('sgcm_data_audit/fixed_pose_time_review/actual_temporal_interpretation_cpu.json','独立解释与单位链'), debug('temporal_report_validation.json','时间报告QA')])}</section>

<section id="init"><h2>初始化：原始return追溯与稀疏原生标签见证</h2><p>seed0的600,000 sampled PLY行由8,667,791个既有converted returns、原pose、float32 worldXYZ与模型时间精确回放；原36,406 actor PLY也回放exact。原0.1m exclusion留下595,894 static行，已选进actor的2,568 sampled returns全部排除，精确同return重复=0。</p>{table(['独立原生语义 / 身份审计','数量','限制'],[
['static中仍保留目标native instance','169行 / 160独立raw returns','目标track5XnA21wBZZ__-ZJ2j35hrg；raw25/30/35/40/45稀疏TOP标签。0.02836%仅已见证下界。'],
['static未有直接语义身份','541,052行 / 90.7967%','全49污染率未估计；884同时间box关联行是几何结果，包含ROAD，不等于全部actor污染。'],
['169见证的自身采集帧投影','163 FRONT外；6 FRONT内/body mask外','其它既有FRONT pose下143行/137 returns至少一次满足中心投影与初始temporal/alpha资格；未计算遮挡、Gaussian footprint或raster贡献。'],
['actor PLY中的物理ROAD','24行（raw25=12 / raw30=3 / raw35=9）','都在body mask内、目标box外，距ground0.036–0.140m；ROAD不是移动或shadow标签。'],
['actor PLY稀疏标签覆盖','CAR1,692 / ROAD24 / UNDEFINED4 / 未标注34,686','24 ROAD不能单独解释训练后大范围dynamic-road layer。'],
])}<p class="limit">标签及GT boxes仅作独立审计，没有筛点、改训练输入或替代M-detector预测。169见证可提供潜在渲染路径，尚未证明它们导致confetti。</p>{links([audit('loss_audit/next_diagnostics_cpu/static_initialisation_actor_cpu/static_initialisation_actor_audit.json','原始初始化完整审计'), audit('loss_audit/next_diagnostics_cpu/static_initialisation_actor_cpu/front_visibility_followup_cpu.json','FRONT资格补充'), audit('loss_audit/next_diagnostics_cpu/static_initialisation_actor_cpu/dynamic_actor_native_semantics_followup_cpu.json','ROAD与原enum证据'), audit('loss_audit/next_diagnostics_cpu/static_initialisation_actor_cpu/HANDOFF.md','初始化完整交接'), audit('loss_audit/next_diagnostics_cpu/static_initialisation_actor_cpu/raw49_labels_layout.json','49帧标签/字段布局')])}<p class="muted">逐return sidecar、all600000_initialisation_lineage.npz与大数组保留远端；本页不重复下载。</p></section>

<section id="preprocess"><h2>M-detector / 时钟 / shadow：已到哪一步</h2><div class="grid"><div><h3>M-detector：独立编译通过</h3><p>固定官方revision {code(native['source_revision'])}；20个vendor源码/config文件核验，修改数0。隔离真实PCL头文件/库的standalone core+adapter编译exit0，archive686,864字节。</p><p><b>未构造detector，未处理真实scan，未输出真实检测标签。</b>CPU contract/source测试、编译、ABI与静态RAM census均不等于初始化或扫描通过。已列payload约8.653GiB，加已列帧数组约8.727GiB；动态cluster/allocator/scratch峰值尚未实测。12GiB+8GiB曾是未批准agent预算，不能称作者/用户硬要求。</p></div><div><h3>Shadow：输入仍缺</h3><p>Detect-AnyShadow源码pin {code(shadow['pinned_revision'])}；官方finetuned SAM与LSTNb链接已记录。有限公开检索未找到可用替代；Baidu工具非重试错误不能证明链接失效或下载不可能。</p><p><b>没有下载weights，没有shadow推理，没有mask输入补齐。</b>body-only Md缺shadow是已知方法差异，尚未证明其导致当前退化。</p></div></div>
<h3>明确的自定义clock，不是作者输入等价性</h3><p>{code(time_contract['custom_clock_name'])}：先在整数微秒算 {code('raw_start_us − first_start_us')}，再/1e6为float64参考秒，保留raw时刻作为身份。49帧跨度<b>4.794396s</b>，间隔min/median/max为99,536 / 99,983.5 / 100,031µs，最多180,332点/帧。</p><p>Waymo Frame stamp指起始，Frame.pose约中点且不对应该stamp；TOP per-pixel pose提供几何，不提供显式逐point time。原SDK已将TOP先转world再逆Frame.pose保存bodyXYZ，因此复用bodyXYZ+原Frame.pose，不二次deskew或猜0.05/0.1s偏移。column6是TOP range pixel index，非timestamp；模型归一化时间也不能作为扫描秒。</p><p>固定M-detector以odom stamp给整帧点同一时间；公开FAST_LIO Waymo配置关闭IMU并将end=begin。已检查公开tree和唯一release仍未找到raw TFRecord→ROS bag producer。<b>author_equivalence=false，epoch与relative浮点标签逐位等价也未验证。</b>Packet-only、construction-only和chronological filter都未执行。</p><p class="limit">全scene detector预测与当前single selected-car body Md的其它mover/unknown/warmup/group域关系仍未解决；旧I-only全场景分配草案readiness=false。其它136 tracks和稀疏前视box分析不能证明没有未知或遮挡movers。部分后续审计代理因配额停止，缺失的执行结果不能据此补写为通过。</p>{links([audit('preprocessing_gap_audit/mdetector_adapter_candidate/build_attempt_01/native_build_receipt.json','真实compile-only回执'), audit('preprocessing_gap_audit/mdetector_adapter_candidate/verification/cpu_verification.json','contract/source CPU'), audit('preprocessing_gap_audit/mdetector_adapter_candidate/build_attempt_01/ram_compile_audit.json','静态RAM/ABI'), audit('loss_audit/next_diagnostics_cpu/mdet_time_contract_cpu/HANDOFF.md','最新时间完整交接'), audit('loss_audit/next_diagnostics_cpu/mdet_time_contract_cpu/clock_source_audit_cpu.json','时钟/producer源码证据'), audit('loss_audit/next_diagnostics_cpu/mdet_time_contract_cpu/time_contract_handoff_receipt.json','clock与I readiness'), audit('loss_audit/next_diagnostics_cpu/mdet_time_contract_cpu/other_dynamic_domain_followup_cpu.json','其它动态对象域'), audit('preprocessing_gap_audit/shadow_weight_access_20261012.json','shadow权重可用性')])}</section>

<section id="reports"><h2>全部既有报告与数据入口</h2>{table(['集合','入口','使用范围'],report_rows)}<p class="muted">原3090报告保留历史HTML和21张原PNG，不能以reset后的旧checkpoint重渲染替代；共同21帧与当前49帧不混合统计。旧网页的进行中标签/排程说明保持历史原样，最新状态由本页及取消回执确定。</p></section>

<section id="inventory"><h2>完整清单、版本与远端保留位置</h2><p>本地报告与轻量审计回执按原层级保存；全文件catalog列出路径/大小/已有hash与保存位置。大checkpoint、原始TFRecord、既有float32/lineage数组与已保存媒体不重复搬运。总包保留 {code('outputs/')} 和 {code('work/')} 层级，解包后本页相对链接可用。</p>{links(inv_links + package_links)}<p>{receipt_link}</p>{git_box}
{table(['远端类别（仅wm-vgpu-1011）','实际保存位置 / 清单入口'],[
['项目根',code(BASE)],['解释器 / kernel',code('/root/autodl-tmp/envs/drivingeditor/bin/python')+'；原FP32 renderer/kernel，compact/cleanup未部署。'],
['实际冻结baseline源码',code(BASE+'/campaign/resume_vgpu_baseline_20261011/frozen_repository')],
['原paper50k父run / 10k状态',code(BASE+'/runs/scene0017_000_048_paper_50000_seed0')],
['vGPU正式20k/30k及停止时日志',code(BASE+'/runs/scene0017_000_048_paper_50000_seed0_resume_vgpu1011_baseline')],
['原始49帧与body/sky、PCA输入',code(BASE+'/data/waymo_paper_scenes/scene0017_000_048')],
['当前campaign完整证据',code(D)+'；component_diagnostic、continuation_diagnostic、loss_audit、sgcm_data_audit、preprocessing_gap_audit、root_review等。'],
])}
<details><summary>B1 / B2 / V / S 四件32k checkpoint与SHA</summary>{table(['实验','远端terminal目录','完整pin','短SHA（详细值见manifest）'],model_rows)}<p class="muted">四件为static、dynamic、environment、training_state，均已有CPU schema与完整状态核验。此交接没有再加载模型、复制checkpoint或进行render。</p></details>
<details><summary>四臂实际执行源码位置</summary>{table(['实验','source根目录','来源'],source_rows)}<p class="muted">这些实际执行pins与最终Git归档版本是不同的可追溯记录；候选分支的存在不代表已部署到原正式run。</p></details>
<details><summary>收集范围（包括历史、失败、草案与未执行工作）</summary>{table(['集合','包含内容','保存/入口'],[
['全部outputs集合','环境准备、官网/Waymo场景对应、paper49 body数据、论文核对、基础设施恢复、10k/20k/30k、3090原历史21帧、vGPU排错、其它场景搜索/队列与编辑候选。','上表全部报告 + local_inventory'],
['当前运行与组件证据','6×5组件、两份短续训、B1/B2/V/S长续训，source/import pins、保护原件、首步等价、逐步losses、相机/LR配对、RNG、PNG/视频解码、终态与GPU释放。',code(D+'/component_diagnostic')+' / '+code(D+'/continuation_diagnostic')],
['loss_audit','公式/梯度、全部event窗口、深度碰撞/定义、outside/nearest候选与默认关开关、guard、计划/materializer/绑定、失败启动、next_diagnostics、初始化/clock/其它对象。',code(D+'/loss_audit')],
['sgcm_data_audit','模型几何、SGCM重复及不可见点、mask/time/unit/ground覆盖、visibility guard、原版重复、V/S独立复核、pooled/core/locality、fixed-pose temporal。',code(D+'/sgcm_data_audit')],
['preprocessing_gap_audit','M-detector/Shadow公开源码pins、vendor/source树、transport adapter、CPU contract、standalone编译、日志/依赖/ABI/RAM、权重检索回执。',code(D+'/preprocessing_gap_audit')],
['计划 / 审核 / 运维历史','task_manifest、root_review、暂停其它场景、原shutdown条件策略、automation历史prompts与当前PAUSED回执；旧设计并不授权新执行。',code(D) + '；本地work/vgpu_bugfix_20261012'],
['远端原始大件','data/raw conversion、PLY/rawIDs/npz、原正式与诊断模型、源快照、原媒体、oncoming/red-scene/mirror与历史runs。','remote_inventory完整逐文件catalog；保留远端。'],
])}</details></section>

<section id="unfinished"><h2>尚未完成：停止状态如实交接</h2>{table(['未完成项','已有证据 / 当前状态'],[
['完整论文版本与从零50k','现有公开DrivingEditor只有已核对的2024main，未取得作者完整2026实现。当前受控实验只从退化30k恢复；四场景主表、完整消融、from-zero终态未完成。'],
['初始化operator因果实验I','169目标static见证与24ROAD dynamic见证存在，但污染全率/彩噪因果未知。M-detector真标签、输入时钟实测与其它对象域关系均缺，I readiness=false。'],
['genuine shadow输入H','公开源码与有限weight检索已存；weights、shadow masks和真实预处理缺失。'],
['静态时间训练T','现有模型时间依赖已测；hard-static/opacity语义的单独训练修复及作者意图未验证。'],
['深度 / SGCM / reset 独立干预','nearest、visibility guard与reset方向只有各自CPU/源码及已有机制证据，未在from-zero/下一次reset做独立因果训练。'],
['最终编辑与质量验收','仍有dynamic道路、原生S raw48车辆形彩片及actor/composite代价；缺dense edited GT，移除正确和平移编辑尚未通过验收。'],
['作者参数与损失细节','NPS采样/权重、opacity域、SGCM区域/重复预算、harmonic900sky与部分扰动约定尚未认证。'],
['浏览器播放验证','媒体按真实49 PNG编码并全帧解码；root查看了实际raw10视频帧与native raw48。浏览器交互播放未作为验证证据。'],
])}<p class="callout">用户已取消继续推进。这些未完成项仅用于后续人工接手，不是排程任务；未执行native初始化/扫描、未新增训练、未关机或删除数据。新S报告被保留为当时已完成证据，不能把“审计通过/编译通过/静态重建改善”改写为“debug修复完成”。</p></section>

<section id="integrity"><h2>冻结与可追溯性</h2><p>旧主报告 {code('outputs/drivingeditor_vgpu_debug/index.html')} SHA256：</p><p>{code(FROZEN_SHA)}</p><p>旧S handoff记录138个本地引用、15项聚焦静态QA；新B1/B2/S同步视频实际49帧、10fps、4.9s，原588 PNG pins与全49解码核验。此次只新增独立索引、manifest与QA回执，原主报告hash再次核对一致。</p>{links([debug('S_report_handoff.json','冻结S交接'), debug('S_report_validation.json','S静态QA15'), debug('local_reference_validation.json','旧引用138'), debug('long_S_video_validation.json','视频pins与解码'), audit('task_manifest.json','历史task_manifest'), debug('report_manifest.json','冻结报告资产manifest')] + ([link(DEST/'validation.json','本交接静态QA')] if (DEST/'validation.json').exists() else []))}<p class="muted">本页所有数字来自上述既有结果；无新GPU、native构造/scan、数据清洗、模型加载、训练或shutdown。</p></section>
<footer>这是当前取得证据的交接索引。完整catalog与Git交付回执负责记录实际文件和推送结果，研究结论保留其已验证范围。</footer></main></body></html>'''

# Manifest is created before reference QA so the public link always exists.
manifest = {
    'schema_version':1, 'status':'EVIDENCE_HANDOFF_USER_CANCELLED_DEBUG_NOT_COMPLETE',
    'created_at':stamp, 'host':'wm-vgpu-1011',
    'scope':'New evidence index only; frozen report unchanged; no new experiment/training/native init/filter/shutdown',
    'source_report':{'path':'outputs/drivingeditor_vgpu_debug/index.html','sha256':FROZEN_SHA,'refs':138,'S_static_QA_checks':15},
    'input_receipt_pins':inputs, 'references':refs,
    'remote_base':BASE, 'remote_campaign':D,
    'github':{'repo':repo,'branch':branch,'commit':commit,'push_verified':delivery.get('push_verified',False)},
    'author_equivalence':False, 'paper_complete_reproduction':False, 'debug_repaired':False,
    'gathered_file_catalogs':[p.name for p in inventory_files if p.exists()],
    'package_files':[p['path'] for p in refs if p['path'].endswith(('.zip','.tar.gz','.7z'))],
    'large_artifacts_policy':'All remote datasets/checkpoints/NPZ/media remain preserved; no repeated large downloads',
    'new_index_builder':{'path':Path(__file__).relative_to(ROOT).as_posix(),'sha256':sha(Path(__file__))},
}
(DEST/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if '本页引用与输入pin清单' not in html:
    manifest_link=link(DEST/'manifest.json','本页引用与输入pin清单')
    html=html.replace('<h2>完整清单、版本与远端保留位置</h2>','<h2>完整清单、版本与远端保留位置</h2>'+links([manifest_link]))
(DEST/'index.html').write_text(html,encoding='utf-8')
manifest['references']=refs
manifest['index_sha256']=sha(DEST/'index.html')
(DEST/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

class QA(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True); self.local=[];self.ids=set();self.anchors=[];self.tags=[]
    def handle_starttag(self,tag,attrs):
        d=dict(attrs);self.tags.append(tag)
        if 'id' in d:self.ids.add(d['id'])
        for key in ('href','src'):
            if key in d:
                url=d[key];parts=urlsplit(url)
                if parts.scheme or parts.netloc:continue
                if not parts.path:
                    if parts.fragment:self.anchors.append(parts.fragment)
                else:self.local.append(unquote(parts.path))

q=QA();q.feed(html)
missing=[p for p in q.local if not (DEST/p).resolve().exists()]
bad_anchors=[a for a in q.anchors if a not in q.ids]
checks={
    'frozen_existing_report_SHA_unchanged':sha(DEBUG/'index.html')==FROZEN_SHA,
    'all_local_href_src_resolve':not missing,
    'all_fragment_links_resolve':not bad_anchors,
    'architecture_SVG_title_and_description':all(x in html for x in ['<svg','arch-title','arch-desc','Waymo 49帧','M-detector / shadow']),
    'dark_responsive_compact_design':'color-scheme:dark' in html and '@media(max-width:800px)' in html,
    'B1_B2_V_S_numbers_from_actual_locked_metrics':all(str(round(x,5)) in html for x in [36.43075686122145,35.90405933084622,35.447988574370136,31.7634247318007]),
    'all49_adverse_results_explicit':'46/49' in html and '47/49' in html and '49/49变差' in html,
    'historical3090_21_and_current49_unpooled':'共同21帧与当前49帧不混合统计' in html,
    'native_compile_only_not_initialized':not native['detector_initialized'] and '未构造detector' in html,
    'clock_custom_not_author_parity':not time_contract['author_equivalence'] and 'author_equivalence=false' in html,
    'native_static169_dynamic24_and_sparse_limits':all(x in html for x in ['169行','24行','541,052','未证明它们导致confetti']),
    'cancelled_no_ongoing_automation_or_new_work':'定时任务已暂停' in html and '停止新实验' in html,
    'unfinished_fromzero_and_edit_validation_disclosed':'from-zero终态未完成' in html and '平移编辑尚未通过验收' in html,
    'no_scripts_or_remote_dependencies':'<script' not in html and '<link rel="stylesheet"' not in html,
}
validation={
    'status':'PASS_LOCAL_STATIC_HTML_EVIDENCE_HANDOFF' if all(checks.values()) else 'FAIL',
    'created_at':stamp, 'checks':checks, 'check_count':len(checks),
    'local_reference_count':len(q.local),'distinct_local_references':len(set(q.local)),
    'missing_local_references':missing,'bad_anchors':bad_anchors,
    'index_sha256':sha(DEST/'index.html'),'frozen_report_sha256_after':sha(DEBUG/'index.html'),
    'numeric_source_receipts_sha256':{'S':sha(DEBUG/'evidence/S_observed_static_rgb30k_to32000/receipts/S_vs_originals_locked_metrics.json'),'V':sha(DEBUG/'evidence/V_opacity_outside30k_to32000/receipts/three_arm_locked_metrics.json')},
    'GPU_or_remote_native_or_training_started':False,'browser_playback_or_visual_render_claimed':False,
}
(DEST/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(DEST/'README.md').write_text(f'''# DrivingEditor 现有证据交接

入口：`index.html`。用户已取消继续调试，定时任务已暂停。现有结果不是已修复或完整论文复现。

本页包含架构组件图、B1/B2/V/S实际49帧折中、checkpoint/reset/export、时间控制、depth、SGCM、初始化原生见证、M-detector compile-only与clock/shadow未完成项，以及全部报告和远端保留位置。

原报告`outputs/drivingeditor_vgpu_debug/index.html`保持冻结，SHA256 `{FROZEN_SHA}`。`manifest.json`记录引用、输入pins及Git绑定；`validation.json`记录静态HTML/实际数字/引用核验。

总包请保留`outputs/`与`work/`原层级。`local_inventory.json`和`remote_inventory.json`由root记录完整已收集文件；远端数据、checkpoint、NPZ及既有媒体不重复下载。`delivery_receipt.json`由root记录实际取消与Git推送。

本交接构建仅访问既有本地证据，无新GPU、训练、detector初始化/扫描、模型加载或关机。
''',encoding='utf-8')
print(json.dumps({'status':validation['status'],'output':str(DEST/'index.html'),'checks':checks,'references':validation['local_reference_count'],'missing':missing,'frozen_SHA_unchanged':checks['frozen_existing_report_SHA_unchanged'],'delivery_receipt_bound':bool(delivery)},ensure_ascii=False,indent=2))
if not all(checks.values()):raise SystemExit(1)
