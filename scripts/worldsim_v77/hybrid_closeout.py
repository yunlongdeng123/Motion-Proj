"""本轮收口登记；保留旧状态备份，不启动任何模型。"""
import json,shutil
from pathlib import Path
REPO=Path('/root/autodl-tmp/motion_proj_v77')
ROOT=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927/r2')
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
backup=ROOT/'repo_backups/closeout';backup.mkdir(parents=True,exist_ok=True)
for rel in ['docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md','docs/research_failures/entries/V77-F02.md','docs/RESEARCH_FAILURES.md']:
 dst=backup/rel;dst.parent.mkdir(parents=True,exist_ok=True)
 if not dst.exists():shutil.copy2(REPO/rel,dst)

mask_path=ROOT/'mask_summary.json'
if not (ROOT/'mask_summary_initial.json').exists():shutil.copy2(mask_path,ROOT/'mask_summary_initial.json')
dump(mask_path,dict(scenes=[dict(scene=s,frames=len(read(ROOT/s/'mask_stats.json')),totals={k:sum(r[k] for r in read(ROOT/s/'mask_stats.json')) for k in ['core','delete','protect','generate']}) for s in ['scene_0230','scene_0255']],source='final per-scene mask_stats; replaces stale pre-fence summary',human_verdict=None))
notes=read(ROOT/'execution_notes.json');notes.update(phase='closed_after_failure',generation_attempts=2,packaging_engineering_issue='PATH lacked ffmpeg; reused imageio_ffmpeg bundled binary, 24 videos decoded; no GPU rerun')
dump(ROOT/'execution_notes.json',notes)
observations=dict(scenes=[dict(scene='scene_0230',frames=[18,25,33,47],observation='gray slab and dark shadow early; elongated dark silhouette and brown/black vehicle-like body later; not actor-free'),dict(scene='scene_0255',frames=[65,72,80,94],observation='silver SUV silhouette and smear/striping behind preserved fence; not actor-free')],prior_association='prior already contains smear/vehicle trace; not proven sole cause',visual_judgement_by='assistant',human_verdict=None)
dump(ROOT/'visual_observations.json',observations)
closeout=dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r2',status='STOP_CANDIDATE_FAILURE',failure_ledger_refs=['V77-F02'],failure_ledger_delta='updated V77-F02',scenes=2,generated_frames=60,new_omega_forwards=0,new_glb_assets=0,training=False,default_pipeline_changed=False,new_generation_stopped=True,pixel_validation=read(ROOT/'validation.json')['contract_passed'],video_validation=read(ROOT/'review/video_validation.json'),review_path='C:/Users/dengyunlong/Documents/Codex/2026-09-26/xia/outputs/v77-hybrid-delete/index.html',human_verdict=None)
dump(ROOT/'closeout.json',closeout)
light=REPO/'docs/autoresearch/worldsim_v77/hybrid_bg_20260927/r2';light.mkdir(parents=True,exist_ok=True)
names=['registration.json','execution_notes.json','mask_summary.json','evidence_summary.json','generation_input_preflight.json','generation_state.json','guard_summary.json','validation.json','background_admission.json','visual_observations.json','closeout.json']
for name in names:
 shutil.copy2(ROOT/name,light/name)
 shutil.copy2(ROOT/name,ROOT/'review'/name)
shutil.copy2(ROOT/'review/video_validation.json',light/'video_validation.json')
for s in ['scene_0230','scene_0255']:
 dest=light/s;dest.mkdir(exist_ok=True)
 for name in ['mask_stats.json','evidence_stats.json','fence_refinement.json']:
  if (ROOT/s/name).exists():shutil.copy2(ROOT/s/name,dest/name)

status='''# 当前研究状态

更新：2026-09-27，v77。`WS-V77-HYBRID-BG-20260927/r2`两场景Hybrid DELETE已完成，按用户stop rule停止该候选：0230仍有灰块/再生车形，0255仍有SUV残影；两例不进入Ω。GPU生成与语义检查已收口，没有定时任务或下一轮自动生成。

## 当前结果

scene_0230/actor22/CAM5/f18–47、scene_0255/actor25/CAM3/f65–94，各30帧。三mask + 原RGB证据 + DiffuEraser官方2-Step，seed42，每例一次。删除区真实证据覆盖0.000426%/0.27238%；车辆再生guard拦截21/30与30/30。60帧像素/来源合同通过，24视频720帧实际解码通过，不能将工程通过当质量通过。

0255围栏mask含助手14点图像提示、镂空候选选择和有界相似变换跟踪，未宣称全自动。内部prior已有残影，与最终车形关联但不证明唯一因果。保护mask不变也不保证所有邻车语义保真。

## 验收与下一步

本地审核入口：`C:/Users/dengyunlong/Documents/Codex/2026-09-26/xia/outputs/v77-hybrid-delete/index.html`。主三栏原视频/旧DriveEditor/新DELETE，附mask、evidence、内部prior及详细注释。报告：[HYBRID_BACKGROUND_RESULTS.md](v77/HYBRID_BACKGROUND_RESULTS.md)。run：`/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927/r2`，同一失败卡V77-F02已追加。

默认仍保留旧DriveEditor FULL及已知缺陷，原Ω/GLB与所有失败输出未改。0训练、0新Ω前向、0新GLB、0新MOVE。不继续换seed/叠加成熟模型。若下一阶段进入数据路线，先验证真实背景监督可得性；本轮没有启动训练。人工verdict保持null，等待用户验收；历史关机安排不适用于本轮。
'''
(REPO/'docs/RESEARCH_STATUS.md').write_text(status)
f=REPO/'docs/research_failures/entries/V77-F02.md'
heading='## Hybrid三mask + DiffuEraser：边界保持成立，目标语义仍失败'
if heading not in f.read_text():
 with f.open('a') as w:w.write('''

## Hybrid三mask + DiffuEraser：边界保持成立，目标语义仍失败

`WS-V77-HYBRID-BG-20260927/r2`，两固定开发目标各30帧、960×536/10Hz、seed42、官方PCM2-Step，每例一次。SAM2.1多对象delete/generate/protect + 原RGB的LK/RANSAC与原Ω/GT/LiDAR支持 + DiffuEraser residual completion完成。0230删除区真实支持2/469403像素帧（0.000426%），0255为735/269840（0.27238%）；支持不足不能证明隐藏背景从未观测，也没有充分真实证据条件的有效性结论。

0230仍有灰色立面/暗影和后段棕黑车形；0255围栏后银灰SUV轮廓与涂抹未消失。GroundingDINO+SAM2对最终图拦截21/30与30/30，内部prior为18/30与27/30；原车正控制均30/30、空路面负控制均0、旧0230/f25幻觉控制命中。早期灰块可漏检，零检测不通过。prior已有残影，但未证明其为唯一原因。两例`FAIL_ACTOR_REGENERATION`、background_input_dir=null，0新Ω/GLB，不用后续世界表示掩盖补景失败。

最终protect变化为0；独立源邻车检测mask中变化430/1279616与2862/861732像素，不能将保护合同当作所有邻车保真。0255自动SAM2把围栏孔洞涂实，助手14点提示选择镂空候选后传播，明确非全自动。初次homography退化、修改mask期间的证据准备均在生成前拒绝/重做，保留工程失败证据；最终有界相似变换最少10内点、最大中位误差1.231px。

60帧精确写回/来源合同、2track身份验证通过；24视频720帧实解码。第一次视频打包缺PATH ffmpeg，复用已有imageio_ffmpeg修复，没有重跑模型。旧/新mask、模型与上下文历史不同，不是单变量消融。按用户stop rule停止此候选，不换seed或追加成熟模型；旧FULL默认和全部资产/失败输出保留，旧版仍有缺陷。没有否定GLB、整个world表示或DiffuEraser所有配置，没有启动新训练。

见[实测与architecture](../../v77/HYBRID_BACKGROUND_RESULTS.md)、[证据与收口](../../autoresearch/worldsim_v77/hybrid_bg_20260927/r2/closeout.json)、[准入拒绝](../../autoresearch/worldsim_v77/hybrid_bg_20260927/r2/background_admission.json)。`failure_ledger_refs: [V77-F02]`；`failure_ledger_delta: updated V77-F02`；`human_verdict: null`。
''')
f=REPO/'docs/EXPERIMENTS.md';t=f.read_text();row='| WS-V77-HYBRID-BG-20260927 / r2 | 两scene三mask + factual evidence + DiffuEraser各30帧；再生车/残影失败，停止候选、不进入Ω；原默认保留 | [实测与architecture](v77/HYBRID_BACKGROUND_RESULTS.md)、[收口](autoresearch/worldsim_v77/hybrid_bg_20260927/r2/closeout.json) |\n'
if row not in t:
 at=t.index('| WS-V77-HYBRID-BG-20260927 / r1');f.write_text(t[:at]+row+t[at:])
print('CLOSED candidate after two failures; human verdict remains null')
