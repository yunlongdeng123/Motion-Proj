"""Publish actual synchronized branch videos, without replacing old evidence history."""
from pathlib import Path
import json,os,shutil,re
W=Path(__file__).parent;P=W.parent/'r15_closure_run/publish_repo'
O=W.parents[1]/'outputs/identity_comparison_20261018'
D=Path('docs/autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/identity_20261018')
S=Path('scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/identity_completion_20261018')
RS=S.parent/'repair_4k_20261018'
paths=[]
def long(p):return Path('\\\\?\\'+str(p.resolve())) if os.name=='nt' else p
def cp(src,relative):
    target=long(P/relative);target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(long(src),target);paths.append(relative.as_posix())
state=json.loads((O/'player_status.json').read_text(encoding='utf-8'))
assert state['models']['old5'] and state['models']['r15']
assert len(list((O/'sync_exports').glob('*.mp4')))==5
# Preserve the previously committed lightweight report, not the local full-PNG matrix.
diag=long(P/D/'diagnostics.html')
if not diag.exists():cp(P/D/'baseline_complete.html',D/'diagnostics.html')
for name in ['index.html','sync_player.html','baseline_complete.html']:
    cp(O/'sync_player.html',D/name)
for name in ['player_status.json','player_validation.json']:
    cp(O/name,D/name)
for folder in ['branch_videos','sync_exports']:
    for file in (O/folder).glob('*.mp4'):cp(file,D/folder/file.name)
for model,ready in state['models'].items():
    if ready:cp(W/'branches'/('branches_'+model)/'branch_receipt.json',D/'branch_receipts'/(model+'.json'))
for root,link in [(O/'tristate','../index.html'),(P/D/'tristate','../baseline_complete.html')]:
    file=long(root/'index.html');text=file.read_text(encoding='utf-8')
    if '主要检查入口：五身份动态' not in text:
        text=text.replace('<h1>',f'<p class="note"><a href="{link}"><strong>主要检查入口：五身份动态／静态／Delete 同步视频，可播放及逐帧拖动</strong></a>。本页保留为三维证据补充。</p><h1>',1)
        file.write_text(text,encoding='utf-8')
paths.append((D/'tristate/index.html').as_posix())
for name in ['build_sync_player.py','sync_player_template.html','build_sync_exports.py','test_player_sync.cjs','collect_branches.py','publish_sync_player.py','validate_sync_player.py','build_tristate_report.py','stage_sync_delivery.py']:
    cp(W/name,S/name)
for name in ['render_branches.py','pack_branches.py']:cp(W/'bundle'/name,S/name)
for name in ['evaluate_repair.py','build_evaluator.py']:cp(W.parent/'r15_repair_4k'/name,RS/name)
p=P/'docs/RESEARCH_STATUS.md';text=p.read_text(encoding='utf-8')
start='五身份分支同步视频已交付'
paragraph='''五身份分支同步视频已交付：旧五身份30K与原R15冻结4K，a6/a37/a5/a26/a18，共49帧raw89—137。主入口改为同一时间轴的删除前后整景、全部Actor动态分支、静态分支，可切换alpha、独立资产、观测图与物理阴影；另导出五段合成总览MP4，所有分支在同一视频内逐帧对齐。默认关闭物理阴影，删除前后整景保持相同开关；开阴影时目标从实际太阳caster中移除，静态接收面独立光栅化。旧5重放full逐像素匹配既有PNG，所有删除target-alpha=0；R15静态几何/外观不改。浏览器安全策略拒绝file://验证，未宣称实际浏览器验收；帧数/编码/帧序和异步解码同步逻辑另有验证。[同步播放器与组件图](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/identity_20261018/baseline_complete.html)。三维中心页仅作补充，human_verdict=null。
'''
if start in text:text=re.sub(r'五身份分支同步视频已交付.*?(?=\n\n|\Z)',paragraph.strip(),text,flags=re.S)
else:text=text.replace('\n\n旧GN11', '\n\n'+paragraph+'\n旧GN11',1)
p.write_text(text,encoding='utf-8');paths.append('docs/RESEARCH_STATUS.md')
p=P/'docs/EXPERIMENTS.md';text=p.read_text(encoding='utf-8')
text=text.replace('旧5完整30K及五身份49帧完成，三版637身份帧/31视频；旧5与原R15三维中心、统一框统计、三值真实支撑投影完成','旧5完整30K；五身份49帧动态/静态/delete共用时间轴，五段合成总览MP4；三维中心与三值支撑为补充')
p.write_text(text,encoding='utf-8');paths.append('docs/EXPERIMENTS.md')
p=P/'docs/research_failures/entries/V82-F06.md';text=p.read_text(encoding='utf-8')
if '五身份同步分支视频' not in text:
    text+='''\n\n## 2026-10-18：五身份同步分支视频\n\n根据用户复核要求，主入口改为旧5/原R15的五身份49帧同步动态、静态、逐身份delete视频。时间轴统一解码后再绘制全部画面，五段合成MP4保证视频内部帧序一致。静态无阴影前后不变用于检查外观残影；开阴影则从太阳caster删除目标，背景接收面使用独立静态光栅结果。主场景删除前后使用相同阴影开关。此为诊断交付修正，不产生新的训练质量结论。真实浏览器file://验收被安全策略拒绝，明确保留该限制；human_verdict=null。\n'''
p.write_text(text,encoding='utf-8');paths.append('docs/research_failures/entries/V82-F06.md')
(W/'sync_publish_paths.json').write_text(json.dumps(paths,indent=2),encoding='utf-8')
print(json.dumps(dict(files=len(paths),video_files=sum(x.endswith('.mp4') for x in paths))))
