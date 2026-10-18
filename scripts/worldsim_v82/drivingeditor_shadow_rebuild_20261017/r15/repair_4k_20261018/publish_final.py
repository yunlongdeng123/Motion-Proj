"""Publish measured repaired4K evidence after the synchronized video delivery is concrete."""
from pathlib import Path
import json,shutil,os,re
W=Path(__file__).parent;I=W.parent/'identity_completion';P=W.parent/'r15_closure_run/publish_repo'
O=W.parents[1]/'outputs/r15_repair_4k_20261018'
D=Path('docs/autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/repair_4k_20261018')
S=Path('scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/repair_4k_20261018')
paths=json.loads((I/'sync_publish_paths.json').read_text(encoding='utf-8'))
def long(p):return Path('\\\\?\\'+str(p.resolve())) if os.name=='nt' else p
def cp(src,relative):
    target=long(P/relative);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(long(src),target);paths.append(relative.as_posix())
status=json.loads((O/'status.json').read_text(encoding='utf-8'));assert status['new_training_updates']==4000 and status['actual_branch_render_complete']
for file in O.rglob('*'):
    if file.is_file():cp(file,D/file.relative_to(O))
for name in ['pack_final.py','collect_final.py','build_final_results.py','publish_final.py','build_report.py','evaluate_repair.py','build_evaluator.py','shutdown_preflight.py']:
    cp(W/name,S/name)
geo=json.loads((W/'geometry_comparison.json').read_text(encoding='utf-8'))
audits=[json.loads((W/'final_metadata/train/ownership'/f'{s:06d}/receipt.json').read_text()) for s in [1500,2500,3500]]
paragraph=f'''修复R15实际完成4000步，训练总用时{json.loads((W/'final_metadata/train_receipt.json').read_text())['seconds']/60:.2f}分钟（含周期审计/保存，非同点数单步基准）；1K/2K/3K/4K检查点保留。三次46帧当前证据审计实际owner变化分别为{[a['actual_owner_changes'] for a in audits]}。五身份49帧独立/删除及真实动态/静态分支完成，已加入同步播放器第三模型；全部有限GPU队列结束，尚需保存推送、空闲核验及授权关机。实际raw113中a6下部白带、a37关阴影删除后的静态车形仍在，整体资产质量未通过。[完整实际结果与组件](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/repair_4k_20261018/index.html)。human_verdict=null。
'''
p=P/'docs/RESEARCH_STATUS.md';text=p.read_text(encoding='utf-8')
text=re.sub(r'(?:R15修复真实GPU自检已通过：|修复R15实际完成4000步，).*?(?=\n\n)',paragraph.strip(),text,flags=re.S)
text=re.sub(r'下一步等待新4K三次归属审计和训练完成，.*?(?=\n\n)',
    '新旧五身份实际分支已经交付为同步逐帧视频。原GN11、旧3、旧5、原R15及新修复4K检查点和原始输入保留，用户其他dirty不动。同一重建窗口，无独立编辑GT，human_verdict=null，problem_solved=null。',text,flags=re.S)
text=text.replace('GPU仍用于新4K；旧两队列完成不是关机条件。新训练/评估、报告保存及v82推送全部完成，确认所有worker和会启动任务的controller结束后，按用户授权关闭AutoDL并记录SSH关闭观察。',
    '新4K和三版分支GPU任务均已结束。报告及视频保存、v82推送核对后，检查无worker及会启动任务的controller，再按既有用户授权关闭AutoDL；实际观察单独记录。')
p.write_text(text,encoding='utf-8');paths.append('docs/RESEARCH_STATUS.md')
p=P/'docs/EXPERIMENTS.md';text=p.read_text(encoding='utf-8').replace('累计几何与增密预算、当前证据归属审计、FP64阴影；同原始输入有界4K及五身份49帧评估','同原始输入4K及五身份49帧实际评估完成；累计几何预算、三次归属审计、FP64阴影；同步视频第三模型')
p.write_text(text,encoding='utf-8');paths.append('docs/EXPERIMENTS.md')
p=P/'docs/research_failures/entries/V82-F06.md';text=p.read_text(encoding='utf-8')
if '同原始输入修复4K实际结果' not in text:
    text+='\n\n## 2026-10-18：同原始输入修复4K实际结果\n\n'+paragraph+'\n'
    text+='|Actor|原4K点数|修复4K点数|修复框外中心率|内部unknown|\n|---|---:|---:|---:|---:|\n'
    for new in geo['repair4k']:
        old=next(x for x in geo['original4k'] if x['actor']==new['actor'])
        text+=f"|a{new['actor']}|{old['points']}|{new['points']}|{new['outside_exact_box']/new['points']:.2%}|{new['raw_status_counts']['0']}|\n"
    text+='\n原始输入/学习率时间表相同；这是组合工程修复，预算使点数明显不同，吞吐不能归因于单独的阴影内核。保体统计使用同一R15正像素域，非独立GT。完整49帧视频用于复核a6下部碎片、a37静态外观残影及另外三个真实身份；未把几何边界结果当成整体质量通过。\n'
    text+='\n实际锚帧复核：a6 raw113下部白带仍在，a37 raw113关闭物理阴影删除后静态车形仍在；整体资产质量未通过。a5/raw107、a26/raw133、a18/raw137保留车体、周边散点更紧凑，但这不等于全49帧人工验收。\n'
p.write_text(text,encoding='utf-8');paths.append('docs/research_failures/entries/V82-F06.md')
closure=D.parent/'closure_20261018/report.html'
for file,player,repair in [(O.parent/'r15_closure_report_20261018.html','identity_comparison_20261018/index.html','r15_repair_4k_20261018/index.html'),
    (P/closure,'../identity_20261018/baseline_complete.html','../repair_4k_20261018/index.html')]:
    file=long(file);text=file.read_text(encoding='utf-8')
    banner=f'<p class="warn" id="latest-identity-video">最新交付：<a href="{player}">五身份动态／静态／Delete 同步视频（49帧可拖动，旧5／原R15／修复R15）</a> · <a href="{repair}">修复4K实际结果与组件</a>。本轮4000步及全部渲染已完成；a6白带、a37静态车形仍未解决。本页保留此前实验与历史电源记录。</p>'
    if 'id="latest-identity-video"' in text:text=re.sub(r'<p class="warn" id="latest-identity-video">.*?</p>',banner,text,count=1,flags=re.S)
    elif '后续身份对照：' in text:text=re.sub(r'<p class="warn">后续身份对照：.*?</p>',banner,text,count=1,flags=re.S)
    else:text=text.replace('<h1>',banner+'<h1>',1)
    file.write_text(text,encoding='utf-8')
paths.append(closure.as_posix())
(W/'final_publish_paths.json').write_text(json.dumps(sorted(set(paths)),indent=2),encoding='utf-8')
print('Measured repair4K results copied to isolated v82 checkout')
