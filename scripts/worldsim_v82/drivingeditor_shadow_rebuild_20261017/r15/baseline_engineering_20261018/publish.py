"""Scoped publication; preserve unrelated source and generated-index changes."""
from pathlib import Path
import os,json,shutil,re,zipfile
W=Path(__file__).parent;P=W.parent/'r15_closure_run/publish_repo';O=W.parents[1]/'outputs/baseline_engineering_20261018'
D=Path('docs/autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/baseline_engineering_20261018')
S=Path('scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/baseline_engineering_20261018')
paths=[]
def long(p):return Path('\\\\?\\'+str(p.resolve())) if os.name=='nt' else p
def cp(src,rel):
    p=long(P/rel);p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(long(src),p);paths.append(rel.as_posix());return p
def main():
    for f in O.rglob('*'):
        if f.is_file():
            p=cp(f,D/f.relative_to(O))
            if p.suffix=='.html':p.write_text(p.read_text(encoding='utf-8').replace('../identity_comparison_20261018/index.html','../identity_20261018/baseline_complete.html'),encoding='utf-8')
    for f in W.glob('*.py'):cp(f,S/f.name)
    # Keep exact upstream bytes and licenses, including intentional Markdown
    # hard breaks. Git text normalization must not invalidate source hashes.
    archive=W/'actual_source_audit_snapshot.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in (W/'source').iterdir():
            if f.is_file():z.write(f,f.name)
    cp(archive,S/archive.name)
    cp(W/'source_manifest.json',S/'source_manifest.json')
    cp(W/'upstream/provenance.json',S/'upstream_provenance.json')
    for f in (W/'pilots').rglob('*'):
        if f.is_file() and f.suffix in ['.json','.jsonl']:cp(f,D/'pilot_metadata'/f.relative_to(W/'pilots'))
    for f in (W/'matched_metadata').glob('*.json'):cp(f,D/'matched_metadata'/f.name)
    banner='<p id="baseline-engineering-warning" style="background:#5d3314;color:#fff1db;padding:16px;border:1px solid #e6a45b;border-radius:8px">旧GN11＋五实例结果尚未通过工程与编辑质量验收。已发现GN11与实例阶段的身份集合不一致；本页保留历史真实输出。<a style="color:#fff" href="../baseline_engineering_20261018/index.html">查看重新安排的工程验收与两阶段重跑</a>。</p>'
    identity=D.parent/'identity_20261018'
    for n in ['baseline_complete.html','sync_player.html','index.html']:
        rel=identity/n;p=long(P/rel)
        if not p.exists():continue
        text=p.read_text(encoding='utf-8')
        if 'id="baseline-engineering-warning"' in text:text=re.sub(r'<p id="baseline-engineering-warning".*?</p>',banner,text,count=1,flags=re.S)
        else:text=text.replace('<body>','<body>'+banner,1)
        p.write_text(text,encoding='utf-8');paths.append(rel.as_posix())
    current=P/'docs/RESEARCH_STATUS.md';text=current.read_text(encoding='utf-8')
    intro='更新：2026-10-18。当前 task=WS-V82-R15-BASELINE-ENGINEERING-20261018；用户要求撤出旧五实例干净基线结论，先排工程再比较算法。failure_ledger_refs=[V82-F06]，不新建failureID、R16/R17，无subagent；原45K暂停保留。'
    text=re.sub(r'更新：2026-10-18。.*?(?=\n\n)',intro,text,count=1,flags=re.S)
    paragraph='''用户最新授权下已重新安排完整五身份GN11 30K＋五实例30K，旧结果保留为历史诊断而非合格基线。确认原GN11动态集合只有a5/a26/a18；a6/a37共45418/555950个五身份正像素在原canonical动态mask中覆盖均0，两车直到实例阶段才退出静态RGB监督。raw128原始/GN11/旧3/旧5同帧链路确认旧5蓝白条带显著恶化；49帧native静态renderer与适配器最大RGB差约2.4e-7。实际CUDA第二批通道、单ID、换序及梯度通过。较早本地快照导致的学习率10倍断言已撤回，实际归档调用路径会补乘0.1。

原raw动态body α100在刚体适配中被0权重及可见ID α替代，遮挡语义不等价；固定拓扑也不能称完整继承原增密/剪枝。已完成4×300步诊断：旧3复现、旧5复现、只恢复raw-body项、只恢复每ID系数100×5/3；短训不代表资产质量通过。五身份一致的独立原始测量输入已完成实际49相机、.1米尺度及SGCM CPU检查；新五身份GN11 30K已经实际启动，独立run为campaign/r15_baseline_owner5_20261018，完成后核对原约束、实际损失梯度、Adam/RNG恢复及编辑表现再启动fresh五实例30K。原GN11/旧3/旧5/原R15/修复R15检查点和原始输入不改，不因旧队列complete提前关机。[本轮组件图、逐帧溯源与固定协议](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/baseline_engineering_20261018/index.html)。human_verdict=null。
'''
    marker='用户最新授权下已重新安排完整五身份GN11'
    if marker not in text:text=text.replace(intro,intro+'\n\n'+paragraph.strip(),1)
    text=text.replace('全部有限GPU队列结束，尚需保存推送、空闲核验及授权关机。','该轮R15队列已结束；当前新增基线工程重跑仍在执行，不能据旧队列状态关机。')
    text=text.replace('新4K和三版分支GPU任务均已结束。报告及视频保存、v82推送核对后，检查无worker及会启动任务的controller，再按既有用户授权关闭AutoDL；实际观察单独记录。','此前R15 4K和三版分支渲染结束；当前用户新增的基线两阶段工程重跑优先。全部新任务、controller、评估、报告与Git交付真正完成后才按既有授权关机，记录实际SSH观察。')
    current.write_text(text,encoding='utf-8');paths.append('docs/RESEARCH_STATUS.md')
    p=P/'docs/EXPERIMENTS.md';text=p.read_text(encoding='utf-8')
    if 'WS-V82-R15-BASELINE-ENGINEERING-20261018' not in text:
        lines=text.splitlines();idx=next(i for i,line in enumerate(lines) if line.startswith('|') and '---' in line)+1
        lines.insert(idx,'| WS-V82-R15-BASELINE-ENGINEERING-20261018 | 旧5未验收；两阶段身份输入审计、实际CUDA通道/梯度、4×300诊断完成；一致五身份GN11 30K已启动，后接经约束/恢复验收的实例30K | [组件与固定协议](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/baseline_engineering_20261018/index.html) |')
        p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    paths.append('docs/EXPERIMENTS.md')
    p=P/'docs/research_failures/entries/V82-F06.md';text=p.read_text(encoding='utf-8')
    if '旧五身份基线重新进行工程验收' not in text:
        text+='\n\n## 2026-10-18：旧五身份基线重新进行工程验收\n\n'+paragraph.strip()+'\n\nGN11与实例阶段新增两车的前景域不一致是实际输入合同变化，不能据此判原算法固有局限。原raw-body α和visible-ID α的替换也不是等价监督，固定拓扑与原生增密/剪枝须单独登记。原有完整49帧真实视频保留并加未验收标记；新的同帧GN11/旧3/旧5图用于追溯退化阶段，不是干净道路GT。学习率初始误判及撤回已记录，后续只绑定实际训练归档源码。\n'
        p.write_text(text,encoding='utf-8')
    paths.append('docs/research_failures/entries/V82-F06.md')
    (W/'publish_paths.json').write_text(json.dumps(sorted(set(paths)),indent=2),encoding='utf-8')
    print('Scoped new baseline evidence files:',len(set(paths)))
if __name__=='__main__':main()
