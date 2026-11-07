from pathlib import Path
import json,shutil,subprocess
RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_road_fusion_bake_20261107_r1')
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42');OWN='docs/autoresearch/scene008_road_fusion_bake_20261107'
qa=REPO/OWN/'publication_qa'
for p in (RUN/'publication_qa').glob('*'):
    if p.is_file():shutil.copy2(p,qa/p.name)
shutil.copy2(Path(__file__),qa/Path(__file__).name)
(qa/'README.md').write_text('''# 发布核验

完整202,358,434字节交付已取回，1496项manifest逐文件一致，1137张PNG可解码，全部49帧动态图集和13条本地静态引用存在；inline JS仅做语法检查。
独立科学复核已确认指标、分母、负例及392张native PNG源字节。
自动审批拒绝了无头浏览器启动，浏览器工具也因仅允许HTTP/HTTPS拒绝本地file页面；未完成浏览器交互／移动布局验收，未绕过限制。
此发布QA位于不可变科学packet之外，完整目标另有精确私有备份；详见相应收据。
''')
status=REPO/'docs/RESEARCH_STATUS.md';text=status.read_text();marker='v44＋a5车顶恢复已完成';suffix=text[text.index(marker):]
note='发布核验：完整交付逐文件与PNG静态检查通过；自动浏览器交互／移动布局验收被本地file协议策略拦截，未执行，也未宣称通过。\n\n'
assert note not in text
status.write_text(text[:text.index(marker)]+note+suffix)
assert status.read_text().endswith(suffix)
def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True)
git('add','--',OWN+'/publication_qa','docs/RESEARCH_STATUS.md')
allowed=[OWN+'/','docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md','docs/research_failures/entries/V82-F13.md']
staged=git('diff','--cached','--name-only').splitlines()
assert all(p.startswith(allowed[0]) or p in allowed[1:] for p in staged)
subprocess.run(['git','diff','--cached','--check'],cwd=REPO,check=True)
branch=git('branch','--show-current').strip()
result=subprocess.run(['git','commit','-m','research(scene008): fuse observed road targets into persistent material'],cwd=REPO,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(RUN/'publication_commit.log').write_text(result.stdout)
if result.returncode:raise RuntimeError(result.stdout[-4000:])
receipt=dict(commit=git('rev-parse','HEAD').strip(),branch=branch,files=len(staged),diff_check_passed=True,
    unrelated_remaining=git('diff','--name-only').splitlines(),browser_interaction_QA='BLOCKED_BY_POLICY_NOT_PASSED',AutoDL='KEEP_ON')
(RUN/'publication_commit_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
