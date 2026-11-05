"""Stage only this reviewed publication and two research navigation documents.

Run remotely after uploading publication_stage_v1.tgz. Never commits or pushes.
The unrelated user render-bridge edit must remain byte-exact and unstaged.
"""
import argparse,hashlib,json,re,shutil,subprocess,tarfile
from pathlib import Path,PurePosixPath

REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a6_insert_artifacts_20261105')
DOC=REPO/'docs/autoresearch/scene008_a6_insert_artifacts_20261105'
BRANCH='research/worldsim-v8.2-drivingeditor-v42'
USERFILE=REPO/'scripts/run_worldsim_v72_e3_render_bridge.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args],text=True).strip()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--overlay',required=True,action='append');p.add_argument('--phase',default='fixed_LOD49_review');args=p.parse_args()
    assert git('branch','--show-current')==BRANCH
    user_before=sha(USERFILE);head=git('rev-parse','HEAD');before=git('status','--porcelain')
    backup=RUN/'publication/backups'/args.phase;backup.mkdir(parents=True,exist_ok=False)
    DOC.mkdir(parents=True,exist_ok=True)
    for rel in ['docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md']:
        dest=backup/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/rel,dest)
    prior=DOC/'prior_research_status.md'
    if not prior.exists():shutil.copy2(REPO/'docs/RESEARCH_STATUS.md',prior)
    for overlay in args.overlay:
        with tarfile.open(overlay) as tar:
            for m in tar.getmembers():
                rel=PurePosixPath(m.name)
                assert m.isfile() and not rel.is_absolute() and '..' not in rel.parts
                target=DOC/Path(*rel.parts)
                assert target.resolve().is_relative_to(DOC.resolve())
                if target.exists():
                    old=backup/'publication'/Path(*rel.parts);old.parent.mkdir(parents=True,exist_ok=True)
                    if not old.exists():shutil.copy2(target,old)
            tar.extractall(DOC)
    manifest=json.loads((DOC/'publication_manifest.json').read_text())
    for row in manifest['files']:
        target=DOC/row['path']
        if not target.exists():
            # Avoid transferring an already retained immutable canonical asset.
            assert row.get('remote_origin')
            source=Path(row['remote_origin']);assert sha(source)==row['sha256']
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        assert target.stat().st_size==row['bytes'] and sha(target)==row['sha256'],row['path']
    old_status_bytes=(REPO/'docs/RESEARCH_STATUS.md').read_bytes()
    # Preserve the other active/recent research blocks byte-for-byte. Only the
    # a6 main-line snapshot is replaced, with its old complete text archived.
    other_start=old_status_bytes.find('RCR-1支线'.encode())
    history_start=old_status_bytes.find('历史：'.encode(),other_start)
    assert other_start>=0 and history_start>other_start,'Review changed status layout before writing'
    other_blocks=old_status_bytes[other_start:history_start].rstrip(b'\r\n')
    status='''# 当前研究状态

更新：2026-11-06。

当前主线为 Scene008 a6 的实际几何与外观恢复。PVD、预训练模型及背景参数冻结，原 M2 与旧 projective 资产保留，研究候选没有自动采用。

本轮已完成删 a37 后黑斑的实际颜色载体与几何位置诊断、观测锚定的 C2 连续深度后验、固定前景 matte RGBA 组合，以及相同 BUILD40 像素域的颜色拟合和固定49帧检查。新 chart 几何确实改变；原生480的 RGB 退化、HR/深度改善与外域局部变化须分别解释，不能把工程 PASS 当画质通过。详细证据见[本轮报告](autoresearch/scene008_a6_insert_artifacts_20261105/README.md)。

同一连续 surface / 固定 texture 的 SurfaceLOD 已完成固定49帧和必要的覆盖率读出；RGB与深度对照改善，同时保留少量低alpha及大幅深度前移的限制。报告、可重放入口与取回验收已完成，结果保留为研究候选，无后续训练控制器，没有启动 PVD 后训练或自由参数搜索。HELD9 未进入本次颜色目标，但此前已用于表示诊断与内部开发，不是独立未触碰泛化集。

执行边界：subagent 仅 gpt-6.1-sol/max；本会话明确保持 AutoDL 开机，不自动关机。failure_ledger_refs=[]，failure_ledger_delta=none；人工画质 verdict 未代填。

__OTHER_RESEARCH_BLOCKS__

历史状态保存在[更新前快照](autoresearch/scene008_a6_insert_artifacts_20261105/prior_research_status.md)，实验导航见[索引](EXPERIMENTS.md)。
'''
    status_bytes=status.encode().replace(b'__OTHER_RESEARCH_BLOCKS__',other_blocks)
    assert other_blocks in status_bytes
    (REPO/'docs/RESEARCH_STATUS.md').write_bytes(status_bytes)
    index=REPO/'docs/EXPERIMENTS.md';text=index.read_text();marker='scene008_a6_insert_artifacts_20261105'
    if marker not in text:
        text=text.replace('# 实验索引\n','# 实验索引\n\n## Scene008 a6 插入成像诊断与连续表面（2026-11-05）\n\n- `'+marker+'`：[实际颜色载体、C2 深度与前景 RGBA、viewport 和固定域评价](autoresearch/'+marker+'/README.md)。配置、资源、完整结果与输入角色见报告；failure_ledger_delta=none。\n',1)
        index.write_text(text)
    # Only the existing tracked historical entry receives a navigation banner.
    # Its 62KB body is retained; the separate 7.34MB local viewer is not imported.
    oldrel='docs/autoresearch/scene008_a6_appearance_formal_20261105/delivery/index.html'
    oldpath=REPO/oldrel;oldbytes=oldpath.read_bytes()
    banner_pattern=rb'<aside id="current-a6-diagnosis-link"[^>]*>.*?</aside>'
    original_body=re.sub(banner_pattern,b'',oldbytes,flags=re.S)
    head_body=subprocess.check_output(['git','-C',str(REPO),'show','HEAD:'+oldrel])
    assert original_body==re.sub(banner_pattern,b'',head_body,flags=re.S),'Historical entry has unrelated edits'
    navurl='../../scene008_a6_insert_artifacts_20261105/delivery/index.html'
    banner=('<aside id="current-a6-diagnosis-link" style="padding:16px 22px;background:#e9f5ff;border:2px solid #357da3;color:#12364a;font:16px/1.6 system-ui,sans-serif">'
        '<strong>本页保留上一轮结果。</strong> <a style="color:#064f7d;font-weight:700" href="'+navurl+'">查看本轮来源诊断、连续表面与固定LOD结果及其限制</a></aside>').encode()
    tag=re.search(rb'<body[^>]*>',original_body);assert tag
    newbytes=original_body[:tag.end()]+banner+original_body[tag.end():]
    assert re.sub(banner_pattern,b'',newbytes,flags=re.S)==original_body
    for q in [oldpath,oldpath.parent/'file_manifest.json']:
        target=backup/q.relative_to(REPO);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(q,target)
    oldpath.write_bytes(newbytes)
    old_manifest=oldpath.parent/'file_manifest.json';mb=old_manifest.read_bytes();oldjson=json.loads(mb)
    pattern=rb'("path": "index\.html",\s*"bytes": )\d+(,\s*"sha256": ")[0-9a-f]+(")'
    newmb,n=re.subn(pattern,lambda m:m[1]+str(len(newbytes)).encode()+m[2]+sha(oldpath).encode()+m[3],mb)
    assert n==1;newjson=json.loads(newmb)
    for a,b in zip(oldjson['files'],newjson['files']):
        if a['path']!='index.html':assert a==b
    old_manifest.write_bytes(newmb)
    navigation=dict(path=oldrel,previous_body_sha256=hashlib.sha256(original_body).hexdigest(),new_sha256=sha(oldpath),
        repo_relative_target=navurl,historical_body_and_all_original_assets_unchanged=True,
        manifest_only_index_size_hash_updated=True,local_viewer_fixed_html_not_imported=True,
        local_outputs_UI_navigation_and_Git_historical_entry_have_different_scope=True)
    write(DOC/'historical_navigation_receipt.json',navigation)
    assert sha(USERFILE)==user_before
    git('add','--','docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md',str(DOC.relative_to(REPO)),oldrel,str(old_manifest.relative_to(REPO)))
    # Three small, explicitly published canonical/replay NPZ files are
    # globally ignored by design. Stage only these reviewed hashes, never
    # matrices, raw datasets, the large retrieval archive or checkpoints.
    forced=[]
    for row in manifest['files']:
        if row['path'].endswith('.npz'):
            assert row['bytes']<4_000_000 and row.get('remote_origin'),row['path']
            path=str((DOC/row['path']).relative_to(REPO));git('add','-f','--',path);forced.append(dict(path=path,bytes=row['bytes']))
    staged=git('diff','--cached','--name-only').splitlines()
    allowed=('docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md',str(DOC.relative_to(REPO))+'/',oldrel,str(old_manifest.relative_to(REPO)))
    assert all(x in allowed[:2]+allowed[3:] or x.startswith(allowed[2]) for x in staged),staged
    assert 'scripts/run_worldsim_v72_e3_render_bridge.py' not in staged
    subprocess.run(['git','-C',str(REPO),'diff','--cached','--check'],check=True)
    receipt=dict(status='STAGED_ONLY_WAITING_PARENT_REVIEW',branch=BRANCH,head_unchanged=head,
        initial_status=before,staged_files=len(staged),staged_worktree_bytes=sum((REPO/x).stat().st_size for x in staged),
        staged_scope=list(allowed),publication_files=len(manifest['files']),
        explicitly_forced_small_NPZ_files=forced,
        historical_navigation=navigation,
        RCR1_a5_other_status_blocks_byte_exact_preserved=True,
        user_unrelated_path=str(USERFILE.relative_to(REPO)),user_unrelated_sha256=user_before,
        unrelated_byte_exact_and_unstaged=True,commit_performed=False,push_performed=False,
        pending=['Parent review before commit and push'])
    write(RUN/'publication'/('staging_'+args.phase+'.json'),receipt);print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':main()
