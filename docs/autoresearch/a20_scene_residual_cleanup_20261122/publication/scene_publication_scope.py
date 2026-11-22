"""固定a20场景清理的发布边界；不要求训练摘要，不运行模型。"""
from pathlib import Path
import copy
import hashlib
import json
import re

ARCHIVE='docs/autoresearch/a20_scene_residual_cleanup_20261122'
SHARED_REPORT='docs/autoresearch/a20_joint_reconstruction_20261122'
REMOTE='/root/autodl-tmp/drivingeditor_repro/campaign/a20_scene_residual_cleanup_20261122_r1'
ACTOR_SHA='ab4fa110440d579bcfd57e4bd0b8a7f61c3f7821189450b58bd027c2f28f67a7'
REPORT_UPDATES={SHARED_REPORT+'/index.html',SHARED_REPORT+'/scene_report_ui.js',SHARED_REPORT+'/source_manifest.json'} | {
    SHARED_REPORT+'/report_source/'+name for name in ['assemble_result.py','build_report.py','report_ui.js','qa_report.py','report_style.css','report_groups.json','result.json']}


def validate_publication_scope(value):
    if value.get('task_id')!='A20-SCENE-RESIDUAL-CLEANUP-20261122' or value.get('archive_root')!=ARCHIVE:
        raise ValueError('任务或归档根目录不符')
    if not (value.get('publication_ready') is True and value.get('root_report_complete') is True):
        raise ValueError('root最终选中sidecar与限定结论尚未冻结')
    if value.get('fixed_actor_sha256')!=ACTOR_SHA or not value.get('actual_result_kind'):
        raise ValueError('必须绑定固定a20及root实际限定结论')
    sidecar=value.get('selected_sidecar')
    if not isinstance(sidecar,dict) or not re.fullmatch('[a-f0-9]{64}',sidecar.get('sha256','')):
        raise ValueError('缺少最终选中sidecar的实际SHA')
    path=Path(sidecar.get('path','')).resolve()
    if not path.is_relative_to(Path(REMOTE)) or not path.is_file():raise ValueError('sidecar必须来自本轮真实远端任务')
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=sidecar['sha256'] or len(data)!=sidecar.get('bytes'):
        raise ValueError('最终sidecar与实际文件字节不符')


def preserve_group_prefix(old,new):
    if new.get('groups',[])[:len(old.get('groups',[]))]!=old.get('groups',[]):
        raise ValueError('历史实际组被修改或丢失')
    if 'history' in old and new.get('history',[])[:len(old['history'])]!=old['history']:
        raise ValueError('历史导航被修改或丢失')
    for key in ['final_iteration','asset_verification','initial_rows']:
        if key in old and new.get(key)!=old[key]:raise ValueError('历史候选或原始实测被改写：'+key)
    ids={g['id'] for g in new.get('groups',[])}
    for key in ['default_group','current_group']:
        if key in new and new[key] not in ids:raise ValueError('默认阶段缺少实际组')


def validate_shared_report_delta(name,old,new):
    if name not in REPORT_UPDATES:raise ValueError('非明确允许的报告追加路径')
    if name.endswith('/result.json') or name.endswith('/report_groups.json'):
        preserve_group_prefix(json.loads(old),json.loads(new))
    elif name.endswith('/index.html'):
        extract=lambda b:json.loads(re.search(rb'<script>const D=(.*?);</script>',b,re.S).group(1))
        preserve_group_prefix(extract(old),extract(new))
    elif name.endswith('/source_manifest.json'):
        prior=json.loads(old);after=json.loads(new)
        for key,value in prior.items():
            if key not in ['source_files','report_append_revisions'] and after.get(key)!=value:
                raise ValueError('历史科学来源metadata改变：'+key)
        for path,rec in prior['source_files'].items():
            if path not in REPORT_UPDATES and after['source_files'].get(path)!=rec:
                raise ValueError('历史非报告源绑定被改写：'+path)
        previous=prior.get('report_append_revisions',[])
        if after.get('report_append_revisions',[])[:len(previous)]!=previous:raise ValueError('报告追加谱系丢失')
    elif not old and name!=SHARED_REPORT+'/scene_report_ui.js':
        raise ValueError('仅允许新加scene UI脚本，其余报告源码应已存在')


def protected_objects(git,tree):
    roots=['AGENTS.md']
    for name in git.call('ls-tree','-d','--name-only',tree+':docs/autoresearch').decode().splitlines():
        path='docs/autoresearch/'+name
        if path not in [ARCHIVE,SHARED_REPORT]:roots.append(path)
    roots += [p for p in git.call('ls-tree','-r','--name-only',tree,'--',SHARED_REPORT).decode().splitlines() if p not in REPORT_UPDATES]
    return {p:git.call('rev-parse',tree+':'+p).decode().strip() for p in roots}


def append_shared_source_manifest(git,base,selected,payload,task_id):
    touched={name:rec for name,rec in selected.items() if name in REPORT_UPDATES}
    if not touched:return None
    path=SHARED_REPORT+'/source_manifest.json'
    if path in selected:raise ValueError('共享报告source_manifest由明确source绑定自动追加，不接受覆盖副本')
    old=git.call('show',base+':'+path);value=json.loads(old)
    value=copy.deepcopy(value)
    value['source_files'].update(touched)
    value.setdefault('report_append_revisions',[]).append(dict(task_id=task_id,parent=base,
        source_manifest=ARCHIVE+'/source_manifest.json',updated_report_source_sha256={k:v['sha256'] for k,v in touched.items()}))
    data=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
    validate_shared_report_delta(path,old,data);payload[path]=data
    return path
