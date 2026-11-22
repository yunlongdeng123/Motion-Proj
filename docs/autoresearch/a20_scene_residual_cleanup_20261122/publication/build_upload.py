"""按显式最终白名单封装轻量上传包；不联网、不操作Git或GPU。"""
from pathlib import Path
import argparse
import copy
import json
import re
import zipfile
from archive_common import ALLOWED_ROLES, dump_json, digest, lightweight_text, read_json, valid_destination

PUB=Path(__file__).resolve().parent
WORK=PUB.parent
REMOTE='/root/autodl-tmp/drivingeditor_repro/campaign/a20_scene_residual_cleanup_20261122_r1'


def build(config,whitelist,revision):
    m=copy.deepcopy(read_json(config));w=read_json(whitelist)
    if not (m.get('publication_ready') is True and m.get('root_report_complete') is True and w.get('finalized') is True):
        raise ValueError('等待root最终选中sidecar、限定结论及显式白名单冻结')
    if w.get('pending_required_sections'):raise ValueError('最终必要来源尚未完成')
    if not re.fullmatch('[a-zA-Z0-9_-]+',revision):raise ValueError('revision格式不符')
    out=PUB/('upload_'+revision)
    if out.exists():raise FileExistsError('保留旧封包，使用新revision')
    remote_payload=REMOTE+'/publication/'+out.name
    sources=[];seen=set();counter=0
    def local(path):
        nonlocal counter
        p=Path(path);p=p if p.is_absolute() else WORK/p
        data=lightweight_text(p);rel='payload/'+str(counter).zfill(3)+p.suffix;counter+=1
        q=out/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(data)
        return remote_payload+'/'+rel,dict(path=str(p.resolve()),sha256=digest(data))
    for rec in w['entries']:
        name=valid_destination(rec['destination'],allow_link_updates=True)
        if name in seen or rec['role'] not in ALLOWED_ROLES:raise ValueError('重复路径或非法来源角色：'+name)
        seen.add(name);item=dict(destination=name,role=rec['role'])
        if rec['origin']=='local':
            item['source'],item['local_origin']=local(rec['source']);item['sha256']=item['local_origin']['sha256']
        elif rec['origin']=='remote':
            if not rec['source'].startswith(REMOTE+'/'):raise ValueError('远端来源必须在本轮任务根下')
            item['source']=rec['source']
            if rec.get('sha256'):item['sha256']=rec['sha256']
        else:raise ValueError('origin须明确local或remote')
        sources.append(item)
    for section in ['status_update','experiments_append','failure_append']:
        if not m.get(section):continue
        src=m[section]['source']
        if not src.startswith('/root/'):
            m[section]['source'],_=local(src)
    m['sources']=sources
    manifest=PUB/('manifest.'+revision+'.json');dump_json(manifest,m)
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out).as_posix())
    if archive.stat().st_size>4_000_000:raise ValueError('上传包超过4MB；精简显式来源，禁止加入媒体/模型')
    receipt=PUB/('upload_receipt.'+revision+'.json')
    dump_json(receipt,dict(path=str(archive),bytes=archive.stat().st_size,sha256=digest(archive.read_bytes()),source_files=len(sources),manifest=str(manifest),no_models_or_media=True))
    print(json.dumps(dict(archive=str(archive),manifest=str(manifest),receipt=str(receipt),bytes=archive.stat().st_size)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--whitelist',required=True);p.add_argument('--revision',required=True)
    a=p.parse_args();build(a.config,a.whitelist,a.revision)
