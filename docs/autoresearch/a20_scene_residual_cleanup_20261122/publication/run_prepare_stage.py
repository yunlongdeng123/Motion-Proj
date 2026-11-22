"""FINALIZED后显式调用：安全解压本轮上传包、准备私有树、暂存并验收；不提交。"""
from pathlib import Path
import argparse
import os
import subprocess
import zipfile
from archive_common import read_json,sha

PUB=Path('/root/autodl-tmp/drivingeditor_repro/campaign/a20_scene_residual_cleanup_20261122_r1/publication')
def main(manifest,archive,receipt,output,proxy):
    manifest=Path(manifest).resolve();archive=Path(archive).resolve();output=Path(output).resolve()
    m=read_json(manifest);r=read_json(receipt)
    if not (m.get('publication_ready') is True and m.get('root_report_complete') is True):raise ValueError('root尚未FINALIZED')
    if archive.stat().st_size!=r['bytes'] or sha(archive)!=r['sha256']:raise ValueError('上传包与本地收据不符')
    target=PUB/archive.stem
    if target.exists():raise FileExistsError('保留旧目录，使用新revision')
    if not output.is_relative_to(PUB) or output.exists():raise ValueError('仅使用本publication的新stage目录')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or any(not (target/n).resolve().is_relative_to(target.resolve()) for n in names):raise ValueError('上传包路径越界或重复')
        target.mkdir()
        for name in names:
            p=target/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
    py='/root/autodl-tmp/envs/motionproj/bin/python'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    plan=output/'plan.json'
    commands=[[py,'-B',str(PUB/'prepare_archive.py'),str(manifest),'--output',str(output),'--proxy',proxy],
              [py,'-B',str(PUB/'publish_selected.py'),str(plan),'--proxy',proxy],
              [py,'-B',str(PUB/'validate_archive.py'),str(plan)]]
    for command in commands:subprocess.run(command,check=True,env=env)
    print('Private stage and source validation complete; no commit/push.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--archive',required=True);p.add_argument('--receipt',required=True);p.add_argument('--output',required=True);p.add_argument('--proxy',required=True);a=p.parse_args();main(a.manifest,a.archive,a.receipt,a.output,a.proxy)
