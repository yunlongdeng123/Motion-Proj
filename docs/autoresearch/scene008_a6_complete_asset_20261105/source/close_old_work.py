"""Close the superseded clipping branch without restarting partial experiments."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np

CAMPAIGN=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
OLD=CAMPAIGN/'scene008_a6_full_basis_20261105_r2'
OUT=CAMPAIGN/'scene008_a6_complete_asset_20261105/old_work_closeout'

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    stop=json.loads((OLD/'STOP_STATE.json').read_text())
    assert stop['controller_absent'] and stop['experiment_absent']
    inventory=[]
    for p in sorted((OLD/'basis_frames').glob('*.npz')):
        valid=True;error=None
        try:
            with zipfile.ZipFile(p) as z:assert z.testzip() is None
            with np.load(p,allow_pickle=False) as f:
                for key in f.files:_=f[key]
        except Exception as exception:valid=False;error=str(exception)
        inventory.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                              size=p.stat().st_size,archive_readable=valid,error=error))
    for name in ('STOP_STATE.json','cpu_moments.json','smoke.json','contract.json','smoke.log','evaluation.log'):
        if (OLD/name).is_file():shutil.copyfile(OLD/name,OUT/name)
    shutil.copytree(OLD/'source',OUT/'source',ignore=shutil.ignore_patterns('__pycache__'))
    result=dict(status='CLOSED_DIRECTION_SUPERSEDED',reason='User requested complete single-object asset recovery instead of native-a6 clipping plus partial PVD',
        old_run=str(OLD),formal_evaluation_complete=False,adopted_asset=False,
        mathematical_cpu_checks='PASS',engineering_smoke_cases=12,
        partial_basis_inventory=inventory,source_models_modified=False,new_runs_started=False,
        preserved_remote=True,autodl='kept on')
    (OUT/'closeout.json').write_text(json.dumps(result,indent=2)+'\n')
    architecture='''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 115"><defs><marker id="a" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7Z" fill="#527180"/></marker></defs><g stroke="#bdccd4"><rect x="5" y="20" width="250" height="65" rx="7" fill="#eef4f7"/><rect x="325" y="20" width="250" height="65" rx="7" fill="#fff0de"/><rect x="645" y="20" width="250" height="65" rx="7" fill="#e4f1eb"/></g><g stroke="#527180" stroke-width="2" marker-end="url(#a)"><path d="M255 52H324"/><path d="M575 52H644"/></g><g font-family="sans-serif" font-size="16" fill="#193441" text-anchor="middle"><text x="130" y="59">旧表示竞争与裁剪诊断</text><text x="450" y="59">停止追加，保存未完现场</text><text x="770" y="59">转向完整单一 a6 资产</text></g></svg>'''
    (OUT/'architecture.svg').write_text(architecture,encoding='utf-8')
    (OUT/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Scene008 旧裁剪分支收尾</title><style>body{font:17px/1.7 system-ui;max-width:1050px;padding:26px;margin:auto;color:#183341}img{width:100%}a{color:#006577}</style><h1>旧裁剪与矩重表达分支收尾</h1><img src="architecture.svg" alt="旧工作归档与新方向"><p>按新研究方向停止追加旧高斯裁剪实验。已保存数学核对、12个工程smoke场景、源码、日志及部分派生文件清单。完整49帧质量评价没有完成，S/U没有执行，未采用新的场景或资产。</p><p>旧现场与模型保留，AutoDL保持开机。接下来以真实观测、原生a6可靠参考和完整PVD提案，恢复一份完整资产后整体替换。</p><p><a href="closeout.json">收尾回执与文件清单</a> · <a href="STOP_STATE.json">停止状态</a> · <a href="smoke.json">工程smoke</a></p></html>''',encoding='utf-8')
    print(json.dumps(dict(closeout=str(OUT),partial_npz=len(inventory),unreadable=sum(not i['archive_readable'] for i in inventory))))

if __name__=='__main__':main()
