"""在真实公共加载器上导出小规模各阶段视图；不更新任何模型。"""
from pathlib import Path
import argparse
import gc
import json
import os
import time

for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(key,'2')


def main(plan_path):
    import torch
    from drivingeditor_dynamic_edit.road_asset_pipeline import render
    plan=json.loads(Path(plan_path).read_text())
    rows=[]
    for job in plan['jobs']:
        out=Path(job['output'])
        if out.exists():
            raise FileExistsError('保留已有渲染，请使用新输出目录: '+str(out))
        start=time.monotonic()
        result=render(job['manifest'],job['proof'],out,[86,110,134,179,198])
        rows.append(dict(name=job['name'],output=str(out),completed=result['completed'],seconds=time.monotonic()-start))
        Path(plan['receipt']).write_text(json.dumps(dict(training_updates=0,jobs=rows,GPU='keep_on'),indent=2))
        print(json.dumps(rows[-1]),flush=True)
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('plan');main(parser.parse_args().plan)
