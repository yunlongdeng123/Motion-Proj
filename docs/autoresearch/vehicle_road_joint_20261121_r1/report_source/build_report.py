"""把实际车辆/道路实验导出整理为可离线查看的统一报告。"""
from pathlib import Path
import argparse, json, shutil

W = Path(__file__).resolve().parent
OUT = W.parents[1] / 'outputs/vehicle_road_joint_20261121'

def build(data_file):
    data = json.loads(Path(data_file).read_text('utf-8'))
    OUT.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace('</', '<\\/')
    template = (W / 'report_template.html').read_text('utf-8')
    (OUT / 'report_data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', 'utf-8')
    (OUT / 'index.html').write_text(template.replace('__REPORT_DATA__', payload), 'utf-8')
    shutil.copy2(W / 'viewer.js', OUT / 'viewer.js')
    return OUT / 'index.html'

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('data'); a = p.parse_args()
    print(build(a.data))
