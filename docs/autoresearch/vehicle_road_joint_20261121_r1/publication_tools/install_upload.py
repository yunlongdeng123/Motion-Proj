"""核验并展开本轮自己生成的轻量source zip到新目录；不调用Git。"""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,zipfile

def main(archive,output,expected):
    archive=Path(archive);output=Path(output).resolve()
    allowed=Path('/root/autodl-tmp/drivingeditor_repro/campaign/vehicle_road_joint_20261121_r1/publication')
    if not output.is_relative_to(allowed) or output.exists():raise ValueError('仅接受publication下新目录')
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected:raise ValueError('上传zip不符')
    with zipfile.ZipFile(archive) as z:
        catalog=json.loads(z.read('catalog.json'))
        expected_names={'catalog.json'}|{'tree/'+r['destination'].split('vehicle_road_joint_20261121_r1/',1)[1] for r in catalog['sources']}
        if set(z.namelist())!=expected_names or len(z.namelist())!=len(set(z.namelist())):raise ValueError('zip正清单不符')
        for name in z.namelist():
            p=PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts:raise ValueError('zip越界')
            q=output/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(z.read(name))
        for r in catalog['sources']:
            q=Path(r['source'])
            if not q.is_relative_to(output) or hashlib.sha256(q.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('展开来源不符')
    print(json.dumps(dict(installed=True,files=len(catalog['sources']),output=str(output),formal_stage=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--output',required=True);p.add_argument('--sha256',required=True)
    a=p.parse_args();main(a.archive,a.output,a.sha256)
