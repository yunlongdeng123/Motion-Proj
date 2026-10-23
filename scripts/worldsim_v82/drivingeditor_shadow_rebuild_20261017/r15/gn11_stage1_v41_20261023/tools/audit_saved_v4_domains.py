"""CPU 只读旧 R 保存图与量化贡献图；不充当新 V4.1 或正式浮点指标。"""
import argparse
import base64
import json
from pathlib import Path
import re
import zlib
import numpy as np
from PIL import Image


def unpack(path):
    meta = json.loads(re.search(r'window\.GSDATA\["[^"]+"\]=(.*);', path.read_text(encoding='utf-8'), re.S).group(1))
    raw = zlib.decompress(base64.b64decode(meta['data']))
    return {k:np.frombuffer(raw, dtype=v['dtype'], count=int(np.prod(v['shape'])),
        offset=v['offset']).reshape(v['shape']) for k,v in meta['arrays'].items()}


def rgb(path): return np.asarray(Image.open(path).convert('RGB'), np.float64)/255


def metrics(pred, gt, mask):
    if not mask.any(): return dict(pixels=0, mae=None, psnr=None)
    error = pred[mask]-gt[mask]; mse = float(np.square(error).mean())
    return dict(pixels=int(mask.sum()), mae=float(np.abs(error).mean()), psnr=float(-10*np.log10(mse)) if mse else None)


def main():
    p = argparse.ArgumentParser()
    for key in ('dense', 'viewer', 'native-png', 'output'): p.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args(); meta = json.loads((args.dense/'manifest.json').read_text()); rows = []
    actors = [5, 26, 18, 6, 37, 'other']
    for fid in meta['frame_order']:
        raw = int(fid[-3:]); maps = unpack(args.viewer/f'data/v4_R/{raw}.js')
        gt = rgb(args.native_png/f'gt/image_0_{fid}.png')
        native = rgb(args.native_png/f'composite/image_0_{fid}.png')
        joint = rgb(args.viewer/f'assets/v4_R/joint/{raw:03d}.webp')
        with np.load(args.dense/'frames'/f'{fid}.npz', allow_pickle=False) as z:
            m, protect = z['visible'], z['protect']
            affected = (maps['road_joint']/65535 > .05) & (maps['body_mask'] == 0)
            domains = dict(reliable_road=m, protected=protect, affected=affected,
                affected_unknown=affected & ~m & ~protect)
            for i,a in enumerate(actors):
                domains[f'a{a}_body'] = z['vehicle_body'][i]; domains[f'a{a}_edge'] = z['vehicle_edge'][i]
            row = dict(frame=fid, affected=int(affected.sum()), affected_reliable=int((affected & m).sum()),
                affected_protected=int((affected & protect).sum()), affected_unknown=int((affected & ~m & ~protect).sum()),
                per_domain={k:dict(v1=metrics(native, gt, d), composite=metrics(joint, gt, d)) for k,d in domains.items()})
        rows.append(row)
    totals = {key:sum(r[key] for r in rows) for key in ('affected', 'affected_reliable', 'affected_protected', 'affected_unknown')}
    result = dict(scope='same V4-R 30K saved 8-bit RGB and quantized contribution, CPU read-only',
        not_formal_float_metrics=True, not_V41_quality_results=True, frame_count=49,
        GPU_used=False, real_scene_updates=0, totals=totals,
        affected_fractions={key:totals[key]/totals['affected'] for key in ('affected_reliable', 'affected_protected', 'affected_unknown')},
        mean_per_frame={}, records=rows)
    for key in rows[0]['per_domain']:
        values = [r['per_domain'][key] for r in rows if r['per_domain'][key]['v1']['pixels']]
        result['mean_per_frame'][key] = dict(nonempty_frames=len(values),
            **{name:{metric:float(np.mean([r[name][metric] for r in values])) if values else None
                for metric in ('mae','psnr')} for name in ('v1','composite')})
    with args.output.open('x', encoding='utf-8') as f: json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}, indent=2))


if __name__ == '__main__': main()
