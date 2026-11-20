"""Package existing BUILD scalar results; no metric, renderer or training call."""
from pathlib import Path
import argparse,json,tarfile,hashlib

def finish(root):
    root=Path(root);out=root/'evaluation_r1';d=json.loads((out/'results.json').read_text('utf-8'))
    assert d['complete']and d['final_endpoint_set_complete']and len(d['rows'])==54 and len(d['packets'])==15
    assert all(r['split']=='BUILD_fit_all_qualified'for r in d['rows'])
    lines=['# 全合格标签BUILD拟合结果','',
        'F0复用已核初始ci与旧F0逐字节一致的公共原生packet。新FIELD与NUMERIC都是固定方法的全标签资产拟合；原held现在属于BUILD_fit，不再是验证。',
        '', '| 帧 | 源 | 端点 | σ1 core n | RGB MAE | HF cos | HF能量比 | flat HF RMS | edge P / R |',
        '|---:|---|---|---:|---:|---:|---:|---:|---|']
    def f(v,n=4):return 'N/A'if v is None else f'{v:.{n}f}'
    for r in d['rows']:
        if r['scale']!=1:continue
        lines.append(f"| {r['raw']} | {r['source']} | {r['arm']} | {r['core_n']} | {f(r['RGB_MAE'],5)} | {f(r['HF_cosine'])} | {f(r['HF_energy_ratio'],2)} | {f(r['flat_pred_RMS'],5)} | {f(r['edge_precision'],3)} / {f(r['edge_recall'],3)} |")
    lines+=['','完整54行包含三尺度，见metrics.csv；context先union后重算core，未复用旧held边界核心。',
        '', '风险：这是已知标签上的拟合，不是盲测、隐藏道路GT或新方法排行榜。新BUILD normalizer不能与旧QUERY训练loss数值直接比较。',
        '', '覆盖、无交点、静态alphaT、K与只读RGB范围代理独立保留；HF能量增大不等于正确质感。没有自动视觉PASS、全局表示极限证明或生产晋升。']
    (out/'summary.md').write_text('\n'.join(lines)+'\n','utf-8')
    package=root/'evaluation_r1_light.tar.gz'
    if package.exists():raise FileExistsError('preserve existing lightpackage')
    selected=[out/'results.json',out/'metrics.csv',out/'summary.md',root/'qa/evaluation_plan_r1.json',root/'qa/evaluation_binding_r1.json']
    with tarfile.open(package,'w:gz')as t:
        for p in selected:t.add(p,arcname=p.name,recursive=False)
    m=dict(archive=str(package),bytes=package.stat().st_size,SHA256=hashlib.sha256(package.read_bytes()).hexdigest(),
        files=[dict(name=p.name,SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)for p in selected],
        GPU_executed=False,native_calls=0,optimizer_updates=0)
    (root/'evaluation_r1_light_manifest.json').write_text(json.dumps(m,indent=2)+'\n','utf-8')
    print(json.dumps(m))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();finish(a.root)
