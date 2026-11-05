"""Extend the road review with a paired PVD column and a diagnostic close-up."""
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent.parent
ROAD=ROOT.parent/'scene008_native_road1k_20261105_r1'
OLD=ROOT.parent/'scene008_a37_only_B10k_road_20261103_r3'
REF=ROOT.parent/'scene008_a6_a37_dynamic_20261104_r1/inputs/a6_probe_reference.npz'
OBS=ROOT.parent/'scene008_instances_20261105_r1/observations/frames'
PILOTS=(105,113,130)
LABEL={
    'a6_rgb':'可见 a6 RGB MAE',
    'a37_rgb':'可见 a37 RGB MAE',
    'pvd_on_observed_a6':'PVD 在可见 a6 上的贡献',
    'pvd_on_observed_a37':'PVD 在可见 a37 上的贡献',
    'pvd_on_fixed_unknown_proxy':'PVD 在固定补全足迹上的贡献',
    'a6_total_on_reprojected':'实测回投射线上的 a6 总贡献',
    'reobserved_road_rgb':'显露道路 RGB MAE',
    'a6_measured_depth':'a6 实测深度 MAE (m)',
}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def pixels(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'))


def pool(values):
    count=sum(value['count'] for value in values)
    return dict(count=count,mean=sum(value['sum'] for value in values)/count if count else None)


def main():
    render=ROOT/'render'
    metrics=load(render/'metrics.json')
    rows=metrics['rows']
    assert [row['raw'] for row in rows]==list(range(89,138))
    summary={}
    for state in ('delete','keep'):
        summary[state]={}
        for key, item in rows[0][state].items():
            if 'count' in item:
                summary[state][key]={'pvd':pool([r[state][key] for r in rows])}
            else:
                summary[state][key]={arm:pool([r[state][key][arm] for r in rows]) for arm in item}
    metrics['pooled']=summary
    dest=ROOT/'pvd_addon'
    dest.mkdir(exist_ok=False)
    images=dest/'pvd_images'
    images.mkdir()
    evaluation=ROAD/'evaluation/all49'
    with np.load(REF,allow_pickle=False) as ref:
        proxy=ref['depth_m']
    for raw in range(89,138):
        base=evaluation/f'{raw:03d}_baseline.png'
        no_pvd=evaluation/f'{raw:03d}_repaired.png'
        pvd=render/f'{raw:03d}_delete.png'
        old=OLD/'evaluation/all49'/f'{raw:03d}_repaired.png'
        gt=ROAD/'native_delete'/f'{raw:03d}'/'gt.png'
        no_edit0=evaluation/f'{raw:03d}_noedit_parent.png'
        no_edit1=evaluation/f'{raw:03d}_noedit_road.png'
        no_edit2=render/f'{raw:03d}_keep.png'
        panels=(base,no_pvd,old,pvd,gt,no_edit0,no_edit1,no_edit2)
        grid=np.vstack([np.hstack([pixels(path) for path in panels[:4]]),
                        np.hstack([pixels(path) for path in panels[4:]])])
        Image.fromarray(grid).save(images/f'{raw:03d}.jpg',quality=91)
        if raw in PILOTS:
            with np.load(OBS/f'{raw:03d}.npz',allow_pickle=False) as source:
                mask=source['visible'][0] | (proxy[raw-89,2]>0)
            y,x=np.nonzero(mask)
            xmin,xmax=max(0,int(x.min())-14),min(480,int(x.max())+15)
            ymin,ymax=max(0,int(y.min())-10),min(320,int(y.max())+12)
            stack=[pixels(path)[ymin:ymax,xmin:xmax] for path in (no_pvd,pvd,render/f'{raw:03d}_delete_pvd_w.png')]
            canvas=Image.fromarray(np.hstack(stack)).resize((3*(xmax-xmin)*3,(ymax-ymin)*3),Image.Resampling.NEAREST)
            view=Image.new('RGB',(canvas.width,canvas.height+28),'#172837')
            view.paste(canvas,(0,28))
            draw=ImageDraw.Draw(view)
            for i,label in enumerate(('10K + road 1K','+ fixed PVD','PVD alpha*T')):
                draw.text((i*(xmax-xmin)*3+7,8),label,fill='white')
            view.save(dest/f'pvd_detail_{raw:03d}.png')
    (dest/'pvd_compare.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
    html=(ROAD/'delivery/index.html').read_text(encoding='utf-8')
    replacements={
        'Scene008 原生 10K + R3 道路恢复 1K</h1>':'Scene008 原生 10K + R3 道路恢复 1K · PVD 对照</h1>',
        'max-width:1100px':'max-width:1760px',
        '上排从左到右为本轮原生 10K 删车、本轮 +1K 删车、旧强 R3 删车。下排依次为原始 RGB、本轮原生无编辑、本轮 +1K 无编辑。':
        '上排从左到右为原生 10K 删车、+道路 1K 删车、旧强 R3 删车、+道路 1K + PVD 删车。下排依次为原始 RGB、原生 10K 无编辑、+道路 1K 无编辑、+道路 1K + PVD 无编辑。',
        'images/113.jpg':'pvd_images/113.jpg',
        "image.src='images/'":"image.src='pvd_images/'",
        '六格对照':'八格对照',
    }
    for a,b in replacements.items():
        assert a in html,a
        html=html.replace(a,b)
    metric_rows=[]
    for state, variants in summary.items():
        for key, item in variants.items():
            before=item.get('before')
            after=item['pvd']
            metric_rows.append(f"<tr><th>{'删车' if state=='delete' else '无编辑'} · {LABEL[key]}</th>"
                               f"<td>{before['mean']:.6f}</td>" if before else
                               f"<tr><th>{'删车' if state=='delete' else '无编辑'} · {LABEL[key]}</th><td>—</td>")
            metric_rows[-1]+=f"<td>{after['mean']:.6f}<small>{after['count']:,}</small></td></tr>"
    section='''<section><h2>固定 PVD 补全面</h2>
<p>复用既有阶段 D 的 1,477 个 a6 生成面高斯及其推断颜色。它们沿 a6 轨迹进入原生联合深度排序，两种编辑状态共用。原生 10K、道路 1K 和 PVD 资产均冻结，训练更新为零。固定补全足迹用来测量显露程度，隐藏面缺独立准确率真值。</p>
<p>删车局部放大。三栏依次为道路 1K、加 PVD、PVD 可见贡献。</p>
<p><a href="pvd_detail_105.png">raw105</a> · <a href="pvd_detail_113.png">raw113</a> · <a href="pvd_detail_130.png">raw130</a></p>
<img src="pvd_detail_113.png" alt="raw113 PVD 局部放大">
<table><tr><th>固定域指标</th><th>道路 1K</th><th>道路 1K + PVD</th></tr>'''+''.join(metric_rows)+'''</table>
<p><a href="pvd_compare.json">完整 49 帧 PVD 指标</a> · <a href="before_pvd.html">原三列报告</a></p></section>'''
    old_title='<section><h2>可核对文件</h2>'
    assert old_title in html
    html=html.replace(old_title,section+old_title)
    (dest/'index_pvd.html').write_text(html,encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
