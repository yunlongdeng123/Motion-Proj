"""Acceptance and report for fixed-geometry road calibration."""
import csv
import json
from pathlib import Path
import shutil
import sys

RUN=Path(__file__).resolve().parent.parent
OLD=RUN.parent/'scene008_a6_residual_replay_20261105'
sys.path.insert(0,str(OLD/'source'))
from report import image_grid

ARCH='''<svg viewBox="0 0 940 152" role="img" aria-label="道路颜色校正数据流">
<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#607282"/></marker></defs>
<g fill="none" stroke="#607282" stroke-width="2" marker-end="url(#arr)">
<path d="M178 41H210V73H236"/><path d="M178 112H210V73H236"/><path d="M451 73H485"/><path d="M688 73H725"/>
</g><g stroke="#bccbd7">
<rect x="10" y="17" width="168" height="49" rx="8" fill="#edf4f8"/><rect x="10" y="87" width="168" height="49" rx="8" fill="#edf4f8"/>
<rect x="236" y="48" width="215" height="50" rx="8" fill="#e1f1eb"/><rect x="485" y="48" width="203" height="50" rx="8" fill="#f7f0e0"/>
<rect x="725" y="48" width="203" height="50" rx="8" fill="#edf4f8"/></g>
<g fill="#172938" font-family="sans-serif" font-size="15" text-anchor="middle">
<text x="94" y="47">冻结场景与道路行</text><text x="94" y="117">RGB 与跨时刻道路观测</text>
<text x="344" y="79">联合排序输出空间基权重</text><text x="586" y="79">全局或世界网格颜色校正</text>
<text x="826" y="79">两种编辑状态逐帧验证</text></g></svg>'''
LABELS={'parent':'原道路 1K','global':'全局颜色偏移','world':'世界坐标网格','colour_oracle':'逐像素颜色上界诊断'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def pool(rows,state,arm,key):
    scores=[r['states'][state][arm][key] for r in rows]
    n=sum(m['count'] for m in scores)
    return dict(n=n,mean=sum(m['sum'] for m in scores)/n if n else None)


def mean(m):
    return m['sum']/m['count'] if m['count'] else 0.


def main():
    result=read(RUN/'result.json')
    design=read(RUN/'design.json')
    rows=result['rows']
    assert len(rows)==49 and result['geometry_parity']==0.
    fit=[r for r in rows if not r['holdout']]
    held=[r for r in rows if r['holdout']]
    stats={}
    for scope,items in [('all',rows),('fit',fit),('holdout',held)]:
        stats[scope]={}
        for state in ('keep','delete'):
            stats[scope][state]={}
            for arm in items[0]['states'][state]:
                stats[scope][state][arm]={key:pool(items,state,arm,key) for key in items[0]['states'][state][arm]}
    def score(scope,state,arm,key='road'):
        return stats[scope][state][arm][key]['mean']
    variants=('parent','global','world')
    # Use fit views for ranking. Held views only test the chosen arm.
    ranking=[]
    for name in variants:
        if all(score('fit',s,name)<=score('fit',s,'parent') for s in ('keep','delete')):
            ranking.append((sum(score('fit',s,name) for s in ('keep','delete')),name))
    trial=min(ranking)[1]
    guard={}
    for name in variants:
        issues=[]
        for row in rows:
            for state in ('keep','delete'):
                m=row['states'][state][name]
                failures=[k for k,limit in [('a6_delta',.0005),('a37_delta',.0005),('protected_delta',.001)]
                          if mean(m[k])>limit]
                if failures:
                    issues.append(dict(raw=row['raw'],state=state,failures=failures))
        guards=dict(frame_failures=issues,mean_failures=[])
        for scope in ('all','holdout'):
            for state in ('keep','delete'):
                for key in ('road','seam'):
                    if score(scope,state,name,key)>score(scope,state,'parent',key)+1e-7:
                        guards['mean_failures'].append(dict(scope=scope,state=state,metric=key))
        if score('all','keep',name,'full')>score('all','keep','parent','full')+1e-7:
            guards['mean_failures'].append(dict(scope='all',state='keep',metric='full'))
        guard[name]=guards
    accepted=trial if not any(guard[trial].values()) else 'parent'
    output=RUN/'delivery'
    output.mkdir(exist_ok=False)
    images=output/'frames'
    images.mkdir()
    for raw in (89,105,113,130,137):
        src=RUN/'render'
        paths=[src/f'{raw:03d}_{s}_{a}.png' for s in ('keep','delete') for a in ('parent',trial)]
        labels=['No edit | parent',f'No edit | {trial}', 'Delete a37 | parent', f'Delete a37 | {trial}']
        image_grid(paths,labels).save(images/f'{raw:03d}.jpg',quality=92)
    for name in ('result.json','coefficients.npz','design.json','shader_validation.json','parent_replay.json'):
        shutil.copyfile(RUN/name,output/name)
    receipt=dict(status='PASS',selected_on_fit=trial,accepted=accepted,stats=stats,guards=guard,geometry_parity=0.,
                 holdout=result['holdout'],freeze='All geometry, coverage, and a6/a37/PVD/street representation frozen.',
                 solver='No reconstruction training; constrained robust road colour solve.')
    (output/'summary.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    with (output/'fixed_metrics.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        csvout=csv.writer(handle,lineterminator='\n')
        csvout.writerow(['raw','held_from_colour_fit','state','arm','metric','count','sum','mean'])
        for row in rows:
            for state,arms in row['states'].items():
                for arm,meters in arms.items():
                    for key,m in meters.items():
                        csvout.writerow([row['raw'],row['holdout'],state,arm,key,m['count'],m['sum'],mean(m) if m['count'] else None])
    table=''
    for name in ('parent','global','world','colour_oracle'):
        a=score('all','keep',name);b=score('all','delete',name)
        c=score('holdout','keep',name);d=score('holdout','delete',name)
        table+=f'<tr><th>{LABELS[name]}</th><td>{a:.6f}</td><td>{b:.6f}</td><td>{c:.6f}</td><td>{d:.6f}</td></tr>'
    old=score('all','keep','parent','seam');new=score('all','keep',trial,'seam')
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Scene008 道路颜色校正</title>
<style>
body{{background:#f4f7fa;color:#172938;font:16px/1.6 system-ui,sans-serif;margin:0}}main{{max-width:1110px;margin:auto;padding:24px}}
section{{background:white;border:1px solid #d7e2e9;border-radius:11px;margin:16px 0;padding:18px;overflow:auto}}
svg,img{{display:block;width:100%}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}th,td{{text-align:left;padding:6px 9px;border-bottom:1px solid #dbe5ec}}
a{{color:#066371}}
</style><main><h1>Scene008 · 道路颜色校正</h1>
<p>按拟合帧选出的候选为 <strong>{LABELS[trial]}</strong>。逐帧及留出检查后，采用 <strong>{LABELS[accepted]}</strong>。全部帧的 a6/PVD 可见权重、累计透明度和深度逐像素保持一致。</p>
<section><h2>组件与数据流</h2>{ARCH}<p>将全局颜色和 6×5 的世界坐标网格通过原生 αT 前向累积。用无遮挡道路 RGB 和跨时刻显露道路观测做有界稳健求解，颜色每通道调整至多 0.09 sRGB。保护区颜色差作为正则。两种编辑状态共用世界颜色。</p>
<p>原始 49 帧中的 {len(held)} 帧完全不参与本次颜色拟合。两种道路参考均来自同场景。无遮挡可见道路拟合排除已有阴影代理，评价保留完整固定域。</p></section>
<section><h2>固定道路 RGB MAE</h2><table><tr><th>方法</th><th>无编辑 49 帧</th><th>删车 49 帧</th><th>无编辑留出帧</th><th>删车留出帧</th></tr>{table}</table>
<p>逐像素颜色上界允许各像素独立调整，作为固定支撑和颜色幅度下的诊断。无编辑的道路覆盖边界带 RGB MAE 在候选中为 {old:.6f} → {new:.6f}。</p></section>
<section><h2>街景试看片</h2><p>左为原道路，右为训练帧选出的候选。上排无编辑，下排删 a37。</p>
<button data-raw="89">89</button><button data-raw="105">105</button><button data-raw="113">113</button><button data-raw="130">130</button><button data-raw="137">137</button>
<img id="frame" src="frames/113.jpg" alt="raw113 四格配对"></section>
<section><h2>逐帧与复现</h2><a href="fixed_metrics.csv">固定域指标</a> · <a href="summary.json">验收结果</a> · <a href="design.json">拟合与留出回执</a> · <a href="coefficients.npz">世界坐标颜色系数</a></section>
<script>
const frame=document.getElementById('frame');
document.querySelectorAll('button[data-raw]').forEach(b=>b.addEventListener('click',()=>{{const raw=b.dataset.raw.padStart(3,'0');frame.src='frames/'+raw+'.jpg';frame.alt='raw'+raw+' 四格配对';}}));
</script></main></html>'''
    (output/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(trial=trial,accepted=accepted,guard=guard,
                          table={scope:{state:{a:score(scope,state,a) for a in ('parent','global','world','colour_oracle')}
                                         for state in ('keep','delete')} for scope in ('all','holdout')}),
                     indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
