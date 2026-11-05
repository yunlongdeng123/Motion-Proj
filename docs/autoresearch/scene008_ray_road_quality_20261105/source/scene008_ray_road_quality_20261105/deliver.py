"""Compare coarse and fine road radiance and generate the 49-view delivery."""
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np

RUN=Path(__file__).resolve().parent.parent
COARSE=RUN.parent/'scene008_ray_road_colour_20261105'
FINE=RUN.parent/'scene008_ray_road_colour_fine_20261105'
PREVIOUS=RUN.parent/'scene008_a6_residual_replay_20261105'
sys.path[:0]=[str(COARSE/'source'),str(PREVIOUS/'source')]
from report import image_grid
from report_colour import ARCH

LABEL={'parent':'原道路 1K','global':'全局偏移','coarse':'6×5 网格','fine':'11×9 网格'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def mean(m):
    return m['sum']/m['count'] if m['count'] else None


def pooled(items):
    n=sum(v['count'] for v in items)
    return dict(count=n,mean=sum(v['sum'] for v in items)/n if n else None)


def prepare():
    coarse=read(COARSE/'result.json')
    fine=read(FINE/'result.json')
    assert coarse['geometry_parity']==fine['geometry_parity']==0.
    data=coarse['rows']
    assert [r['raw'] for r in data]==list(range(89,138))
    assert coarse['holdout']==fine['holdout']
    parity=0.
    for a,b in zip(data,fine['rows']):
        assert a['raw']==b['raw'] and a['holdout']==b['holdout']
        for s in ('keep','delete'):
            old=a['states'][s]
            high=b['states'][s]
            for key,m in old['world'].items():
                value=high['coarse'][key]
                assert m['count']==value['count']
                if m['count']:
                    parity=max(parity,abs(mean(m)-mean(value)))
            old['coarse']=old.pop('world')
            old['fine']=high['world']
    assert parity<3e-7,parity
    scopes={'all':data,'fit':[r for r in data if not r['holdout']],'holdout':[r for r in data if r['holdout']]}
    stats={}
    for scope,rows in scopes.items():
        stats[scope]={}
        for s in ('keep','delete'):
            stats[scope][s]={}
            for arm in rows[0]['states'][s]:
                stats[scope][s][arm]={k:pooled([r['states'][s][arm][k] for r in rows])
                                          for k in rows[0]['states'][s][arm]}
    def score(scope,state,arm,key):
        return stats[scope][state][arm][key]['mean']
    rules={}
    for name,ref in [('coarse','parent'),('fine','coarse')]:
        flags=[]
        for scope in ('fit','holdout','all'):
            for s in ('keep','delete'):
                for key in ('road','seam'):
                    if score(scope,s,name,key)>score(scope,s,ref,key)+1e-7:
                        flags.append((scope,s,key))
        for state in ('keep','delete'):
            if score('all',state,name,'full')>score('all',state,ref,'full')+1e-7:
                flags.append(('all',state,'full'))
        frame_violations=[]
        for row in data:
            for state in ('keep','delete'):
                m=row['states'][state][name]
                d=row['states'][state]['parent']
                issues=[key for key,limit in [('a6_delta',.0005),('a37_delta',.0005),('protected_delta',.001)]
                        if (mean(m[key]) or 0)>limit]
                if mean(m['road'])>mean(d['road'])+1e-7:
                    issues.append('road_vs_original')
                if issues:
                    frame_violations.append(dict(raw=row['raw'],state=state,metrics=issues))
        rules[name]=dict(aggregate=flags,frames=frame_violations)
    choice='parent'
    for name in ('coarse','fine'):
        if not any(rules[name].values()):
            choice=name
    dest=RUN/'delivery'
    dest.mkdir(exist_ok=False)
    (dest/'frames').mkdir()
    info=dict(status='PASS',frames=len(data),winner=choice,coarse_interpolation_max_metric_delta=parity,
              held_out=coarse['holdout'],stats=stats,guards=rules,geometry_parity=0.)
    (dest/'summary.json').write_text(json.dumps(info,indent=2,allow_nan=False)+'\n')
    (dest/'rows.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    for source,label in ((COARSE,'coarse'),(FINE,'fine')):
        shutil.copyfile(source/'coefficients.npz',dest/f'{label}_coefficients.npz')
        shutil.copyfile(source/'design.json',dest/f'{label}_design.json')
    with (dest/'fixed_metrics.csv').open('w',encoding='utf-8-sig',newline='') as h:
        out=csv.writer(h,lineterminator='\n')
        out.writerow(['raw','held','state','arm','metric','n','sum','mean'])
        for r in data:
            for s,arms in r['states'].items():
                for arm,m in arms.items():
                    for k,v in m.items():
                        out.writerow([r['raw'],r['holdout'],s,arm,k,v['count'],v['sum'],mean(v)])
    print(json.dumps(dict(winner=choice,rules=rules,coarse_interpolation=parity),indent=2),flush=True)
    return info


def render_all(info):
    import torch
    import calibrate
    calibrate.RUN=RUN
    name=info['winner']
    calibrate.GRID=(11,9) if name=='fine' else (6,5)
    from calibrate import Experiment,RAWS,save,metric
    directory=RUN/'delivery'
    expected=read(directory/'rows.json')
    torch.set_num_threads(1)
    with torch.no_grad():
        e=Experiment()
        if name=='parent':
            colour=e.colour
        else:
            path=(FINE if name=='fine' else COARSE)/'coefficients.npz'
            with np.load(path,allow_pickle=False) as data:
                w=e.basis.new_tensor(data['world'])
            colour=(e.colour+e.basis@w).clamp(0,1)
        scratch=RUN/'delivery_renders'
        scratch.mkdir()
        for raw in RAWS:
            paths=[]
            for state in ('keep','delete'):
                target=e.evidence(raw,state=='delete')
                scene=e.assemble(raw,state=='delete')
                old=e.render(raw,scene)
                e.check_parent(raw,state,old,target)
                e.colourize(scene,colour)
                new=e.render(raw,scene)
                assert all(torch.equal(old[k],new[k]) for k in ('alpha','depth_sum','extra'))
                m=metric((new['rgb']-target['target']).abs().mean(0),target['road'])
                ref=expected[raw-89]['states'][state][name]['road']
                assert m['count']==ref['count'] and abs(m['sum']-ref['sum'])<.002,(raw,state)
                for label,pkg in [('parent',old),('new',new)]:
                    p=scratch/f'{raw:03d}_{state}_{label}.png'
                    save(p,pkg['rgb'])
                    paths.append(p)
            image_grid(paths,['No edit | parent',f'No edit | {name}',
                              'Delete a37 | parent',f'Delete a37 | {name}']).save(directory/'frames'/f'{raw:03d}.jpg',quality=90)
        assert e.trial.exp.unchanged()
    (directory/'receipt.json').write_text(json.dumps(dict(status='PASS',frames=49,winner=name,
         both_edit_states=True,all_geometry_and_contribution_bitwise_equal=True,road_metric_replay=True),indent=2)+'\n')


def html(info):
    dest=RUN/'delivery'
    rows=''
    for arm in ('parent','global','coarse','fine','colour_oracle'):
        cells=[info['stats'][scope][state][arm]['road']['mean']
               for scope in ('all','holdout') for state in ('keep','delete')]
        label=LABEL.get(arm,'逐像素上界诊断')
        rows+=f"<tr><th>{label}</th>"+''.join(f"<td>{x:.6f}</td>" for x in cells)+'</tr>'
    status=[]
    for arm in ('coarse','fine'):
        r=info['guards'][arm]
        status.append(f"{LABEL[arm]} {'通过' if not any(r.values()) else '存在保护或验证失败'}")
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Scene008 道路整体校正</title>
<style>
body{{background:#f3f7fa;color:#172938;font:16px/1.6 system-ui,sans-serif;margin:0}}main{{max-width:1100px;margin:auto;padding:24px}}
section{{background:white;border:1px solid #d6e2e8;border-radius:10px;padding:17px;margin:16px 0;overflow:auto}}
svg,img{{display:block;width:100%}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}
th,td{{text-align:left;padding:6px 9px;border-bottom:1px solid #dde7ec}}input{{width:50%}}a{{color:#086675}}
</style><main><h1>Scene008 · 道路整体校正</h1><p>正式采用 <strong>{LABEL[info['winner']]}</strong>。道路几何、透明度、前后车和 PVD 资产冻结。全部 49 帧的深度、透明度和各层可见贡献逐像素一致。</p>
<section><h2>组件与数据流</h2>{ARCH}<p>使用真实 RGB 和同场景跨时刻道路参考。网格在世界坐标共享，通过原生 alpha*T 的精确颜色响应拟合。细网格由粗网格精确插值起步，再拟合剩余误差。校正范围每色彩通道至多 0.09 sRGB。</p></section>
<section><h2>道路 RGB MAE</h2><table><tr><th>方案</th><th>无编辑 49 帧</th><th>删车 49 帧</th><th>无编辑留出</th><th>删车留出</th></tr>{rows}</table>
<p>{'。'.join(status)}。9 个留出视角未用于本次任何颜色拟合。逐像素上界独立调整各像素，只用于估计固定几何下的颜色余量。</p></section>
<section><h2>49 帧对照</h2><p>左列为原道路，右列为采用版本。上排无编辑，下排删 a37。</p>
<label for="raw">raw<span id="no">113</span></label><input type="range" id="raw" min="89" max="137" value="113"><button data-raw="105">105</button><button data-raw="113">113</button><button data-raw="130">130</button>
<img id="frame" src="frames/113.jpg" alt="raw113 道路四格对照"></section>
<section><h2>逐帧与复现</h2><a href="fixed_metrics.csv">逐帧 CSV</a> · <a href="summary.json">验收检查</a> · <a href="receipt.json">全 49 帧实际重放</a> · <a href="coarse_coefficients.npz">粗网格系数</a> · <a href="fine_coefficients.npz">细网格系数</a></section>
<script>
const raw=document.getElementById('raw'),no=document.getElementById('no'),frame=document.getElementById('frame');
function update(){{const k=raw.value.padStart(3,'0');no.textContent=k;frame.src='frames/'+k+'.jpg';}}
raw.addEventListener('input',update);document.querySelectorAll('button[data-raw]').forEach(b=>b.addEventListener('click',()=>{{raw.value=b.dataset.raw;update();}}));
</script></main></html>'''
    (dest/'index.html').write_text(page,encoding='utf-8')


if __name__=='__main__':
    data=prepare()
    render_all(data)
    html(data)
