"""r36同源条件前后对照；显式区分条件与DriveEditor生成。"""
from pathlib import Path
import html,json,shutil
import numpy as np
from PIL import Image
from build_role_condition_review import encode
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929');O=T/'r36';DEST=O/'review'
def read(p):return json.loads(p.read_text())
def esc(v):return html.escape(str(v))


def main():
    DEST.mkdir(exist_ok=True)
    comp=read(O/'paired_comparison.json');cases={c['case_id']:c for c in read(T/'r28/prepared.json')['cases']}
    qa={r['case_id']:r for r in read(O/'r36_independent_quality.json')['cases']} if (O/'r36_independent_quality.json').exists() else {}
    cards=[];manifest=[]
    for row in comp['cases']:
        cid=row['case_id'];c=cases[cid];base=T/row['before_run'];dest=DEST/cid;dest.mkdir(exist_ok=True)
        clips=[('Y','真实Y：仅用于质检',Path(c['source_Y_quality_only']),'{i:05}.png'),
               ('input','实际输入：最终H已擦除',O/'observed'/cid/'rgb','{i:05}.png'),
               ('before','原条件F：灰色是未知',base/'state'/cid,'{i:05}_projected_rgb.jpg'),
               ('after','反证检查后F：仍是条件图',O/'state'/cid,'{i:05}_projected_rgb.jpg')]
        media='<div class="videos">';videos=[]
        for key,name,root,pattern in clips:
            encode(dest/f'{key}.mp4',(np.asarray(Image.open(root/pattern.format(i=i)).convert('RGB')) for i in range(30)))
            videos.append(f'{cid}/{key}.mp4')
            media+=f'<div><b>{name}</b><video src="{cid}/{key}.mp4" controls muted preload="none" playsinline></video></div>'
        media+='</div>'
        for name in sorted(O.glob(f'{cid}_slot*.jpg')):
            shutil.copy2(name,dest/name.name)
            media+=f'<details><summary>{esc(name.stem)} · Y / 合法X与可见SAM / 单身份F</summary><a href="{cid}/{name.name}"><img src="{cid}/{name.name}" loading="lazy"></a></details>'
        own=qa.get(cid,{});grade=own.get('condition_assistant_grade',own.get('assistant_grade','待独立复核'))
        reason=own.get('reason',own.get('decision','技术指标不自动等于质量通过。'))
        metrics=[]
        for label,k in [('洞内O精度','rendered_O_precision_inside_H'),('被遮B覆盖','hidden_B_coverage_by_O'),('洞内未知U','unknown_H_fraction'),('洞内真实背景证据N','N_H_fraction')]:
            metrics.append(f'<tr><td>{label}</td><td>{row["before"][k]:.2%}</td><td>{row["after"][k]:.2%}</td></tr>')
        cards.append(f'<article id="{cid}" data-case="{cid}"><h2>{cid} · {esc(c["scene"])} · 条件AI评分 {grade}</h2><p>{esc(reason)}</p><table><tr><th>相对Y-SAM的参考指标</th><th>原条件</th><th>反证检查后</th></tr>{"".join(metrics)}</table>{media}<button class="play">同步播放／暂停</button><input class="seek" type="range" min="0" max="29" value="0"><label>人工评分 <select class="score"><option value="">未评</option><option>0</option><option>1</option><option>2</option></select></label><input class="note" placeholder="逐帧观察／备注"></article>')
        manifest.append({'case_id':cid,'scene':c['scene'],'videos':videos,'human_verdict':None,'condition_assistant_grade':grade})
    for name in ['paired_comparison.json','r36_independent_quality.json','run.json','actor_identity_evaluation.json']:
        if (O/name).exists():shutil.copy2(O/name,DEST/name)
    old=(T/'r34/review/index.html').read_text();js=old.split('<script>')[1].split('</script>')[0].replace('v77-r34-human-v1','v77-r36-human-v1').replace('v77_r34_human_review.json','v77_r36_human_review.json')
    css=old.split('<style>')[1].split('</style>')[0]+'.videos{grid-template-columns:repeat(4,1fr)}@media(max-width:1000px){.videos{grid-template-columns:repeat(2,1fr)}}'
    diagram='<div class="flow"><span>最终H遮后30帧<br>合法RGB与SAM</span> → <span>逐身份代理点<br>跨帧重新投影</span> → <span>可见轮廓外的反证<br>隐藏／遮挡处不判断</span> → <span>保留一致点<br>O / F / Q，未知仍为U</span> ⇢ <span>后续条件训练<br>尚未进行</span></div>'
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v77 r36 · 可见证据约束条件</title><style>{css}</style><h1>r36：先修条件边缘，再判断条件是否值得训练</h1>
    <p class="warn">本页4列是同一真实片段的Y、带洞输入、两版投影条件。后两列不是DriveEditor生成视频。本轮训练0步，尚未获得新的真实DELETE收益。所有人工分数留空。</p>{diagram}
    <p>Q046主车身份稳定，但邻车边缘投影到道路。这里对同一批代理点做一次固定检查：若某点在未被洞遮住、也未被其他保留对象遮住的输入帧中，落到本实例SAM轮廓2像素容差外，就把该点作为不一致证据移除。洞中未知不当空背景，不读隐藏Y，不新增表面。</p>
    <p>两例洞内O精度都上升，旧覆盖分别保留99.58%／92.62%；N没有改变。独立质检给Q060 2分、Q046仍1分。Q046两邻车精度从26.54%／57.75%升到76.46%／66.39%，仍有边缘误差；主B投影删去的2600点中，2397点原本落在近似真实B标签上。数值门通过未带来两例共同质量通过，因此不把这个过滤器升为全局默认，也不继续调阈值。它还不是补景收益。</p>
    <p>这一控制借鉴<a href="https://homes.cs.washington.edu/~seitz/papers/kutu-ijcv00.pdf">Space Carving论文</a>的可见观测一致性思路；这里只检查冻结框面上的动态实例点，不是该算法的完整实现，也没有它在理想观测下的保证。</p>
    <p><a href="../v77-target-protected-r34/index.html">r31–r35数据入口与失败对照</a> · <a href="../v77-target-protected-r28/index.html">保留的旧条件与87例输入</a> · <a href="paired_comparison.json">固定前后指标</a> · <a href="r36_independent_quality.json">独立质量复核</a></p><button id="export">导出人工评分</button>{''.join(cards)}<script>{js}</script></html>'''
    (DEST/'index.html').write_text(page)
    (DEST/'manifest.json').write_text(json.dumps({'cases':manifest,'training_steps':0,'no_generated_output':True,'human_verdict':None},ensure_ascii=False,indent=2)+'\n')
    print('CONSISTENCY_REVIEW',len(manifest),sum(len(c['videos']) for c in manifest))


if __name__=='__main__':main()
