"""CPU材料就绪与方法结果分开；网页仅展示本次真实存在的条件预览。"""
from common import *
import html, argparse
from collections import Counter
import numpy as np, cv2
from PIL import Image

def main(conditions_only=False):
    plan=read(O/'manifest.json');summary=read(O/'condition_summary.json')
    byid={c['case_id']:c for c in summary['cases']};out=O/'review';out.mkdir(exist_ok=True)
    cards=[];rows=[]
    for c in plan['cases']:
        cid=c['case_id'];d=byid[cid];stats=d['statistics'];area=sum(s['H'] for s in stats)
        known={k:sum(s[k+'_H'] for s in stats)/max(1,area) for k in ['O','N','U']}
        rows.append({'case_id':cid,'split':c['split'],'scene':c['scene'],'type':c['type'],
            'hole_condition_fraction':known,'background_returns':sum(s['visible_background_points'] for s in d['lidar']),
            'human_verdict':None,'model_benefit':None})
        dest=out/'assets'/cid;dest.mkdir(parents=True,exist_ok=True)
        holes=None
        if not conditions_only or not all((dest/f'{f:02}.jpg').exists() for f in [0,5,9]):
            holes=images(c,'hole')>0
        for f in [0,5,9]:
            if conditions_only and (dest/f'{f:02}.jpg').exists():continue
            picture=np.asarray(Image.open(O/'conditions'/cid/f'review_{f:02}.jpg').convert('RGB')).copy()
            contours,_=cv2.findContours(holes[f].astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            contours=[v+np.array([1024,0]) for v in contours]
            cv2.drawContours(picture,contours,-1,(255,195,45),2)
            Image.fromarray(picture).save(dest/f'{f:02}.jpg',quality=92)
        figures=[]
        for f in [0,5,9]:
            s=stats[f];denom=max(1,s['H'])
            caption=f"f{f:02} · 当帧洞内 O {s['O_H']/denom:.2%} / N {s['N_H']/denom:.3%} / U {s['U_H']/denom:.2%}"
            figures.append(f'<figure><figcaption>{caption} · 点击看原尺寸</figcaption><a href="assets/{cid}/{f:02}.jpg" target="_blank" rel="noopener"><img loading="lazy" src="assets/{cid}/{f:02}.jpg"></a></figure>')
        diagnostic=''
        if cid=='A022' and (out/'a022_condition_audit/index.html').is_file():
            diagnostic='<p class="note"><a href="a022_condition_audit/index.html">A022 unknown 逐层诊断与真实显露对照</a>：首帧洞内全 U 来自当前两份稀疏 LiDAR 的取样规则，后续 RGB 已有真实背景显露，却未进入 N 构建。不能解释为整段视频没有证据。</p>'
        cards.append(f'''<article id="{cid}"><h2>{cid} · {html.escape(c['scene'])} · {c['split']}</h2>{diagnostic}
        <p>{html.escape(c['type'])}｜10帧累计洞内 O {known['O']:.1%} / N {known['N']:.2%} / U {known['U']:.1%}。
        O 是 GT 车辆框的保守内核先验（Q=0.5），不是精确分割；N 是排除实体后确实有观测支持的背景返回（道路、护栏等）；没有依据的区域保持灰色 U。</p>
        <p>左：实际遮洞输入；右：绿 O / 蓝 N / 灰 U，黄线为洞边界。此处不是 DriveEditor 的生成结果。</p>
        <div class="frames">{''.join(figures)}</div></article>''')
    counts=Counter(c['split'] for c in plan['cases'])
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>v77 r45 · CPU 准备页：O/N/U/Q 先验</title><style>
    body{font:16px/1.65 system-ui;background:#121922;color:#e4ecf5;margin:0 auto;padding:28px;max-width:1400px}h1,h2{line-height:1.3}a{color:#8bc5ff}article{padding:20px;background:#1c2633;margin:22px 0;border-radius:10px}img{width:100%;display:block}.frames{display:grid;gap:12px;grid-template-columns:1fr}figure{margin:0}svg{max-width:100%;height:auto}.note{background:#3e341c;padding:14px}nav{display:flex;gap:12px;flex-wrap:wrap}small{color:#b9c6d5}</style>
    <h1>CPU 准备页：当前 O/N/U/Q 空间先验</h1><p class="note">此页展示实际用于该 run 的准备条件。绿色 O、蓝色 N、灰色 U 是输入先验；模型生成效果另见对照页。</p>
    <svg viewBox="0 0 1260 135" role="img" aria-label="已有几何证据到ON UQ、小Adapter、冻结DriveEditor和固定DELETE对照">
    <defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#91b7e7"/></marker></defs>
    <g fill="#253e5c" stroke="#74a3d5"><rect x="5" y="25" width="205" height="76" rx="8"/><rect x="258" y="25" width="188" height="76" rx="8"/><rect x="492" y="25" width="178" height="76" rx="8"/><rect x="716" y="25" width="236" height="76" rx="8"/><rect x="998" y="25" width="250" height="76" rx="8"/></g>
    <g stroke="#91b7e7" marker-end="url(#arrow)"><path d="M211,63 H250"/><path d="M446,63 H484"/><path d="M670,63 H708"/><path d="M952,63 H990"/></g>
    <g fill="white" text-anchor="middle" font-size="19"><text x="108" y="57">相机／车辆框／LiDAR</text><text x="108" y="83">已有几何证据</text><text x="352" y="70">O / N / U / Q</text><text x="581" y="57">小 Adapter</text><text x="581" y="83">仅训练此分支</text><text x="834" y="57">DriveEditor 主干冻结</text><text x="834" y="83">保留原始权重</text><text x="1123" y="57">固定真实 DELETE</text><text x="1123" y="83">关分支 / 全未知 / 有条件</text></g></svg>
    <p>第一轮固定 160 步，原始 diffusion loss，seed 6201；推理 seed 42、25 steps。只训练 157,888 参数。
    全未知是同一支路的输入消融；关分支恢复原模型。以幻觉车、后车与邻车保持来判断实际收益，不能只看训练 loss。</p>
    <p>GT 相机和车辆框是明确使用的额外 POC 输入。车辆框边缘设为未知，背景不等于“框外”；N 不凭空铺满路面。
    真实 DELETE 没有去车 GT；全部属于已曝光开发评测。输入 RGB/隐藏 Y 不进入条件分支；Y 只供合成训练损失。</p>'''
    page+='''<p class="note">当前 occ 先验是投影到每张图像的二维 O/N/U/Q 条件，不是密集三维占据场。
    绿 O：需要保留的车辆 GT 包络保守内核，已排除待删目标，不能当完整车体分割；蓝 N：有实测 LiDAR 观测依据的背景点，点间没有依据的路面仍是 U；灰 U：未知，不代表空地或确定背景。
    黄线是修复洞 H，H 内也可能存在需要恢复的保护车 B，因此可以有绿色。Q 是置信度通道，本页颜色主要展示 O/N/U。</p>'''
    if (O/'time_fix_audit.json').exists():
        page+='<h2>r46 时间边界修复检查</h2><p>关键帧直接取关联 sample 标注，非关键帧用官方 SDK 插值。A013、A007 原首帧整张灰色是漏标，已恢复；A022 首帧洞内全 U 是当前稀疏条件规则的结果，不代表其他帧没有背景显露。下表为当前 f00 洞内像素数，不是模型输出。</p><nav><a href="#A013">先看 A013</a><a href="#A007">先看 A007</a><a href="#A022">先看 A022</a></nav><table><tr><th>case</th><th>H 像素</th><th>O</th><th>N</th><th>U</th></tr>'
        for cid in ['A013','A007','A022']:
            s=byid[cid]['statistics'][0]
            page+=f'<tr><td><a href="#{cid}">{cid}</a></td><td>{s["H"]}</td><td>{s["O_H"]}</td><td>{s["N_H"]}</td><td>{s["U_H"]}</td></tr>'
        page+='</table>'
    if (O/'results_summary.json').exists():page+='<p><a href="index.html">返回模型三组视频对照</a></p>'
    page+=f'<p>固定样本：{counts["train"]} 训练 / {counts["validation"]} 合成评测 / {counts["real_DEV"]} 真实评测。两条密集车列训练例共享同一场景和位置，H 不同；不把它们计作独立场景。</p>'
    page+='<nav>'+''.join(f'<a href="#{c["case_id"]}">{c["case_id"]}</a>' for c in plan['cases'])+'</nav>'
    page+=''.join(cards)+'<small>WS-V77-TARGET-PROTECTED-20260929 / r45 · human verdict 未填写</small></html>'
    target=out/('conditions.html' if conditions_only else 'index.html')
    target.write_text(page.replace('r45',O.name))
    if not conditions_only:
        dump(out/'condition_inventory.json',{'cases':rows,'model_quality_claim':False})
        dump(O/'condition_inventory.json',{'cases':rows,'model_quality_claim':False})
    print('REVIEW_READY',len(cards),len(cards)*3)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--conditions-only',action='store_true');a=p.parse_args()
    main(a.conditions_only)
