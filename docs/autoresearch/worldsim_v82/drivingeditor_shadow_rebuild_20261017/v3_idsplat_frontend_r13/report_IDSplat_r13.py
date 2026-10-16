from pathlib import Path
import json,zipfile,shutil,html,statistics
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W=Path(__file__).resolve().parent;ROOT=W.parents[1];O=ROOT/'outputs/shadow_rebuild_v3/idsplat_frontend_r13';O.mkdir(parents=True,exist_ok=True);I=W/'pointset_review_r13';I.mkdir(exist_ok=True)
with zipfile.ZipFile(W/'pointset_review_r13.zip') as z:z.extractall(I)
for name in ['protocol.json','frontend_complete.json','pointset_QUERY_gate.json','module_coverage_diagnostic.json']:
    shutil.copyfile(I/name,O/name)
shutil.copyfile(W/'dinov3_r13_result.json',O/'dinov3_validation.json');shutil.copyfile(W/'USER_STOP_BC_receipt.json',O/'BC_stop_receipt.json')
gate=json.loads((O/'pointset_QUERY_gate.json').read_text());stages=json.loads((O/'module_coverage_diagnostic.json').read_text());dino=json.loads((O/'dinov3_validation.json').read_text());targets=[5,6,12,18,25,26,37];contact=[]
plt.rcParams.update({'font.size':9,'axes.titlesize':11,'figure.facecolor':'white'})
for a in targets:
    n=np.load(I/f'a{a:02d}_review.npz');rgbpath=next(I.glob(f'a{a:02d}_raw*_RGB.png'));image=Image.open(rgbpath).convert('RGB');overlay=np.asarray(image).copy();mask=n['sample_mask'];edge=mask.copy();edge[1:]|=mask[:-1];edge[:-1]|=mask[1:];edge[:,1:]|=mask[:,:-1];edge[:,:-1]|=mask[:,1:];edge&=~mask;overlay[edge]=[255,180,0]
    fig,axes=plt.subplots(2,3,figsize=(13,6),layout='constrained');fig.suptitle(f'Actor {a} | {rgbpath.stem} | measured seeds only, fixed GT trajectory',fontsize=14)
    axes[0,0].imshow(overlay);axes[0,0].set_title('Real RGB + common predicted body outline');axes[0,0].axis('off')
    frame=int(n['sample_frame']);take=n['candidate_frame']==frame;pixel=n['candidate_pixel'][take];accepted=n['accepted'][take];reason=n['reason'][take];inside=(pixel[:,0]>=0)&(pixel[:,0]<480)&(pixel[:,1]>=0)&(pixel[:,1]<320)
    axes[1,0].imshow(image);axes[1,0].scatter(pixel[inside&~accepted,0],pixel[inside&~accepted,1],s=7,c='#db4646',label='uncertain OBB returns');axes[1,0].scatter(pixel[inside&accepted,0],pixel[inside&accepted,1],s=7,c='#00e070',label='accepted true LiDAR');axes[1,0].set_xlim(0,480);axes[1,0].set_ylim(320,0);axes[1,0].axis('off');axes[1,0].set_title('Same-frame actual LiDAR projection');axes[1,0].legend(loc='lower left',fontsize=7)
    for row,(ix,iy,label) in enumerate([(0,2,'local x / z (m)'),(1,2,'local y / z (m)')]):
        allpoints=n['A_xyz'];lo=np.quantile(allpoints[:,[ix,iy]],.001,axis=0)-.15;hi=np.quantile(allpoints[:,[ix,iy]],.999,axis=0)+.15
        for col,name in [(1,'A'),(2,'D')]:
            ax=axes[row,col];p=n[name+'_xyz'];ax.scatter(p[:,ix],p[:,iy],s=.8,c='#758ba7' if name=='A' else '#079563',rasterized=True)
            q=n['query_body'];l=n['query_lowerbody'];ax.scatter(q[:,ix],q[:,iy],s=4,c='#ed8d0a',alpha=.5,label='heldout body proxy');ax.scatter(l[:,ix],l[:,iy],s=13,c='#bb2793',marker='+',label='lowerbody proxy')
            ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal',adjustable='box');ax.set_xlabel(label);ax.grid(alpha=.15);ax.set_title(('A: box initialization' if name=='A' else 'Adapted IDSplat frontend')+f' ({len(p)} seeds)')
            if row==1:ax.legend(loc='upper right',fontsize=6)
    fig.savefig(O/f'a{a:02d}_pointset.png',dpi=120);plt.close(fig)
    thumb=Image.open(O/f'a{a:02d}_pointset.png').convert('RGB');thumb.thumbnail((1200,555));contact.append(thumb.copy())
canvas=Image.new('RGB',(1200,sum(x.height for x in contact)),(255,255,255));y=0
for im in contact:canvas.paste(im,(0,y));y+=im.height
canvas.save(O/'all7_pointset_contact.jpg',quality=92)
summ=[]
for a in targets:
    values=[r for r in dino['records'] if r['actor']==a and r['status']=='INDEPENDENT_REGISTRATION_MEASURED']
    def med(key):return float(np.median([x[key] for x in values])) if values else None
    summ.append(dict(actor=a,pairs=len(values),planned_pairs=8,insufficient_matches=8-len(values),fitness_passed=sum(x['fitness_passed_official_half'] for x in values),target_coverage_median=med('target_coverage10cm'),rotation_vs_GT_deg_median=med('rotation_vs_GT_deg'),center_vs_GT_m_median=med('vehicle_center_vs_GT_m'),surface_vs_GT_m_median=med('surface_transfer_vs_GT_median_m'),world_residual_median=med('matched_world_residual_median_m'),GTlocal_residual_median=med('matched_GT_local_residual_median_m')))
(O/'dinov3_summary.json').write_text(json.dumps(summ,indent=2),encoding='utf8')
architecture='''<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="290" viewBox="0 0 1100 290"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#52657c"/></marker></defs><style>text{font:16px sans-serif;fill:#172d45}.box{fill:#f1f6fd;stroke:#7c96b6;rx:10}.arrow{stroke:#52657c;stroke-width:2;fill:none;marker-end:url(#a)}</style><rect class="box" x="15" y="100" width="180" height="80"/><text x="30" y="130">真实 RGB / LiDAR</text><text x="30" y="157">共同 mask / 固定 GT</text><path class="arrow" d="M195 125H235V55H275"/><path class="arrow" d="M195 140H275"/><path class="arrow" d="M195 165H235V245H275"/><rect class="box" x="275" y="20" width="210" height="70"/><text x="295" y="49">A：框内点分给车辆</text><text x="295" y="74">保留原点集作对照</text><rect class="box" x="275" y="110" width="210" height="70"/><text x="292" y="138">mask 提升＋深度检查</text><text x="292" y="163">DBSCAN＋细部保护</text><path class="arrow" d="M485 145H525"/><rect class="box" x="525" y="110" width="190" height="70"/><text x="542" y="138">车辆局部实测点集</text><text x="542" y="163">ID / 可信度 / 来源</text><rect class="box" x="275" y="210" width="440" height="65"/><text x="292" y="237">DINOv3＋官方 RANSAC → 对照 GT</text><text x="292" y="260">独立验证；不改点集、不覆盖轨迹</text><path class="arrow" d="M485 55H755V125H790"/><path class="arrow" d="M715 145H790"/><rect class="box" x="790" y="105" width="285" height="80"/><text x="810" y="134">道路误入＋车体 / 下部覆盖</text><text x="810" y="160">验收通过 → 同 infra 训练</text><text x="805" y="221" fill="#b13636">本轮：6 车保体通过，a12 未通过</text><text x="805" y="247">尚未启动任何新场景训练</text></svg>'''
(O/'architecture.svg').write_text(architecture,encoding='utf8')
def pct(v):return '无分母' if v is None else f'{100*v:.1f}%'
def num(v):return '—' if v is None else f'{v:.3f}'
rows=''
for x,y,z in zip(gate['models'][0]['actors'],gate['models'][1]['actors'],gate['screen']):
    a=x['actor'];rows+=f"<tr class='{'' if z['body_passed'] and z['lowerbody_passed'] else 'bad'}'><td><a href='#a{a}'>{a}</a></td><td>{x['Actor_Gaussians']} → {y['Actor_Gaussians']}</td><td>{x['road_only_seed_proxy']} → {y['road_only_seed_proxy']}</td><td>{pct(z['road_reduction']) if z['road_reliable_denominator'] else '样本不足 / 无正道路证据'}</td><td>{pct(x['heldout_body_coverage8cm'])} → {pct(y['heldout_body_coverage8cm'])}</td><td>{pct(x['heldout_lowerbody_coverage8cm'])} → {pct(y['heldout_lowerbody_coverage8cm'])}</td><td>{'通过' if z['body_passed'] and z['lowerbody_passed'] else '未通过'}</td></tr>"
dr=''.join(f"<tr><td>{x['actor']}</td><td>{x['fitness_passed']}/{x['pairs']}</td><td>{pct(x['target_coverage_median'])}</td><td>{num(x['rotation_vs_GT_deg_median'])}°</td><td>{num(x['center_vs_GT_m_median'])} m</td><td>{num(x['surface_vs_GT_m_median'])} m</td><td>{num(x['world_residual_median'])} / {num(x['GTlocal_residual_median'])} m</td></tr>" for x in summ)
sr=''
for x in stages['actors']:
    if x['actor']!=12:continue
    names=['原框全部 BUILD','相机深度可见','共同 mask＋深度','聚类＋细部保护','道路冲突剔入待定池']
    sr=''.join(f"<tr><td>{name}</td><td>{v['measured_points']}</td><td>{pct(v['body_coverage'])}</td><td>{pct(v['lowerbody_coverage'])}</td></tr>" for name,v in zip(names,x['stages']))
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>⑥ IDSplat 前端点集对照 R13</title><style>body{{font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;background:#f3f6fa;color:#213146;max-width:1280px;margin:30px auto;padding:0 20px}}section,.card{{background:white;padding:22px;border:1px solid #dce4ee;border-radius:12px;margin:18px 0}}h1{{font-size:30px}}h2{{font-size:23px}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #dae1eb;padding:9px;text-align:left}}th{{background:#eef3fa}}.bad{{background:#fff0ef}}a{{color:#1a59a4}}.notice{{border-left:5px solid #b83d32;background:#fff5f2}}.muted{{color:#637084}}details{{margin:12px 0}}code{{background:#eef2f8;padding:2px 5px}}.scroll{{overflow:auto}}</style>
<h1>⑥ IDSplat 前端：先看点集，不先训练</h1><p class="muted">2026-10-17 · 同原始测量 / 冻结共同 mask / 固定 Waymo GT / 同 infra 后端 · 本代理检查，无 subagent</p>
<div class="card notice"><strong>结果：道路污染代理明显减少，6 辆车保体达标；a12 仍不达标，因此整体点集门槛未通过，新增训练 0 步。</strong><p>这说明 IDSplat 前端策略在本场景有实际价值，不能判成“完全无效”；也不能把 6 辆车的改善当作 7 辆车均可安全编辑。a12 的主要缺口发生在共同 mask 提升阶段，聚类又损失少量细部。未放宽门槛、未据 QUERY 重调阈值。</p></div>
<p><a href="../ownership_controls_r12_sam3/A_quick_report/report.html">A 组及旧 18k 视觉对照</a> · <a href="../controls49_comparison.html">既有 49 帧动态 / 编辑视频</a> · <a href="pointset_QUERY_gate.json">本轮逐车原始指标</a> · <a href="module_coverage_diagnostic.json">模块覆盖诊断</a> · <a href="protocol.json">运行前固定协议</a></p>
<section><h2>组件与数据流</h2><img src="architecture.svg" alt="原始测量经过框或mask深度聚类得到点集，对照验收；DINO独立核验GT"/><p>共同输入为既有 <code>shared_sam3_masks_r2</code>：a37 全 49 帧使用 SAM3，其他车辆含明确记录的 DEVA / 冲突回退；不能声称全部换成原生 SAM3。没有重跑分割或改轨迹。框仅作候选范围，真实相机 body mask 和 LiDAR 深度决定可接纳点。</p></section>
<section><h2>7 辆车点集验收</h2><div class="scroll"><table><tr><th>Actor</th><th>种子点 A → 本轮</th><th>道路误入代理计数</th><th>道路代理降幅</th><th>车体 8cm 覆盖</th><th>下部 8cm 覆盖</th><th>保体门槛</th></tr>{rows}</table></div><p>固定门槛：每车车体与下部覆盖均保留 A 的 ≥90%；A 道路代理计数 ≥30 的车辆，要求减少 ≥50%。a6 无正道路证据；a12/a18 道路分母过小，不据此宣布去路成功。a12 的 204 个 body 查询点、38 个下部点均来自训练帧保留返回；三张测试帧 body 分母为 0。</p><p class="muted">这些是实际保留 LiDAR 的几何覆盖和正道路代理，不是真实车影比例、手工三维归属真值或删除成功率。下部为可见 mask 框底部 25% 的诊断域，不能当精确轮胎标注。QUERY 先前已用于开发，本轮没有新盲测。</p></section>
<section><h2>a12 在哪一步损失</h2><table><tr><th>步骤（体素前实际返回）</th><th>返回数</th><th>body 覆盖</th><th>下部覆盖</th></tr>{sr}</table><p>相机深度检查后覆盖仍为 95.6% / 89.5%；要求同帧共同 mask 后降到 78.4% / 65.8%；聚类与细部支持筛选后为 76.0% / 60.5%。本轮 a12 没有正道路冲突返回，因此这次保体失败不能归因于“按道路证据裁掉轮胎”。但 mask 缺失、投影偏差或查询标签错误仍需更细的真实证据区分。</p></section>
<section><h2>真实点集与投影证据</h2><p>绿色为接纳的同帧真实 LiDAR，红色为框内待定返回；红色不等于道路。右侧点图橙色 / 紫色来自保留 body / 下部查询代理。所有图均使用实际测量与固定 GT；点图展示种子，不能替代训练后编辑视频。</p>
{''.join(f'<details id="a{a}" {"open" if a in [5,12,37] else ""}><summary>Actor {a}：RGB、共同轮廓、实际点投影、局部点集对照</summary><a href="a{a:02d}_pointset.png"><img src="a{a:02d}_pointset.png" alt="Actor {a}实际点集对照" loading="lazy"/></a></details>' for a in targets)}
</section><section><h2>DINOv3＋官方 RANSAC：独立验证，不替换 GT</h2><p>实际对 46 张训练图像提取 ViT-B/16 第 7 层特征，移除 CLS＋4 register token；原生 20×30×768 特征双线性采样。7 辆车各取 8 个跨帧配对，共 56 对，调用官方余弦匹配＋三点 Umeyama / 100000 次 RANSAC。没有对 QUERY 提特征，没有 GTSAM 或估计轨迹覆盖 GT。每帧最多 512 个真实表面样本，规范帧按采样后点数选；这是适配验证，不是完整原版无标注轨迹复现。</p><div class="scroll"><table><tr><th>Actor</th><th>目标覆盖≥50%的配对</th><th>10cm 覆盖中位数</th><th>旋转对 GT 中位差</th><th>车辆中心对 GT 中位差</th><th>表面对 GT 中位差</th><th>世界 / GT 局部对应残差</th></tr>{dr}</table></div><p>高配准覆盖不保证运动正确；低纹理和重复特征可能产生错误匹配，必须同时看 GT 偏差。停车、遮挡、冲突继续待定；没有用绑定 Actor 的高斯随车运动来证明归属。<a href="dinov3_validation.json">56 对完整真实结果</a>。</p></section>
<section><h2>迁移边界与本轮停止点</h2><p>采用官方 IDSplat 的 mask 提升、DBSCAN 和真实 DINO / RANSAC 机制；适配保留多簇、边界实测细部、停车、待定来源，禁随机框补种。保持其他 36 个 Actor、静态种子属性、RGB / 监督文件不变；仅修正邻居索引偏移。GT 轨迹前后 SHA256 完全相同。未搬入 SplatAD 物体管理、Nerfstudio 相机 y/z 翻转或 LiDAR 强度 / raydrop / LOS 专用损失。</p><p>B/C 按用户要求停止：B 已完成 3000 步、在视频导出时停下；1500 / 3000 检查点完整保留，C 尚未开始。<a href="BC_stop_receipt.json">退出及检查点记录</a>。本轮不改变透明度、不训练、不输出虚构的新编辑结果，①②④也未开始。</p><p>来源：<a href="https://arxiv.org/html/2511.19235v1#S3.SS2">IDSplat §3.2–3.3</a>；官方代码固定 commit <code>e3e08fcd794d18ab61fa4550235a6bbf41643726</code>。置信度为证据分数，并非已校准概率。</p><p id="closeout">等待根代理视觉复核和保存退出交接。</p></section></html>'''
(O/'report.html').write_text(page,encoding='utf8')
print(json.dumps({'report':str(O/'report.html'),'pointset_numerical_passed':gate['numerical_passed'],'dino_summary':summ},ensure_ascii=False))
