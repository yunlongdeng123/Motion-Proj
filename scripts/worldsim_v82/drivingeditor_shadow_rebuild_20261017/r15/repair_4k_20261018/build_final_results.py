"""Append measured final evidence to the existing repair component/contract report."""
from pathlib import Path
import json,shutil,re
import numpy as np
from PIL import Image,ImageDraw,ImageFont
W=Path(__file__).parent;O=W.parents[1]/'outputs/r15_repair_4k_20261018'
I=W.parent/'identity_completion';C=O.parent/'identity_comparison_20261018';M=W/'final_metadata'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
assert read(M/'queue_status.json')['state']=='ALL_REPAIR_GPU_WORK_COMPLETE_AWAITING_REVIEW_AND_DELIVERY'
assert read(M/'train/progress.json')['global_step']==4000
assert read(C/'player_status.json')['models']['repair4k']
shutil.copytree(M,O/'evidence',dirs_exist_ok=True)
geo=read(M/'geometry_comparison.json');progress=read(M/'train/progress.json')
audits=[read(M/'train/ownership'/f'{s:06d}/receipt.json') for s in [1500,2500,3500]]
status=read(O/'status.json');status.update(actual_gpu_smoke='PASSED',new_training_updates=4000,
    evaluated_frames=49,evaluated_identities=5,actual_branch_render_complete=True,
    queue_state=read(M/'queue_status.json')['state'],quality_verdict='INCOMPLETE_A6_FRAGMENT_AND_A37_STATIC_GHOST_REMAIN',human_verdict=None)
(O/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
body=[]
for actor in [6,37,5,26,18]:
    for model in ['old5','r15','repair4k']:
        mass=covered=pixels=0
        for raw in range(89,138):
            fid=f'0008{raw:03d}'
            positive=np.asarray(Image.open(C/'assets/r15'/fid/f'a{actor:02d}_mask.png').convert('L'))>127
            alpha=np.asarray(Image.open(C/'assets'/model/fid/f'a{actor:02d}_visible.png').convert('L'),dtype=float)/255
            mass+=float(alpha[positive].sum());covered+=int((alpha[positive]>.1).sum());pixels+=int(positive.sum())
        body.append(dict(actor=actor,model=model,R15_positive_pixels=pixels,
            visible_alpha_mean=mass/pixels if pixels else None,positive_coverage_alpha_gt_0p1=covered/pixels if pixels else None))
(O/'body_retention.json').write_text(json.dumps(body,indent=2),encoding='utf-8')
rows=''
for actor in [6,37,5,26,18]:
    old,new=[next(x for x in geo[key] if x['actor']==actor) for key in ['original4k','repair4k']]
    last=next(x for x in progress['actor_geometry'] if x['actor']==actor)
    b=[next(x for x in body if x['actor']==actor and x['model']==m) for m in ['r15','repair4k']]
    rows+=f"<tr><td>a{actor}</td><td>{old['points']:,} → {new['points']:,}</td><td>{old['growth']:.2f}× → {new['growth']:.2f}×</td><td>{old['outside_exact_box']/old['points']:.2%} → {new['outside_exact_box']/new['points']:.2%}</td><td>{new['outside_padded_box']:,}</td><td>{last['outside_allowed_envelope']} / {last['drift_violations']} / {last['max_origin_displacement_m']:.6f}m</td><td>{old['maximum_sigma_quantile'][-1]:.3f} → {new['maximum_sigma_quantile'][-1]:.3f}</td><td>{new['raw_status_counts']['0']}</td><td>{b[0]['visible_alpha_mean']:.3f} → {b[1]['visible_alpha_mean']:.3f}</td></tr>"
auditrows=''
for a in audits:
    auditrows+=f"<tr><td>{a['step']}</td><td>{a['frames']}</td><td><code>{json.dumps(a['actual_owner_changes'],ensure_ascii=False)}</code></td><td><code>{json.dumps(a['status_transition_raw_0U_1Actor_2Static'])}</code></td><td>{a['seconds']:.2f}s</td></tr>"
# Native-image contact sheets, same input-derived display crop for every version.
page=(C/'sync_player.html').read_text(encoding='utf-8');data=json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>',page,re.S).group(1))
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
for actor in [6,37,5,26,18]:
    raw=113 if actor in [6,37] else data['best_frame'][str(actor)];fid=f'0008{raw:03d}';box=data['rois'][str(actor)][raw-89]
    sheet=Image.new('RGB',(1440,885),(10,20,32));draw=ImageDraw.Draw(sheet)
    for j,(model,label) in enumerate([('old5','旧5 30K'),('r15','原R15 4K'),('repair4k','修复R15 4K')]):
        draw.text((10,j*295+3),f'{label} | a{actor} | raw{raw} | 整景物理阴影关闭',font=font,fill='white')
        folder=C/'assets'/model/fid;br=I/'branches'/('branches_'+model)/fid
        files=[(folder/('full.png' if model=='old5' else 'full_no_shadow.png'),'完整场景'),
            (folder/f'a{actor:02d}_isolated.png','目标独立资产'),(br/'static.png','静态分支'),
            (folder/(f'a{actor:02d}_delete.png' if model=='old5' else f'a{actor:02d}_delete_no_shadow.png'),'删除目标后的整景')]
        for k,(file,label2) in enumerate(files):
            x=k*360;y=j*295+29;draw.text((x+7,y),label2,font=font,fill=(210,230,245))
            tile=Image.open(file).convert('RGB').crop(box);scale=min(354/tile.width,240/tile.height)
            tile=tile.resize((round(tile.width*scale),round(tile.height*scale)),Image.Resampling.LANCZOS)
            sheet.paste(tile,(x+(360-tile.width)//2,y+24+(240-tile.height)//2))
    sheet.save(O/f'a{actor:02d}_raw{raw}.png')
block=f'''<section id="final-results"><h2>实际 4K 结果：工程边界与质量分开检查</h2>
<p class="notice"><b>相同原始初始化的 4,000 步、三次归属审计、五身份全 49 帧独立／删除以及动态／静态分支均已完成。</b>原长训仍暂停。以下数值证明几何边界和审计通路实际生效，尚不等于资产质量通过。</p>
<p><a href="../identity_comparison_20261018/index.html">打开五身份同步视频，模型选择“原R15 与 修复R15”或“三版对照”</a>。一条时间轴检查全部49帧；三维中心不再作为主要验收方式。</p>
<div style="overflow-x:auto"><table><tr><th>身份</th><th>原4K → 修复4K点数</th><th>相对原始种子增长</th><th>真实框外中心率</th><th>修复框+0.1m外点数</th><th>容许域外 / 漂移超限 / 最大累计位移</th><th>最大σ(m)</th><th>内部unknown</th><th>同一正域可见α均值</th></tr>{rows}</table></div>
<p>真实框外与容许域外是两项统计：初始已越框种子保留原包络，所以真实框外率不必为零。可见α使用三版相同的R15正像素域（49帧像素加权，PNG量化）；它是保体诊断，不是独立GT或编辑质量分数。<a href="body_retention.json">完整三版保体统计</a> · <a href="evidence/geometry_comparison.json">全量几何统计</a>。</p>
<h2>归属审计确实执行了什么</h2><table><tr><th>步数</th><th>帧</th><th>实际owner变化</th><th>状态转移矩阵（行旧、列新；U/Actor/Static）</th><th>审计耗时</th></tr>{auditrows}</table>
<p>矩阵中的unknown是证据状态；硬owner仍负责渲染。反证撤销状态不会把点自动改为道路。owner变化仍需要重复实际身体3D证据，不把a37道路上的车形整体转成Actor。<a href="evidence/train/ownership/001500/receipt.json">1.5K记录</a> · <a href="evidence/train/ownership/002500/receipt.json">2.5K记录</a> · <a href="evidence/train/ownership/003500/receipt.json">3.5K记录</a>。</p>
<h2>同帧直接检查（完整动态过程见同步视频）</h2>'''
for actor in [6,37,5,26,18]:
    raw=113 if actor in [6,37] else data['best_frame'][str(actor)]
    block+=f'<h3>a{actor} · raw{raw} · 上旧5 / 中原R15 / 下修复R15</h3><img loading="lazy" src="a{actor:02d}_raw{raw}.png" alt="a{actor}三版相同帧完整、独立、静态、删除对照">'
block+='''<p>没有配对干净道路编辑真值；减少点数和框外率不替代对车体、a6下部碎片与a37静态车形的视觉复核。human_verdict=null；组合修复结果不归因单一改动。</p></section>'''
if (W/'visual_review.json').is_file():
    review=read(W/'visual_review.json');shutil.copy2(W/'visual_review.json',O/'visual_review.json')
    findings='<h2>实际图像复核：仍有未解决项</h2><table><tr><th>身份／锚帧</th><th>实际观察</th></tr>'
    for case in review['cases']:findings+=f"<tr><td>a{case['actor']} / raw{case['raw_frame']}</td><td>{case['finding']}</td></tr>"
    findings+='</table><p><b>整体资产质量未通过：a6下部白带和a37静态车形仍未解决。</b>以上是明确列出的锚帧观察，不能当成全49帧人工验收。<a href="visual_review.json">复核范围与限制</a>。自动浏览器拒绝file://，未冒称交互浏览器播放验收；视频帧序和同步逻辑另有检查。</p>'
    block=block.replace('</section>',findings+'</section>')
p=O/'index.html';text=p.read_text(encoding='utf-8')
text=text.replace('</main>',block+'</main>') if 'id="final-results"' not in text else re.sub(r'<section id="final-results">.*?</section>',block,text,flags=re.S)
text=re.sub(r'当前：真实 GPU 自检通过，新 4K 正在训练；最近读取第 [\d,]+ / 4,000 步。','当前：真实GPU自检、4K训练、五身份49帧评估及动态／静态分支渲染已完成；质量未自动判为通过。',text)
text=text.replace('控制器 PID 20154 正在从头训练 4K，随后自动执行五身份 49 帧独立／删除评估。','有限控制器已完成全部GPU工作；检查点和逐帧证据保留。')
text=text.replace('新 4K 已启动；','新 4K 已完成；')
p.write_text(text,encoding='utf-8')
print(json.dumps(dict(status='MEASURED_RESULTS_APPENDED',points=[(r['actor'],r['points']) for r in geo['repair4k']]),ensure_ascii=False))
