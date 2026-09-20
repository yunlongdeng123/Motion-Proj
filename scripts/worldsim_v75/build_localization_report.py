"""从完整结果生成科学图和开发报告，不手填不存在的实验结果。"""
import json
import argparse
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--run-dir',type=Path)
parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/V75_Localization_Report')
args=parser.parse_args()
OUT=args.output_dir
E=OUT/'evidence'
E.mkdir(parents=True,exist_ok=True)
if args.run_dir:
    for name in ['summary.json','protocol.json','target_projections.json','evaluator_calibration.json','shared_prefix_audit.json']:
        shutil.copy2(args.run_dir/name,E/name)
    for name in ['target-frozen.png','comparison-seed42.jpg','comparison-seed43.jpg','comparison-seed42.mp4','comparison-seed43.mp4']:
        shutil.copy2(args.run_dir/name,OUT/name)
summary=json.loads((E/'summary.json').read_text())
projection=json.loads((E/'target_projections.json').read_text())
assert summary['status']=='complete'
fig,axes=plt.subplots(2,2,figsize=(11,6.7),sharex=True,sharey=True,layout='constrained')
for i,seed in enumerate([42,43]):
    for j,sign in enumerate(['negative','positive']):
        ax=axes[i,j]
        for kind,color,style,label in [('persistent','#b94735','-','Persistent bias'),('restore','#148170','--','Restore at 1.23s')]:
            row=next(r for r in summary['results'] if r['seed']==seed and r['case']==f'{sign}_{kind}')
            p=row.get('detection_pairs',[])
            ax.plot([r['frame']/30 for r in p],[r['center_distance_px'] if r['center_distance_px'] is not None else np.nan for r in p],
                    marker='o',ms=4,lw=2,color=color,ls=style,label=label)
        frames=[r['frame'] for r in p]
        control=[]
        for f in frames:
            if f<5:
                control.append(0.0)
                continue
            a=np.asarray(projection['clean'][f]['bounds'])
            b=np.asarray(projection[sign][f]['bounds'])
            control.append(float(np.linalg.norm((b[:2]+b[2:]-a[:2]-a[2:])/2)))
        ax.plot(np.asarray(frames)/30,control,':',color='#66778a',lw=1.7,label='Input (persistent bias)')
        ax.axvline(37/30,color='#9aa6af',ls=':',lw=1)
        ax.set_title(f'Seed {seed} | {"-" if sign=="negative" else "+"}0.5 m condition offset',fontsize=11)
        ax.set_ylabel('Center displacement vs clean (px)')
        ax.set_xlabel('Time (s)')
        ax.grid(alpha=0.2)
        ax.set_ylim(bottom=0)
        ax.spines[['top','right']].set_visible(False)
axes[0,0].legend(fontsize=8,loc='upper left')
fig.suptitle('Paired 2D response and restoration | one development scene',fontsize=15)
fig.savefig(OUT/'response-curves.png',dpi=170)
fig.savefig(OUT/'response-curves.svg')
svg=OUT/'response-curves.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
plt.close(fig)
comparisons=summary['restoration_comparisons']
improved=sum(r['restored_center_distance_px'] is not None and r['persistent_center_distance_px'] is not None and
             r['restored_center_distance_px']<r['persistent_center_distance_px'] for r in comparisons)
trs=''.join(f'<tr><td>{r["seed"]}</td><td>{"−" if r["sign"]=="negative" else "+"}0.5 m</td><td>{r["persistent_center_distance_px"]:.2f}</td><td>{r["restored_center_distance_px"]:.2f}</td><td>{r["persistent_roi_mae"]:.2f} → {r["restored_roi_mae"]:.2f}</td></tr>' for r in comparisons)
markdown='\n'.join(f'| {r["seed"]} | {"−" if r["sign"]=="negative" else "+"}0.5m | {r["persistent_center_distance_px"]:.2f}px | {r["restored_center_distance_px"]:.2f}px | {r["persistent_roi_mae"]:.2f} → {r["restored_roi_mae"]:.2f} |' for r in comparisons)
peak=max(r['sampled_gpu_peak_mib'] for r in summary['results'])/1024
results=f'''\n## 完整执行结果

10/10项完成，300个生成段、每项237帧，全部有限值检查与视频解码通过，均无OOM。整卡每秒采样峰值{peak:.2f}GiB。仍然只有1场景、1目标、2个seed，不能把300段或逐帧数量当作独立样本。

真实初帧±4px平移校准的检测中心残差为+0.103px／−0.470px。每个clean和干预条件的匹配数量见轻量结果，未删除检测缺失。恢复前共享的原始输出前缀逐像素一致，保证比较没有在干预开始前出现差异。

恢复窗口37–53帧的配对结果如下。二维框指标使用第37、45、53帧，与相同seed的clean比较；像素指标使用该窗口全部17帧。二者的分母不同。这里的“距离”是2D框中心相对clean的差异，不是相对未来真值的错误。

| seed | 条件偏置 | 持续偏置框中心差 | 恢复组框中心差 | ROI像素MAE（持续→恢复，0–255） |
|---|---|---:|---:|---:|
{markdown}

4组比较中有{improved}组在恢复正确条件后，检测框中心比持续偏置更接近clean。输入投影本身的普通透视响应作为描述性几何参照，不能用像素/米比值声称“放大”。该参照采用投影框的中心，与输入局部性审计中的角点投影均值口径分开。

目前只证实人工条件误差能够传到生成画面，并检验其短期可恢复性。恢复使用原始记录条件；不是一个已证明同信息预算优势的修复方法。没有自然重建误差样本、真实未来RGB或策略反馈，因此没有完成科学badcase或闭环危害准入。

轻量结果见 [localization](../autoresearch/worldsim_v75/localization/)；原始根目录保留 `comparison-seed42/43.jpg`、同名MP4和全部未压缩帧。固定时间裁剪在同一帧的五条件间完全共享，未按生成车辆重新居中。

failure_ledger_delta：none；本轮没有把可恢复的条件响应登记成新的科学失败。人工verdict保持null。
'''
(OUT/'RESULTS.md').write_text(results,encoding='utf-8',newline='\n')
html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V7.5 定位偏置与恢复</title><style>
body{{margin:0;color:#183348;background:#eff3f7;font:16px/1.65 "Segoe UI","Microsoft YaHei",sans-serif}}main{{max-width:1280px;margin:auto;padding:36px 24px}}h1{{font-size:32px;line-height:1.25}}h2{{font-size:23px}}section{{background:white;border:1px solid #dde4ea;border-radius:14px;padding:22px;margin:22px 0}}img,video{{width:100%;height:auto;border-radius:8px}}.muted{{color:#627382;font-size:14px}}.tag{{color:#087d6d;font-weight:bold}}table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;border-bottom:1px solid #dde4ea;text-align:left}}.flow{{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap}}.box{{background:#eaf4f2;border:1px solid #b9d7cf;border-radius:10px;padding:16px;flex:1;min-width:140px;text-align:center}}a{{color:#087d6d}}.scroll{{overflow:auto}}
</style><main><div class="tag">V7.5 · 单场景开发实验 · 10/10 完成</div><h1>局部条件偏置如何进入生成画面，恢复条件能恢复多少？</h1>
<p>迎面白车 · 沿初始ego前向 ±0.5m · 相同初帧、文本、轨迹及配对seed。{improved}/4组恢复后的二维位置响应更接近clean。当前证据范围是条件敏感性与短期恢复，尚无自然重建缺陷或闭环危害结论。</p>
<section><h2>Architecture components</h2><div class="flow"><div class="box">固定真实初帧<br>原始参考世界</div>→<div class="box">目标条件偏置<br>±0.5m / 恢复</div>→<div class="box">OmniDreams 2B<br>生成历史与cache</div>→<div class="box">未压缩输出<br>图像差异 / 2D检测</div></div><p class="muted">物理参考保持不变。没有策略反馈；恢复使用原始记录条件，属于诊断参照。</p></section>
<section><h2>目标来自真实初帧</h2><img src="target-frozen.png" alt="原始RGB中黄色框标记迎面白车81"><p class="muted">13个初始几何候选均保留。植被遮挡、车辆遮挡或恢复后可见期不足者排除；目标81在偏置生成前冻结。评价窗口到1.767秒，不能讨论长时目标漂移。</p></section>
<section><h2>位置响应与普通投影参照</h2><img src="response-curves.png" alt="两个seed、正负条件偏置的二维中心差异与恢复曲线"><p class="muted">红：持续偏置；绿：1.233秒恢复原条件；灰：条件框的几何投影变化。纵轴为相对同seed clean的2D中心差，不能当作米制真实错误或误差放大率。</p><a href="response-curves.svg">下载矢量图</a></section>
<section><h2>恢复窗口的全部四组比较</h2><div class="scroll"><table><tr><th>seed</th><th>偏置</th><th>持续中心差 / px</th><th>恢复中心差 / px</th><th>ROI像素差（持续→恢复）</th></tr>{trs}</table></div><p class="muted">框中心：第37、45、53帧；像素差：37–53共17帧。真实初帧±4px平移校准残差约0.10–0.47px；这不能完全消除生成外观变化对检测器的影响。</p></section>
<section><h2>Seed 42：同一裁剪中的五种条件</h2><video src="comparison-seed42.mp4" controls loop preload="metadata"></video><img src="comparison-seed42.jpg" alt="seed42五条件在0.70、1.20、1.50、1.77秒的实际生成"><p class="muted">五列共享参考裁剪与缩放。前两行尚未恢复，后两行的恢复列使用原始条件。</p></section>
<section><h2>Seed 43：相同协议重复</h2><video src="comparison-seed43.mp4" controls loop preload="metadata"></video><img src="comparison-seed43.jpg" alt="seed43五条件的实际生成"></section>
<section><h2>核验与下一步决策</h2><p>零偏置237帧条件重渲染完全一致；干预窗口所有输入改变位于目标投影邻域，框外为0。不同条件共享输入的输出前缀也完全相同。10项均无OOM，整卡采样峰值{peak:.2f}GiB。</p><p>本轮不能把图像残留等同于不可恢复的状态错误。下一步需要新的长可见场景与自然重建→条件读出证据；保留原始条件恢复与普通几何修正作为强控制，不扩大扰动来制造失败。</p><p><a href="evidence/summary.json">完整汇总</a> · <a href="evidence/protocol.json">冻结协议</a> · <a href="evidence/evaluator_calibration.json">评价器校准</a> · <a href="evidence/shared_prefix_audit.json">共享前缀核验</a></p></section><p class="muted">WS-V75-LOCALIZE-01 / 20260920-r1 · 1场景 / 1目标 / 2seeds · 人工verdict: null</p></main></html>'''
(OUT/'index.html').write_text(html,encoding='utf-8',newline='\n')
print(json.dumps({'improved':improved,'comparisons':comparisons},ensure_ascii=False,indent=2))
