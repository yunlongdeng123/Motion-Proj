from pathlib import Path
import json,datetime,shutil
W=Path(__file__).resolve().parent;ROOT=W.parents[2];O=ROOT/'outputs/shadow_rebuild_v3/r14'
preview=json.loads((O/'geometry_003000/preview/summary.json').read_text(encoding='utf8'))
owner=json.loads((O/'geometry_003000/gates/003000/owner/owner.json').read_text(encoding='utf8'))
monitor=json.loads((W/'latest_monitor.json').read_text(encoding='utf8'))
review=dict(step=3000,reviewer='root agent; no subagents',human_verdict=None,quality_accepted=False,observed_all7_contact=True,detail_review_actors=[5,12,18,37],observations=['a5独立车体本帧未见旧的大面积路板；下沿仍有细小毛刺，不能只凭单帧确认干净','a12仍是遮挡后可见的局部片段；小MAE不能证明完整车体','a18白车外观仍偏亮、失真，bodyMAE约0.103，不接受为完成','a37删除后出现明显浅色缺口；背景覆盖与道路重建未完成','天空、远处建筑与道路明显模糊；全场景质量尚未通过'],single_frame_only=True,full49_acceptance_pending=True,next='continue registered13000/7000/2400; no extra routes, no early shutdown')
(W/'root_review_003000.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf8');(O/'geometry_003000/root_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf8')
shutil.copy2(W/'architecture.svg',O/'architecture.svg')
rows=''.join(f'<tr><td>a{x["actor"]}</td><td>raw{x["raw"]}</td><td>{x["body_MAE"]:.4f}</td><td>{x["valid_body_pixels"]}</td><td>{x["unknown_mask_pixels"]}</td></tr>' for x in preview['actors'])
images=''.join(f'<h3>Actor {a}</h3><a href="geometry_003000/preview/a{a:02d}.png"><img src="geometry_003000/preview/a{a:02d}.png" alt="a{a} 真实RGB/全场景/独立资产/原始alpha/删除预览"></a>' for a in [5,6,12,18,25,26,37])
notes=''.join('<li>'+x+'</li>' for x in review['observations'])
html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>R14 3k几何检查：完整实验继续</title><style>body{{background:#0c1726;color:#e1ebf5;max-width:1400px;margin:28px auto;padding:0 20px;font:16px/1.7 system-ui,Microsoft YaHei}}h1{{font-size:28px}}h2{{margin-top:32px}}a{{color:#8dc7ff}}img{{max-width:100%;display:block;margin:12px 0}}table{{border-collapse:collapse;width:100%}}td,th{{padding:7px 14px;border-bottom:1px solid #34485d;text-align:left}}.card{{padding:16px;background:#17283b;border-radius:8px}}.warning{{color:#ffce88}}code{{background:#20344a;padding:3px 6px}}</style>
<h1>R14：3k 几何检查（完整实验继续）</h1><div class="card">本页是中间诊断，<strong>尚未完成质量验收</strong>。最近监测训练步数 {monitor['progress']['global_step']}/22400；当前阶段 {monitor['pipeline']['status']}。完整预算：几何13000＋材质光照7000＋联合2400。<br><a href="mask_review/report.html">完整49帧 mask / LiDAR 核查</a> · <a href="geometry_003000/root_review.json">本代理逐车检查记录</a></div>
<img src="architecture.svg" alt="真实观测到实例前端、持续归属约束、几何阴影与编辑验证组件图">
<h2>3k 看到了什么</h2><ul class="warning">{notes}</ul><p>以上是固定7帧的几何阶段预览，尚未启用最终材质阴影合成。低车体误差不代表道路归属正确；这不是完整49帧编辑验收，也不与旧18k作为相同预算的因果对照。</p>
<h2>归属审计实际执行</h2><p>46张训练帧、全43身份参与真实全场景 α×T 审计；拆分前后各一轮。本次满足严格混合拆分条件的父点0，Actor→静态0，静态→Actor0。真实待定车体点接纳 {owner['unknown_measured_body_admitted']}，仍保留未知 {owner['unknown_pool_retained']}。没有为了产生非零修改而放宽阈值；没有将待定点默认放入背景。</p>
<p>原控制器在3k记录状态时发生参数重名异常，审计前退出；已保留原日志和检查点，用修复控制器从同一3k模型、Adam、RNG、输入继续。两检查点逐字节相同，重复SGD0，训练模型代码未改。CPU加载和状态函数检查通过，实际审计/预览通过后已继续后续训练。</p>
<h2>固定帧车体误差</h2><p>线性 RGB 的有效 mask 内 MAE，仅用于此快照；未知像素不计入正负mask监督。小分母、局部遮挡与域变化必须结合原图看。</p><table><tr><th>Actor</th><th>相机帧</th><th>body MAE</th><th>有效正像素</th><th>该实例未知像素</th></tr>{rows}</table>
<h2>逐车预览</h2><p>每行从左到右：真实RGB → 全场景 → 独立Actor RGB → 未乘mask的原始alpha → 该Actor删除。点击原图放大。7车均有同帧相机与真实LiDAR准入证据。</p>{images}
<h2>后续验收与边界</h2><p>继续完整训练，再交付49帧65条动态与编辑流，包括7车删除、旧三车同轴平移和15°/30°连续旋转。阴影另检查同时间改位姿、删除对应遮光体、关阴影保持基础材质等反事实。原GN11/v2/18k只读；其余4车没有旧独立资产，不称同资产编辑配对。</p><p>没有编辑/无影道路反事实真值；mask与道路证据均为预测或几何代理。DINOv3/RANSAC只独立对照，不替换GT。已看过的QUERY属于开发验证。旧车身自身明暗没有完成完整材质分解。</p><p>本页时间：{datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}。human_verdict=null；problem_solved=null。完成完整训练、49帧验收、报告与源码发布并核对所有任务空闲后才关机。</p></html>'''
(O/'report.html').write_text(html,encoding='utf8')
unified=ROOT.parent.parent/'2026-10-15/file-c-users-dengyunlong-documents-codex/outputs/DrivingEditor_unified_report.html';s=unified.read_text(encoding='utf8');start='<!-- R14_TRAINING_MILESTONE_BEGIN -->';end='<!-- R14_TRAINING_MILESTONE_END -->'
entry=f'{start}<section style="padding:20px;margin:20px 0;border:2px solid #b88e43;border-radius:10px;background:#252132;color:#eee"><h2>R14 3k 几何检查：完整训练继续</h2><p>7车独立资产/原始alpha/删除诊断，当前质量尚未通过；归属审计与控制器修复证据保留。</p><a style="color:#8dc7ff" href="{(O/"report.html").as_uri()}">查看3k阶段报告</a></section>{end}'
if start in s:
    a=s.index(start);b=s.index(end,a)+len(end);s=s[:a]+entry+s[b:]
else:
    a=s.index('>',s.lower().find('<body'))+1;s=s[:a]+entry+s[a:]
unified.write_text(s,encoding='utf8')
print(json.dumps(dict(report=str(O/'report.html'),quality_accepted=False,root_review=True,unified_updated=True),ensure_ascii=False))
