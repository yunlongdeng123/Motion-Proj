"""本轮新产物独立评分；原始静帧、被拒窗口和旧审核均保留。"""
import argparse,html,json,shutil,sys
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from geometry_factory import read,dump
from build_synthetic_review import build

def main(root):
    factory=root/'factory';out=root/'review';out.mkdir(exist_ok=True)
    build(factory,out)
    manifest=read(out/'review_manifest.json');manifest['human_scoring_scope']='training_input_five_checks_r3'
    page=(out/'index.html').read_text();start=page.index('const DATA=');end=page.index(';\nconst $=',start)
    page=page[:start]+'const DATA='+json.dumps(manifest,ensure_ascii=False).replace('</','<\\/')+page[end:]
    page=page.replace("KEY='v77-target-protected-synthetic-frames-v1'","KEY='v77-training-input-frames-r3'").replace('20260929 / r1','20260929 / r3')
    page=page.replace('合成数据逐帧全检','新造训练输入逐帧全检').replace('V77_Target_Protected_frame_review.json','V77_training_input_r3_frame_review.json')
    page=page.replace('本帧问题：身份、接地、遮挡、边缘、清晰度、闪烁等','本帧问题：洞形、尺度位置/压ego、时间连续、遮挡次序、合成像素漏出等')
    banner='<div class="card warning"><b>按五项训练输入判据评分。</b>合理车辆洞形；尺度/位置合理且不压自车；轨迹与mask连续；A在B前且B可见部分保持；合成影响被最终H全部清除。被洞遮住的车身材质/受光/贴片感不单独扣分。<a href="index.html">本轮报告</a> · <a href="../v77-target-protected-r2/data_review.html">旧37例复核</a>。只对独立通过的新例开放人工全帧评分。</div>'
    page=page.replace('<main>','<main>'+banner,1);(out/'data_review.html').write_text(page);dump(out/'review_manifest.json',manifest)
    asset=out/'source_masks';asset.mkdir(exist_ok=True)
    for p in (root/'mask_review').glob('*.jpg'):shutil.copy2(p,asset/p.name)
    for p in (root/'temporal_windows/mask_review').glob('*.jpg'):shutil.copy2(p,asset/p.name)
    temporal=read(root/'temporal_windows/source_manifest.json')['clips'];still=read(root/'still_mask_reviews.json')['reviews'];nq=read(root/'temporal_windows/mask_review/mask_audit.json')['clips'];tq=read(root/'temporal_windows/independent_mask_reviews.json')['clips'];pairs=read(factory/'pair_candidates.json')
    coupling=read(root/'coupled_source_control.json')
    stats={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r3','still_masks':len(still),'still_pass':sum(r['status']=='pass' for r in still),
           'temporal_proposed':len(temporal),'temporal_geometry_pass':sum(c['geometry_pass'] for c in temporal),'temporal_segmented':len(nq),'temporal_numeric_pass':sum(c['numeric_gate_pass'] for c in nq),
           'temporal_independent_pass':sum(c['donor_mask_status']=='pass' for c in tq),'new_cases':len(manifest['clips']),'new_case_status':manifest['counts'],'new_case_types':manifest['type_counts'],
           'receiver_scenes':manifest['scene_count'],'training_ready':0,'model_training_steps':0,'human_verdict':None,'new_video_count':len(manifest['clips'])*4,
           'five_checks':'hole silhouette, placement/ego, continuity, A-before-B, full synthetic support masked','new_masks_only_SAM2':True,
           'new_multiview_donor_cases':coupling['new_multiview_donor_cases'],'duplicate_old_proposals_excluded':coupling['duplicates_not_rerendered'],
           'quality_sampling':'one preset maximum-occlusion frame per single-actor case, middle frame for background; all-frame numerical and pixel validation, human full-frame review required',
           'multiview_limit':'new source masks passed but no geometric placements; actual new synthetic cases reuse historically approved clean sources; no synchronized multicamera rendering or multiview neural conditioning'}
    dump(root/'summary.json',stats);dump(out/'summary.json',stats)
    rows=''.join(f'<tr><td>{c["case_id"]}</td><td>{c["scene"]} / {c["camera"]}</td><td>{c["donor_source_id"]}</td><td>{c["type"]}</td><td>{c["review"]["synthetic_status"]}</td><td>{html.escape(c["review"]["note"])}</td></tr>' for c in manifest['clips'])
    sources=''.join(f'<figure><img src="source_masks/{c["source_id"]}.jpg"><figcaption>{c["source_id"]}：{html.escape(c["note"])}</figcaption></figure>' for c in tq)
    h=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v77 r3 · 连续轮廓与新造数据</title><style>body{{background:#101725;color:#e2eaf5;font:16px/1.65 system-ui;margin:30px auto;padding:0 24px;max-width:1500px}}a{{color:#8ccfff}}section{{background:#182438;padding:22px;border:1px solid #34435c;border-radius:12px;margin:20px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px;border-bottom:1px solid #3a4860;text-align:left}}img{{max-width:100%}}.row{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}svg{{width:100%;height:auto}}@media(max-width:800px){{.row{{display:block}}}}</style>
<h1>v77 r3：连续车辆洞形与几何配对造数</h1><p>本轮沿用真实 Y。新多视角来源完成分割但没有通过放置；实际新合成改由已审干净来源先配对几何。只运行冻结 SAM2，未训练 DriveEditor。<a href="data_review.html">进入新数据逐帧打分页</a> · <a href="../v77-target-protected-r2/index.html">上一轮27个候选与旧评分</a></p>
<section><svg viewBox="0 0 1200 100" role="img" aria-label="已审干净来源，经几何配对、连续3D放置和洞检查，形成masked-X与真实Y监督"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#93c9ef"/></marker></defs><g fill="#254368" stroke="#7ea4c8">{''.join(f'<rect x="{i*240+5}" y="10" width="205" height="75" rx="8"/>' for i in range(5))}</g><g stroke="#93c9ef" marker-end="url(#a)">{''.join(f'<path d="M{i*240+211} 48h25"/>' for i in range(4))}</g><g fill="white" text-anchor="middle" font-size="17">{''.join(f'<text x="{i*240+106}" y="40">{a}<tspan x="{i*240+106}" dy="24">{b}</tspan></text>' for i,(a,b) in enumerate([('已有真实干净来源','复用已审连续mask'),('先做几何配对','整段可放置才合成'),('连续3D放置','真实Y / 保护车B'),('五项输入检查','masked-X + 真实Y'),('独立抽帧QA','人工逐帧确认')]))}</g></svg></section>
<section><h2>实际结果</h2><p>4张静帧均通过轮廓检查。固定居中窗口只有1/4通过整段几何/曝光检查；保留原提案后，以统一的有界窗口搜索得到 {stats['temporal_geometry_pass']}/4 个有效窗口。连续SAM2完成 {len(nq)} 段，数值检查通过 {stats['temporal_numeric_pass']} 段，独立抽帧通过 {stats['temporal_independent_pass']} 段。</p><p>新合成 <b>{stats['new_cases']} 例 / {stats['receiver_scenes']} 个receiver scene</b>，类别 {html.escape(str(stats['new_case_types']))}。独立状态 {html.escape(str(stats['new_case_status']))}，人工评分未填写，训练准入仍为0。旧例和本轮重复背景不能合并宣称独立scene数量。</p><p>mask先清零再编码，完整synthetic-X不作为DriveEditor条件；这些视频是训练数据预览，不是模型补景输出。外观差异仅在漏到条件里或破坏hole/空间关系时构成拒绝依据。</p></section>
<section><h2>配对控制与边界</h2><p>仅用新多视角供体：相机相对回放与既有固定世界位移两种有界控制均为0个合法提案。随后复用此前独立通过的干净来源，保留污染黑名单和同样的空间门槛，按整段几何配对得到新样本；排除 {stats['duplicate_old_proposals_excluded']} 个旧重复提案。新样本中使用新多视角供体的为 {stats['new_multiview_donor_cases']} 个，不能宣称多视角方案改善了合成质量。</p><p>所有类别都先检查最终洞是否侵入画面底部64px保守自车区域，并在渲染入口复查；这只是避免已知机盖失败的保守门槛，不等于像素级ego分割。位置、尺度、yaw、A→B次序与LiDAR支持沿用逐帧检查；原始RGB和所有合成影响域、最终H、真实Y均保存。</p><p>整段使用同一真实供体camera-track，禁止逐帧换角度；不是同步多相机共享世界的联合渲染，也未增加网络条件或修改架构。每例10个真实曝光，仅为短窗。为节省重复AI视觉开销，本轮新合成按预先固定规则每例看一帧：有B用最大遮挡帧，纯背景用中帧；工程全帧检查照常，时序最终由用户全帧确认，不从一帧判视频通过。</p><p>SAM2可选CUDA扩展缺失，官方小孔洞后处理跳过；本轮保留实际原推理输出和日志，没有静默修补mask。</p></section>
<section><h2>逐例结果</h2><table><tr><th>case</th><th>真实背景</th><th>实际供体</th><th>类型</th><th>独立判断</th><th>依据</th></tr>{rows}</table></section><section><h2>新连续来源的证据</h2><div class="row">{sources}</div></section><section><details><summary>有限候选搜索的拒绝计数（工程代理，非模型效果）</summary><pre>{html.escape(json.dumps(pairs['rejection_counts'],ensure_ascii=False,indent=2))}</pre></details><p>完整来源、失败和无损训练对保存在远端 {root}。人工评分导出单独使用r3命名空间，r1/r2评分不覆盖。</p></section></html>'''
    (out/'index.html').write_text(h);print('REPORT',stats)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
