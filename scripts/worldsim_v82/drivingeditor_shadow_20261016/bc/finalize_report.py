"""Finalize reviewed decisions, including failed numerical winners and rollback."""
import datetime,hashlib,json,subprocess,sys
from pathlib import Path
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(HERE/'update_union_report.py')],check=True)
OUT=HERE.parent if HERE.name=='scripts' else HERE.parents[1]/'outputs/shadow_bc_experiments'
BASE='/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_BC_20261016_v1'
def get(source,target):
    p=OUT/target;p.parent.mkdir(exist_ok=True,parents=True)
    subprocess.run(['scp','wm-vgpu-1011:'+BASE+'/'+source,str(p)],capture_output=True,text=True,timeout=40,check=True)
for name in ['status.json','runtime_contract.json']:
    get('safe_coverage_probe_v1/'+name,'rollback/'+name)
get('safe_coverage_probe_v1/R4/evaluation/metrics.json','rollback/metrics.json')
get('safe_coverage_probe_v1/R4/evaluation/unshaded_static.png','rollback/unshaded_static_NOT_ACCEPTED.png')
get('safe_coverage_probe_v1/R4/evaluation/observed_composite.png','rollback/observed_composite_NOT_ACCEPTED.png')
read=lambda p:json.loads((OUT/p).read_text(encoding='utf-8'))
summary=read('measured_summary.json');review=read('evidence/visual_review.json');rollback=read('rollback/metrics.json')
summary['read_only_probes']={'receiver_only_rollback':{'metrics':rollback,'accepted':False,
    'reason':'Changing the jointly optimized candidate composition without refitting makes clean-road MAE0.1986, worse than original0.1039; no training or deployment.'}}
summary['recommended_variant']='deferred_R4'
summary['visually_rejected_variants']=['union_R4','receiver_only_rollback']
summary['recommendation_is_partial_local_prototype']=True
(OUT/'measured_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
review['rejected_variants']['receiver_only_rollback']={'reason':'Zero-update candidate rollback failed clean reference error (0.1986 > original0.1039); jointly fitted parameters cannot simply be recombined.','training_updates':0}
review['final_preferred_variant']='deferred_R4';review['final_reviewed_at']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
(OUT/'evidence/visual_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
canvas=Image.new('RGB',(1080,664),'white');d=ImageDraw.Draw(canvas)
for k,(title,path) in enumerate([
    ('Original Static: baked vehicle shadow','results/R0/unshaded_static.png'),
    ('Deferred model: delete target vehicle','deferred/delete_target.png'),
    ('Deferred model: Actor with learned shadow','deferred/observed_composite.png'),
    ('Deferred model: Actor translated +1m','deferred/translate_+1_m.png')]):
    im=Image.open(OUT/path).crop((25,205,205,305)).resize((540,300));x=k%2*540;y=k//2*332
    canvas.paste(im,(x,y+32));d.text((x+8,y+9),title,fill='black')
canvas.save(OUT/'recommended_result.png')
m=summary['methods']['deferred_R4'];base=summary['methods']['R0'];contact=summary['methods']['R3']
clean_gain=1-m['diagnostic_footprint_clean_linear_MAE']/base['diagnostic_footprint_clean_linear_MAE']
shadow_gain=1-m['observed_shadow_linear_MAE']/contact['observed_shadow_linear_MAE']
section=f'''<section style="border:2px solid #29775a;background:#edf7f2;padding:18px;margin:24px 0"><h2>本轮结论与保留版本</h2>
<p><b>保留 deferred_R4：真实跨时刻道路监督＋有LiDAR支持的接收面＋Actor条件投影阴影＋道路可见贡献延迟着色。</b>在已查看的原位、删除和平移图中，它比接触阴影和逐高斯着色更可靠。仍是局部原型，没有通过全部质量门槛。</p>
<p>原黑块区干净参考误差 {base['diagnostic_footprint_clean_linear_MAE']:.6f} → {m['diagnostic_footprint_clean_linear_MAE']:.6f}（降低{clean_gain:.1%}）；车影拟合误差 {contact['observed_shadow_linear_MAE']:.6f} → {m['observed_shadow_linear_MAE']:.6f}（相对接触基线降低{shadow_gain:.1%}）。删除所有Actor后的阴影联动与未着色Static逐值一致。固定相机时间误差 {m['fixed_camera_mean_nonanchor_time_MAE']:.6f}，仍未达到1e−5门槛。</p>
<img src="recommended_result.png" alt="保留模型的原始背景、删除车辆、原位带阴影与平移对照">
<p class="note"><b>数值更优的 union_R4 已否决：</b>它引入原区域外的新黑片，表明全场景控制区均值不足以保护局部质量。随后做了零训练回退，但干净道路误差变成{rollback['diagnostic_footprint_clean_linear_MAE']:.6f}，同样否决。没有将失败版本覆盖到保留模型。</p>
<p>跨时刻覆盖应扩展到有明确道路归属的接收面；不能因为高斯核碰到道路，就把整个高斯的时态冻结。当前数据支持B＋C路线，但保留模型仍有时间残差和阴影近似误差。大角度旋转的车体缺陷在原R0中已经存在；复制/新增Actor与真实重叠编辑尚未认证。此次没有转向完整重光照D。</p>
<p><a href="evidence/visual_review.json">实际视觉审查与否决记录</a> · <a href="measured_summary.json">全部数值与失败探针</a> · <a href="rollback/metrics.json">零训练回退结果</a></p>
<pre>保留addon：{BASE}/runs/deferred_1000_v1/R4/addon_001000.pt
对应渲染实现：{BASE}/implementation_deferred_v1/run_deferred_refinement.py
使用时需加载只读父checkpoint；这是addon，不是可直接替代原checkpoint的完整场景权重。
本轮C更新：5×1000首轮＋3×1000追加；另有B多起点低参数拟合、接口短测和只读诊断。
所有训练worker已结束；没有再排更多训练。</pre></section>'''
report=OUT/'report.html';text=report.read_text(encoding='utf-8');text=text.replace('<h1>车影 B＋C：真实 GPU 实验与自验证</h1>','<h1>车影 B＋C：真实 GPU 实验与自验证</h1>'+section)
report.write_text(text,encoding='utf-8')
print(json.dumps({'recommended':'deferred_R4','rejected':['union_R4','receiver_only_rollback'],'clean_error_reduction':clean_gain,'shadow_error_reduction_vs_contact':shadow_gain},ensure_ascii=False))
