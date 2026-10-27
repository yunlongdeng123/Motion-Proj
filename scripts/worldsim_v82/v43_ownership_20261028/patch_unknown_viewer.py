"""CPU-only UI: separate abstention causes and preview semantic removal explicitly."""
from pathlib import Path
import re
import subprocess,sys,json
B=Path(__file__).resolve().parents[2];W=B/'work/persistent_world_20261027'
p=W/'scene_viewer.html';s=p.read_text(encoding='utf8')
s=s.replace('v43-badcases-4','v43-unknown-cpu-5')
if 'semantic_unknown_reason' not in s:s=s.replace('<option value="semantic_conflict">','<option value="semantic_unknown_reason">unknown 来源</option><option value="semantic_conflict">')
s=s.replace('<option value="0">关闭</option>','<option value="0">关闭额外轮廓带</option>')
s=s.replace('单位为480×320实拍像素；控制语义暂缓分类，不裁点、不改owner。','所有车辆内外两侧，单位为480×320实拍像素；外侧包含路面。关闭只取消此项，基础unknown仍保留。')
if 'id="semanticAblation"' not in s:
    controls='''<div class="subtools"><label><input type="checkbox" id="semanticStaticUnknown" checked>静态车影外观候选 → unknown（49帧）</label><label>语义剥离预览<select id="semanticAblation"><option value="all">保留全部</option><option value="hide_unknown">隐藏unknown</option><option value="hide_vehicle">隐藏车辆类14–19</option><option value="hide_both">隐藏车辆＋unknown</option></select></label></div><div id="semanticFilterStatus" class="note">此开关只过滤右侧三维中心；左侧保存的RGB不会变化。真实高斯RGB剥离待GPU重渲染。静态车影候选只暂缓分类，不更改owner或删除模型。</div>'''
    s=s.replace('<div id="semanticStatus"',controls+'<div id="semanticStatus"',1)
s=s.replace('查看 V43 1 / 0 / unknown 候选','旧CPU身份候选 1 / 0 / unknown（独立于语义）')
p.write_text(s,encoding='utf8')
p=W/'scene_edit_geometry.js';s=p.read_text(encoding='utf8')
stats="S.semanticFilterStats={considered:0,hidden:0,unknown:0,vehicle:0,contour:0,static_vehicle:0,base_unknown:0,road_to_unknown:0};"
if stats not in s:s=s.replace("instances=geometryInstances(),counts={};","instances=geometryInstances(),counts={};"+stats)
while stats+stats in s:s=s.replace(stats+stats,stats)
needle="   const foreign=instance.cloud.startsWith('foreign:');"
if 'semanticFilterReject(e)' not in s:
    assert needle in s
    s=s.replace(needle,"""   const e=(mode.startsWith('semantic')||($('semanticAblation')?.value||'all')!=='all')?semanticRow(instance,j):null;
   if(e){const q=S.semanticFilterStats;q.considered++;if(e.class_id===0)q.unknown++;if(e.class_id>=14&&e.class_id<=19)q.vehicle++;if(e.boundary_abstained)q.contour++;if(e.static_vehicle_quarantined)q.static_vehicle++;if(e.base_class_id===0)q.base_unknown++;if(e.base_class_id===1&&e.class_id===0)q.road_to_unknown++;if(semanticFilterReject(e)){q.hidden++;continue;}}
"""+needle,1)
    s=s.replace('semanticColor(semanticRow(instance,j),mode)','semanticColor(e,mode)',1)
needle=" semanticLegend();if($('geometryColorLegend'))"
if "$('semanticFilterStatus').textContent=" not in s:
    assert needle in s
    s=s.replace(needle,""" if($('semanticFilterStatus')){const q=S.semanticFilterStats;$('semanticFilterStatus').textContent=`当前画内抽样中心：unknown ${q.unknown}（基础低置信/混合 ${q.base_unknown}，轮廓规则 ${q.contour}，静态车影候选 ${q.static_vehicle}；原因可重叠），车辆类 ${q.vehicle}，本开关隐藏 ${q.hidden}。road→unknown ${q.road_to_unknown}。不是全模型比例。仅右侧几何预览；左侧RGB未重渲染，不能据此判定残影已剥离。`;}
"""+needle,1)
reason="mode==='semantic_unknown_reason'?'unknown来源：灰=基础低置信/混合；黄=额外轮廓带；橙=跨帧静态车影候选；蓝=本帧无贡献证据。多原因优先显示橙/黄，其余类别暗显。':"
if reason not in s:s=s.replace("mode.startsWith('semantic')?'语义：",reason+"mode.startsWith('semantic')?'语义：")
while reason+reason in s:s=s.replace(reason+reason,reason)
p.write_text(s,encoding='utf8')
p=W/'scene_viewer.js';s=p.read_text(encoding='utf8')
s=s.replace("'semanticBoundary','semanticBoundaryMass'].forEach", "'semanticBoundary','semanticBoundaryMass','semanticStaticUnknown','semanticAblation'].forEach")
s=s.replace("boundary_fraction:Number($('semanticBoundaryMass').value),profile:","boundary_fraction:Number($('semanticBoundaryMass').value),static_vehicle_unknown:$('semanticStaticUnknown').checked,geometry_ablation:$('semanticAblation').value,profile:")
needle="S.channel=j.channel;$('caseNote').value=j.note||'';"
if "j.semantic_parameters?.boundary_px" not in s:
    assert needle in s
    s=s.replace(needle,needle+"if(j.semantic_parameters){const v=j.semantic_parameters;for(const [key,id] of [['boundary_px','semanticBoundary'],['boundary_fraction','semanticBoundaryMass'],['geometry_ablation','semanticAblation'],['profile','semanticProfile'],['radius_px','semanticRadius'],['depth_m','semanticDepth'],['max_sigma_m','semanticSigma']])if(v[key]!==undefined)$(id).value=String(v[key]);if(v.static_vehicle_unknown!==undefined)$('semanticStaticUnknown').checked=v.static_vehicle_unknown;}/* j.semantic_parameters?.boundary_px restored before display */")
dup="if(m?.isolated_alpha){metric('车身独立 alpha',fmt(m.isolated_alpha[i]/255));metric('合成可见比例',fmt(m.visibility_ratio[i]/255));}"
s=s.replace(dup+dup,dup)
dup="isolated_alpha:'车身独立 alpha（覆盖率）',visibility_ratio:'合成贡献 / 独立 alpha（可见比例）',"
s=s.replace(dup+dup,dup)
s=s.replace("if($('boundaryTools'))$('boundaryTools').hidden=!window.V43_BADCASE_AVAILABLE;\n if($('boundaryTools'))$('boundaryTools').hidden=!window.V43_BADCASE_AVAILABLE;","if($('boundaryTools'))$('boundaryTools').hidden=!window.V43_BADCASE_AVAILABLE;")
p.write_text(s,encoding='utf8')
p=W/'scene_semantics.js';s=p.read_text(encoding='utf8')
s=s.replace("await ensureSemantic();draw3D();semanticLegend();","await ensureSemantic();S.selected=null;if(S.channel==='semantic_2d')drawSemanticImage();draw3D();semanticLegend();")
p.write_text(s,encoding='utf8')
subprocess.run([sys.executable,str(Path(__file__).with_name('patch_viewer.py'))],check=True)
print('Unknown causes and geometry-only ablation installed; GPU renders not run')
