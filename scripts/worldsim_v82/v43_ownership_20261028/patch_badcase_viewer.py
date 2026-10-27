"""Shared template upgrades for diagnostic uncertainty, visibility and V1 routing."""
from pathlib import Path
B=Path(__file__).resolve().parents[2];W=B/'work/persistent_world_20261027'
p=W/'scene_semantics.js';s=p.read_text(encoding='utf8')
s=s.replace("'#505968','#804080'","'#c8cdd4','#804080'").replace("'#2863ff'","'#2de5aa'")
s=s.replace("return e.branch==='static'?'#536d89':'#b6c0ce'","return '#c8cdd4'")
s=s.replace("return {class_id:14,class_name:'car / 已核验目标车身种子'","return {class_id:0,class_name:'unknown / 新资产逐行贡献语义尚未加载'")
if '/* BOUNDARY_RUNTIME */' not in s:s='/* BOUNDARY_RUNTIME */\n'+s
s=s.replace("function gpuSemanticRow(instance,j){","function gpuSemanticRow(instance,j){\n const boundary=boundarySemanticRow(instance,j);if(boundary)return boundary;",1) if 'const boundary=boundarySemanticRow' not in s else s
s=s.replace("await ensureGPUSemantic();return;","await ensureGPUSemantic();await ensureBoundarySemantic();return;")
s=s.replace("const color=GPU_SEMANTIC_COLORS[a.semantic_image[i]]","const band=S.boundaryFrames?.get(S.raw)?.image,rad=Number($('semanticBoundary')?.value||0),bi=[3,5,8].indexOf(rad),unknown=band&&bi>=0&&band.bands[bi*W*H+i];const color=GPU_SEMANTIC_COLORS[unknown?0:a.semantic_image[i]]")
s=s.replace("灰色 unknown。目标支持模式","浅灰 unknown；绿色 car，静态car为深绿、动态car为亮绿。同一高斯可以是car语义＋原static分支，不能把分支当语义。目标支持模式")
s=s.replace("若没有","若没有")
p.write_text(s,encoding='utf8')
p=W/'scene_viewer.html';s=p.read_text(encoding='utf8')
if 'id="semanticBoundary"' not in s:
    s=s.replace('<div id="semanticStatus"','<div class="subtools" id="boundaryTools"><label>车辆轮廓内外 unknown 带 <select id="semanticBoundary"><option value="0">关闭</option><option value="3">3 px</option><option value="5" selected>5 px</option><option value="8">8 px</option></select></label><label>边界贡献比例 <select id="semanticBoundaryMass"><option value="0.1">10%</option><option value="0.2" selected>20%</option><option value="0.5">50%</option></select></label><span class="note">单位为480×320实拍像素；控制语义暂缓分类，不裁点、不改owner。</span></div><div id="semanticStatus"',1)
s=s.replace('<div class="legend">蓝：静态高斯','<div id="geometryColorLegend" class="legend">蓝：静态高斯')
s=s.replace('v43-first-structure-3','v43-badcases-4')
p.write_text(s,encoding='utf8')
p=W/'scene_viewer.js';s=p.read_text(encoding='utf8')
s=s.replace("dynamic_weight:'合成中的动态贡献'","isolated_alpha:'车身独立 alpha（覆盖率）',visibility_ratio:'合成贡献 / 独立 alpha（可见比例）',dynamic_weight:'合成中的动态贡献'")
s=s.replace("['alpha','dynamic_weight','depth_m'].forEach","['alpha','dynamic_weight','depth_m','isolated_alpha','visibility_ratio'].forEach")
s=s.replace("if(f.maps)['alpha'","if(f.maps)['alpha'")
s=s.replace("['semanticProfile','semanticRadius','semanticDepth','semanticSigma'].forEach","['semanticProfile','semanticRadius','semanticDepth','semanticSigma','semanticBoundary','semanticBoundaryMass'].forEach")
s=s.replace("metric('合成动态贡献',m?fmt(m.dynamic_weight?.[i]/255):'未导出');","metric('合成动态贡献',m?fmt(m.dynamic_weight?.[i]/255):'未导出');if(m?.isolated_alpha){metric('车身独立 alpha',fmt(m.isolated_alpha[i]/255));metric('合成可见比例',fmt(m.visibility_ratio[i]/255));}")
if "['默认无编辑主通路'" not in s:s=s.replace("const actions=[[", "const actions=[['默认无编辑主通路','v43_main_noedit','rgb'],[",1)
if "['横移3米'" not in s:s=s.replace("['旧横移','motion_translate_lateral_1m','rgb'],", "['旧横移','motion_translate_lateral_1m','rgb'],['横移3米','v43_shift_3m','rgb'],['不透明度对照','v43_density2_turn','rgb'],['静态车影抑制对照','v43_static_affinity_20','rgb'],")
s=s.replace("function initV43Focus(){\n", "function initV43Focus(){\n if($('boundaryTools'))$('boundaryTools').hidden=!window.V43_BADCASE_AVAILABLE;\n")
p.write_text(s,encoding='utf8')
p=W/'scene_edit_geometry.js';s=p.read_text(encoding='utf8')
s=s.replace("transform:null,label:'STATIC'","transform:null,label:'STATIC',suppressed_rows:st.static_suppressed_rows||null")
if 'const suppressed=instance.suppressed_rows' not in s:s=s.replace("const {data}=cloud;let count=0;","const {data}=cloud;let count=0;const suppressed=instance.suppressed_rows?new Set(instance.suppressed_rows):null;")
if 'if(suppressed?.has' not in s:s=s.replace("for(let j=0;j<data.ids.length;j++){","for(let j=0;j<data.ids.length;j++){\n   if(suppressed?.has(data.ids[j]))continue;")
s=s.replace("Math.max(.25,pt.opacity)","Math.max(0,Math.min(1,pt.opacity))")
if 'if(cloud.opacity_density_scale)' not in s:s=s.replace("return pt;","if(cloud.opacity_density_scale)pt.opacity=1-Math.pow(1-pt.opacity,cloud.opacity_density_scale);return pt;")
if 'geometryColorLegend' not in s:
    s=s.replace("semanticLegend();const count=IA?.frames[S.raw].visible_count;","semanticLegend();if($('geometryColorLegend'))$('geometryColorLegend').textContent=mode.startsWith('semantic')?'语义：亮绿=car×dynamic，深绿=car×static，浅灰=unknown；紫=road。static是原渲染分支，不是背景语义。橙框=所选目标，蓝框=其他原始车辆。':mode==='ownership'?'归属候选：绿1，蓝0，橙unknown；原branch与语义分别保留。':'蓝=原static分支，橙=目标dynamic分支，青=原LiDAR；橙框目标，蓝框其他原始车辆。';const count=IA?.frames[S.raw].visible_count;")
p.write_text(s,encoding='utf8')
# The runtime is inserted once at build time, not dynamically fetched from a network.
p=B/'work/v43_gpu_20261028/patch_viewer.py';s=p.read_text(encoding='utf8')
if "scene_boundary_semantics.js" not in s:s=s.replace("viewer=viewer.replace('/* '+marker+'_RUNTIME */',(W/name).read_text(encoding='utf8'))","viewer=viewer.replace('/* '+marker+'_RUNTIME */',(W/name).read_text(encoding='utf8'))\nviewer=viewer.replace('/* BOUNDARY_RUNTIME */',(W/'scene_boundary_semantics.js').read_text(encoding='utf8'))")
p.write_text(s,encoding='utf8')
print('Badcase shared viewer source updated')
