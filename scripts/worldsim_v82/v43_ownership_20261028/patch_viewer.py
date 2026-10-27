"""Add GPU evidence to the same template, preserving all previous experiments."""
from pathlib import Path
import json
B=Path(__file__).resolve().parents[2];W=B/'work/persistent_world_20261027';O=B/'outputs/GS_Scene_Diagnostics_20261027'
p=W/'scene_viewer.js';s=p.read_text(encoding='utf8')
if "semantic_2d:'GPU" not in s:
    s=s.replace("const channels={rgb:","const channels={semantic_2d:'GPU 二维语义预测（原始观测）',rgb:")
    s=s.replace("const available=new Set(['rgb','gt']);","const available=new Set(['rgb','gt','semantic_2d']);")
    s=s.replace("if(S.channel==='sensor_overlay'){drawSensorOverlay();","if(S.channel==='semantic_2d')drawSemanticImage();if(S.channel==='sensor_overlay'){drawSensorOverlay();")
    s=s.replace("$('cloudColor').value='semantic_branch';enableSemantics();","$('cloudColor').value='semantic_branch';$('semanticProfile').value='gpu';enableSemantics();")
    s=s.replace("[['summary','原生执行凭据']","[['summary','原生执行凭据']")
    p.write_text(s,encoding='utf8')
s=p.read_text(encoding='utf8')
s=s.replace("metric('当前显示 RGB · PNG8',rgb?rgb.join(' / '):'—');","metric(S.channel==='semantic_2d'?'原始实拍 RGB · PNG8':'当前显示 RGB · PNG8',rgb?rgb.join(' / '):'—');if(S.channel==='semantic_2d'){const a=S.gpuSemanticFrames?.get(S.raw),k=a?.semantic_image[i];metric('二维语义预测',k===undefined?'未加载':k+' '+GPU_SEMANTIC_NAMES[k]);metric('二维预测概率',a?fmt(a.confidence_image[i]):'未加载');}")
p.write_text(s,encoding='utf8')
s=p.read_text(encoding='utf8').replace("const categories={reconstruction:","const categories={v43:'V43 身份与结构实验',reconstruction:")
s=s.replace("g.category==='continuous'?'evidence/continuous_summary.json':'evidence/motion_summary.json'","g.category==='continuous'?'evidence/continuous_summary.json':g.category==='v43'?'evidence/'+g.id+'_metrics.json':'evidence/motion_summary.json'")
p.write_text(s,encoding='utf8')
p=W/'scene_viewer.html';s=p.read_text(encoding='utf8')
if 'value="gpu"' not in s:
    s=s.replace('<select id="semanticProfile">','<select id="semanticProfile"><option value="gpu" selected>GPU 完整可见贡献 × 19类语义</option>')
    s=s.replace('<option value="depth" selected>','<option value="depth">')
    start=s.index('<div id="semanticStatus"');end=s.index('</div>',start)+6
    s=s[:start]+'<div id="semanticStatus" class="notice">GPU 已完成三场景147帧语义与完整高斯 alpha*T 贡献归因。点击加载后按帧显示；可在二维通道查看原始语义预测图。GPU map v2：0 unknown，1 road，2 sidewalk，3 building，4 wall，5 fence，6 pole，7 traffic light，8 traffic sign，9 vegetation，10 terrain，11 sky，12 person，13 rider，14 car，15 truck，16 bus，17 train，18 motorcycle，19 bicycle。与旧 CPU map 独立，预测不是GT。<a href="../../V43_Ownership_20261027/gpu_v43/index.html">GPU实验与诊断 →</a></div>'+s[end:]
    s=s.replace('v43-semantic-1','v43-semantic-gpu-2')
    p.write_text(s,encoding='utf8')
s=p.read_text(encoding='utf8')
if 'src="v43_groups.js' not in s:s=s.replace('<script src="../viewer.js','<script src="v43_groups.js"></script><script src="../viewer.js')
if 'src="v43_groups.js?' in s:s=s.replace('<script src="v43_groups.js"></script>','')
s=s.replace('零训练更新 · 本地离线查看','原30K与V43实验分别记录 · 本地离线查看')
p.write_text(s,encoding='utf8')
for name in ['scene_viewer.js','scene_edit_geometry.js']:
    p=W/name;s=p.read_text(encoding='utf8');s=s.replace("['gt','sensor_overlay']","['gt','sensor_overlay','semantic_2d']");p.write_text(s,encoding='utf8')
viewer=(W/'scene_viewer.js').read_text(encoding='utf8')
for marker,name in [('EDIT_GEOMETRY','scene_edit_geometry.js'),('SAMPLING','scene_sampling.js'),('SEMANTIC','scene_semantics.js')]:viewer=viewer.replace('/* '+marker+'_RUNTIME */',(W/name).read_text(encoding='utf8'))
(O/'viewer.js').write_text(viewer,encoding='utf8')
for scene in json.loads((O/'scenes.json').read_text(encoding='utf8')):
    (O/scene['context']/'index.html').write_text((W/'scene_viewer.html').read_text(encoding='utf8'),encoding='utf8')
    if not (O/scene['context']/'v43_groups.js').exists():(O/scene['context']/'v43_groups.js').write_text('// V43 nodes pending; never substitute smoke for formal.\n')
print('GPU viewer patched; all prior groups retained')
