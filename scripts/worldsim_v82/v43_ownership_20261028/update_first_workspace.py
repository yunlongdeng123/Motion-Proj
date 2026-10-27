"""Make first-scene structural questions directly reviewable in its existing HTML."""
from pathlib import Path
B=Path(__file__).resolve().parents[2];W=B/'work/persistent_world_20261027';O=B/'outputs/GS_Scene_Diagnostics_20261027'
p=W/'scene_viewer.html';s=p.read_text(encoding='utf8')
s=s.replace('<span class="badge">研究任务已暂停</span>','<span class="badge">V43 GPU 实验 · 逐组更新</span>')
if 'id="v43Focus"' not in s:
    marker='<div class="case-row">'
    panel='''<section id="v43Focus" class="notice" hidden><strong>V43 问题对照：车体结构 / 静态残影 / 语义归属</strong><p>新车身使用实测点与刚体轨迹；原 30K 静态和环境冻结。先比较同帧的旧/新车体，再检查删除后的静态残影。右侧显示当前组的真实高斯几何，无 RGB 底图；车框与相机随同一请求变化。</p><div id="v43Actions" class="subtools"></div><p id="v43FocusStatus"></p></section>'''
    s=s.replace(marker,panel+marker,1)
p.write_text(s,encoding='utf8')
p=W/'scene_viewer.html';s=p.read_text(encoding='utf8')
s=s.replace('新车身使用实测点与刚体轨迹；原 30K 静态和环境冻结。','本组另建实测点＋刚体轨迹的局部车身，原百万行资产保留作对照；原 30K 静态和环境冻结。')
s=s.replace('v43-semantic-gpu-2','v43-first-structure-3')
s=s.replace('src="v43_groups.js"','src="v43_groups.js?v=v43-first-structure-3"')
if 'first_structural_contact.jpg' not in s:
    s=s.replace('<p id="v43FocusStatus"></p>','<p id="v43FocusStatus"></p><details><summary>查看 raw91 / 109 / 114 的旧新原生 RGB 排查图</summary><p><a href="../../V43_Ownership_20261027/gpu_v43/first_structural_contact.jpg" target="_blank">打开完整对照图 →</a>（场景1）</p></details>')
p.write_text(s,encoding='utf8')
p=W/'scene_viewer.js';s=p.read_text(encoding='utf8')
s=s.replace("['benchmark','motion','continuous'].includes(selected().category)","['benchmark','motion','continuous','v43'].includes(selected().category)")
if 'function initV43Focus()' not in s:
    fn='''function initV43Focus(){
 const panel=$('v43Focus');if(!panel||!allGroups.v43_actor_10000)return;panel.hidden=false;
 const actions=[['新车身 10K','v43_actor_10000','rgb'],['旧资产 30K','reconstruction_30000','rgb'],['新车身独立','v43_actor_10000','dynamic'],['新转向','v43_motion_turn_left15','rgb'],['旧转向','motion_turn_left15','rgb'],['新横移','v43_motion_translate_lateral_1m','rgb'],['旧横移','motion_translate_lateral_1m','rgb'],['删除后残影','v43_motion_delete_target','rgb'],['原静态车辆支持','reconstruction_30000','rgb','semantic_conflict']];
 for(const [label,id,channel,color] of actions){if(!allGroups[id])continue;const button=el('button',{},label);button.onclick=async()=>{S.channel=channel;S.cameraView=true;S.cameraReference='current';$('geometryState').value='current';$('static3d').checked=channel!=='dynamic';$('dynamic3d').checked=true;$('cloudColor').value=color||'branch';setCategory(allGroups[id].category,id);const valid=Object.keys(allGroups[id].frames).map(Number);if(!valid.includes(S.raw))S.raw=valid.reduce((a,b)=>Math.abs(a-S.raw)<Math.abs(b-S.raw)?a:b);await show();if(color)await enableSemantics();};$('v43Actions').append(button);}
 $('v43FocusStatus').textContent='初步目视对照：raw109/114 转向时，新车身比旧资产集中，旧长条拖散减轻；原位静态车影仍明显，车辆未见表面和外观仍有缺陷。语义提示是候选证据，尚未据此重划静态归属。1K / 5K / 10K 各49帧，原生编辑各31帧；请继续逐帧核验。';
}
'''
    s=s.replace('async function init(){',fn+'async function init(){',1)
    s=s.replace('createComparisons();trainTable();editingCompare();','createComparisons();trainTable();editingCompare();initV43Focus();',1)
s=s.replace("const met=f?.metrics?.full;$('frameMetrics').textContent=met?", "const met=f?.metrics?.full;$('frameMetrics').textContent=g.category==='v43'?'V43 重点核对：车身空间结构、编辑后是否拉散，以及原静态层的残影。像素分数保存在执行记录中。':met?")
s=s.replace("if(params.get('channel')==='sensor_overlay'&&IA){S.channel='sensor_overlay';$('channel').value=S.channel;}","if(params.has('channel')&&[...$('channel').options].some(o=>o.value===params.get('channel'))){S.channel=params.get('channel');$('channel').value=S.channel;}")
s=s.replace('零训练更新、远端无大型打包写入。','原 V42 只读导出与 V43 新实验分组记录。')
s=s.replace('已完成：1K / 5K / 10K 新车身（各49帧）；已交付的新编辑在按钮和实验组中显示。结构集中与刚体稳定性已有对照，静态残影尚未修复；语义提示仍是证据候选，不直接改写原归属。','初步目视：raw109/114 转向时，新车身比旧资产集中，旧长条拖散减轻；原位静态车影仍明显，车辆侧面仍有缺陷。语义提示是候选证据，尚未据此重划静态归属。1K / 5K / 10K 各49帧，原生编辑各31帧；请继续逐帧核验。')
p.write_text(s,encoding='utf8')
p=W/'scene_edit_geometry.js';s=p.read_text(encoding='utf8').replace('30K 几何加载量：','当前组几何加载量：');p.write_text(s,encoding='utf8')
p=O/'index.html';s=p.read_text(encoding='utf8');ctx='14705303724557273004_3105_000_3125_000'
s=s.replace('研究任务暂停 · 质量待人审','V43 GPU 推进中 · 质量待人审')
s=s.replace('<h2>场景 1 · 30K 完整</h2>','<h2>场景 1 · V43 10K 已更新</h2>')
s=s.replace(ctx+'/index.html#workspace',ctx+'/index.html#workspace?group=v43_actor_10000&raw=109')
s=s.replace('此次只读导出，不恢复训练或定时任务。','原 V42 结果保留；V43 新车身、原生编辑与固定后端按实际完成状态接入。旧定时任务未恢复。')
s=s.replace('同场景共用一个工作台</p>','同场景共用一个工作台</p>',1)
# A compact entry stays visible; existing scene cards and their baseline thumbnails are retained.
if 'id="v43FirstUpdate"' not in s:
    s=s.replace('<main id="workspace">','<main id="workspace"><section class="panel" id="v43FirstUpdate"><h2>第一场景 · V43 结构实验</h2><p>已接入新车身 1K / 5K / 10K 与实际完成的删除、横移、转向。点击进入后，左侧为对应 RGB，右侧为同组高斯几何；拖动帧轴同步查看，顶部按钮切换旧/新资产与静态残影。</p><a class="primary" href="'+ctx+'/index.html#workspace?group=v43_actor_10000&raw=109">查看第一场景 V43 10K →</a></section>',1)
p.write_text(s,encoding='utf8')
print('First-scene entry and shared template updated')
