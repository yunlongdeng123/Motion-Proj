"""Actual native RGB + actual static adapter/background geometry in existing UI."""
from pathlib import Path
import json,copy,subprocess,sys,argparse
import numpy as np
from install_gpu_review import unpack,pack,copy_images
B=Path(__file__).resolve().parents[2];O=B/'outputs/V43_Ownership_20261027/gpu_v43';S=B/'outputs/GS_Scene_Diagnostics_20261027/14705303724557273004_3105_000_3125_000';W=B/'work/persistent_world_20261027';D=O/'target_unknown_delivery_v1'

def readjs(p):
    s=p.read_text(encoding='utf8');return json.loads(s[s.index('=')+1:].rstrip(';'))

def main():
    global D
    p=argparse.ArgumentParser();p.add_argument('--conditional',action='store_true');args=p.parse_args()
    if args.conditional:D=O/'target_conditional_delivery_v1'
    assert json.loads((D/'delivery_validation.json').read_text())['archive_and_files_verified'];v=json.loads((S/'v43_groups_data.json').read_text());G=readjs(S/'edit_geometry.js');L=np.array(G['local_from_world']);v.setdefault('static_adapters',{})
    if (D/'target_evidence.json').exists():v['target_unknown_evidence']=json.loads((D/'target_evidence.json').read_text())
    supp=json.loads((D/'suppression.json').read_text()) if (D/'suppression.json').exists() else {};localmasks=json.loads((D/'time_local_masks.json').read_text()) if args.conditional else {};added=[]
    plans=[('conditional_joint','按时刻联合分解＋视频先验10K'),('conditional_split','按时刻固定剥离＋同视频先验10K'),('time_local75','按时刻unknown · 75%贡献纯度'),('time_local95','按时刻unknown · 95%贡献纯度')] if args.conditional else [('baseline','删动态资产基线'),('target_upper_bound','目标unknown全移除（上界诊断）'),('target_preserve_observed_bg','目标unknown剥离＋背景证据保护'),('joint199','199帧证据 · 联合分解10K'),('split199','199帧证据 · 固定剥离后修复10K'),('joint_dense199','199帧表面插值 · 联合分解10K')]
    for name,title in plans:
        rel=Path('assets/v43_target_unknown')/name;copy_images(D/name/'native_rgb',S/rel);frames={};geom={};adapter=None;bgkey=None
        if (D/name/'adapter.json').exists():
            adapter=name;v['static_adapters'][adapter]=json.loads((D/name/'adapter.json').read_text());cloud=json.loads((D/name/'cloud.json').read_text());data=unpack(cloud['data']);data['xyz']=(data['xyz'].astype(float)@L[:3,:3].T+L[:3,3]).astype('float32');cloud['data']=pack(data);cloud['coordinate_space']='local_meters';bgkey='v43:bg_'+name;v['clouds'][bgkey]=cloud
        for p in sorted((S/rel).glob('*.png')):
            raw=str(int(p.stem));q=copy.deepcopy(G['groups']['reconstruction_30000'][raw]);q['objects']=[];q['static_visible']=True;q.pop('static_suppressed_rows',None);q.pop('semantic_render_filter',None)
            original=G['groups']['motion_no_edit'].get(raw,G['groups']['reconstruction_30000'][raw]);target=next((o for o in original['objects'] if o['key']=='target'),None)
            if target:q['reference_boxes']=[dict(target=True,label='TARGET 原位参考',corners_local=target['corners_local'])]
            if name in supp:q['static_suppressed_rows']=supp[name]
            if name.startswith('time_local'):
                q['static_suppressed_rows']=localmasks['purity'+name.removeprefix('time_local')][raw];bgkey='v43:bg_joint_dense199';q['objects']=[dict(key='background_repair',cloud=bgkey,label='实证背景补层',query_time=q['world_time'],transform=None,render_only=True)]
            if adapter:
                q['static_adapter']=adapter;q['objects']=[dict(key='background_repair',cloud=bgkey,label='实证背景补层',query_time=q['world_time'],transform=None,render_only=True)]
            q['note']='目标专属unknown实验；橙框为原位参考，目标动态资产已删除。原static批次几何参数冻结，自带时变运动仍保留；保留率与DC为实际学习参数，背景补层使用真实导出几何。语义携带原观测证据，未在此新模型重新归因；保留率不是身份概率。尚未达到干净剥离验收。'+(' 使用原始观测的贡献历史按时刻门控；视频先验是生成结果，实测证据优先，最终图为原生高斯重渲染。' if name.startswith('conditional_') else '');geom[raw]=q;frames[raw]=dict(channels=dict(rgb=(rel/p.name).as_posix()))
        gid='v43_unknown_v2_'+name;receipt=json.loads((D/name/'summary.json').read_text()) if (D/name/'summary.json').exists() else json.loads((D/('time_local_summary.json' if args.conditional else 'probe_summary.json')).read_text());(S/'evidence'/(gid+'_metrics.json')).write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8');v['groups']=[g for g in v['groups'] if g['id']!=gid]+[dict(id=gid,title=title,category='v43',kind='v43_target_unknown',frames=frames,summary='evidence/'+gid+'_metrics.json')];v['geometry'][gid]=geom;added.append(gid)
    (S/'v43_groups_data.json').write_text(json.dumps(v,separators=(',',':')),encoding='utf8');(S/'v43_groups.js').write_text('window.V43_BADCASE_AVAILABLE=true;window.V43_DELIVERY='+json.dumps(v,separators=(',',':'))+';window.SCENE_DATA.groups.push(...window.V43_DELIVERY.groups);Object.assign(window.EDIT_GEOMETRY.groups,window.V43_DELIVERY.geometry);Object.assign(window.FOREIGN_CLOUDS,window.V43_DELIVERY.clouds);',encoding='utf8')
    p=W/'scene_edit_geometry.js';s=p.read_text(encoding='utf8');s=s.replace("suppressed_rows:st.static_suppressed_rows||null,suppressed_rows:st.static_suppressed_rows||null", "suppressed_rows:st.static_suppressed_rows||null,static_adapter:st.static_adapter||null")
    if '/* target-unknown-adapter */' not in s:
        s=s.replace(' if(cloud.opacity_density_scale)',""" /* target-unknown-adapter */
 const a=S.staticAdapters?.[instance.static_adapter],idx=a?sparseIndex(a.ids,cloud.data.ids[j]):-1;
 if(idx>=0){pt.opacity*=a.opacity_factor[idx];pt.rgb=[0,1,2].map(k=>Math.round(Math.max(0,Math.min(255,cloud.data.rgb[j*3+k]+255*a.dc_rgb_delta[idx*3+k]))));}
 if(cloud.opacity_density_scale)""",1)
        s=s.replace("return [...others,...st.objects.map", "return [...others,...(st.reference_boxes||[]),...st.objects.filter(o=>!o.render_only).map")
        s=s.replace("const part=instance.cloud==='static'?'static':'dynamic';", "const part=instance.cloud==='static'||S.clouds[instance.cloud]?.branch==='static'?'static':'dynamic';")
        s=s.replace("mode==='rgb'?`rgb(${data.rgb[j*3]},${data.rgb[j*3+1]},${data.rgb[j*3+2]})`", "mode==='rgb'?`rgb(${(pt.rgb||Array.from(data.rgb.slice(j*3,j*3+3))).join(',')})`")
        s=s.replace("x.fillStyle=mode.startsWith('semantic')?", "x.fillStyle=mode==='target_unknown_v2'?targetUnknownColor(instance,data.ids[j]):mode.startsWith('semantic')?")
        s=s.replace("if(mode==='ownership')$('ownershipStatus')", "if(mode==='target_unknown_v2')$('ownershipStatus').textContent='V2目标专属证据：橙=目标相关但未分清；紫=目标与真实背景混合；蓝=重复背景深度支持（未核验静止）；灰=此目标之外。青=新增实证背景补层（语义仍unknown）。渲染保留率不是身份概率。';if(mode==='ownership')$('ownershipStatus')")
    if "mode==='target_unknown_v2'?'V2颜色" not in s:
        s=s.replace("$('geometryColorLegend').textContent=mode===", "$('geometryColorLegend').textContent=mode==='target_unknown_v2'?'V2颜色：橙=目标相关未分清；紫=目标/背景混合；蓝=重复背景深度支持（未核验静止）；灰=此目标之外；青=新增背景补层。保留率不是身份概率。':mode===")
    if '/* target-unknown-selected */' not in s:
        s=s.replace("if(mode.startsWith('semantic'))$('gaussianInfo')", """/* target-unknown-selected */
  if(mode==='target_unknown_v2'){const a=S.targetUnknown,k=inst.cloud==='static'&&a?sparseIndex(a.ids,d.ids[j]):-1,ad=S.staticAdapters?.[inst.static_adapter],ai=ad?sparseIndex(ad.ids,d.ids[j]):-1;$('gaussianInfo').textContent+=k>=0?` V2证据状态 ${['背景深度支持（未核验静止）','车辆候选','目标unknown','混合支撑'][a.role[k]]}；目标支持 ${a.target_frames[k]} 帧，背景深度支持 ${a.background_frames[k]} 帧。`:' V2：此目标之外，或新增实证背景补层（语义仍unknown）。';if(ai>=0)$('gaussianInfo').textContent+=` 实际学习保留率 ${fmt(ad.opacity_factor[ai])}，不是身份概率。`;}
  if(mode.startsWith('semantic'))$('gaussianInfo')""")
    s=s.replace("pt.opacity*=a.opacity_factor[idx];pt.rgb=[0,1,2].map(k=>Math.round(Math.max(0,Math.min(255,cloud.data.rgb[j*3+k]+255*a.dc_rgb_delta[idx*3+k]))));", "const ti=a.raws?a.raws.indexOf(S.raw):-1,phi=a.time_phi&&ti>=0?a.time_phi[ti*a.ids.length+idx]:1;pt.opacity*=1-(1-a.opacity_factor[idx])*phi;pt.rgb=[0,1,2].map(k=>Math.round(Math.max(0,Math.min(255,cloud.data.rgb[j*3+k]+255*phi*a.dc_rgb_delta[idx*3+k]))));")
    p.write_text(s,encoding='utf8')
    p=W/'scene_viewer.js';s=p.read_text(encoding='utf8')
    if 'S.staticAdapters={}' not in s:
        s=s.replace('S.clouds=loaded;draw3D();',"S.clouds=loaded;S.staticAdapters={};for(const [k,a] of Object.entries(window.V43_DELIVERY?.static_adapters||{}))S.staticAdapters[k]=await unpack(a);if(window.V43_DELIVERY?.target_unknown_evidence)S.targetUnknown=await unpack(window.V43_DELIVERY.target_unknown_evidence);draw3D();")
    p.write_text(s,encoding='utf8')
    p=W/'scene_semantics.js';s=p.read_text(encoding='utf8')
    if 'function targetUnknownColor(' not in s:
        s+="""
function targetUnknownColor(instance,row){
 if(S.clouds[instance.cloud]?.row_namespace==='v43_measured_background')return '#55ddd5';
 if(instance.cloud!=='static')return '#35dc98';
 const a=S.targetUnknown,i=a?sparseIndex(a.ids,row):-1;return i<0?'#465568':['#559bdf','#35dc98','#ffad45','#df6bc7'][a.role[i]];
}
"""
        s=s.replace('function semanticRow(instance,j){',"function semanticRow(instance,j){\n if(S.clouds[instance.cloud]?.row_namespace==='v43_measured_background')return {class_id:0,branch:'static',map_version:'measured-world-background-v1',reason:'跨帧实证背景或受限表面插值补层；没有重新语义归因，保留unknown。'};")
    p.write_text(s,encoding='utf8')
    p=W/'scene_viewer.html';s=p.read_text(encoding='utf8');s=s.replace('v43-native-review-6','v43-target-unknown-8').replace('v43-target-unknown-7','v43-target-unknown-8')
    if 'value="target_unknown_v2"' not in s:s=s.replace('<option value="ownership">','<option value="target_unknown_v2">V2 目标专属unknown / 背景证据</option><option value="ownership">')
    p.write_text(s,encoding='utf8');subprocess.run([sys.executable,str(Path(__file__).with_name('patch_viewer.py'))],check=True);print(json.dumps(dict(installed=added,all_prior_groups_retained=True)))

if __name__=='__main__':main()
