"""Current-case integration: original bypass, new controls, uncertainty and real backend."""
from pathlib import Path
import json,shutil,base64,copy
import numpy as np
B=Path(__file__).resolve().parents[2];O=B/'outputs/V43_Ownership_20261027/gpu_v43';D=O/'first_badcase_delivery_v1';S=B/'outputs/GS_Scene_Diagnostics_20261027/14705303724557273004_3105_000_3125_000'
assert json.loads((D/'delivery_validation.json').read_text())['archive_and_files_verified'],'Finish and verify inbound delivery before installation'
def readjs(p):
    s=p.read_text(encoding='utf8');return json.loads(s[s.index('=')+1:].rstrip(';'))
def copywrap(source,target):
    shutil.copytree(source,target,dirs_exist_ok=True)
    for p in target.rglob('*.png'):
        rel=p.relative_to(S).as_posix();p.with_name(p.name+'.js').write_text('window.IMAGE_DATA['+json.dumps(rel)+']='+json.dumps('data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode())+';',encoding='utf8')
v=json.loads((S/'v43_groups_data.json').read_text());G=readjs(S/'edit_geometry.js');scene=readjs(S/'scene.js');groups={g['id']:g for g in v['groups']};diag=json.loads((D/'diagnostics_summary.json').read_text());fade=json.loads((D/'fade_summary.json').read_text());ghost=json.loads((D/'ghost_summary.json').read_text());L=np.array(G['local_from_world']);poses=json.loads((S/'assets/v43/10000/poses_world.json').read_text())
shutil.copytree(D/'boundary',S/'gpu_semantics',dirs_exist_ok=True);v['badcase_available']=True
def put(group,geom,receipt):
    ids={g['id'] for g in v['groups']};v['groups']=[g for g in v['groups'] if g['id']!=group['id']]+[group];v['geometry'][group['id']]=geom;(S/'evidence'/(group['id']+'_metrics.json')).write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
base=copy.deepcopy(next(g for g in scene['groups'] if g['id']=='reconstruction_30000'));base.update(id='v43_main_noedit',title='V43 默认无编辑 · V1 原生通路',category='v43',kind='v43_bypass',summary='evidence/v43_main_noedit_metrics.json')
put(base,copy.deepcopy(G['groups']['reconstruction_30000']),dict(route='Direct original V1 native branch; no new local actor substitution',actual_native_float_max_difference=fade['no_edit_V1_routing_max_difference'],source='Original30K saved images, no reencoding'))
for case in ['no_edit','translate_lateral_1m','turn_left15']:
    gid='v43_motion_'+case;group=groups[gid];copywrap(D/'diagnostics'/case,S/'assets/v43_diagnostics'/case)
    for raw,f in group['frames'].items():
        src=S/'assets/v43_diagnostics'/case/('%03d_maps.js'%int(raw));target=S/'evidence/v43_badcase_maps'/case/(raw+'.js');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target);f['maps']=target.relative_to(S).as_posix()
        for ch,sub in [('dynamic','actor_rgb')]:
            p=S/'assets/v43_diagnostics'/case/sub/('%03d.png'%int(raw))
            if p.exists():f['channels'][ch]=p.relative_to(S).as_posix()
    put(group,v['geometry'][gid],dict(diagnostics=diag,case=case))
for case in ['delete_target','translate_lateral_1m','turn_left15']:
    copywrap(D/'backend'/case/'generated',S/'assets/v43_motion'/case/'generated');group=next(g for g in v['groups'] if g['id']=='v43_motion_'+case)
    for raw,f in group['frames'].items():f['channels']['generated']='assets/v43_motion/'+case+'/generated/%03d.png'%int(raw)
    (S/'evidence'/('v43_backend_'+case+'.json')).write_text((D/'backend'/case/'summary.json').read_text());group['backend']='evidence/v43_backend_'+case+'.json'
# Three-meter displacement, using the exact native requested center and same tracked box.
case='translate_lateral_3m';copywrap(D/'diagnostics'/case,S/'assets/v43_diagnostics'/case);frames={};geom={}
for r in [r for r in diag['records'] if r['case']==case]:
    raw=str(r['raw']);q=copy.deepcopy(G['groups']['motion_no_edit'][raw]);pose=np.array(poses[raw]);delta=L[:3,:3]@(np.array(r['requested_center_world'])-pose[:3,3]);pose[:3,:3]=np.array(r['requested_rotation_world'])@pose[:3,:3];pose[:3,3]=r['requested_center_world'];obj=q['objects'][0];obj['cloud']='v43:10000';obj['transform']=(L@pose).tolist();obj['center_local']=(np.array(obj['center_local'])+delta).tolist();obj['corners_local']=(np.array(obj['corners_local'])+delta).tolist();obj['label']='V43 SHIFT 3m';q['note']='只读较大偏移压力对照；未补道路、未作道路支撑或无碰撞认证。';geom[raw]=q
    mp=S/'evidence/v43_badcase_maps'/case/(raw+'.js');mp.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/'assets/v43_diagnostics'/case/('%03d_maps.js'%int(raw)),mp);frames[raw]=dict(channels=dict(rgb='assets/v43_diagnostics/'+case+'/native_rgb/%03d.png'%int(raw)),maps=mp.relative_to(S).as_posix())
put(dict(id='v43_shift_3m',title='V43 10K · 横移3米（压力对照）',category='v43',kind='v43_control',frames=frames,summary='evidence/v43_shift_3m_metrics.json'),geom,dict(diagnostics=diag,case=case,not_new_benchmark_case=True))
# Optical-density-only control: the web opacity matches the changed rendered opacity.
copywrap(D/'density2_turn_left15',S/'assets/v43_density2');geom=copy.deepcopy(v['geometry']['v43_motion_turn_left15']);frames={};cloud=copy.deepcopy(v['clouds']['v43:10000']);cloud['opacity_density_scale']=2;v['clouds']['v43:density2']=cloud
for raw,q in geom.items():
    q['objects'][0]['cloud']='v43:density2';q['note']='仅光学密度翻倍的诊断，未写回模型；不作为已接受修复。';mp=S/'evidence/v43_badcase_maps/density2'/(raw+'.js');mp.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(S/'assets/v43_density2'/('%03d_maps.js'%int(raw)),mp);frames[raw]=dict(channels=dict(rgb='assets/v43_density2/native_rgb/%03d.png'%int(raw)),maps=mp.relative_to(S).as_posix())
put(dict(id='v43_density2_turn',title='诊断 · 转向仅密度×2（未采纳）',category='v43',kind='v43_control',frames=frames,summary='evidence/v43_density2_turn_metrics.json'),geom,fade)
for probe in ghost['probes']:
    name=probe['name'];copywrap(D/'ghost'/name,S/'assets/v43_static_probe'/name);ids=json.loads((D/'ghost'/(name+'_rows.json')).read_text());geom=copy.deepcopy(G['groups']['motion_delete_target']);frames={}
    for raw,q in geom.items():q['static_suppressed_rows']=ids;q['note']='静态车影亲和候选的整行抑制诊断；归属未确认，可能损伤真实背景；未补全。';frames[raw]=dict(channels=dict(rgb='assets/v43_static_probe/'+name+'/native_rgb/%03d.png'%int(raw)))
    key='v43_static_'+name;put(dict(id=key,title='诊断 · 静态车影候选抑制 '+name+'（未采纳）',category='v43',kind='v43_control',frames=frames,summary='evidence/'+key+'_metrics.json'),geom,dict(probe=probe,sweep=ghost['sweep']))
(S/'v43_groups_data.json').write_text(json.dumps(v,separators=(',',':')),encoding='utf8');(S/'v43_groups.js').write_text('window.V43_BADCASE_AVAILABLE=true;window.V43_DELIVERY='+json.dumps(v,separators=(',',':'))+';window.SCENE_DATA.groups.push(...window.V43_DELIVERY.groups);Object.assign(window.EDIT_GEOMETRY.groups,window.V43_DELIVERY.geometry);Object.assign(window.FOREIGN_CLOUDS,window.V43_DELIVERY.clouds);',encoding='utf8')
print(json.dumps(dict(groups=len(v['groups']),frames=sum(len(g['frames']) for g in v['groups']),unknown_band_available=True,backend_groups=3,second_scene_touched=False)))
