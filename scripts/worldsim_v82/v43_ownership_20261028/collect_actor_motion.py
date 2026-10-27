"""Install native+actual backend edits in the same case; exact same request geometry."""
from pathlib import Path
import subprocess,json,hashlib,tarfile,shutil,base64,sys
import numpy as np
B=Path(__file__).resolve().parents[2];O=B/'outputs/V43_Ownership_20261027/gpu_v43';R='/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028';name=sys.argv[1]
receipt=json.loads(subprocess.check_output(['ssh','wm-vgpu-1011','cat '+R+'/'+name+'_receipt.json']));archive=O/(name+'.tar.gz');tmp=archive.with_suffix('.download')
if not archive.exists():subprocess.run(['scp','wm-vgpu-1011:'+receipt['archive'],str(tmp)],check=True);tmp.replace(archive)
assert archive.stat().st_size==receipt['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
with tarfile.open(archive) as t:
    for m in t.getmembers():assert (O/m.name).resolve().is_relative_to(O.resolve()) and (m.isfile() or m.isdir()),m.name
    t.extractall(O)
root=O/name
for rel,m in json.loads((root/'manifest.json').read_text()).items():
    p=root/rel;assert p.stat().st_size==m['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256']
index=json.loads((root/'index.json').read_text());ctx=index['context'];scene=B/'outputs/GS_Scene_Diagnostics_20261027'/ctx
old=json.loads((scene/'v43_groups_data.json').read_text());summary=json.loads((root/ctx/'summary.json').read_text())
s=(scene/'edit_geometry.js').read_text(encoding='utf8');G=json.loads(s[s.index('=')+1:].rstrip(';'));L=np.array(G['local_from_world']);poses=json.loads((scene/'assets/v43/10000/poses_world.json').read_text())
titles=dict(no_edit='无编辑',delete_target='删目标车',translate_lateral_1m='横移 1 米',turn_left15='左转 15°')
rows=[];closure=0.
for item in index['cases']:
    case=item['case'];key='v43_motion_'+case;source=root/ctx/case;target=scene/'assets/v43_motion'/case;shutil.copytree(source,target,dirs_exist_ok=True)
    for p in target.rglob('*.png'):
        rel=p.relative_to(scene).as_posix();p.with_name(p.name+'.js').write_text('window.IMAGE_DATA['+json.dumps(rel)+']='+json.dumps('data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode())+';',encoding='utf8')
    geometry={};frames={};records={str(r['raw']):r for r in summary['records'] if r['case']==case}
    for raw,state in G['groups']['motion_'+case].items():
        q=json.loads(json.dumps(state));base=L@np.array(poses[raw]);r=records[raw]
        for obj in q['objects']:
            transform=np.array(obj['transform'])@base if obj['transform'] else base
            obj['transform']=transform.tolist();obj['cloud']='v43:10000';obj['label']='V43 TARGET'
            # Compare displayed Actor pose against the actual saved native request.
            expected=np.array(poses[raw]);expected[:3,:3]=np.array(r['requested_rotation_world'])@expected[:3,:3];expected[:3,3]=r['requested_center_world'];closure=max(closure,float(np.abs(transform-L@expected).max()))
        q['note']='V43 10K新车身；原静态/环境冻结。与左侧使用同一物理请求、相机和时刻。';geometry[raw]=q
        channels=dict(rgb='assets/v43_motion/'+case+'/native_rgb/%03d.png'%int(raw),mask='assets/v43_motion/'+case+'/mask/%03d.png'%int(raw))
        if item['generated']:channels['generated']='assets/v43_motion/'+case+'/generated/%03d.png'%int(raw)
        frames[raw]=dict(channels=channels)
    assert len(frames)==31 and closure<1e-7,(case,closure)
    evidence=dict(native_summary=summary,case=case,backend=json.loads((source/'backend_summary.json').read_text()) if item['generated'] else None,geometry_request_max_abs=closure)
    (scene/'evidence'/(key+'_metrics.json')).write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf8')
    group=dict(id=key,title='V43 10K · '+titles[case],category='v43',kind='v43_motion',frames=frames,summary='evidence/'+key+'_metrics.json')
    old['groups']=[g for g in old['groups'] if g['id']!=key]+[group];old['geometry'][key]=geometry;rows.append(dict(group=key,frames=31,generated=item['generated']))
(scene/'v43_groups_data.json').write_text(json.dumps(old,separators=(',',':')),encoding='utf8')
(scene/'v43_groups.js').write_text('window.V43_DELIVERY='+json.dumps(old,separators=(',',':'))+';window.SCENE_DATA.groups.push(...window.V43_DELIVERY.groups);Object.assign(window.EDIT_GEOMETRY.groups,window.V43_DELIVERY.geometry);Object.assign(window.FOREIGN_CLOUDS,window.V43_DELIVERY.clouds);',encoding='utf8')
result=dict(delivery=name,groups=rows,geometry_request_max_abs=closure,archive_and_all_files_sha_verified=True,models_downloaded=False)
(O/(name+'_installation.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result))
