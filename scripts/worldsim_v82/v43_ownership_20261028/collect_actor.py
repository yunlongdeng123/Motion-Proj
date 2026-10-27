from pathlib import Path
import subprocess,json,hashlib,tarfile,shutil,base64,sys
import numpy as np
B=Path(__file__).resolve().parents[2];O=B/'outputs/V43_Ownership_20261027/gpu_v43';R='/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028';name=sys.argv[1]
receipt=json.loads(subprocess.check_output(['ssh','wm-vgpu-1011','cat '+R+'/'+name+'_receipt.json']));archive=O/(name+'.tar.gz');tmp=archive.with_suffix('.download')
if not archive.exists():
    subprocess.run(['scp','wm-vgpu-1011:'+receipt['archive'],str(tmp)],check=True);assert tmp.stat().st_size==receipt['bytes'] and hashlib.sha256(tmp.read_bytes()).hexdigest()==receipt['sha256'];tmp.replace(archive)
with tarfile.open(archive) as t:
    for m in t.getmembers():assert (O/m.name).resolve().is_relative_to(O.resolve()) and (m.isfile() or m.isdir()),m.name
    t.extractall(O)
root=O/name
for rel,m in json.loads((root/'manifest.json').read_text()).items():
    p=root/rel;assert p.stat().st_size==m['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256']
for item in json.loads((root/'index.json').read_text()):
    ctx=item['context'];scene=B/'outputs/GS_Scene_Diagnostics_20261027'/ctx;groups=[];geometries={};clouds={}
    previous=scene/'v43_groups_data.json';old=json.loads(previous.read_text()) if previous.exists() else dict(groups=[],geometry={},clouds={})
    for node in item['nodes']:
        step=node['step'];key='v43_actor_'+str(step);source=root/ctx/('%06d'%step);target=scene/'assets/v43'/str(step);target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(source,target,dirs_exist_ok=True)
        for p in target.rglob('*.png'):
            rel=p.relative_to(scene).as_posix();p.with_name(p.name+'.js').write_text('window.IMAGE_DATA['+json.dumps(rel)+']='+json.dumps('data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode())+';')
        original=(scene/'edit_geometry.js').read_text(encoding='utf8');G=json.loads(original[original.index('=')+1:].rstrip(';'));L=np.array(G['local_from_world']);poses=json.loads((source/'poses_world.json').read_text(encoding='utf8'));gs={}
        for raw,state in G['groups']['reconstruction_30000'].items():
            q=json.loads(json.dumps(state));q['note']='V43 新Actor局部结构组；原静态背景未修复；与左侧同一checkpoint/时刻。';q['supported']=True
            for obj in q['objects']:obj['cloud']='v43:'+str(step);obj['transform']=(L@np.array(poses[raw])).tolist();obj['label']='V43 TARGET'
            gs[raw]=q
        records={str(f['raw']):dict(channels=dict(rgb='assets/v43/'+str(step)+'/rgb/%03d.png'%f['raw'],dynamic='assets/v43/'+str(step)+'/actor/%03d.png'%f['raw']),metrics=dict(full=dict(composite_psnr_db=f['psnr']),actor=dict(composite_psnr_db=f['target_psnr']))) for f in node['summary']['frames']}
        evidence=scene/'evidence'/(key+'_metrics.json');evidence.write_text(json.dumps(node,ensure_ascii=False,indent=2))
        groups.append(dict(id=key,title='V43 局部车身约束 · '+str(step//1000)+'K',category='v43',kind='v43_structure',frames=records,summary='evidence/'+key+'_metrics.json'));geometries[key]=gs;clouds['v43:'+str(step)]=json.loads((source/'cloud.json').read_text())
    ids={g['id'] for g in groups};old['groups']=[g for g in old['groups'] if g['id'] not in ids]+groups;old['geometry'].update(geometries);old['clouds'].update(clouds);previous.write_text(json.dumps(old,separators=(',',':')))
    script='window.V43_BADCASE_AVAILABLE='+json.dumps(old.get('badcase_available',False))+';window.V43_DELIVERY='+json.dumps(old,separators=(',',':'))+';window.SCENE_DATA.groups.push(...window.V43_DELIVERY.groups);Object.assign(window.EDIT_GEOMETRY.groups,window.V43_DELIVERY.geometry);Object.assign(window.FOREIGN_CLOUDS,window.V43_DELIVERY.clouds);'
    (scene/'v43_groups.js').write_text(script,encoding='utf8')
print(json.dumps(dict(delivery=name,archive_SHA_verified=True,models_downloaded=False,scenes=[dict(context=x['context'],nodes=[n['step'] for n in x['nodes']]) for x in json.loads((root/'index.json').read_text())])))
