"""用已保存结果构建离线逐帧工作台；不重渲染、不访问远端、不启动GPU。"""
from pathlib import Path
import json,subprocess,shutil,base64,gzip,hashlib,datetime
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
REPORT=BASE/'outputs/waymo_val029_experiment'
PLATFORM=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-22/ni-x-a/outputs/GS_Scene_Diagnostics_20261027')
CONTEXT='12820461091157089924_5202_916_5222_916'
KEYS=[130,155,160,190]
EXACT=[130,155,160,170,190,197]
def read(p):return json.loads(p.read_text(encoding='utf8'))
def dumpjs(p,name,data):p.write_text(f'window.{name}='+json.dumps(data,ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf8')
def blob(a):return base64.b64encode(gzip.compress(np.ascontiguousarray(a).tobytes(),compresslevel=5,mtime=0)).decode()
def main():
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');backup=HERE/'backups'/stamp;backup.mkdir(parents=True)
    for p,name in [(PLATFORM/'index.html','platform_index.html'),(PLATFORM/'scenes.json','scenes.json'),(REPORT/'index.html','experiment_index.html')]:shutil.copy2(p,backup/name)
    if not (REPORT/'report.html').exists():shutil.copy2(REPORT/'index.html',REPORT/'report.html')
    out=REPORT/'viewer';out.mkdir(exist_ok=True)
    for n in ['frames','exact','layers','evidence','observed']:(out/n).mkdir(exist_ok=True)
    # 无编辑直通与来源ID删除均使用既有原生10K的原PNG，不能从完整方法倒推。
    native_receipt=read(BASE/'work/val065_cpu/scene029_existing_native/receipt.json')
    import tarfile
    with tarfile.open(BASE/'work/val065_cpu/scene029_existing_native/native_keep_delete_109_197.tar') as tar:
        for row in native_receipt['files']:
            payload=tar.extractfile(row['member']).read()
            assert hashlib.sha256(payload).hexdigest()==row['sha256']
            path=out/'native10k'/row['member'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
    # GPU独立入口已补全：89帧V1，10帧来源ID删除；保留旧归属导出档案。
    completion=BASE/'work/val065_gpu/scene029_v1_completion.tar'
    if completion.exists():
        assert hashlib.sha256(completion.read_bytes()).hexdigest()=='787635854bf8a19bba7563c70eb0836c2eddfe9f0d2a4362babfe629780034ae'
        with tarfile.open(completion) as bundle:
            for raw in range(109,198):
                member=f'scene029_v1_all/{raw:03d}/keep.png';dst=out/'native10k'/str(raw)/'keep.png';dst.parent.mkdir(exist_ok=True);dst.write_bytes(bundle.extractfile(member).read())
            for raw in range(109,119):
                dst=out/'native10k'/str(raw)/'delete.png';dst.write_bytes(bundle.extractfile(f'scene029_native_held_delete/{raw:03d}/delete.png').read())
            for source in ['scene029_v1_parity','scene029_v1_all','scene029_native_held_delete']:
                (out/'evidence'/f'{source}_summary.json').write_bytes(bundle.extractfile(source+'/summary.json').read())
        native_receipt.update(status='PASS_PURE_V1_89_AND_SOURCE_ID_DELETE_89',direct_raws=list(range(109,198)),missing_raws=[],independent_v1_entry='render_v1.py; no edit package imported',v1_parity_four_raws_maxabs=0.0,updates_this_export=0,files=[dict(member=str(p.relative_to(out/'native10k')).replace(chr(92),'/'),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted((out/'native10k').glob('*/*.png'))])
        (out/'evidence/native_route_audit.json').write_text(json.dumps(native_receipt,ensure_ascii=False,indent=2),encoding='utf8')
    else:
        shutil.copy2(BASE/'work/val065_cpu/scene029_existing_native/receipt.json',out/'evidence/native_route_audit.json')
    existing=list((out/'frames').glob('*.png'))
    if len(existing)!=89:
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(REPORT/'assets/media/full198_comparison.mp4'),'-vf',r'select=gte(n\,109),crop=1920:320:0:28','-fps_mode','vfr','-start_number','109','-threads','2','-compression_level','4',str(out/'frames/%03d.png')],check=True)
    assert len(list((out/'frames').glob('*.png')))==89
    photos=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-31/g/outputs/V43_R4_GPU_20261031/actor_candidates/scenes')/CONTEXT/'frames'
    for raw in range(109,198):
        with Image.open(photos/f'{raw:06d}.jpg') as im:
            assert im.size==(960,640)
            im.convert('RGB').resize((480,320),Image.Resampling.BILINEAR).save(out/'observed'/f'{raw}.png')
    for raw in EXACT:
        dest=out/'exact'/str(raw);dest.mkdir(exist_ok=True)
        for stage,short in [('formal10k_eval','native'),('semantic_all_eval','semantic'),('semantic_asset_all_r2','surface'),('final_all_eval','final')]:
            for state in ['keep','delete']:
                shutil.copy2(REPORT/'assets/frames'/f'{raw:03d}'/stage/f'{state}.png',dest/f'{short}_{state}.png')
        shutil.copy2(REPORT/'assets/frames'/f'{raw:03d}/final_all_eval/observed.png',dest/'observed.png')
    T=np.loadtxt(HERE/'recovered/inputs/native/transform_matrix.txt');scale=float(np.linalg.det(T[:3,:3])**(1/3))
    anchor=read(BASE/'outputs/waymo_edit_case_selection/verified_correspondences.json');meta=read(BASE/'outputs/waymo_val029_preparation/evidence/media.json');byraw={f['raw']:f for f in meta['frames']}
    observation=read(BASE/'outputs/waymo_val029_preparation/evidence/inputs/observations/155.json');body=next(b for b in observation['boxes'] if b['track']=='DwjZr_bYUSZOr-XDGregnA');P=np.array(body['world_from_actor']).reshape(4,4);xyz=np.array([p['actor_local_xyz'] for p in anchor['pairs']]);world=xyz@P[:3,:3].T+P[:3,3];c={s.split(':')[0]:np.fromstring(s.split(':')[1],sep=' ') for s in (HERE/'recovered/cpu_inputs/inputs/calib/0029155.txt').read_text().splitlines()};C=np.eye(4);C[:3]=c['Tr_velo_to_cam_0'].reshape(3,4);C=C@np.linalg.inv(np.array(observation['ego_world']).reshape(4,4));camera=world@C[:3,:3].T+C[:3,3];K=c['P0'].reshape(3,4)[:,:3]/4;K[2,2]=1;uv=(camera@K.T)[:,:2]/camera[:,2:];pixels=np.floor(uv).astype(int)
    for raw in KEYS:
        packs={};N=480*320
        for state in ['keep','delete']:
            pack=np.load(REPORT/'evidence_package/key_float/final_all_eval'/f'{raw:03d}/{state}.npz');alpha=pack['alpha'];depth=pack['depth_sum']/np.maximum(alpha,1e-5)/scale;arrays=np.concatenate([pack['group_alpha'],alpha,depth,pack['rgb']],axis=0).astype('<f4');assert arrays.shape==(9,320,480);packs[state]=blob(arrays)
        labels=np.array(Image.open(REPORT/'evidence_package/masks'/f'{raw:03d}.png'),dtype=np.uint8);payload=dict(raw=raw,shape=[9,320,480],dtype='Float32LE',encoding='gzip-base64',keep=packs['keep'],delete=packs['delete'],labels=blob(labels))
        (out/'layers'/f'{raw}.js').write_text('window.VAL029_LAYER_PAYLOADS=window.VAL029_LAYER_PAYLOADS||{};window.VAL029_LAYER_PAYLOADS['+str(raw)+']='+json.dumps(payload,separators=(',',':'))+';\n',encoding='utf8')
    rows=read(REPORT/'evidence_package/evidence/final_all_eval/summary.json')['frames'];metrics={r['raw']:r for r in rows}
    stage_metrics={name:{r['raw']:r for r in read(REPORT/'evidence_package/evidence'/folder/'summary.json')['frames']} for name,folder in [('semantic','semantic_all_eval'),('surface','semantic_asset_all_r2'),('final','final_all_eval')]}
    roles=[dict(id=70,key='target_red',label='待删除红色轿车',track='lvxXvuSB3kueTyP2ignh7w',colour='#ffad52'),dict(id=25,key='revealed_white_suv',label='后方白色 SUV',track='DwjZr_bYUSZOr-XDGregnA',colour='#43d9ee'),dict(id=20,key='preserved_gray',label='保持的右侧灰车',track='AyRpKRmelCT4A709Lo8pvA',colour='#63dda4')]
    frames=[]
    for raw in range(109,198):
        f=byraw[raw];boxes=[]
        for role in roles:
            b=next((b for b in f['boxes'] if b['track']==role['track']),None)
            if b:boxes.append(dict(role=role['id'],visible=b['official_FRONT_visible'],corners=[None if p is None else [p[0]/2,p[1]/2] for p in b['corners2D']],center_world=b['center_world']))
        frames.append(dict(raw=raw,time_s=raw/10,role_boxes=boxes,holdout=79<=raw<=118,metrics=metrics.get(raw),stage_metrics={k:v.get(raw) for k,v in stage_metrics.items()},exact_png=raw in EXACT,float_layers=raw in KEYS))
    data=dict(schema='waymo.val029.viewer.v1',context=CONTEXT,scene='Waymo · scene-029',split='验证集',width=480,height=320,min_raw=109,max_raw=197,default_raw=155,fps_source=10,keys=KEYS,exact=EXACT,roles=roles,frames=frames,anchor155=dict(pixels=pixels.tolist(),depth_m=camera[:,2].tolist(),query_photo_uv=[[p['query_uv'][0]/2,p['query_uv'][1]/2] for p in anchor['pairs']],reference_photo_uv=[[p['reference_uv'][0]/2,p['reference_uv'][1]/2] for p in anchor['pairs']],scope='67个BUILD实测点，非独立隐藏真值'),attribution=read(REPORT/'reveal_audit.json'),summary=read(REPORT/'metrics.json'))
    dumpjs(out/'data.js','VAL029_DATA',data)
    asset=np.load(HERE/'recovered/revealed_material_r2/complete_asset.npz');obs=np.load(HERE/'recovered/revealed_prior/observations.npz');aidx=np.linspace(0,len(asset['xyz'])-1,min(6500,len(asset['xyz'])),dtype=int);oidx=np.linspace(0,len(obs['sample_xyz_local_m'])-1,min(5000,len(obs['sample_xyz_local_m'])),dtype=int)
    geometry=dict(coordinates='SUV局部坐标，米；不是整场景原生高斯',asset=dict(xyz=asset['xyz'][aidx].round(6).tolist(),rgb=asset['rgb'][aidx].round(5).tolist(),ids=aidx.tolist(),kind=asset['representation_kind'][aidx].tolist(),inferred_colour=asset['colour_inferred'][aidx].tolist()),measured=dict(xyz=obs['sample_xyz_local_m'][oidx].round(6).tolist(),ids=oidx.tolist(),source_raw=obs['sample_raw_frame'][oidx].tolist()),full_counts=dict(surface=len(asset['xyz']),measured=len(obs['sample_xyz_local_m'])))
    dumpjs(out/'geometry.js','VAL029_GEOMETRY',geometry)
    for name in ['measured_correspondence.png','architecture.svg']:shutil.copy2(REPORT/name,out/'evidence'/name)
    for name in ['viewer.js','viewer.css']:shutil.copy2(HERE/name,out/name)
    page=(HERE/'viewer.html').read_text(encoding='utf8');local=page.replace('@@STYLE@@','viewer/theme/style.css').replace('@@WORKSPACE@@','viewer/theme/workspace.css').replace('@@ROOT@@','viewer/').replace('@@NAV@@','<span>逐帧查看</span>')
    theme=out/'theme';theme.mkdir(exist_ok=True)
    for name in ['style.css','workspace.css']:shutil.copy2(PLATFORM/name,theme/name)
    (REPORT/'index.html').write_text(local,encoding='utf8')
    child=PLATFORM/CONTEXT
    if child.exists():shutil.copytree(child,backup/'previous_child')
    child.mkdir(exist_ok=True);shutil.copytree(out,child/'viewer',dirs_exist_ok=True)
    (child/'index.html').write_text(page.replace('@@STYLE@@','../style.css').replace('@@WORKSPACE@@','../workspace.css').replace('@@ROOT@@','viewer/').replace('@@NAV@@','<a href="../index.html">全部场景</a>'),encoding='utf8')
    p=PLATFORM/'index.html';text=p.read_text(encoding='utf8');card=f'<article class="panel" data-scene="{CONTEXT}"><h2>Waymo · scene-029</h2><details class="dataset-origin"><summary>验证集 · 原始路段编号</summary><code>{CONTEXT}</code></details><img src="{CONTEXT}/viewer/exact/155/final_delete.png" alt="路口车辆删除预览"><p>逐帧对照红车删除、SUV显露与灰车保持，检查像素贡献和表面证据。</p><a class="primary" href="{CONTEXT}/index.html">打开诊断 →</a></article>'
    if f'data-scene="{CONTEXT}"' not in text:
        assert '<div class="scene-cards">' in text;text=text.replace('<div class="scene-cards">','<div class="scene-cards">'+card,1);p.write_text(text,encoding='utf8')
    scenes=read(PLATFORM/'scenes.json')
    if not any(s['context']==CONTEXT for s in scenes):scenes.append(dict(context=CONTEXT,scene_key='waymo-val-029',display_name='Waymo · scene-029',dataset_split='validation',registered_scene_index=29,raws=list(range(109,198)),groups=4,entry=f'{CONTEXT}/index.html',note='登记号沿用候选库零起始清单，不是全局官方ID'));(PLATFORM/'scenes.json').write_text(json.dumps(scenes,ensure_ascii=False,indent=2),encoding='utf8')
    receipt=dict(status='BUILT_FROM_SAVED_OUTPUTS',child=str(child/'index.html'),local=str(REPORT/'index.html'),default_window=[109,197],source_seconds=[10.9,19.7],frames=89,float_frames=KEYS,original_png_frames=EXACT,continuous_frames_source='H264 full198 comparison decoded then lossless PNG; not original float imagery',GPU_started=False,neural_updates=0,rerender_calls=0,backup=str(backup),existing_viewer_runtime_changed=False)
    (HERE/'build_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
