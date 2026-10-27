"""Only new visual evidence. No models, no held-scene processing."""
import json,shutil,hashlib,tarfile,time
import numpy as np
from segmentation import ROOT,CONTEXTS,write
from export_diagnostics import pack

def main():
    ctx=CONTEXTS[0];dest=ROOT/'first_badcase_delivery_v1';dest.mkdir(exist_ok=False);boundary=ROOT/'boundary_semantics_v2'/ctx;b=json.loads((boundary/'summary.json').read_text());assert len(b['records'])==49
    target=dest/'boundary';target.mkdir();index=dict(version='cityscapes19-contour-abstention-v3',frames={},radii=[3,5,8],default_px=5,default_fraction=.2,branch_sizes=b['branch_sizes'],original_owner_unchanged=True,evidence_source='Real observed RGB, original/no-edit V43 alpha*T; labels carried with row into edits, not counterfactual ground truth')
    for r in b['records']:
        raw=r['raw'];im=np.load(boundary/('image_%03d.npz'%raw));parts={v:pack({k:np.load(boundary/('%s_%03d.npz'%(v,raw)))[k] for k in ['ids','class_id','mass','purity','target_fraction','vehicle_fraction','boundary_fraction']}) for v in ['original','actor10k']};parts['image']=pack(dict(semantic_image=im['semantic_image'],bands=im['bands'].astype('uint8')))
        filename='boundary_%03d.js'%raw;(target/filename).write_text('window.BOUNDARY_FRAMES=window.BOUNDARY_FRAMES||{};window.BOUNDARY_FRAMES['+str(raw)+']='+json.dumps(parts,separators=(',',':'))+';');index['frames'][str(raw)]='gpu_semantics/'+filename
    (target/'boundary_index.js').write_text('window.BOUNDARY_INDEX='+json.dumps(index,separators=(',',':'))+';');write(dest/'boundary_summary.json',b)
    diag=ROOT/'first_badcase_diagnostics_v1';fade=ROOT/'first_fade_control_v1';ghost=ROOT/'first_static_ghost_sweep_v1'
    for source,name in [(diag,'diagnostics'),(fade,'fade'),(ghost,'ghost')]:
        assert json.loads((source/'status.json').read_text())['phase']=='complete';write(dest/(name+'_summary.json'),json.loads((source/'summary.json').read_text()))
    for case in ['no_edit','translate_lateral_1m','turn_left15','translate_lateral_3m']:
        source=diag/case;target=dest/'diagnostics'/case;target.mkdir(parents=True)
        if case.endswith('3m'):shutil.copytree(source/'native_rgb',target/'native_rgb')
        for name in ['actor_rgb','isolated_alpha','visible_weight','visibility_ratio']:
            if (source/name).exists():shutil.copytree(source/name,target/name)
        for p in source.glob('*_maps.npz'):
            raw=int(p.name.split('_')[0]);d=np.load(p);arrays={k:(d[k]*255).clip(0,255).round().astype('uint8') if k!='depth_m' else d[k] for k in d.files};path='evidence/v43_badcase_maps/'+case+'/'+str(raw)+'.js';(target/('%03d_maps.js'%raw)).write_text('window.EVIDENCE_MAPS['+json.dumps(path)+']='+json.dumps(pack(arrays),separators=(',',':'))+';')
    shutil.copytree(fade/'density2_turn_left15',dest/'density2_turn_left15')
    for p in fade.glob('*_maps.npz'):
        raw=int(p.name.split('_')[0]);d=np.load(p);arrays={k:(d[k]*255).clip(0,255).round().astype('uint8') if k!='depth_m' else d[k] for k in d.files};path='evidence/v43_badcase_maps/density2/'+str(raw)+'.js';(dest/'density2_turn_left15'/('%03d_maps.js'%raw)).write_text('window.EVIDENCE_MAPS['+json.dumps(path)+']='+json.dumps(pack(arrays),separators=(',',':'))+';')
    for probe in ['affinity_50','affinity_20']:
        shutil.copytree(ghost/probe,dest/'ghost'/probe);write(dest/'ghost'/(probe+'_rows.json'),np.load(ghost/(probe+'_rows.npy')).tolist())
    for case in ['delete_target','translate_lateral_1m','turn_left15']:
        source=ROOT/'actor_motion_v1'/ctx/case/'propainter';d=json.loads((source/'summary.json').read_text());assert d['status']=='complete' and d['frames']==31;shutil.copytree(source/'model_png',dest/'backend'/case/'generated');write(dest/'backend'/case/'summary.json',d)
    files={str(p.relative_to(dest)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in dest.rglob('*') if p.is_file()};write(dest/'manifest.json',files);archive=ROOT/'first_badcase_delivery_v1.tar.gz'
    with tarfile.open(archive,'w:gz') as t:t.add(dest,arcname=dest.name)
    write(ROOT/'first_badcase_delivery_v1_receipt.json',dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(files),models_exported=False));print(json.dumps(dict(files=len(files),bytes=archive.stat().st_size)))
if __name__=='__main__':main()
