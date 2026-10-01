"""仅复用已有显式GLB作为多视角遮挡轮廓，不生成GT、不调用资产模型。"""
from pathlib import Path
import json,numpy as np,trimesh,importlib.util
T=Path('/root/autodl-tmp/runs/worldsim_v77');O=T/'WS-V77-TARGET-PROTECTED-20260929/r8/assets';O.mkdir(exist_ok=True)
rows=[]
for name,scene in [('sedan','scene_0230'),('suv','scene_0255')]:
    source=T/'WS-V77-NINE-FULL-20260928/r1'/scene/'actor.glb'
    mesh=trimesh.load(source,force='scene').dump(concatenate=True)
    # glTF是Y-up；恢复Blender/GT的Z-up。必须在规范化前转换，避免宽/高混淆。
    mesh.vertices=np.asarray(mesh.vertices)[:,[0,2,1]]*np.array([1,-1,1])
    lo,hi=mesh.bounds;size=hi-lo
    assert size.argmax()==0,(name,size,'需要已校正的长轴X资产')
    v=(np.asarray(mesh.vertices)-(lo+hi)/2)/size
    plain=trimesh.Trimesh(vertices=v,faces=mesh.faces,process=True)
    print('SIMPLIFIER_AVAILABILITY',{m:importlib.util.find_spec(m) is not None for m in ['open3d','fast_simplification','vtk','pymeshlab']},flush=True)
    faces=np.asarray(plain.faces);v=np.asarray(plain.vertices)
    np.savez_compressed(O/(name+'_full.npz'),vertices=v.astype('float32'),faces=faces.astype('int32'))
    assert np.isfinite(v).all() and faces.max()<len(v)
    np.savez_compressed(O/(name+'.npz'),vertices=v.astype('float32'),faces=faces.astype('int32'))
    rows.append({'name':name,'source':str(source),'source_role':'existing exposed DEV-generated shape used only for silhouette; no its RGB/reference in training input or GT','bounds':mesh.bounds.tolist(),'faces':len(faces),'vertices':len(v),'canonical_axes':'glTF Y-up -> GT X length Y width Z up, centered unit bbox','simplification':'none; identical geometry, UV seam duplicates merged by position','front_direction':'axis-corrected retained asset; no identity/texture fidelity claim'})
(O/'manifest.json').write_text(json.dumps({'assets':rows,'GT':'real nuScenes train Y only','RGB_condition':'all asset pixels erased inside H','new_generation_network':False,'asset_scene_leakage_boundary':'these shapes derive from exposed DEV scenes; they are NOT independent shape/test evidence; final scenes untouched'},ensure_ascii=False,indent=2)+'\n')
print(rows)
