import sys,json,hashlib,time
from pathlib import Path
from types import SimpleNamespace
import bpy,numpy as np
from mathutils import Vector
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030')
sys.path.insert(0,str(ROOT/'code/simulation'))
import generate_scene as g
out=ROOT/'heldout_edit_refs_r1'
if out.exists(): raise RuntimeError('Preserve existing output')
out.mkdir()
p=json.loads((ROOT/'dataset_r2/dataset_manifest.json').read_text())['parameters']
c=SimpleNamespace(**p)
s,cam,car,objects,device=g.build_scene(c)
g.set_frame(8,c,cam,car,'none',True)
cam.location.y+=1
cam.rotation_euler=(car.location+Vector((.5,0,.85))-cam.location).to_track_quat('-Z','Y').to_euler()
bpy.context.view_layer.update()
fixed=cam.matrix_world.copy()
cal=g.camera_metadata(8,c,cam)
original=json.loads((ROOT/'dataset_r2/privileged/heldout/manifest.json').read_text())['frames'][0]
assert np.max(np.abs(np.array(cal['camera_to_world'])-np.array(original['camera_to_world'])))==0
entries=[]
for edit in g.EDIT_NAMES:
 g.set_frame(8,c,cam,car,edit,True)
 cam.matrix_world=fixed.copy()
 bpy.context.view_layer.update()
 path=out/edit/'000008.png'
 secs=g.render(s,path)
 entries.append({'edit':edit,'rgb':str(path.relative_to(out)),'render_s':secs,'camera':g.camera_metadata(8,c,cam)})
receipt={'schema':'v43_heldout_edit_reference_v1','status':'rendered','source_generator':str(ROOT/'code/simulation/generate_scene.py'),'generator_sha256':hashlib.sha256((ROOT/'code/simulation/generate_scene.py').read_bytes()).hexdigest(),'frame_id':8,'samples':32,'base_seed':431030,'effective_seed':431038,'device':device,'blender':bpy.app.version_string,'train_input':False,'algorithm_input':False,'purpose':'independent heldout evaluation only; edit RGB not available to M1/M2 construction','camera_contract':'Original none heldout camera fixed across edits; worldY+1m, aimed at unedited car','camera_parity_max_abs':0,'references':entries,'renderer_contract':g.renderer_contract()}
g.dump_json(out/'manifest.json',receipt)
