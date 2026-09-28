import bpy,sys,json,math
from pathlib import Path
from mathutils import Matrix
R=Path(sys.argv[-1]);reg=json.loads((R/'orientation.json').read_text());yaw=reg['selected_yaw_deg']
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False);bpy.ops.import_scene.gltf(filepath=str(R/'actor_axis_aligned.glb'))
objects=[o for o in bpy.context.scene.objects if o.type=='MESH']
for o in objects:o.matrix_world=Matrix.Rotation(math.radians(yaw),4,'Z')@o.matrix_world
bpy.ops.object.select_all(action='DESELECT')
for o in objects:o.select_set(True)
bpy.context.view_layer.objects.active=objects[0];bpy.ops.export_scene.gltf(filepath=str(R/'actor.glb'),export_format='GLB',use_selection=True)
print('BAKED_YAW',yaw,flush=True)
