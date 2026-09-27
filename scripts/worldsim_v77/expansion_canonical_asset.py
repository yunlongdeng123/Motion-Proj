"""显式记录Hunyuan原生长轴Y→GT车辆前轴X；不按渲染好坏搜索朝向。"""
import bpy,sys,json,math
from pathlib import Path
from mathutils import Matrix,Vector
R=Path(sys.argv[-1]);bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False);bpy.ops.import_scene.gltf(filepath=str(R/'actor_native.glb'));meshes=[o for o in bpy.context.scene.objects if o.type=='MESH'];assert meshes
for o in meshes:o.matrix_world=Matrix.Rotation(-math.pi/2,4,'Z')@o.matrix_world
bpy.ops.object.select_all(action='DESELECT')
for o in meshes:o.select_set(True)
bpy.context.view_layer.objects.active=meshes[0];bpy.ops.export_scene.gltf(filepath=str(R/'actor.glb'),export_format='GLB',use_selection=True)
points=[o.matrix_world@Vector(c) for o in meshes for c in o.bound_box];dims=[max(p[i] for p in points)-min(p[i] for p in points) for i in range(3)]
assert dims[0]>dims[1]>dims[2],dims
(R/'canonical.json').write_text(json.dumps(dict(rotation_blender_z_deg=-90,basis='Shape8-view visual review: front+Y, rear-Y, upZ. Rotate-90 mapsfront+X. GT NuScenes pose forward+X; yaw0/180 observed-source control follows.',dimensions=dims,human_verdict=None),indent=2),encoding='utf-8')
