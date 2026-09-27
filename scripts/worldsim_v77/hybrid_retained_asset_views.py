"""Blender独立多视角检查，先看新保留资产自身，尚不放进DELETE。"""
import bpy,sys,math,json,os
from pathlib import Path
from mathutils import Vector,Matrix
ROOT=Path(os.environ.get('V77_ASSET_REVIEW_ROOT','/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927/r40'));mode=sys.argv[-1];assert mode in ['shape','pbr'];out=ROOT/f'{mode}_views_checked';assert not out.exists();out.mkdir()
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
if mode=='shape':bpy.ops.import_scene.gltf(filepath=str(ROOT/'shape_untextured.glb'))
else:
 if hasattr(bpy.ops.wm,'obj_import'):bpy.ops.wm.obj_import(filepath=str(ROOT/'actor_pbr.obj'),forward_axis='Y',up_axis='Z')
 else:bpy.ops.import_scene.obj(filepath=str(ROOT/'actor_pbr.obj'),axis_forward='Y',axis_up='Z')
 for o in bpy.context.selected_objects:
  if o.type=='MESH':o.matrix_world=Matrix.Rotation(math.pi/2,4,'X')@o.matrix_world
meshes=[o for o in bpy.context.scene.objects if o.type=='MESH'];assert meshes
mat=bpy.data.materials.new('review_surface');mat.use_nodes=True;mat.node_tree.nodes.clear();bs=mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled');material_output=mat.node_tree.nodes.new('ShaderNodeOutputMaterial');mat.node_tree.links.new(bs.outputs[0],material_output.inputs[0]);bs.inputs['Base Color'].default_value=(.48,.52,.56,1);bs.inputs['Roughness'].default_value=.55
if mode=='pbr':
 for suffix,slot in [('', 'Base Color'),('_metallic','Metallic'),('_roughness','Roughness')]:
  tex=mat.node_tree.nodes.new('ShaderNodeTexImage');tex.image=bpy.data.images.load(str(ROOT/f'actor_pbr{suffix}.jpg'))
  if suffix:tex.image.colorspace_settings.name='Non-Color'
  mat.node_tree.links.new(tex.outputs['Color'],bs.inputs[slot])
for o in meshes:o.data.materials.clear();o.data.materials.append(mat)
points=[o.matrix_world@Vector(c) for o in meshes for c in o.bound_box];lo=Vector([min(p[i] for p in points) for i in range(3)]);hi=Vector([max(p[i] for p in points) for i in range(3)]);center=(lo+hi)/2;size=hi-lo
scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=16;scene.cycles.seed=7740;scene.render.threads_mode='FIXED';scene.render.threads=4;scene.render.resolution_x=512;scene.render.resolution_y=384;scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA';scene.world.use_nodes=True;scene.world.node_tree.nodes.clear();background=scene.world.node_tree.nodes.new('ShaderNodeBackground');world_output=scene.world.node_tree.nodes.new('ShaderNodeOutputWorld');scene.world.node_tree.links.new(background.outputs[0],world_output.inputs[0]);background.inputs['Color'].default_value=(.65,.65,.65,1);background.inputs['Strength'].default_value=.8;scene.view_settings.view_transform='Standard';scene.render.film_transparent=True
bpy.ops.object.camera_add();cam=bpy.context.object;scene.camera=cam;cam.data.type='ORTHO';cam.data.ortho_scale=max(size)*1.45
bpy.ops.object.light_add(type='AREA',location=center+Vector((2,-2,4))*max(size));light=bpy.context.object;light.data.energy=700;light.data.shape='DISK';light.data.size=max(size)*3
for az in [0,45,90,135,180,225,270,315]:
 t=math.radians(az);cam.location=center+Vector((math.cos(t)*3,math.sin(t)*3,1.0))*max(size);cam.rotation_euler=(center-cam.location).to_track_quat('-Z','Y').to_euler();scene.render.filepath=str(out/f'az{az:03}.png');bpy.ops.render.render(write_still=True)
if mode=='pbr':
 bpy.ops.object.select_all(action='DESELECT')
 for o in meshes:o.select_set(True)
 bpy.context.view_layer.objects.active=meshes[0];bpy.ops.export_scene.gltf(filepath=str(ROOT/'retained52_unaligned.glb'),export_format='GLB',use_selection=True)
(out/'view_state.json').write_text(json.dumps(dict(mode=mode,blender=bpy.app.version_string,scene_axes_bounds=[list(lo),list(hi)],dimensions=list(size),views=8,renderer='Cycles CPU16 samples fixed neutral lighting; no fitted scene illumination',human_verdict=None),indent=2)+'\n');print('ASSET_VIEWS_COMPLETE',mode,list(size),flush=True)
