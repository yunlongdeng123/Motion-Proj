"""每例同一OBJ/PBR坐标转换；不按输出挑朝向或材质。"""
import bpy,sys,math,json,numpy as np
from pathlib import Path
from mathutils import Matrix,Vector
R=Path(sys.argv[-1]);out=R/'asset_views';out.mkdir(exist_ok=False)
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.ops.wm.obj_import(filepath=str(R/'asset/actor_pbr.obj'),forward_axis='Y',up_axis='Z')
meshes=[o for o in bpy.context.selected_objects if o.type=='MESH'];assert meshes
for o in meshes:o.matrix_world=Matrix.Rotation(math.pi/2,4,'X')@o.matrix_world
# 水平PCA确定车身长轴，消除生成网格轴向变化；前后符号交给固定真实参考双候选检查。
xyz=np.array([list(o.matrix_world@v.co) for o in meshes for v in o.data.vertices],dtype=float)
xy=xyz[:,:2]-xyz[:,:2].mean(0);eigval,eigvec=np.linalg.eigh(xy.T@xy/len(xy));axis=eigvec[:,-1]
if axis[0]<0:axis=-axis
angle=-math.atan2(float(axis[1]),float(axis[0]))
for o in meshes:o.matrix_world=Matrix.Rotation(angle,4,'Z')@o.matrix_world
mat=bpy.data.materials.new('hunyuan_pbr');mat.use_nodes=True;mat.node_tree.nodes.clear();bs=mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled');mo=mat.node_tree.nodes.new('ShaderNodeOutputMaterial');mat.node_tree.links.new(bs.outputs[0],mo.inputs[0])
for suffix,slot in [('', 'Base Color'),('_metallic','Metallic'),('_roughness','Roughness')]:
    tex=mat.node_tree.nodes.new('ShaderNodeTexImage');tex.image=bpy.data.images.load(str(R/f'asset/actor_pbr{suffix}.jpg'))
    if suffix:tex.image.colorspace_settings.name='Non-Color'
    mat.node_tree.links.new(tex.outputs['Color'],bs.inputs[slot])
for o in meshes:o.data.materials.clear();o.data.materials.append(mat)
bpy.ops.object.select_all(action='DESELECT')
for o in meshes:o.select_set(True)
bpy.context.view_layer.objects.active=meshes[0];bpy.ops.export_scene.gltf(filepath=str(R/'actor.glb'),export_format='GLB',use_selection=True)
points=[o.matrix_world@Vector(c) for o in meshes for c in o.bound_box];lo=Vector([min(p[i] for p in points) for i in range(3)]);hi=Vector([max(p[i] for p in points) for i in range(3)]);center=(lo+hi)/2;size=hi-lo
scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=16;scene.cycles.seed=7740;scene.render.threads_mode='FIXED';scene.render.threads=4;scene.render.resolution_x=512;scene.render.resolution_y=384;scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA';scene.render.film_transparent=True;scene.view_settings.view_transform='Standard'
scene.world.use_nodes=True;bg=next(n for n in scene.world.node_tree.nodes if n.type=='BACKGROUND');bg.inputs['Color'].default_value=(.65,.65,.65,1);bg.inputs['Strength'].default_value=.8
bpy.ops.object.camera_add();cam=bpy.context.object;scene.camera=cam;cam.data.type='ORTHO';cam.data.ortho_scale=max(size)*1.45
bpy.ops.object.light_add(type='AREA',location=center+Vector((2,-2,4))*max(size));light=bpy.context.object;light.data.energy=700;light.data.shape='DISK';light.data.size=max(size)*3
for az in [0,45,90,135,180,225,270,315]:
    t=math.radians(az);cam.location=center+Vector((math.cos(t)*3,math.sin(t)*3,1))*max(size);cam.rotation_euler=(center-cam.location).to_track_quat('-Z','Y').to_euler();scene.render.filepath=str(out/f'az{az:03}.png');bpy.ops.render.render(write_still=True)
(R/'canonical.json').write_text(json.dumps(dict(scene=R.name,dimensions=list(size),matrix='Rx(+90deg) after OBJ import; horizontal vertex PCA rotate long axis toX; same rule all9',pca_rotation_z_deg=math.degrees(angle),pca_eigenvalues=eigval.tolist(),yaw_correction_deg=None,orientation_passed=None,axes_dimension_sanity=bool(size.x>size.y and size.x>size.z),material='existing Hunyuan color/metallic/roughness; no fit',human_verdict=None),indent=2)+'\n')
print('CANONICAL_DONE',R.name,list(size),flush=True)
