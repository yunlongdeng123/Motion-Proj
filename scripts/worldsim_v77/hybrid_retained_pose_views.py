"""在原本可见的参考姿态检查固定SUV网格，显式展示车头180度歧义。"""
import bpy,os,json,math
from pathlib import Path
from mathutils import Matrix,Vector
from bpy_extras.object_utils import world_to_camera_view
ROOT=Path(os.environ['V77_ASSET_REVIEW_ROOT']);MODE=os.environ.get('V77_ASSET_MODE','shape');assert MODE in ['shape','pbr'];OUT=ROOT/('observed_controls/render' if MODE=='shape' else 'observed_controls/render_pbr');assert not OUT.exists();OUT.mkdir();reg=json.loads((ROOT/'observed_controls/camera_frames.json').read_text());W,H=1024,576
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False);bpy.ops.import_scene.gltf(filepath=str(ROOT/('shape_untextured.glb' if MODE=='shape' else 'retained52_unaligned.glb')));meshes=[o for o in bpy.context.scene.objects if o.type=='MESH'];assert meshes
def bounds():
 pts=[o.matrix_world@Vector(c) for o in meshes for c in o.bound_box];return Vector([min(v[i] for v in pts) for i in range(3)]),Vector([max(v[i] for v in pts) for i in range(3)])
lo,hi=bounds();axis_rotation=90 if (hi-lo).y>(hi-lo).x else 0
for o in meshes:o.matrix_world=Matrix.Rotation(math.radians(axis_rotation),4,'Z')@o.matrix_world
lo,hi=bounds();center=(lo+hi)/2;dims=hi-lo;root=bpy.data.objects.new('retained_actor52',None);bpy.context.collection.objects.link(root)
for o in meshes:old=o.matrix_world.copy();o.parent=root;o.matrix_parent_inverse=Matrix.Identity(4);o.matrix_world=Matrix.Translation(-center)@old
mat=bpy.data.materials.new('geometry_only');mat.use_nodes=True;mat.node_tree.nodes.clear();bs=mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled');mo=mat.node_tree.nodes.new('ShaderNodeOutputMaterial');mat.node_tree.links.new(bs.outputs[0],mo.inputs[0]);bs.inputs['Base Color'].default_value=(.42,.5,.57,1);bs.inputs['Roughness'].default_value=.6
if MODE=='shape':
 for o in meshes:o.data.materials.clear();o.data.materials.append(mat)
scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=16;scene.cycles.seed=7740;scene.render.threads_mode='FIXED';scene.render.threads=4;scene.render.resolution_x=W;scene.render.resolution_y=H;scene.render.resolution_percentage=100;scene.render.film_transparent=True;scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA';scene.view_settings.view_transform='Standard';scene.world.use_nodes=True;nodes=scene.world.node_tree.nodes;nodes.clear();bg=nodes.new('ShaderNodeBackground');wo=nodes.new('ShaderNodeOutputWorld');scene.world.node_tree.links.new(bg.outputs[0],wo.inputs[0]);bg.inputs['Strength'].default_value=.8;bg.inputs['Color'].default_value=(.65,.65,.65,1)
cd=bpy.data.cameras.new('camera');cam=bpy.data.objects.new('camera',cd);bpy.context.collection.objects.link(cam);scene.camera=cam;cd.sensor_fit='HORIZONTAL';cd.sensor_width=36;cd.clip_end=500
rows=[];origin=Vector(reg['rows'][0]['actor']['pose'][i][3] for i in range(3))
for row in reg['rows']:
 f=row['frame'];pose=Matrix(row['actor']['pose']);pose.translation-=origin;v=row['view'];c2w=Matrix(v['c2w']);c2w.translation-=origin;cam.matrix_world=c2w@Matrix.Diagonal(Vector((1,-1,-1,1)));sx=W/v['original_wh'][0];sy=H/v['original_wh'][1];k=v['intrinsics'];fx=k[0][0]*sx;fy=k[1][1]*sy;cx=k[0][2]*sx+(sx-1)/2;cy=k[1][2]*sy+(sy-1)/2;scene.render.pixel_aspect_x=1;scene.render.pixel_aspect_y=fx/fy;cd.lens=fx*36/W
 def uv(p):
  bpy.context.view_layer.update();q=world_to_camera_view(scene,cam,c2w@Vector(p));return q.x*W,(1-q.y)*H
 cd.shift_x=0;cd.shift_y=0;u0,v0=uv((0,0,10));cd.shift_x=.01;u1,_=uv((0,0,10));cd.shift_x=0;cd.shift_y=.01;_,v1=uv((0,0,10));cd.shift_x=.01*(cx-u0)/(u1-u0);cd.shift_y=.01*(cy-v0)/(v1-v0);error=max(math.hypot(uv((x,y,10))[0]-(fx*x/10+cx),uv((x,y,10))[1]-(fy*y/10+cy)) for x in [-2,0,2] for y in [-1,0,1]);assert error<.02
 for yaw in [0,180]:
  scale=[row['actor']['size_lwh'][i]/dims[i] for i in range(3)];root.matrix_world=pose@Matrix.Rotation(math.radians(yaw),4,'Z')@Matrix.Diagonal(Vector((*scale,1)));scene.render.filepath=str(OUT/f'f{f:03}_yaw{yaw:03}.png');bpy.ops.render.render(write_still=True);rows.append(dict(frame=f,camera=3,yaw_diagnostic=yaw,axis_rotation_deg=axis_rotation,projection_max_error_px=error,scale_xyz=scale))
(OUT/'render_state.json').write_text(json.dumps(dict(rows=rows,blender=bpy.app.version_string,mode=MODE,role='original pose diagnostic, not DELETE output; two yaw candidates are axis disambiguation, not admitted counterfactuals',limitations='No fitted lighting or neighbor occlusion; raw layer must not be treated as a composited background',human_verdict=None),indent=2)+'\n');print('POSE_VIEWS_COMPLETE',len(rows))
