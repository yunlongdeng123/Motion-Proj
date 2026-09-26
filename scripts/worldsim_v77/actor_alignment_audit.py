"""同GLB与GT相机的方向/投影控制；不改网格或训练任何模型。"""
import argparse,json,math,pathlib,sys
import bpy
from mathutils import Matrix,Vector
from bpy_extras.object_utils import world_to_camera_view

p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);a.out=a.out.resolve();a.root=a.root.resolve();a.out.mkdir(parents=True,exist_ok=True)
W,H=688,384
ref,cam,actor={'scene_0230':(5,2,'22'),'scene_0255':(20,3,'25')}[a.scene]
reg=json.loads((a.root/'registration.json').read_text());spec=next(s for s in reg['scenes'] if s['name']==a.scene)
fa=json.loads((a.root/a.scene/'instances_snapshot.json').read_text())[actor]['frame_annotations'];j=fa['frame_idx'].index(ref)
pose=Matrix(fa['obj_to_world'][j]);size=fa['box_size'][j];origin=Vector(spec['spec']['origin_world'])
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.ops.import_scene.gltf(filepath=str(a.root/a.scene/'actor/actor.glb'))
meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
bounds=[(min((o.matrix_world@Vector(c))[i] for o in meshes for c in o.bound_box),max((o.matrix_world@Vector(c))[i] for o in meshes for c in o.bound_box)) for i in range(3)]
dims=[hi-lo for lo,hi in bounds];center=Vector([(lo+hi)/2 for lo,hi in bounds])
root=bpy.data.objects.new('GT_actor',None);bpy.context.collection.objects.link(root)
for o in meshes:
    old=o.matrix_world.copy();o.parent=root;o.matrix_parent_inverse=Matrix.Identity(4);o.matrix_world=Matrix.Translation(-center)@old
scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=16
scene.render.resolution_x=W;scene.render.resolution_y=H;scene.render.resolution_percentage=100;scene.render.film_transparent=True
scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA';scene.view_settings.view_transform='Standard'
try:scene.view_settings.look='None'
except TypeError:pass
scene.world.use_nodes=True
bg=next(n for n in scene.world.node_tree.nodes if n.type=='BACKGROUND');bg.inputs['Color'].default_value=(0.65,0.68,0.72,1);bg.inputs['Strength'].default_value=.8
sd=bpy.data.lights.new('Sun','SUN');sd.energy=2;sun=bpy.data.objects.new('Sun',sd);bpy.context.collection.objects.link(sun);sun.rotation_euler=(.52,-.44,-.52)
cd=bpy.data.cameras.new('calibrated_camera');camera=bpy.data.objects.new('calibrated_camera',cd);bpy.context.collection.objects.link(camera);scene.camera=camera
cd.sensor_fit='HORIZONTAL';cd.sensor_width=36
c2w=Matrix([[float(x) for x in l.split()] for l in (a.root/'source'/a.scene/'extrinsics'/f'{ref:03d}_{cam}.txt').read_text().splitlines()]);c2w.translation-=origin
camera.matrix_world=c2w@Matrix.Diagonal(Vector((1,-1,-1,1)))
v=spec['spec']['views'][cam];sx=W/v['original_wh'][0];sy=H/v['original_wh'][1]
fx=v['intrinsics'][0][0]*sx;fy=v['intrinsics'][1][1]*sy;cx=v['intrinsics'][0][2]*sx+(sx-1)/2;cy=v['intrinsics'][1][2]*sy+(sy-1)/2
scene.render.pixel_aspect_x=1;scene.render.pixel_aspect_y=fx/fy;cd.lens=fx*36/W
def uv(cp):
    bpy.context.view_layer.update();q=world_to_camera_view(scene,camera,c2w@Vector(cp));return [q.x*W,(1-q.y)*H]
def errors():
    vals=[]
    for x in [-2.,0.,2.]:
      for y in [-1.,0.,1.]:
        u,v=uv((x,y,10));vals.append(math.hypot(u-(fx*x/10+cx),v-(fy*y/10+cy)))
    return {'max_error_px':max(vals),'mean_error_px':sum(vals)/len(vals),'principal_projected':uv((0,0,10))}
cd.shift_x=(W/2-v['intrinsics'][0][2]*sx)/W;cd.shift_y=(v['intrinsics'][1][2]*sy-H/2)/H
old_error=errors()
cd.shift_x=0;cd.shift_y=0;u0,v0=uv((0,0,10));cd.shift_x=.01;u1,_=uv((0,0,10));cd.shift_x=0;cd.shift_y=.01;_,v1=uv((0,0,10))
cd.shift_x=.01*(cx-u0)/(u1-u0);cd.shift_y=.01*(cy-v0)/(v1-v0)
new_error=errors()
assert new_error['max_error_px']<.02,new_error
scales=[size[i]/dims[i] for i in range(3)];scale=Matrix.Diagonal(Vector((*scales,1)))
loc=pose.copy();loc.translation-=origin
for yaw in [0,180]:
    root.matrix_world=loc@Matrix.Rotation(math.radians(yaw),4,'Z')@scale
    bpy.context.view_layer.update();scene.render.filepath=str(a.out/f'{a.scene}_yaw{yaw}.png');bpy.ops.render.render(write_still=True)
report={'scene':a.scene,'frame':ref,'cam':cam,'actor_id':actor,'glb_dimensions_xyz':dims,'gt_size_lwh':size,'axis_scales':scales,
 'old_projection':old_error,'corrected_projection':new_error,'corrected_shift_xy':[cd.shift_x,cd.shift_y],
 'camera_K_resized':[[fx,0,cx],[0,fy,cy],[0,0,1]],'world_actor_forward':list(pose.to_3x3()@Vector((1,0,0))), 'human_verdict':None}
(a.out/f'{a.scene}_projection.json').write_text(json.dumps(report,indent=2)+'\n');print('AUDIT',json.dumps(report),flush=True)
