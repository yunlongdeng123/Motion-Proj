"""复用原 GLB 修复逐帧放置；只输出原位诊断，不执行未准入 MOVE。"""
import argparse,json,math,pathlib,sys
import bpy
from mathutils import Matrix,Vector
from bpy_extras.object_utils import world_to_camera_view
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);a.out=a.out.resolve();a.root=a.root.resolve();a.out.mkdir(parents=True,exist_ok=True)
plan=json.loads((a.out/'view_plan.json').read_text())[a.scene]
spec=next(s for s in json.loads((a.root/'registration.json').read_text())['scenes'] if s['name']==a.scene)
fa=json.loads((a.root/a.scene/'instances_snapshot.json').read_text())[plan['actor_id']]['frame_annotations']
origin=Vector(spec['spec']['origin_world']);W,H=688,384
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.ops.import_scene.gltf(filepath=str(a.root/a.scene/'actor/actor.glb'))
meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
bounds=[(min((o.matrix_world@Vector(c))[i] for o in meshes for c in o.bound_box),max((o.matrix_world@Vector(c))[i] for o in meshes for c in o.bound_box)) for i in range(3)]
dims=[hi-lo for lo,hi in bounds];center=Vector([(lo+hi)/2 for lo,hi in bounds])
root=bpy.data.objects.new('GT_actor',None);bpy.context.collection.objects.link(root)
for o in meshes:
    old=o.matrix_world.copy();o.parent=root;o.matrix_parent_inverse=Matrix.Identity(4);o.matrix_world=Matrix.Translation(-center)@old
scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=16;scene.cycles.seed=77
scene.render.resolution_x=W;scene.render.resolution_y=H;scene.render.resolution_percentage=100;scene.render.film_transparent=True
scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA';scene.view_settings.view_transform='Standard'
try:scene.view_settings.look='None'
except TypeError:pass
scene.world.use_nodes=True
bg=next(n for n in scene.world.node_tree.nodes if n.type=='BACKGROUND');bg.inputs['Color'].default_value=(.65,.68,.72,1);bg.inputs['Strength'].default_value=.8
sd=bpy.data.lights.new('Sun','SUN');sd.energy=2;sun=bpy.data.objects.new('Sun',sd);bpy.context.collection.objects.link(sun);sun.rotation_euler=(.52,-.44,-.52)
cd=bpy.data.cameras.new('calibrated_camera');camera=bpy.data.objects.new('calibrated_camera',cd);bpy.context.collection.objects.link(camera);scene.camera=camera
cd.sensor_fit='HORIZONTAL';cd.sensor_width=36
reports=[]
for row in plan['frames']:
    f,c=row['frame'],row['camera'];j=fa['frame_idx'].index(f);size=fa['box_size'][j]
    pose=Matrix(fa['obj_to_world'][j]);pose.translation-=origin
    c2w=Matrix([[float(x) for x in l.split()] for l in (a.root/'source'/a.scene/'extrinsics'/f'{f:03d}_{c}.txt').read_text().splitlines()]);c2w.translation-=origin
    camera.matrix_world=c2w@Matrix.Diagonal(Vector((1,-1,-1,1)))
    v=spec['spec']['views'][c];sx=W/v['original_wh'][0];sy=H/v['original_wh'][1]
    fx=v['intrinsics'][0][0]*sx;fy=v['intrinsics'][1][1]*sy;cx=v['intrinsics'][0][2]*sx+(sx-1)/2;cy=v['intrinsics'][1][2]*sy+(sy-1)/2
    scene.render.pixel_aspect_x=1;scene.render.pixel_aspect_y=fx/fy;cd.lens=fx*36/W
    def uv(cp):
        bpy.context.view_layer.update();q=world_to_camera_view(scene,camera,c2w@Vector(cp));return q.x*W,(1-q.y)*H
    cd.shift_x=0;cd.shift_y=0;u0,v0=uv((0,0,10));cd.shift_x=.01;u1,_=uv((0,0,10));cd.shift_x=0;cd.shift_y=.01;_,v1=uv((0,0,10))
    cd.shift_x=.01*(cx-u0)/(u1-u0);cd.shift_y=.01*(cy-v0)/(v1-v0)
    error=max(math.hypot(uv((x,y,10))[0]-(fx*x/10+cx),uv((x,y,10))[1]-(fy*y/10+cy)) for x in [-2,0,2] for y in [-1,0,1])
    assert error<.02,error
    scales=[size[i]/dims[i] for i in range(3)];rotation=Matrix.Rotation(math.radians(plan['yaw_correction_deg']),4,'Z')
    root.matrix_world=pose@rotation@Matrix.Diagonal(Vector((*scales,1)))
    folder=a.out/a.scene;folder.mkdir(exist_ok=True)
    scene.render.filepath=str(folder/f'f{f:03d}_cam{c}_corrected.png');bpy.ops.render.render(write_still=True)
    if f=={'scene_0230':5,'scene_0255':20}[a.scene]:
        # Uniform length scale preserves the GLB proportions; anchor its bottom
        # to the same GT box bottom so height differences do not create a float.
        uni=scales[0];z=(dims[2]*uni-size[2])/2
        root.matrix_world=pose@Matrix.Translation((0,0,z))@rotation@Matrix.Diagonal(Vector((uni,uni,uni,1)))
        scene.render.filepath=str(folder/'uniform_scale.png');bpy.ops.render.render(write_still=True)
    reports.append({'frame':f,'camera':c,'projection_max_error_px':error,'yaw_correction_deg':plan['yaw_correction_deg'],'axis_scales':scales,'legal_move_executed':False})
(a.out/f'{a.scene}_render_validation.json').write_text(json.dumps(reports,indent=2)+'\n')
print('CORRECTED_RENDER_DONE',a.scene,len(reports),flush=True)
