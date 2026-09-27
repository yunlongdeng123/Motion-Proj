"""原GLB的连续原位RGBA/深度层；保留已验证的车头和相机修正。"""
import argparse,json,math,pathlib,sys,time
import bpy,numpy as np
from mathutils import Matrix,Vector
from bpy_extras.object_utils import world_to_camera_view
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--scene',required=True);p.add_argument('--limit',type=int);a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
reg=json.loads((a.root/'registration.json').read_text(encoding='utf-8'));spec=next(s for s in reg['scenes'] if s['name']==a.scene);frames=json.loads((a.root/a.scene/'camera_frames.json').read_text(encoding='utf-8'))
out=a.root/a.scene/'actor_layers';out.mkdir(exist_ok=True);W,H=688,384
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.ops.import_scene.gltf(filepath=str(a.root/a.scene/'actor.glb'))
meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
bounds=[(min((o.matrix_world@Vector(c))[i] for o in meshes for c in o.bound_box),max((o.matrix_world@Vector(c))[i] for o in meshes for c in o.bound_box)) for i in range(3)]
dims=[hi-lo for lo,hi in bounds];center=Vector([(lo+hi)/2 for lo,hi in bounds]);root=bpy.data.objects.new('explicit_actor',None);bpy.context.collection.objects.link(root)
for o in meshes:
    old=o.matrix_world.copy();o.parent=root;o.matrix_parent_inverse=Matrix.Identity(4);o.matrix_world=Matrix.Translation(-center)@old
scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=16;scene.cycles.seed=77
scene.render.threads_mode='FIXED';scene.render.threads=6
scene.render.resolution_x=W;scene.render.resolution_y=H;scene.render.resolution_percentage=100;scene.render.film_transparent=True
scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA';scene.view_settings.view_transform='Standard'
try:scene.view_settings.look='None'
except TypeError:pass
scene.world.use_nodes=True;bg=next(n for n in scene.world.node_tree.nodes if n.type=='BACKGROUND');bg.inputs['Color'].default_value=(.65,.68,.72,1);bg.inputs['Strength'].default_value=.8
ld=bpy.data.lights.new('Sun','SUN');ld.energy=2;sun=bpy.data.objects.new('Sun',ld);bpy.context.collection.objects.link(sun);sun.rotation_euler=(.52,-.44,-.52)
cd=bpy.data.cameras.new('camera');cam=bpy.data.objects.new('camera',cd);bpy.context.collection.objects.link(cam);scene.camera=cam;cd.sensor_fit='HORIZONTAL';cd.sensor_width=36;cd.clip_end=500
scene.view_layers[0].use_pass_z=True
group=bpy.data.node_groups.new('ActorDepth','CompositorNodeTree');scene.compositing_node_group=group
rl=group.nodes.new('CompositorNodeRLayers');output=group.nodes.new('CompositorNodeOutputFile');output.directory=str(out/'depth_exr');output.format.media_type='IMAGE';output.format.file_format='OPEN_EXR';output.format.color_depth='32';output.format.color_mode='BW';output.file_output_items.new('FLOAT','Depth');group.links.new(rl.outputs['Depth'],output.inputs[0])
reports=[];new_count=0
for fr in frames:
    f=fr['frame'];actor=next(b for b in fr['all_boxes'] if b['actor_id']==spec['actor']);origin=Vector(fr['origin_world'])
    pose=Matrix(actor['pose']);pose.translation-=origin;size=actor['size_lwh'];scales=[size[i]/dims[i] for i in range(3)]
    root.matrix_world=pose@Matrix.Rotation(math.radians(spec['yaw']),4,'Z')@Matrix.Diagonal(Vector((*scales,1)))
    for stream in spec['streams']:
        c=stream['camera']
        if f not in stream['active_frames']:continue
        stem=f'f{f:03}_cam{c}';path=out/(stem+'.png');depthpath=out/(stem+'_depth.npy');meta=out/(stem+'.json')
        if path.exists() and depthpath.exists() and meta.exists():reports.append(json.loads(meta.read_text()));continue
        v=fr['views'][c];c2w=Matrix(v['c2w']);c2w.translation-=origin;cam.matrix_world=c2w@Matrix.Diagonal(Vector((1,-1,-1,1)))
        sx=W/v['original_wh'][0];sy=H/v['original_wh'][1];K=v['intrinsics'];fx=K[0][0]*sx;fy=K[1][1]*sy;cx=K[0][2]*sx+(sx-1)/2;cy=K[1][2]*sy+(sy-1)/2
        scene.render.pixel_aspect_x=1;scene.render.pixel_aspect_y=fx/fy;cd.lens=fx*36/W
        def uv(cp):
            bpy.context.view_layer.update();q=world_to_camera_view(scene,cam,c2w@Vector(cp));return q.x*W,(1-q.y)*H
        cd.shift_x=0;cd.shift_y=0;u0,v0=uv((0,0,10));cd.shift_x=.01;u1,_=uv((0,0,10));cd.shift_x=0;cd.shift_y=.01;_,v1=uv((0,0,10));cd.shift_x=.01*(cx-u0)/(u1-u0);cd.shift_y=.01*(cy-v0)/(v1-v0)
        error=max(math.hypot(uv((x,y,10))[0]-(fx*x/10+cx),uv((x,y,10))[1]-(fy*y/10+cy)) for x in [-2,0,2] for y in [-1,0,1]);assert error<.02,error
        scene.render.filepath=str(path);output.file_name=stem+'_';t=time.monotonic();bpy.ops.render.render(write_still=True)
        exr=out/'depth_exr'/f'{stem}_Depth.exr';assert exr.exists(),exr
        zimg=bpy.data.images.load(str(exr),check_existing=False);pixels=np.asarray(zimg.pixels[:],dtype=np.float32).reshape(H,W,4)[::-1,:,0].copy();bpy.data.images.remove(zimg)
        depth=pixels;depth[(pixels>1e8)|(~np.isfinite(pixels))]=np.inf;np.save(depthpath,depth.astype('float32'))
        row={'frame':f,'camera':c,'actor_id':spec['actor'],'yaw':spec['yaw'],'projection_max_error_px':error,'elapsed_s':time.monotonic()-t,'depth_convention':'Blender 5.2 Cycles Z pass = camera z; verified plane z5 across image','factual_pose_only':True}
        meta.write_text(json.dumps(row,indent=2)+'\n');reports.append(row);new_count+=1;print('LAYER_DONE',a.scene,f,c,flush=True)
        if a.limit and new_count>=a.limit:break
    if a.limit and new_count>=a.limit:break
(out/'render_summary.json').write_text(json.dumps({'scene':a.scene,'rendered_views':len(reports),'expected_views':sum(len(s['active_frames']) for s in spec['streams']),'rows':reports,'human_verdict':None},indent=2)+'\n')
