"""Render fixed explicit actor meshes at GT poses for factual and +2 m MOVE."""
import argparse
import json
import math
import pathlib
import sys

import bpy
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view

DEFAULT_ROOT = pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPLICIT-POC-20260926/r1')
SCENES = ['scene_0230', 'scene_0255']
FRAMES = {'scene_0230': list(range(0, 50, 5)), 'scene_0255': list(range(0, 100, 10))}
W, H = 688, 384

argv = sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
p = argparse.ArgumentParser()
p.add_argument('--scene', choices=SCENES, required=True)
p.add_argument('--frame', type=int)
p.add_argument('--camera', type=int)
p.add_argument('--asset', choices=['pbr','shape'], default='pbr')
p.add_argument('--root', type=pathlib.Path, default=DEFAULT_ROOT)
p.add_argument('--moves-only', action='store_true')
a = p.parse_args(argv)
ROOT = a.root

spec = next(s for s in json.loads((ROOT/'registration.json').read_text())['scenes'] if s['name']==a.scene)
actor_root = ROOT/a.scene/'actor'
selection = json.loads((ROOT/a.scene/'selection.json').read_text())
asset = actor_root/('actor_pbr.obj' if a.asset=='pbr' else 'shape_untextured.obj')
if not asset.is_file():
    raise FileNotFoundError(asset)
instances = json.loads((ROOT/a.scene/'instances_snapshot.json').read_text())
track = instances[spec['actor']]
fa = track['frame_annotations']
origin = Vector(spec['spec']['origin_world'])

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
if hasattr(bpy.ops.wm, 'obj_import'):
    bpy.ops.wm.obj_import(filepath=str(asset), forward_axis='Y', up_axis='Z')
else:
    bpy.ops.import_scene.obj(filepath=str(asset), axis_forward='Y', axis_up='Z')
meshes = [o for o in bpy.context.selected_objects if o.type=='MESH']
if not meshes: raise RuntimeError('empty imported actor')
# Hunyuan OBJ uses local +Y as height and +Z as width for these generated
# vehicles. Normalize it once to the v77 actor contract: X length, Y width,
# Z height. A fixed +90 degree X rotation maps local +Y to world +Z.
for o in meshes:
    o.matrix_world=Matrix.Rotation(math.pi/2,4,'X') @ o.matrix_world
if a.asset=='pbr':
    mat=bpy.data.materials.new('Hunyuan_PBR')
    mat.use_nodes=True
    nodes=mat.node_tree.nodes
    links=mat.node_tree.links
    nodes.clear()
    principled=nodes.new('ShaderNodeBsdfPrincipled')
    output=nodes.new('ShaderNodeOutputMaterial')
    links.new(principled.outputs['BSDF'],output.inputs['Surface'])
    for suffix,slot in [('', 'Base Color'),('_metallic','Metallic'),('_roughness','Roughness')]:
        path=actor_root/f'actor_pbr{suffix}.jpg'
        if not path.is_file():raise FileNotFoundError(path)
        tex=nodes.new('ShaderNodeTexImage')
        tex.image=bpy.data.images.load(str(path),check_existing=True)
        if suffix:
            try:tex.image.colorspace_settings.name='Non-Color'
            except TypeError:tex.image.colorspace_settings.name='Linear'
        links.new(tex.outputs['Color'],principled.inputs[slot])
    for o in meshes:
        o.data.materials.clear()
        o.data.materials.append(mat)
        print('PBR_BIND',o.name,'uv_layers',len(o.data.uv_layers),
              'materials',len(o.data.materials),'links',[(l.from_node.name,l.to_node.name,l.to_socket.name) for l in links],flush=True)
    out_glb=actor_root/'actor.glb'
    if not out_glb.exists():
        bpy.ops.object.select_all(action='DESELECT')
        for o in meshes:o.select_set(True)
        bpy.context.view_layer.objects.active=meshes[0]
        bpy.ops.export_scene.gltf(filepath=str(out_glb),export_format='GLB',use_selection=True)
        print('ACTOR_GLB',str(out_glb),out_glb.stat().st_size,flush=True)
asset_bounds = [[min((o.matrix_world @ Vector(c))[i] for o in meshes for c in o.bound_box),
                 max((o.matrix_world @ Vector(c))[i] for o in meshes for c in o.bound_box)] for i in range(3)]
dimensions = [hi-lo for lo,hi in asset_bounds]
if min(dimensions)<=0: raise RuntimeError(f'bad actor size {dimensions}')
local_center = Vector([(lo+hi)/2 for lo,hi in asset_bounds])
root = bpy.data.objects.new('explicit_actor', None)
bpy.context.collection.objects.link(root)
for o in meshes:
    original = o.matrix_world.copy()
    o.parent = root
    o.matrix_parent_inverse = Matrix.Identity(4)
    o.matrix_world = root.matrix_world @ Matrix.Translation(-local_center) @ original

camdata = bpy.data.cameras.new('GT_camera')
camera = bpy.data.objects.new('GT_camera', camdata)
bpy.context.collection.objects.link(camera)
bpy.context.scene.camera = camera
camdata.type='PERSP'
camdata.sensor_fit='HORIZONTAL'
camdata.sensor_width=36
scene=bpy.context.scene
scene.render.engine='CYCLES'
scene.cycles.device='CPU'
scene.cycles.samples=16
scene.render.resolution_x=W
scene.render.resolution_y=H
scene.render.resolution_percentage=100
scene.render.film_transparent=True
scene.render.image_settings.file_format='PNG'
scene.render.image_settings.color_mode='RGBA'
scene.view_settings.view_transform='Standard'
try:scene.view_settings.look='Medium High Contrast'
except TypeError:scene.view_settings.look='None'
scene.view_settings.exposure=0
scene.view_settings.gamma=1
scene.world.color=(0.6,0.6,0.6)
scene.world.use_nodes=True
scene.world.node_tree.nodes['Background'].inputs['Color'].default_value=(0.65,0.68,0.72,1)
scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=0.8
scene.render.film_transparent=True

sun_data=bpy.data.lights.new('Sun','SUN')
sun_data.energy=2.0
sun=bpy.data.objects.new('Sun',sun_data)
bpy.context.collection.objects.link(sun)
sun.rotation_euler=(math.radians(30), math.radians(-25), math.radians(-30))

ref_frame,ref_cam={'scene_0230':(5,2),'scene_0255':(20,3)}[a.scene]
ref_source=ROOT/'source'/a.scene if (ROOT/'source'/a.scene).exists() else pathlib.Path(spec['root'])
ref_c2w=Matrix([[float(x) for x in line.split()] for line in (ref_source/'extrinsics'/f'{ref_frame:03d}_{ref_cam}.txt').read_text().splitlines()])
delta=Vector((ref_c2w[0][0],ref_c2w[1][0],0))
delta.normalize();delta*=2
(actor_root/'placement.json').write_text(json.dumps({'move_delta_world_m':list(delta),
  'reference_frame':ref_frame,'reference_cam':ref_cam,
  'rule':'horizontal projection of GT reference-camera screen-right vector, fixed 2m for all frames'},indent=2)+'\n')
print('MOVE_DELTA',a.scene,tuple(delta),flush=True)

for frame in ([a.frame] if a.frame is not None else FRAMES[a.scene]):
    if frame not in fa['frame_idx']: continue
    j=fa['frame_idx'].index(frame)
    pose=Matrix(fa['obj_to_world'][j])
    size=fa['box_size'][j]
    scale=Matrix.Diagonal(Vector((size[0]/dimensions[0],size[1]/dimensions[1],size[2]/dimensions[2],1)))
    for cam in ([a.camera] if a.camera is not None else range(6)):
        stream=selection['camera_streams'].get(str(cam))
        if not stream or stream['first'] is None or not stream['first']<=frame<=stream['last']:continue
        source=ROOT/'source'/a.scene if (ROOT/'source'/a.scene).exists() else pathlib.Path(spec['root'])
        c2w_file=source/'extrinsics'/f'{frame:03d}_{cam}.txt'
        c2w=Matrix([[float(x) for x in line.split()] for line in c2w_file.read_text().splitlines()])
        c2w.translation -= origin
        camera.matrix_world = c2w @ Matrix.Diagonal(Vector((1,-1,-1,1)))
        view=spec['spec']['views'][cam]
        fx=view['intrinsics'][0][0]*W/view['original_wh'][0]
        fy=view['intrinsics'][1][1]*H/view['original_wh'][1]
        cx=view['intrinsics'][0][2]*W/view['original_wh'][0]
        cy=view['intrinsics'][1][2]*H/view['original_wh'][1]
        # Blender horizontal fit with pixel aspect correction for fx != fy.
        scene.render.pixel_aspect_x=1
        scene.render.pixel_aspect_y=fx/fy
        camdata.lens=fx*36/W
        camdata.shift_x=(W/2-cx)/W
        camdata.shift_y=(cy-H/2)/H
        camera.data=camdata
        edits=[('move',delta)] if a.moves_only else [('factual',Vector((0,0,0))),('move',delta)]
        for kind,offset in edits:
            loc=pose.copy()
            loc.translation=pose.translation-origin+offset
            root.matrix_world=loc @ scale
            bpy.context.view_layer.update()
            # Diagnose intrinsics/coordinate conventions against direct K projection.
            if kind=='factual' and frame==FRAMES[a.scene][0]:
                center=world_to_camera_view(scene,camera,loc.translation)
                print('ACTOR_PROJECT',a.scene,frame,cam,center.x*W,(1-center.y)*H,flush=True)
                mesh_center=meshes[0].matrix_world @ Vector((0,0,0))
                q=world_to_camera_view(scene,camera,mesh_center)
                print('MESH_PROJECT',tuple(mesh_center),q.x*W,(1-q.y)*H,q.z,
                      'dimensions',dimensions,'size',size,'scale',tuple(root.scale),flush=True)
            out=ROOT/a.scene/'actor_render'/f'{frame:03d}'
            out.mkdir(parents=True,exist_ok=True)
            scene.render.filepath=str(out/f'{kind}_cam{cam}.png')
            bpy.ops.render.render(write_still=True)
            print('ACTOR_RENDER',a.scene,frame,cam,kind,str(scene.render.filepath),flush=True)
