import bpy,numpy as np,pathlib
from mathutils import Vector
s=bpy.context.scene
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.ops.mesh.primitive_plane_add(size=40,location=(0,0,-5))
bpy.ops.object.camera_add(location=(0,0,0));s.camera=bpy.context.object
s.camera.data.lens=20;s.camera.data.sensor_width=36
s.render.engine='CYCLES';s.cycles.samples=1;s.render.resolution_x=64;s.render.resolution_y=32;s.render.resolution_percentage=100
s.view_layers[0].use_pass_z=True
g=bpy.data.node_groups.new('DepthExport','CompositorNodeTree');s.compositing_node_group=g
rl=g.nodes.new('CompositorNodeRLayers');o=g.nodes.new('CompositorNodeOutputFile')
print('ITEMS',[(x.name,x.socket_type) for x in o.file_output_items]);print('NEW',o.file_output_items.bl_rna.functions['new'].description,[(p.identifier,p.type) for p in o.file_output_items.bl_rna.functions['new'].parameters])
print('FORMAT',[(p.identifier,p.type,([i.identifier for i in p.enum_items] if p.type=='ENUM' else '')) for p in o.format.bl_rna.properties])
o.directory=str(pathlib.Path(__file__).parent/'depth_probe');o.file_name='depth_single';o.format.media_type='IMAGE';o.format.file_format='OPEN_EXR';o.file_output_items.new('FLOAT','Depth');o.format.color_depth='32';o.format.color_mode='BW'
g.links.new(rl.outputs['Depth'],o.inputs[0])
bpy.ops.render.render()
print('FILES',list(pathlib.Path(o.directory).glob('*')))
for p in pathlib.Path(o.directory).glob('depth_single*.exr'):
 im=bpy.data.images.load(str(p));a=np.array(im.pixels[:]).reshape(32,64,4)[::-1,:,0];print('Z_VALUES',a[16,32],a[16,0],a.min(),a.max())
