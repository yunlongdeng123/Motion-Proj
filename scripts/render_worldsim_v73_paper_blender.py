"""用原始三角面渲染失败几何；裁切只用于局部展示，不改评价面。"""
import bpy,sys,json,math,os
from pathlib import Path
import numpy as np
from mathutils import Vector
ROOT=Path(os.environ.get('WORLDSIM_PAPER_FORENSICS_ROOT','/root/autodl-tmp/motion_proj/docs/autoresearch/worldsim_v73/paper_forensics/20260909T184200Z__saved-surface-oracles-r1'))
OUT=ROOT/'renders';OUT.mkdir(exist_ok=True)
def mat(name,c,rough=.7):
 m=bpy.data.materials.new(name);m.diffuse_color=(*c,1);m.use_nodes=True
 p=m.node_tree.nodes.get('Principled BSDF');p.inputs['Base Color'].default_value=(*c,1);p.inputs['Roughness'].default_value=rough;return m
def mesh(name,v,f,mats,mi=None):
 data=bpy.data.meshes.new(name);data.from_pydata(v.tolist(),[],f.tolist());data.update();o=bpy.data.objects.new(name,data);bpy.context.collection.objects.link(o)
 for m in mats:o.data.materials.append(m)
 if mi is not None:
  for face,idx in zip(o.data.polygons,mi):face.material_index=int(idx)
 return o
def spheres(name,points,radius,material):
 # 小型八面体只作为测量标记，不参与原始曲面导出或查询。
 b=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])*radius
 f=np.array([[0,2,4],[2,1,4],[1,3,4],[3,0,4],[2,0,5],[1,2,5],[3,1,5],[0,3,5]])
 if len(points):return mesh(name,np.concatenate([b+p for p in points]),np.concatenate([f+6*i for i in range(len(points))]),[material])
def line(name,a,b,r,m):
 delta=Vector(b)-Vector(a);bpy.ops.mesh.primitive_cylinder_add(vertices=12,radius=r,depth=delta.length,location=(Vector(a)+Vector(b))/2)
 obj=bpy.context.object;obj.name=name;obj.rotation_euler=delta.to_track_quat('Z','Y').to_euler();obj.data.materials.append(m);return obj
def camera(look,pos,scale):
 bpy.ops.object.camera_add(location=pos);c=bpy.context.object;c.rotation_euler=(Vector(look)-c.location).to_track_quat('-Z','Y').to_euler();c.data.type='ORTHO';c.data.ortho_scale=scale;bpy.context.scene.camera=c;return c
def setup():
 bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
 sc=bpy.context.scene;sc.render.engine='CYCLES';sc.cycles.device='CPU';sc.cycles.samples=16;sc.cycles.use_denoising=True;sc.render.threads_mode='FIXED';sc.render.threads=2
 sc.render.resolution_x=1100;sc.render.resolution_y=800;sc.render.resolution_percentage=100;sc.render.image_settings.file_format='PNG'
 sc.world.color=(.8,.8,.8);sc.world.use_nodes=True;sc.world.node_tree.nodes['Background'].inputs[0].default_value=(.93,.95,.98,1);sc.world.node_tree.nodes['Background'].inputs[1].default_value=.8
 sc.view_settings.view_transform='Standard';sc.view_settings.look='Medium High Contrast' if 'Medium High Contrast' in [] else 'None'
 sc.view_settings.exposure=0;sc.view_settings.gamma=1
 for loc,power,size in [((3,-4,7),550,5),((-5,2,4),300,4)]:
  bpy.ops.object.light_add(type='AREA',location=loc);l=bpy.context.object;l.data.energy=power;l.data.size=size;l.rotation_euler=(-l.location).to_track_quat('-Z','Y').to_euler()
 return sc
def main():
 info=json.loads((ROOT/'summary.json').read_text())
 names=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else ['AdaPoinTr','VGGT-native','LiDAR-R8','Open-r3','Attraction-r4','First-r6']
 for name in names:
  data=np.load(ROOT/'cases'/(name+'.npz'));v=data['vertices'];f=data['faces'];rid=int(data['selected_ray']);fid=int(data['face_ids'][rid]);d=data['directions'][rid];o=data['origins'][rid];r=data['ranges'][rid];first=data['first'][rid];x=o+d*first;y=o+d*r;center=(v.min(0)+v.max(0))/2;extent=max(np.ptp(v,axis=0));comp=data['components'];cid=comp[fid]
  for mode in os.environ.get('WORLDSIM_RENDER_MODES','whole,detail').split(','):
   sc=setup();gray=mat('Original surface',(.65,.71,.77));red=mat('Early-hit triangle',(.80,.045,.055));orange=mat('Selected connected chart',(.95,.48,.18));green=mat('Heldout measurement',(.06,.58,.28));blue=mat('Build observation',(.10,.32,.68));dark=mat('Ray direction',(.13,.16,.20));gold=mat('Selected observed endpoint',(.07,.72,.24))
   early=np.isfinite(data['first'])&(data['first']<data['ranges']-.2);earlyfaces=np.unique(data['face_ids'][early]);mi=np.zeros(len(f),int);mi[comp==cid]=2;mi[earlyfaces]=1
   if mode=='whole':
    mesh('Original_surface',v,f,[gray,red,orange],mi);bp=data['build_points'];tp=data['target_points'];bp=bp[np.linspace(0,max(0,len(bp)-1),min(800,len(bp))).astype(int)] if len(bp) else bp;tp=tp[np.linspace(0,max(0,len(tp)-1),min(1000,len(tp))).astype(int)] if len(tp) else tp
    spheres('Build_points',bp,.018,blue);spheres('Heldout_endpoints',tp,.019,green)
    line('Selected_ray',x-d*1.2,y+d*.3,.012,dark);spheres('First_hit',[x],.055,red);spheres('Measured_endpoint',[y],.055,gold)
    camera(center,center+np.array([1.0,-1.5,1.1])*extent,extent*1.2)
   else:
    # 局部视图隔离责任连通面片，避免其他面片遮挡；查询仍使用完整原网格。
    keep=(comp==cid)
    localmi=mi.copy();localmi[:]=0;localmi[comp==cid]=2;localmi[fid]=1
    mesh('Isolated_responsible_chart_original_triangles',v,f[keep],[gray,red,orange],localmi[keep])
    bp=data['build_points'];tp=data['target_points'];bp=bp[np.linalg.norm(bp-(x+y)/2,axis=-1)<1.3];tp=tp[np.linalg.norm(tp-(x+y)/2,axis=-1)<1.3]
    spheres('Build_points',bp,.016,blue);spheres('Heldout_endpoints',tp,.018,green);line('Exact_selected_ray',x-d*.7,y+d*.35,.008,dark);spheres('First_hit',[x],.04,red);spheres('Measured_endpoint',[y],.045,gold)
    tri=v[f[fid]];n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);n=n/max(1e-8,np.linalg.norm(n))
    if n[2]<0:n=-n
    side=np.cross(d,[0,0,1]);side=side/max(1e-8,np.linalg.norm(side));view=n+.55*side+.2*np.array([0,0,1]);view/=np.linalg.norm(view)
    look=(x+y)/2;camera(look,look+4*view,max(1.8,np.linalg.norm(y-x)+1.3))
   sc.render.filepath=str(OUT/(name+'_'+mode+'.png'))
   bpy.ops.wm.save_as_mainfile(filepath=str(OUT/(name+('' if mode=='whole' else '_detail')+'.blend')))
   bpy.ops.render.render(write_still=True);print(name,mode,flush=True)
if __name__=='__main__':main()
