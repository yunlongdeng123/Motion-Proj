"""Render exact actor-axis orthographic views, without material or geometry edits."""
import argparse
import importlib.util
from pathlib import Path
import json
import numpy as np
from PIL import Image

p=argparse.ArgumentParser()
p.add_argument('--audit-script',type=Path,required=True)
p.add_argument('--mesh',type=Path,required=True)
p.add_argument('--audit-dir',type=Path,required=True)
a=p.parse_args()
spec=importlib.util.spec_from_file_location('hq_audit',a.audit_script)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
mesh=m.load_mesh(a.mesh)
bank=np.load(a.audit_dir/'point_surface_residuals.npz',allow_pickle=False)
views=[('Front +X',[1,0,0]),('Rear -X',[-1,0,0]),('Left +Y',[0,1,0]),
       ('Right -Y',[0,-1,0]),('Top +Z',[0,0,1]),('Bottom -Z',[0,0,-1])]
for name,points,dist in [('grey_six_orthographic.png',None,None),
                         ('measured_points_six_orthographic.png',bank['xyz_canonical_m'],bank['point_to_surface_m'])]:
    footer='Exact axis-aligned orthographic; x forward / y left / z up'
    if points is not None:footer='All points projected; blue 0m to red >=0.30m; no visibility claim'
    tiles=[m.ortho(mesh,d,t,footer,size=760,points=points,distances=dist) for t,d in views]
    Image.fromarray(np.concatenate([np.concatenate(tiles[:3],1),np.concatenate(tiles[3:],1)],0)).save(a.audit_dir/name)
topo=mesh.copy();topo.merge_vertices(digits_vertex=6)
comps=m.trimesh.graph.connected_components(topo.face_adjacency,nodes=np.arange(len(topo.faces)),min_len=1)
comps=sorted(comps,key=lambda ids:topo.area_faces[ids].sum(),reverse=True)
if len(comps)>1:
    component=topo.submesh([comps[1]],append=True,repair=False)
    # Position is conveyed by main-mesh views with component vertices shown as red markers.
    pts=np.asarray(component.vertices)
    fake=np.full(len(pts),.3)
    tiles=[m.ortho(mesh,d,t,'Red = second connected component; geometry not edited',size=760,points=pts,distances=fake)
           for t,d in [('Left component locator',[0,1,0]),('Top component locator',[0,0,1]),('Front component locator',[1,0,0])]]
    Image.fromarray(np.concatenate(tiles,1)).save(a.audit_dir/'second_component_locator.png')
record={'mesh':m.digest(a.mesh),'orthographic_directions_actor':[d for _,d in views],
        'geometry_modified':False,'gpu_used':False,'heldout_used':False,
        'images':{name:m.digest(a.audit_dir/name) for name in ['grey_six_orthographic.png','measured_points_six_orthographic.png','second_component_locator.png'] if (a.audit_dir/name).exists()},
        'script':m.digest(__file__)}
m.write_json(a.audit_dir/'exact_orthographic_receipt.json',record)
print(json.dumps(record,indent=2))
