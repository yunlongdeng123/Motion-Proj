from pathlib import Path
import trimesh,numpy as np,json
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
BASE=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927');rows=[]
for run in ['r40','r41']:
 mesh=trimesh.load(BASE/run/'shape_untextured.glb',force='mesh',process=False);e=mesh.edges_unique;g=coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(len(mesh.vertices),len(mesh.vertices))).tocsr();n,labels=connected_components(g,directed=False);face_labels=labels[mesh.faces[:,0]];face_counts=np.bincount(face_labels,minlength=n);area=np.asarray(mesh.area_faces);rows.append(dict(run=run,vertices=len(mesh.vertices),faces=len(mesh.faces),connected_components=n,largest_component_faces=int(face_counts.max()),components_under100_faces=int((face_counts<100).sum()),faces_in_components_under100=int(face_counts[face_counts<100].sum()),zero_area_faces=int((area<1e-12).sum()),finite_vertices=bool(np.isfinite(mesh.vertices).all()),watertight=bool(mesh.is_watertight)))
path=BASE/'r41/topology_audit.json';assert not path.exists();path.write_text(json.dumps(dict(rows=rows,scope='Geometry topology inspection; not proof of actor identity or appearance'),indent=2)+'\n');print(rows)
