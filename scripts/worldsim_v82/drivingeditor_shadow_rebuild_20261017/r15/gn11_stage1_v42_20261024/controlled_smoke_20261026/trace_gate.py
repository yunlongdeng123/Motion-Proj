from editor_common import *
from utils.v4_road_render import settings
import diff_gaussian_rasterization as native
sys.path.insert(0,'/root/autodl-tmp/gsdiag_gpu_20261021_r2/build')
import gsdiag_native_trace_r1 as ext
n=Native();c=n.camera(102);gate,_=make_gate(n);empty=torch.empty(0,device='cuda')
m=read('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json');T=np.asarray(m['world_to_model'])@np.asarray(m['coordinate_anchor']);inv=np.linalg.inv(T)
with torch.no_grad():
    k=prepare_rasterization(c,n.static,n.args);st=settings(c,n.args,n.static.active_sh_degree,n.bg);val=lambda key:k[key] if k.get(key) is not None else empty
    if len(sys.argv)>1:
        gate=torch.as_tensor(np.load(OUT/'fixed_gate_ray_depth.npz')['suppress'],device='cuda');k['mask']=k['mask']&~gate
    r=native._C.rasterize_gaussians(st.bg,k['means3D'].contiguous(),val('colors_precomp').contiguous(),k['features'].contiguous(),k['opacities'].contiguous(),val('scales').contiguous(),val('rotations').contiguous(),st.scale_modifier,val('cov3D_precomp').contiguous(),k['mask'].contiguous(),st.viewmatrix,st.projmatrix,st.tanfovx,st.tanfovy,320,480,val('shs').contiguous(),st.sh_degree,st.campos,False,False)
    points=[(255,240),(256,219)];ti,tr,tc,ts=ext.trace(r[5],r[6],r[7],val('colors_precomp').contiguous(),k['features'].contiguous(),torch.tensor(points,device='cuda',dtype=torch.int32),r[4],len(k['means3D']),480,320,32768)
    rows=[]
    for q,xy in enumerate(points):
        count=int(tc[q,1]);records=cpu(tr[q,:count]);idx=cpu(ti[q,:count]);pick=np.argsort(-records[:,3])[:20]
        top=[]
        for j in pick:
            i=int(idx[j]);xyz=cpu(k['means3D'][i]);local=xyz@inv[:3,:3].T+inv[:3,3]
            top.append(dict(row=i,weight=float(records[j,3]),local=local.tolist(),base=cpu(n.static.get_xyz[i]).tolist(),sigma=(cpu(n.static.get_scaling[i])/m['meter_to_model']).tolist(),gate=bool(gate[i]),uv=records[j,7:9].tolist()))
        rows.append(dict(pixel=xy,top=top))
    (OUT/('gate_trace_remaining.json' if len(sys.argv)>1 else 'gate_trace.json')).write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
