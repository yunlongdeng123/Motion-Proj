"""Attribute the USER-SELECTED preservation C, without altering its objective."""
from editor_common import *
from utils.v4_road_render import settings
import diff_gaussian_rasterization as native
sys.path.insert(0,'/root/autodl-tmp/gsdiag_gpu_20261021_r2/build')
import gsdiag_native_trace_r1 as ext


def main():
    dest=OUT/'c_selected_mechanism';dest.mkdir(exist_ok=True)
    n=Native();c=n.camera(102);empty=torch.empty(0,device='cuda')
    ck=torch.load(OUT/'edit_smoke_preserve/full199/checkpoint1000_from0.pth',weights_only=False)
    assert ck['master']==n.before
    road=Road(FULL/'full199_initialization.npz');road.dc.data.copy_(ck['dc']);road.opacity.data.copy_(ck['opacity'])
    mult=torch.ones(len(n.static.get_xyz),device='cuda').index_copy(0,ck['ids'],ck['gate'].sigmoid())
    meta=read('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json')
    scale=meta['meter_to_model'];inverse=np.linalg.inv(np.asarray(meta['world_to_model'])@np.asarray(meta['coordinate_anchor']))
    points=[(255,240),(256,219),(255,270),(255,290),(230,260),(280,250),(90,255),(365,280)]
    data={};ns=len(n.static.get_xyz)
    with torch.no_grad():
        for name,gate,rd in [('A_original',None,None),('B_gate_only',mult,None),('C_selected',mult,road)]:
            k=prepare_rasterization(c,n.static,n.args);k['means2D']=torch.zeros_like(k['means3D'])
            if gate is not None:k['opacities']=k['opacities']*gate[:,None]
            if rd is not None:k=merge_rows(k,road_kwargs(c,rd,n.args,k['shs'].shape[1] if k['shs'] is not None else 1,precomputed_color=k['colors_precomp'] is not None))
            val=lambda key:k[key] if k.get(key) is not None else empty
            st=settings(c,n.args,n.static.active_sh_degree,n.bg)
            r=native._C.rasterize_gaussians(st.bg,k['means3D'].contiguous(),val('colors_precomp').contiguous(),k['features'].contiguous(),k['opacities'].contiguous(),val('scales').contiguous(),val('rotations').contiguous(),st.scale_modifier,val('cov3D_precomp').contiguous(),k['mask'].contiguous(),st.viewmatrix,st.projmatrix,st.tanfovx,st.tanfovy,320,480,val('shs').contiguous(),st.sh_degree,st.campos,False,False)
            ti,tr,tc,ts=ext.trace(r[5],r[6],r[7],val('colors_precomp').contiguous(),k['features'].contiguous(),torch.tensor(points,device='cuda',dtype=torch.int32),r[4],len(k['means3D']),480,320,32768)
            ref=n.render(c,rd,static_multiplier=gate);environment=n.env(c.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
            replay=r[2]+(1-r[3][1:2])*environment
            assert float((replay-ref['rgb']).abs().max())<2e-5
            for q,xy in enumerate(points):
                count=int(tc[q,1]);rec=cpu(tr[q,:count]);ids=cpu(ti[q,:count]);s=cpu(ts[q]);x,y=xy
                errors=dict(alpha=abs(float(s[4]-r[3][1,y,x])),rgb=float(np.abs(s[:3]-cpu(r[2][:,y,x])).max()),depth_sum=abs(float(s[3]-r[3][0,y,x])))
                assert max(errors.values())<2e-5 and count<32768
                weights=rec[:,3];order=np.argsort(rec[:,0]);cum=np.cumsum(weights[order]);quantiles=[]
                for fraction in [.1,.5,.9]:quantiles.append(float(rec[order[min(np.searchsorted(cum,fraction*cum[-1]),len(order)-1)],0]/scale))
                top=[]
                for j in np.argsort(-weights)[:30]:
                    idx=int(ids[j]);v=rec[j];pos=cpu(k['means3D'][idx]);local=pos@inverse[:3,:3].T+inverse[:3,3]
                    lineage=getattr(n.static,'_sgcm_lineage_ids',None)
                    top.append(dict(row=idx,branch='static' if idx<ns else 'road',weight=float(v[3]),alpha=float(v[1]),T=float(v[2]),z_m=float(v[0]/scale),rgb=v[4:7].tolist(),local_xyz_m=local.tolist(),
                                    opacity_multiplier=float(mult[idx]) if idx<ns else None,lineage=int(lineage[idx]) if idx<ns and lineage is not None else None))
                data.setdefault(f'{x},{y}',{})[name]=dict(rgb=cpu(ref['rgb'][:,y,x]).tolist(),alpha=float(s[4]),environment_weight=float(1-s[4]),expected_depth_m=float(s[3]/max(s[4],1e-8)/scale),depth_weight_quantiles_m=quantiles,
                    static_weight=float(weights[ids<ns].sum()),road_weight=float(weights[ids>=ns].sum()),accepted=count,top=top,replay_errors=errors,
                    static_weights={str(int(i)):float(w) for i,w in zip(ids,weights) if i<ns})
        for pixel,variants in data.items():
            aa=variants['A_original'];bb=variants['B_gate_only'];cc=variants['C_selected']
            oldids=[str(x['row']) for x in aa['top'][:20]]
            variants['attribution']=dict(original_top20_weight_before=sum(aa['static_weights'].get(i,0) for i in oldids),original_top20_weight_after=sum(cc['static_weights'].get(i,0) for i in oldids),
                C_minus_B_rgb_maxabs=float(np.max(np.abs(np.array(cc['rgb'])-bb['rgb']))),
                interpretation='Appearance attribution only. Gaussian centre depth/lineage does not establish true road geometry.')
            for v in [aa,bb,cc]:del v['static_weights']
    assert n.unchanged()
    result=dict(checkpoint='edit_smoke_preserve/full199/checkpoint1000_from0.pth',parameter_updates=0,points=data,master_unchanged=True,
                geometry_and_static_colors_frozen=True,quality_accepted=False,human_verdict=None)
    (dest/'pixel_trace.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:{name:{key:v for key,v in val.items() if key!='top'} for name,val in variants.items()} for k,variants in data.items()}),flush=True)


if __name__=='__main__':main()
