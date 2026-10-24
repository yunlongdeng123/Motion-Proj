"""Frozen native-buffer attribution and causal removals, never a training patch."""
import sys, os, json, gc, argparse, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from PIL import Image

SOURCE = Path('/root/autodl-tmp/motion_proj_v82_gn11_v4/scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/gn11_stage1_v3_20261020/diagnostics_20261021/src')
sys.path.insert(0, str(SOURCE))
from export_v41_results import cpu_adapter, binding, unchanged, init_gpu, camera, save_json, DEFAULT_CAMPAIGN

def main(out):
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    start=datetime.now(timezone.utc).isoformat()
    a=cpu_adapter();run,bindings=binding(DEFAULT_CAMPAIGN,'R')
    run,args,models,road,arrays=init_gpu(DEFAULT_CAMPAIGN,'R',a)
    import torch
    import diff_gaussian_rasterization as native
    from backend import digest, clean, CAMPAIGN as V3
    from gaussian_renderer import prepare_rasterization
    from utils.v4_road_render import settings, road_kwargs, merge_rows, render_composite
    sys.path.insert(0,'/root/autodl-tmp/gsdiag_gpu_20261021_r2/build')
    import gsdiag_native_trace_r1 as ext
    expected='0eaff4d7a9b8ef6e711627f4c19824bf427c4387f7c6453fcb769a2c919c920e'
    if digest(native._C.__file__)!=expected: raise RuntimeError('uninspected native buffer ABI')
    static,dynamic,env=models;ns=len(static._xyz);nd=len(dynamic._xyz)
    bg=torch.tensor([1.,1.,1.] if args.white_background else [0.,0.,0.],device='cuda')
    if bg.any(): raise RuntimeError('trace aggregate currently assumes black raster background')
    empty=torch.empty(0,device='cuda')
    def render(k,st):
        val=lambda n:k[n] if k.get(n) is not None else empty
        r=native._C.rasterize_gaussians(st.bg,k['means3D'].contiguous(),val('colors_precomp').contiguous(),k['features'].contiguous(),k['opacities'].contiguous(),val('scales').contiguous(),val('rotations').contiguous(),st.scale_modifier,val('cov3D_precomp').contiguous(),k['mask'].contiguous(),st.viewmatrix,st.projmatrix,st.tanfovx,st.tanfovy,320,480,val('shs').contiguous(),st.sh_degree,st.campos,False,False)
        if not torch.isfinite(r[2]).all() or not torch.isfinite(r[3]).all():raise FloatingPointError('nonfinite native image')
        return r
    def prep(raw,mode):
        c=camera(a,raw,run);st=settings(c,args,static.active_sh_degree,bg)
        sk=prepare_rasterization(c,static,args);sk['means2D']=torch.zeros_like(sk['means3D'])
        k=sk;branches=[0]*ns;rows=list(range(ns))
        if mode=='joint':
            dk=prepare_rasterization(c,dynamic,args,dynamic=True);dk['means2D']=torch.zeros_like(dk['means3D'])
            k=merge_rows(k,dk);branches += [1]*nd;rows += list(range(nd))
        if mode in ('joint','deleted_five','road_only'):
            rk=road_kwargs(c,road,args,k['shs'].shape[1] if k['shs'] is not None else 1,k['colors_precomp'] is not None)
            if mode=='road_only':k=rk;branches=[2]*len(road.xyz);rows=list(range(len(road.xyz)))
            else:k=merge_rows(k,rk);branches += [2]*len(road.xyz);rows += list(range(len(road.xyz)))
        environment=env(c.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
        return c,st,k,np.array(branches),np.array(rows),environment
    def image(r,e):return r[2]+(1-r[3][-1:])*e
    def webp(rel,im):
        path=out/rel;path.parent.mkdir(parents=True,exist_ok=True)
        v=im.detach().cpu().numpy() if torch.is_tensor(im) else im
        Image.fromarray(np.round(v.clip(0,1).transpose(1,2,0)*255).astype(np.uint8)).save(path,lossless=True)
        return rel
    def js(rel,key,value):
        path=out/rel;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('window.GSDATA=window.GSDATA||{};window.GSDATA['+json.dumps(key)+']='+json.dumps(clean(value),ensure_ascii=False,allow_nan=False)+';\n',encoding='utf-8')
    queries={};cfs={};summaries=[];selections={}
    points=[(255,240),(255,214),(255,290),(90,255),(365,280)]
    with torch.inference_mode():
        for mode in ('ordinary','deleted_five','joint','road_only'):
            c,st,k,branch,rows,e=prep(102,mode);r=render(k,st);P=len(rows)
            cols=k['colors_precomp'] if k['colors_precomp'] is not None else empty
            ti,tr,tc,ts=ext.trace(r[5],r[6],r[7],cols.contiguous(),k['features'].contiguous(),torch.tensor(points,device='cuda',dtype=torch.int32),r[4],P,480,320,32768)
            if mode in ('ordinary','deleted_five','joint'):
                reference=render_composite(c,static,dynamic if mode=='joint' else None,road if mode!='ordinary' else None,args,bg,env)
                full_replay=float((image(r,e)-reference['rgb']).abs().max())
                if full_replay>2e-5:raise RuntimeError('native whole-image replay failed')
                del reference
            else:full_replay=None
            for q,(x,y) in enumerate(points):
                n=int(tc[q,1]);assert n<=32768
                rec=tr[q,:n].cpu().numpy();ids=ti[q,:n].long();local=ids.cpu().numpy();s=ts[q].cpu().numpy()
                f=r[3][:,y,x].cpu().numpy();rgb=r[2][:,y,x].cpu().numpy()
                errors=dict(alpha_abs=float(abs(s[4]-f[-1])),depth_sum_model_abs=float(abs(s[3]-f[-2])),rgb_pre_environment_abs_max=float(abs(s[:3]-rgb).max()),T_abs=float(abs(s[5]-s[6])))
                if max(errors.values())>2e-5:raise RuntimeError(str(errors))
                picked=np.argsort(-rec[:,3],kind='stable')[:40]
                flags=torch.zeros((P,min(20,len(picked))+2),device='cuda')
                for j,i in enumerate(picked[:20]):flags[ids[i],j]=1
                flags[:,-2:]=k['features'][:,-2:]
                fk=dict(k,features=flags,shs=None,colors_precomp=torch.zeros_like(k['means3D']))
                vr=render(fk,st);err=float((vr[3][:len(picked[:20]),y,x].cpu()-torch.from_numpy(rec[picked[:20],3])).abs().max()) if n else 0.
                if err>2e-6:raise RuntimeError('one-hot weight replay failed')
                del flags,fk,vr
                top=[];profile=[]
                for i,v in enumerate(rec):
                    role=int(branch[local[i]]);row=int(rows[local[i]]);name=('ordinary','dynamic','road')[role]
                    profile.append(dict(z_m=float(v[0]/a.scale),weight=float(v[3]),T=float(v[2]),branch=name))
                    if i not in picked:continue
                    xyz=k['means3D'][ids[i]].cpu().numpy();world=a.invA[:3,:3]@xyz+a.invA[:3,3]
                    cpname=('static','dynamic','road')[role];b=bindings[cpname];aa,bb,cc=v[9:12];det=aa*cc-bb*bb
                    m=models[role] if role<2 else road
                    lineage=getattr(m,'_sgcm_lineage_ids',None)
                    top.append(dict(id=f'v4_1_R:{b["sha256"][:12]}:{cpname}:{row}',row=row,branch=name,checkpoint_sha256=b['sha256'],road_id=int(road.unit_ids[row]) if role==2 else -1,
                        source_lineage=int(lineage[row]) if lineage is not None else None,world_m=world,model_xyz=xyz,camera_z_m=v[0]/a.scale,alpha=v[1],T=v[2],weight=v[3],color=v[4:7],color_contribution=v[4:7]*v[3],
                        sigma_m=(m.get_scaling[row] if role<2 else road.scaling[row]).cpu().numpy()/a.scale,
                        quaternion=(m.get_rotation[row] if role<2 else road.rotation[row]).cpu().numpy(),cov2=[cc/det,-bb/det,aa/det],uv=v[7:9],radius_px=v[12],
                        opacity_base=float(m.get_opacity[row]) if role<2 else float(road.opacity[row].sigmoid()),time_center=float(m._t[row]) if role<2 else 0.,sigma_t=float(m._scaling_t[row].exp()) if role<2 else None,
                        velocity_model=m._velocity[row].cpu().numpy() if role<2 else [0,0,0]))
                groups={name:float(rec[branch[local]==j,3].sum()) for j,name in enumerate(('ordinary','dynamic','road'))}
                value=dict(status='NATIVE_BUFFER_CUDA_TRACE_VERIFIED',arm='v4_1_R',raw=102,pixel=[x,y],mode=mode,candidate_count=int(tc[q,0]),accepted_count=n,top=top,profile=profile,
                    reference=dict(rgb_gaussians=s[:3],alpha=s[4],T=s[5],depth_sum_model=s[3]),native=dict(alpha=f[-1],depth_sum_model=f[-2]),
                    replay_errors=errors,saved_final49_replay_errors=None,whole_image_replay_max_abs=full_replay,numeric_alpha_depth_agree=True,
                    id_weight_validation=dict(status='NATIVE_ONE_HOT_ID_WEIGHT_REPLAY_PASSED',count=len(picked[:20]),max_weight_error=err),
                    native_rasterizer_binary_sha256=expected,groups=groups,environment_color_not_in_reference_rgb=True,parameter_updates=0,no_optimizer=True,gpu_used=True,
                    note='Top weights identify causes of rendered pixels, not automatic incorrect-Gaussian labels. No hidden-road RGB ground truth.')
                key=f'gpu/v4_1_R/102/{x}/{y}/{mode}';rel=f'data/gpu/v4_1_R_102_{x}_{y}_{mode}.js';js(rel,key,value);queries[key]=rel
                summaries.append(dict(pixel=[x,y],mode=mode,groups=groups,alpha=float(s[4]),rgb=clean((r[2] if mode=='road_only' else image(r,e))[:,y,x].cpu().numpy()),top=clean(top[:3]),replay_errors=errors))
                if mode=='deleted_five':
                    ordinary=[i for i in np.argsort(-rec[:,3],kind='stable') if branch[local[i]]==0][:20]
                    selections[(x,y)]=[int(rows[local[i]]) for i in ordinary]
            del c,st,k,r,ti,tr,tc,ts;gc.collect();torch.cuda.empty_cache()
            print('traced',mode,flush=True)
        # Causal diagnostic only: mask selected frozen static rows from raster inputs;
        # no checkpoint parameter mutation and no claim these rows are safely deletable.
        for x,y in points:
            sel=selections[(x,y)];all_frames=[];images={}
            for raw in (102,89,113,137):
                c,st,k,branch,rows,e=prep(raw,'deleted_five')
                before=render(k,st);bk=image(before,e).cpu().numpy();ba=before[3][-1].cpu().numpy()
                def road_weight(k):
                    flags=torch.cat((torch.tensor(branch==2,device='cuda')[:,None].float(),k['features'][:,-2:]),1)
                    rr=render(dict(k,features=flags,shs=None,colors_precomp=torch.zeros_like(k['means3D'])),st)
                    return rr[3][0].cpu().numpy()
                rb=road_weight(k);del before
                ak=dict(k,mask=k['mask'].clone());ak['mask'][sel]=False
                after=render(ak,st);ai=image(after,e).cpu().numpy();aa=after[3][-1].cpu().numpy();ra=road_weight(ak)
                delta=np.abs(ai-bk).mean(0);roi=np.zeros((320,480),bool);roi[max(0,y-10):y+11,max(0,x-10):x+11]=True
                with np.load(V3/f'final49/full_r2/0008{raw:03d}/render_float.npz') as z:gt=z['gt'];body=z['body_mask'][0].astype(bool)
                normal=~body
                metrics=dict(selected_pixel_L1_change=float(delta[y,x]),selected_pixel_road_before=float(rb[y,x]),selected_pixel_road_after=float(ra[y,x]),selected_pixel_alpha_before=float(ba[y,x]),selected_pixel_alpha_after=float(aa[y,x]),mean_abs_RGB_change=float(delta.mean()),outside_ROI_RGB_change=float(delta[~roi].mean()),changed_pixel_count=int((delta>1/255).sum()),
                    observed_mask_complement_count=int(normal.sum()),observed_mask_complement_MAE_before=float(np.abs(bk-gt)[:,normal].mean()),observed_mask_complement_MAE_after=float(np.abs(ai-gt)[:,normal].mean()),
                    observed_mask_complement_harmed_pixels=int(((np.abs(ai-gt).mean(0)-np.abs(bk-gt).mean(0)>1/255)&normal).sum()),
                    normal_control_limitation='target-body complement is an observed-output damage screen, not verified road semantics or hidden-road ground truth')
                prefix=f'assets/ghost_v41/{x}_{y}/{raw}'
                paths={}
                for name,im in [('before',bk),('after',ai),('difference',np.abs(ai-bk)*5),('road_before',np.repeat(rb[None],3,0)),('road_after',np.repeat(ra[None],3,0))]:paths[name]=webp(prefix+'_'+name+'.webp',im)
                all_frames.append(dict(raw=raw,metrics=metrics,images=paths))
                if raw==102:images=paths
                del c,st,k,ak,after;gc.collect();torch.cuda.empty_cache()
            removed=[f'v4_1_R:{bindings["static"]["sha256"][:12]}:static:{row}' for row in sel]
            value=dict(status='FROZEN_NATIVE_COUNTERFACTUAL_RENDERED',arm='v4_1_R',raw=102,pixel=[x,y],mode='deleted_five',removed=removed,selection_kind='top20_static_actual_alpha_weight',images=images,metrics=all_frames[0]['metrics'],temporal_controls=all_frames[1:],
                no_optimizer=True,parameter_updates=0,checkpoint_parameters_modified=False,quality_accepted=False,human_verdict=None)
            key=f'cf/v4_1_R/102/{x}/{y}';rel=f'data/counterfactual/v4_1_R_102_{x}_{y}.js';js(rel,key,value);cfs[key]=rel
            print('counterfactual',x,y,flush=True)
        if not unchanged(bindings):raise RuntimeError('protected input changed')
        save_json(out/'ghost_summary.json',dict(started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),queries=summaries,bindings=bindings,parameter_updates=0,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),quality_accepted=False,human_verdict=None))
        text='window.GSDATA=window.GSDATA||{};window.GSDATA.gpu_index=window.GSDATA.gpu_index||{};\n'
        for key,value in [('queries',queries),('counterfactuals',cfs)]:text+='Object.assign(window.GSDATA.gpu_index.'+key+'||(window.GSDATA.gpu_index.'+key+'={}),'+json.dumps(value)+');\n'
        (out/'v41_ghost_index.js').write_text(text,encoding='utf-8')
    print('COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
