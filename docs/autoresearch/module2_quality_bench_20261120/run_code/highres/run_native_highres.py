"""Root-authorized zero-update 1920x1280 native evidence; new independent lease."""
from pathlib import Path
import argparse,json,sys,time,hashlib,traceback,gc,subprocess
import numpy as np

B=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
Q=B/'road_teacher_detail_cpu_20261119_r1';G=Q/'gpu_20261119'
DEFAULT=G/'quality_bench_20261120_r1/highres'
ASSET=B/'road_activation_20261118_r1/B32_beam95_native_r1/asset_guarded'
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
OWNER='quality_bench_highres';MAX_CALLS=12
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}

def setup(out):
    out.mkdir(parents=True,exist_ok=True)
    if not(out/'authorization.json').exists():
        write(out/'authorization.json',dict(schema='read_only_highres_native_bench.v1',GPU_authorized=True,
            user_authorization='user requested improved quality bench and re-evaluation; root explicitly authorized two-frame actual highres probe',
            root_authorization='110 then86; 1920x1280; core8 plus optional M10four; maximum12 native calls; no training or production writes',
            frames=[110,86],resolution=[1920,1280],max_native_forward_calls=12,formal_train=False,optimizer_updates=0,
            write_scope='new highres evidence only',geometry_opacity_scale_active_world_fields_unchanged=True,
            fixed_scope_rows=89887,source_asset=str(ASSET),GPU_keep_on=True,power_actions=False))
        write(out/'gpu_lease.json',dict(owner=OWNER,active=True,gpu_authorized=True,scope='highres readonly probe only',max_native_forward_calls=12))
    if not(out/'native_counter.json').exists():write(out/'native_counter.json',dict(attempts=0,completed=0,rows=[]))

def run(out,do_m10):
    setup(out);auth=read(out/'authorization.json');lease=read(out/'gpu_lease.json')
    if not auth.get('GPU_authorized')or auth.get('max_native_forward_calls')!=12 or auth.get('formal_train')is not False:raise RuntimeError('invalid authorization')
    if not lease.get('active')or lease.get('owner')!=OWNER:raise RuntimeError('no independent lease')
    # No model import before a real capacity check. Do not evict other user tasks.
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
    if int(lines[0])<12288:
        write(out/'capacity_wait.json',dict(free_MiB=int(lines[0]),required_free_MiB=12288,native_attempts=0,other_tasks_untouched=True));return
    import torch
    torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
    sys.path[:0]=[str(B/'road_capacity_gpu_20261118_r1/code'),str(REPO/'scripts/worldsim_v82'),str(Q/'code')]
    from road_capacity_native import NativeRoadContext
    from drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff
    from drivingeditor_dynamic_edit.v44_manifest import SceneManifest,make_native_camera
    manifest=ASSET/'persistent/scene.json';proof=ASSET/'persistent/proof.json'
    scene,m,receipt=load_persistent_handoff(manifest,proof)
    low_headers=m.camera_headers();data=dict(m.data);data['render_wh']=[1920,1280]
    high_headers=SceneManifest(data,manifest.parent).camera_headers();cameras={}
    def camera(raw):
        if raw not in cameras:cameras[raw]=make_native_camera(high_headers[raw])
        return cameras[raw]
    scene.camera_provider=camera
    ctx=NativeRoadContext(scene,m,receipt,ASSET,[110,86],out/'gpu_lease.json',OWNER)
    ctx._manifest_path=manifest;ctx._proof_path=proof
    scopefile=G/'native_qualification_r2/basis/110/row_scope.npz';scope=load(scopefile)
    assert np.array_equal(scope['active_world_rows'],ctx.active_world_rows)
    eligible=scope['eligible_active'];assert eligible.dtype==bool and int(eligible.sum())==89887
    counter=read(out/'native_counter.json')
    def raster(raw,mode,rows):
        ctx.require_lease()
        if counter['attempts']>=MAX_CALLS:raise RuntimeError('native budget exhausted')
        counter['attempts']+=1;row=dict(call=counter['attempts'],raw=raw,mode=mode,status='started');counter['rows'].append(row);write(out/'native_counter.json',counter)
        ctx.native_calls+=1;value=scene._bindings().raster(camera(raw),scene.pipe,0,rows,scene._background)
        torch.cuda.synchronize();row['status']='completed';counter['completed']+=1;write(out/'native_counter.json',counter)
        return value
    def joint_rows(raw):
        rows,owners,cam,coverage=scene._rows(raw,(),(),False)
        rows=dict(rows,mask=rows['mask']&~scene._contains(owners,(10,)))
        ns=len(scene.static.get_xyz);nd=len(scene.dynamic.get_xyz)if scene.dynamic is not None and scene._include_dynamic else 0
        extras=[v for v in coverage if v['available']]
        if any(v['kind']!='road'for v in extras):raise RuntimeError('unexpected nonroad provider; source decomposition must remain explicit')
        nr=sum(v['rows']for v in extras);assert nr==len(ctx.base_rgb)and len(rows['means3D'])==ns+nd+nr
        assert torch.equal(rows['means3D'][ns+nd:],ctx.base_packet(raw)['means3D'])
        labels=torch.empty(len(rows['means3D']),device=ctx.base_rgb.device,dtype=torch.int8);labels[:ns]=0;labels[ns:ns+nd]=1;labels[ns+nd:]=2
        return rows,labels,dict(static_rows=ns,dynamic_rows=nd,road_rows=nr,provider_coverage=coverage)
    def env(raw,alpha):return (1-alpha)*scene.environment(camera(raw).get_world_directions(False).permute(1,2,0)).permute(2,0,1)
    def export(raw,mode,folder):
        if mode=='road':rows=ctx.base_packet(raw);labels=torch.full((len(ctx.base_rgb),),2,dtype=torch.int8,device=ctx.base_rgb.device);layout=dict(road_rows=len(ctx.base_rgb))
        else:rows,labels,layout=joint_rows(raw)
        colors=rows['colors_precomp'];e=torch.zeros(len(colors),dtype=torch.bool,device=colors.device);e[labels==2]=torch.as_tensor(eligible,device=e.device)
        f=colors.new_zeros((len(colors),24));f[:,:2]=rows['features'][:,:2]
        f[:,2:5]=colors*e[:,None];f[:,5]=e.to(colors.dtype)
        f[:,6:9]=colors*(~e)[:,None];f[:,9]=rows['features'][:,0]*e;f[:,10]=rows['features'][:,0].square()*e
        f[:,11]=(labels==2).to(colors.dtype);f[:,12]=(~e).to(colors.dtype)
        f[:,13:16]=colors*(labels==0)[:,None];f[:,16:19]=colors*(labels==1)[:,None];f[:,19:22]=colors*(labels==2)[:,None]
        f[:,22]=(labels==0).to(colors.dtype);f[:,23]=(labels==1).to(colors.dtype)
        with torch.no_grad():v=raster(raw,mode+'_basis',dict(rows,features=f));environment=env(raw,v['alpha'])if mode=='joint'else torch.zeros_like(v['rgb'])
        z=v['extra'];base=v['rgb']+environment
        color_error=float((z[11:14]+z[14:17]+z[17:20]+environment-base).abs().max())
        alpha_error=float((z[20]+z[21]+z[9]-v['alpha'].squeeze()).abs().max())
        # Transfer one extra image once; field names below are NumPy views, not copies.
        Z=z.detach().cpu().numpy();E=environment.detach().cpu().numpy();P=base.detach().cpu().numpy();A=v['alpha'].squeeze().detach().cpu().numpy()
        arrays=dict(baseline_rgb=P,eligible_rgb=Z[0:3],eligible_weight=Z[3],readonly_rgb=P-Z[0:3],direct_readonly_rgb=Z[4:7]+E,
            eligible_depth_mass_model=Z[7],eligible_depth2_mass_model2=Z[8],entire_road_weight=Z[9],readonly_weight=Z[10],
            static_rgb=Z[11:14],dynamic_rgb=Z[14:17],whole_road_rgb=Z[17:20],static_alphaT=Z[20],dynamic_alphaT=Z[21],
            env_rgb=E,alpha=A,K=ctx.K_masks[raw])
        folder.mkdir(parents=True,exist_ok=True);path=folder/(mode+'_basis.npz');np.savez_compressed(path,**arrays)
        with np.load(path,allow_pickle=False)as check:
            readback=all(np.array_equal(check[k],a)for k,a in arrays.items())
            checks=dict(source_RGB_conservation_max=color_error,source_alpha_conservation_max=alpha_error,save_read_exact=readback,
                K_pixels=int(arrays['K'].sum()),K_eligible_weight_max=float(arrays['eligible_weight'][arrays['K']].max())if mode=='joint'else None,
                source_layout=layout,shape=[1280,1920])
        write(folder/(mode+'_basis_receipt.json'),checks)
        del rows,labels,colors,e,f,v,z,base,Z,E,P,A,arrays,environment;gc.collect();torch.cuda.empty_cache()
        return checks
    def constant(raw,mode,folder):
        C=torch.tensor([.17,.53,.81],device=ctx.base_rgb.device);color=ctx.base_rgb.clone();color[torch.as_tensor(eligible,device=C.device)]=C
        if mode=='road':rows=dict(ctx.base_packet(raw),colors_precomp=color)
        else:
            rows,labels,_=joint_rows(raw);c=rows['colors_precomp'].clone();c[labels==2]=color;rows=dict(rows,colors_precomp=c)
        with torch.no_grad():v=raster(raw,mode+'_constant',rows);P=v['rgb']+(env(raw,v['alpha'])if mode=='joint'else 0)
        with np.load(folder/(mode+'_basis.npz'),allow_pickle=False)as z:
            actual=P.detach().cpu().numpy();expected=z['baseline_rgb']+z['eligible_weight'][None]*np.array([.17,.53,.81],np.float32)[:,None,None]-z['eligible_rgb']
            alt=z['direct_readonly_rgb']+z['eligible_weight'][None]*np.array([.17,.53,.81],np.float32)[:,None,None]
            checks=dict(constant=[.17,.53,.81],RGB_max=float(np.abs(actual-expected).max()),direct_readonly_RGB_max=float(np.abs(actual-alt).max()),
                alpha_exact=np.array_equal(v['alpha'].squeeze().cpu().numpy(),z['alpha']),joint_K_RGB_max=float(np.abs(actual-z['baseline_rgb'])[:,z['K']].max())if mode=='joint'else None)
        write(folder/(mode+'_constant_receipt.json'),checks)
        del v,P,rows,color;gc.collect();torch.cuda.empty_cache();return checks
    try:
        camera_checks={}
        for raw in [110,86]:
            a,h=low_headers[raw],high_headers[raw]
            assert np.array_equal(a['W'],h['W'])and a['timestamp']==h['timestamp']
            assert all(h[k]==4*a[k]for k in ['fx','fy','cx','cy'])
            camera_checks[str(raw)]={k:h[k]for k in ['width','height','fx','fy','cx','cy','timestamp']};camera_checks[str(raw)]['world_view_model']=h['W'].tolist()
            folder=out/f'native_r1/{raw:03d}';begin=time.perf_counter()
            if(folder/'completed.json').exists():continue
            receipts={mode:export(raw,mode,folder)for mode in ['road','joint']}
            checks={mode:constant(raw,mode,folder)for mode in ['road','joint']}
            write(folder/'completed.json',dict(raw=raw,native_calls=4,basis=receipts,constant=checks,seconds=time.perf_counter()-begin,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            print('FRAME_COMPLETED',raw,json.dumps(checks),flush=True)
        if do_m10 and counter['attempts']<=8:
            from fixed_material_read_operator import FixedReadOperator
            from world_texture_runtime import WorldTextureRuntime
            op=FixedReadOperator.load(Q/'permission/avg_sparse_gh31_r1/operator')
            model,_=WorldTextureRuntime.load(G/'formal_r1/M10/C800/field.pt',device='cpu')
            with torch.no_grad():lo,hi=model.fields()
            values=op.numpy_query(lo.numpy(),hi.numpy(),'teacher').astype(np.float32)
            loc=np.searchsorted(ctx.active_world_rows,op.row_ids);assert np.array_equal(ctx.active_world_rows[loc],op.row_ids)
            colors=ctx.base_rgb.clone();colors[torch.as_tensor(loc,device=colors.device)]=torch.as_tensor(values,device=colors.device)
            for raw in [110,86]:
                outputs={}
                for mode in ['road','joint']:
                    if mode=='road':rows=dict(ctx.base_packet(raw),colors_precomp=colors)
                    else:
                        rows,labels,_=joint_rows(raw);c=rows['colors_precomp'].clone();c[labels==2]=colors;rows=dict(rows,colors_precomp=c)
                    with torch.no_grad():v=raster(raw,'M10_'+mode,rows);P=v['rgb']+(env(raw,v['alpha'])if mode=='joint'else 0)
                    outputs[mode+'_rgb']=P.cpu().numpy();outputs[mode+'_alpha']=v['alpha'].squeeze().cpu().numpy()
                    del v,P,rows;gc.collect();torch.cuda.empty_cache()
                outputs['K']=ctx.K_masks[raw];folder=out/'rendered/M10';folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/f'{raw:03d}.npz',**outputs)
        ctx.assert_frozen();write(out/'native_completed.json',dict(completed=True,native_attempts=counter['attempts'],native_completed=counter['completed'],
            calls_max=12,optimizer_updates=0,frames=[110,86],resolution=[1920,1280],camera=camera_checks,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),seconds=time.perf_counter()-began,
            original_manifest_SHA256=sha(manifest),proof_SHA256=sha(proof),scope_SHA256=sha(scopefile),production_assets_written=False,GPU_keep_on=True))
    except Exception:
        write(out/'failure.json',dict(traceback=traceback.format_exc(),counter=counter,optimizer_updates=0));raise
    finally:
        ctx.close();lease=read(out/'gpu_lease.json');lease.update(active=False,owner=None);write(out/'gpu_lease.json',lease)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--execute',action='store_true');p.add_argument('--include-M10',action='store_true');a=p.parse_args()
    if not a.execute:print(json.dumps(dict(native_default=False,budget=12,training=0,output=str(a.output))))
    else:run(a.output,a.include_M10)
