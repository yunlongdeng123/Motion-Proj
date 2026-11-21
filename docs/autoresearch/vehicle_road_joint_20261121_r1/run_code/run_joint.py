"""Scene065 a20 对象坐标刚体高斯与已增密道路的少帧联合优化。"""
from pathlib import Path
import argparse,copy,json,os,random,sys,time,traceback
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
sys.path.insert(0,str(REPO/'scripts/worldsim_v82'))
from drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff,save_persistent_road_handoff
from drivingeditor_dynamic_edit.v44_sh_asset import basis3,sh_rotation_matrix
from drivingeditor_dynamic_edit.road_texture_training import differences
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/vehicle_road_joint_20261121_r1')
ROAD=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_surface_residual_20261121_r2/appearance_fit/asset')
TEACHER=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_common1152_20261121/teacher')
BANK=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs')
MASK=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/repaired_masks_all_r4')
FRAMES=(86,110,134,179,198);TRAIN=(86,110);TARGET=21;DELETE=10
TRACK='bzFbWRrSGOgEaXxnuGV0tw'
torch.set_num_threads(2)

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def ten(a,dtype=torch.float32):return torch.as_tensor(np.array(a,copy=True),device='cuda',dtype=dtype)
def cpu(v):
    if torch.is_tensor(v):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return type(v)(cpu(x) for x in v)
    return copy.deepcopy(v)
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');torch.save(cpu(v),q);q.replace(p)
def png(path,a):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);a=a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
    if a.ndim==3 and a.shape[0] in (1,3):a=a.transpose(1,2,0)
    if a.ndim==3 and a.shape[-1]==1:a=a[...,0]
    Image.fromarray(np.round(np.clip(a,0,1)*255).astype(np.uint8)).save(p)
def status(phase,**kw):write(ROOT/'status.json',dict(state='running',phase=phase,PID=os.getpid(),updated_at=time.time(),**kw))
def masked_mean(v,m):return (v*m).sum()/m.sum().clamp_min(1)
def l1(x,y,m):return masked_mean((x-y).abs().mean(0),m)
def ssim_map(x,y):
    x=x[None];y=y[None];mean=lambda a:F.avg_pool2d(a,7,1,3)
    ux,uy=mean(x),mean(y);vx,vy=mean(x*x)-ux*ux,mean(y*y)-uy*uy;c=mean(x*y)-ux*uy
    return (((2*ux*uy+.01**2)*(2*c+.03**2))/((ux.square()+uy.square()+.01**2)*(vx+vy+.03**2))).mean(1)[0]
def edge(x,y,m):
    dx=((x[:,:,1:]-x[:,:,:-1])-(y[:,:,1:]-y[:,:,:-1])).abs().mean(0)
    dy=((x[:,1:]-x[:,:-1])-(y[:,1:]-y[:,:-1])).abs().mean(0)
    return (masked_mean(dx,(m[:,1:]&m[:,:-1]))+masked_mean(dy,(m[1:]&m[:-1])))/2
def hf(x,y,m):
    kernel=x.new_tensor([.054488685,.244201342,.402619947,.244201342,.054488685]);k=(kernel[:,None]*kernel[None,:])[None,None].expand(3,1,5,5)
    a=x-F.conv2d(x[None],k,padding=2,groups=3)[0];b=y-F.conv2d(y[None],k,padding=2,groups=3)[0]
    av=a[:,m];bv=b[:,m]
    if not av.numel():return dict(correlation=None,energy_ratio=None)
    corr=(av*bv).sum()/(av.square().sum().sqrt()*bv.square().sum().sqrt()).clamp_min(1e-12)
    energy=av.square().mean().sqrt()/bv.square().mean().sqrt().clamp_min(1e-12)
    return dict(correlation=float(corr),energy_ratio=float(energy))
def project(xyz,cam):
    view=xyz@cam.world_view_transform[:3,:3]+cam.world_view_transform[3,:3]
    # Camera stores +z forward, +x right and +y down.
    u=view[:,0]/view[:,2].clamp_min(1e-8)*cam.fx+cam.cx;v=view[:,1]/view[:,2].clamp_min(1e-8)*cam.fy+cam.cy
    return torch.stack([u,v],1),view[:,2]
def subset(rows,ids):
    n=len(rows['means3D']);out={}
    for k,v in rows.items():out[k]=v[ids].detach().clone() if torch.is_tensor(v) and v.ndim and len(v)==n else v
    out['mask']=torch.ones(len(out['means3D']),device='cuda',dtype=torch.bool)
    return out
def quat_matrix(q):return Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
def quat_from_matrix(m):return Rotation.from_matrix(m).as_quat()[:,[3,0,1,2]]
def qmul(a,b):
    aw,ax,ay,az=a.unbind(-1);bw,bx,by,bz=b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz,aw*bx+ax*bw+ay*bz-az*by,aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw),-1)

class Vehicle(torch.nn.Module):
    def __init__(self,a):
        super().__init__()
        for k in ('xyz','scales','rotation','coefficients','opacity'):
            self.register_buffer('base_'+k,ten(a[k]))
        self.dc=torch.nn.Parameter(self.base_coefficients[:,0].clone())
        self.opacity_logits=torch.nn.Parameter(torch.logit(self.base_opacity.clamp(1e-5,1-1e-5)))
        self.log_scale_delta=torch.nn.Parameter(torch.zeros_like(self.base_scales))
    def coeff(self):return torch.cat([self.dc[:,None],self.base_coefficients[:,1:]],1)
    def scales(self):return self.base_scales*self.log_scale_delta.clamp(np.log(.75),np.log(1.5)).exp()
    def rows(self,pose,cam):
        R=pose[:3,:3]/pose[:3,:3].norm(dim=0).mean();s=pose[:3,:3].norm(dim=0).mean()
        xyz=self.base_xyz@pose[:3,:3].T+pose[:3,3]
        local=F.normalize(F.normalize(xyz-cam.camera_center,dim=1)@R,dim=1)
        colors=(torch.einsum('ni,nic->nc',basis3(local),self.coeff())+.5).clamp_min(0)
        qp=ten(quat_from_matrix(R.detach().cpu().numpy()[None]))[0]
        rotation=qmul(qp.expand_as(self.base_rotation),self.base_rotation)
        depth=xyz@cam.world_view_transform[:3,2]+cam.world_view_transform[3,2]
        features=xyz.new_zeros((len(xyz),24));features[:,0]=depth;features[:,1]=1;features[:,2]=1
        return dict(means3D=xyz,means2D=torch.zeros_like(xyz),colors_precomp=colors,shs=None,features=features,opacities=self.opacity_logits.sigmoid()[:,None],scales=self.scales()*s,rotations=rotation,cov3D_precomp=None,mask=depth>.00377)
    def export(self):return dict(xyz=cpu(self.base_xyz).numpy(),scales=cpu(self.scales()).numpy(),rotation=cpu(self.base_rotation).numpy(),coefficients=cpu(self.coeff()).numpy(),opacity=cpu(self.opacity_logits.sigmoid()).numpy())

class Experiment:
    def __init__(self,road_source=ROAD):
        status('loading');self.started=time.monotonic();self.native_calls=0;self.backward_calls=0
        self.road_source=Path(road_source)
        self.scene,self.manifest,self.loader=load_persistent_handoff(self.road_source/'persistent/scene.json',self.road_source/'persistent/proof.json')
        self.road=load(self.road_source/'road.npz');self.active=np.flatnonzero(self.road['active']);self.road_rgb=ten(self.road['reference_srgb'][self.active])
        ids=load('/root/autodl-tmp/drivingeditor_repro/campaign/road_surface_residual_20261121_r2/train_inputs/allowed_color_world_rows.npz')['world_row_ids']
        pos=np.searchsorted(self.active,ids);assert np.array_equal(self.active[pos],ids)
        self.road_train_ids=ids;self.road_pos=ten(pos,torch.long);self.road_color=torch.nn.Parameter(self.road_rgb[self.road_pos].clone())
        self.cameras={raw:self.scene.camera_provider(raw) for raw in FRAMES};self.poses={};self.targets={};self.packets={};self.native_packets={};self.env={};self.domains={}
        self.transform=np.asarray(self.manifest.world_to_model);self.scale=float(self.road['meter_to_model'])
        for raw in FRAMES:
            obs=read(BANK/'observations'/f'{raw:03d}.json');box=next(x for x in obs['boxes'] if x['track']==TRACK)
            self.poses[raw]=ten(self.transform@np.asarray(box['world_from_actor']).reshape(4,4))
            rgb=np.asarray(Image.open(BANK/obs['image_path']).convert('RGB').resize((1152,768),Image.Resampling.BOX)).astype(np.float32)/255
            mask=np.asarray(Image.open(MASK/f'{raw:03d}'/'a20.png').resize((1152,768),Image.Resampling.NEAREST))>0
            labels=np.asarray(Image.open(MASK/f'{raw:03d}'/'labels.png').resize((1152,768),Image.Resampling.NEAREST))
            self.targets[raw]=dict(rgb=ten(rgb.transpose(2,0,1)),mask=ten(mask,torch.bool),labels=labels,box=box)
            with torch.no_grad():
                rows,owners,cam,cov=self.scene._rows(raw,(TARGET,),(),False)
                # Keep exact native masks; packing does not change the Gaussian ordering.
                ns=len(rows['means3D'])-len(self.active);idx=torch.nonzero(rows['mask']).reshape(-1)
                packet=subset(rows,idx);packet['features'][:,2:]=0
                own=owners[idx].clone();ri=torch.where(idx>=ns,idx-ns,torch.full_like(idx,-1))
                packet['features'][:,2]=(own==TARGET).float();packet['features'][:,3]=(ri>=0).float()
                self.native_packets[raw]=(packet,own,ri)
                keep=own!=TARGET;self.packets[raw]=(subset(packet,keep),own[keep],ri[keep])
                self.env[raw]=self.scene.environment(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1).detach()
            if raw in TRAIN:
                teacher=load(TEACHER/f'raw{raw:03d}'/'supervision.npz')
                combined=np.where(teacher['A'][None],teacher['target_A_srgb'],teacher['target_C_srgb'])
                self.domains[raw]=dict(teacher=ten(combined),A=ten(teacher['A'],torch.bool),C=ten(teacher['C'],torch.bool),S=ten(teacher['S'],torch.bool),K=ten(teacher['K'],torch.bool),teacher_available=True)
            else:
                domain=load('/root/autodl-tmp/drivingeditor_repro/campaign/road_surface_residual_20261121_r2/proposal/domains/'+f'{raw:03d}.npz')
                self.domains[raw]=dict(teacher=ten(rgb.transpose(2,0,1)),A=ten(domain['fit'],torch.bool),C=ten(np.zeros((768,1152),bool),torch.bool),S=ten(np.zeros((768,1152),bool),torch.bool),K=ten(np.zeros((768,1152),bool),torch.bool),teacher_available=False)
            self.targets[raw]['labels']=ten(labels,torch.long)
        status('canonical_initialization')
        init=ROOT/'initial_vehicle_r2.npz'
        if not init.exists():self.build_vehicle(init)
        self.initial=load(init);self.vehicle=Vehicle(self.initial).cuda();self.original_state=cpu(self.vehicle.state_dict())
        self.build_domains();self.records=[];self.metrics=read(ROOT/'delivery/metrics.json')['rows'] if (ROOT/'delivery/metrics.json').exists() else [];self.gradient_log=[]
        write(ROOT/'loader_receipt.json',self.loader)
    def build_vehicle(self,path):
        # Lift the exact, effective native reference packet, including temporal opacity,
        # shader colors and effective covariance. No cross-time row mixture is used.
        reference=110;packet,owners,ri=self.native_packets[reference];ids=torch.nonzero(owners==TARGET).reshape(-1)
        source=subset(packet,ids);pose=self.poses[reference];R=(pose[:3,:3]/self.scale).cpu().numpy()
        xyz=((source['means3D']-pose[:3,3])@torch.linalg.inv(pose[:3,:3]).T).cpu().numpy()
        if source['cov3D_precomp'] is not None:
            cov=source['cov3D_precomp'].cpu().numpy();full=np.zeros((len(cov),3,3));full[:,[0,0,0,1,1,2],[0,1,2,1,2,2]]=cov;full[:,[1,2,2],[0,0,1]]=cov[:,[1,2,4]]
            local=R.T[None]@full@R[None]/self.scale**2;values,vectors=np.linalg.eigh(local)
            negative=np.linalg.det(vectors)<0;vectors[negative,:,0]*=-1;scales=np.sqrt(np.maximum(values,1e-16));rotation=quat_from_matrix(vectors)
        else:
            scales=source['scales'].cpu().numpy()/self.scale;rotation=quat_from_matrix(R.T[None]@quat_matrix(source['rotations'].cpu().numpy()))
        coeff=np.zeros((len(ids),16,3),np.float32);coeff[:,0]=(source['colors_precomp'].cpu().numpy()-.5)/.28209479177387814
        out=dict(xyz=xyz,scales=scales,rotation=rotation,coefficients=coeff,opacity=source['opacities'].reshape(-1).cpu().numpy(),source_row=ids.cpu().numpy(),source_branch=np.full(len(ids),2,np.int8),source_raw=np.full(len(ids),reference,np.int16))
        np.savez_compressed(path,**out)
        write(ROOT/'canonical_source_r2.json',dict(track=TRACK,actor_id=TARGET,alias='a20',delete_actor_id=DELETE,delete_alias='a9',source='exact effective native rasterization packet at raw110; semantic owner21 includes both branches; no geometry row crop',rows=len(ids),reference=reference,native_reference_colors_baked_to_canonical_SH0=True,effective_native_covariance_and_temporal_opacity=True,source_model_saw_full_sequence=True,world_from_actor_frozen=True,local_xyz_frozen=True,unknown_hidden_geometry='inherited native representation; no invented geometry in this stage'))
        print('CANONICAL_R2',len(ids),flush=True)
        return
        d=self.scene.dynamic;s=self.scene.static;arrays=[];receipts=[]
        # Each inherited Gaussian is lifted once at its strongest training-view lifetime.
        times=[self.cameras[r].timestamp for r in TRAIN]
        marg=torch.stack([d.get_marginal_t(t).reshape(-1) for t in times]);best=marg.argmax(0)
        for branch,model,owner in [('dynamic',d,self.scene._source_d),('static',s,self.scene._owners_s)]:
            xyz_all=model.get_xyz.detach();coeff=model.get_features.detach();scales=model.get_scaling.detach();rot=model.get_rotation.detach();op=model.get_opacity.detach().reshape(-1)
            for fi,raw in enumerate(TRAIN):
                choose=(owner==TARGET)&((best==fi)&(marg[fi]>.1) if branch=='dynamic' else torch.full_like(owner,fi==0,dtype=torch.bool))
                ids=torch.nonzero(choose).reshape(-1)
                xyz=d.get_xyz_SHM(times[fi])[ids].detach() if branch=='dynamic' else xyz_all[ids]
                pose=self.poses[raw];local=(xyz-pose[:3,3])@torch.linalg.inv(pose[:3,:3]).T
                half=ten(self.targets[raw]['box']['dimensions'])/2
                inside=(local.abs()<=half+ten([.15,.15,.10])).all(1)
                ids=ids[inside];local=local[inside]
                if not len(ids):continue
                R=(pose[:3,:3]/self.scale).cpu().numpy();inv_sh=np.linalg.inv(sh_rotation_matrix(R));sh=coeff[ids].cpu().numpy()
                canonical_sh=np.einsum('ij,njc->nic',inv_sh,sh)
                rotation=quat_from_matrix(R.T[None]@quat_matrix(rot[ids].cpu().numpy()))
                opacity=op[ids]*(marg[fi,ids] if branch=='dynamic' else 1)
                a=dict(xyz=local.cpu().numpy(),scales=scales[ids].cpu().numpy()/self.scale,rotation=rotation,coefficients=canonical_sh,opacity=opacity.cpu().numpy(),source_row=ids.cpu().numpy(),source_branch=np.full(len(ids),0 if branch=='dynamic' else 1,np.int8),source_raw=np.full(len(ids),raw,np.int16))
                arrays.append(a);receipts.append(dict(branch=branch,raw=raw,rows=len(ids)))
        out={k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]}
        # Canonical bounds reject learned road/sky leakage, preserving measured vehicle geometry.
        assert len(out['xyz'])>0
        np.savez_compressed(path,**out)
        write(ROOT/'canonical_source.json',dict(track=TRACK,actor_id=TARGET,alias='a20',delete_actor_id=DELETE,delete_alias='a9',source='native30K inherited static/dynamic Gaussians, lifted at max lifetime among raw86/110; 3D actor OBB + fixed metric margin',source_rows=receipts,rows=len(out['xyz']),source_model_saw_full_sequence=True,trained_from_scratch=False,unknown_hidden_geometry='not invented; supported only by inherited native scene geometry',world_from_actor_frozen=True,local_xyz_frozen=True,canonical_SH_rotation=True,dimension_lwh=self.targets[110]['box']['dimensions']))
        print('CANONICAL',receipts,flush=True)
    def colors(self):return self.road_rgb.index_copy(0,self.road_pos,self.road_color.clamp(0,1))
    def render(self,raw,mode='keep',branch='full',native=False):
        cam=self.cameras[raw];packet,owners,road_index=(self.native_packets if native else self.packets)[raw]
        rows={k:v for k,v in packet.items()}
        if branch=='road':
            mask=road_index>=0;rows=subset(packet,mask);road_index=road_index[mask];owners=owners[mask]
        elif branch=='vehicle' and native:
            mask=owners==TARGET;rows=subset(packet,mask);road_index=road_index[mask];owners=owners[mask]
        if branch!='vehicle':
            ids=torch.nonzero(road_index>=0).reshape(-1)
            rows=dict(rows,colors_precomp=rows['colors_precomp'].index_copy(0,ids,self.colors()[road_index[ids]]))
        if not native and branch!='road':
            car=self.vehicle.rows(self.poses[raw],cam)
            if branch=='vehicle':rows=car;owners=torch.full((len(car['means3D']),),TARGET,device='cuda',dtype=torch.long)
            else:
                rows=self.scene._bindings().merge_rows(rows,car);owners=torch.cat([owners,torch.full((len(car['means3D']),),TARGET,device='cuda',dtype=torch.long)])
        if mode=='delete' and branch=='full':rows=dict(rows,mask=rows['mask']&(owners!=DELETE))
        out=self.scene._bindings().raster(cam,self.scene.pipe,0,rows,self.scene._background)
        self.native_calls+=1
        result=dict(rgb=out['rgb'],alpha=out['alpha'].squeeze(),vehicle_alpha=out['extra'][0],road_alpha=out['extra'][1])
        if branch=='full':result['rgb']=result['rgb']+(1-out['alpha'])*self.env[raw]
        return result
    def build_domains(self):
        for raw in FRAMES:
            target=self.targets[raw];half=np.asarray(target['box']['dimensions'])/2
            corners=np.array([[x,y,z] for x in [-half[0],half[0]] for y in [-half[1],half[1]] for z in [-half[2],half[2]]])
            xyz=ten(corners)@self.poses[raw][:3,:3].T+self.poses[raw][:3,3];uv,depth=project(xyz,self.cameras[raw]);a=uv.min(0).values;b=uv.max(0).values
            x0=max(0,int(a[0])-4);y0=max(0,int(a[1])-4);x1=min(1152,int(b[0])+5);y1=min(768,int(b[1])+5)
            bbox=torch.zeros((768,1152),device='cuda',dtype=torch.bool);bbox[y0:y1,x0:x1]=True
            domain=self.domains[raw];legal=(domain['A']|domain['C']|domain['S'])&~domain['K']&~bbox
            # Preserve all retained actors. Original a9-only pixels may be legal road targets after DELETE.
            labels=target['labels'];legal&=~((labels>0)&(labels!=DELETE))
            domain.update(legal=legal,bbox=bbox)
            with torch.no_grad():road=self.render(raw,branch='road')
            domain['legal']&=road['alpha']>.05
            occluder=(labels>0)&(labels!=TARGET)&bbox
            occluder=F.max_pool2d(occluder.float()[None,None],9,1,4)[0,0]>0
            target['negative_alpha_valid']=(~target['mask'])&~occluder
            domain['road_fixed_alpha']=road['alpha'].detach();domain['road_target']=domain['teacher']*domain['road_fixed_alpha'][None]
            ref=ROOT/'delivery/references'/f'{raw:03d}';png(ref/'rgb.png',target['rgb']);png(ref/'vehicle_mask.png',target['mask']);png(ref/'road_teacher.png',domain['teacher']);png(ref/'road_legal.png',domain['legal']);png(ref/'vehicle_potential_bbox.png',bbox);png(ref/'vehicle_negative_valid.png',target['negative_alpha_valid']);png(ref/'vehicle_occlusion_ignore.png',occluder)
        write(ROOT/'supervision_domains.json',dict(rows=[dict(raw=r,vehicle_visible_pixels=int(self.targets[r]['mask'].sum()),road_legal_pixels=int(self.domains[r]['legal'].sum()),a20_box_pixels=int(self.domains[r]['bbox'].sum()),a9_revealed_road_pixels=int((self.domains[r]['legal']&(self.targets[r]['labels']==DELETE)).sum())) for r in FRAMES],visible_alpha='a20 contribution from full KEEP depth/alpha renderer, not isolated alpha; target raw observed body mask, shadow excluded',road='DELETE composite RGB, actual a9 hole included only outside a20 complete projected OBB and retained actors',teacher_vehicle_overlap_allowed=False))
    def metric(self,raw,stage,keep,deleted,road,vehicle):
        mask=self.targets[raw]['mask'];rgb=self.targets[raw]['rgb'];alpha=keep['vehicle_alpha'];legal=self.domains[raw]['legal'];pred=alpha>.5
        iou=float((pred&mask).sum()/(pred|mask).sum().clamp_min(1));valid=mask|self.targets[raw]['negative_alpha_valid'];valid_iou=float((pred&mask&valid).sum()/((pred|mask)&valid).sum().clamp_min(1));h=hf(keep['rgb'],rgb,mask);rh=hf(deleted['rgb'],self.domains[raw]['teacher'],legal)
        return dict(stage=stage,raw=raw,role='train' if raw in TRAIN else 'diagnostic_observed',vehicle_rgb_mae=float(l1(keep['rgb'],rgb,mask)),vehicle_masked_ssim=float(masked_mean(ssim_map(keep['rgb'],rgb),mask)),vehicle_edge_l1=float(edge(keep['rgb'],rgb,mask)),vehicle_HF_sigma1=h,vehicle_alpha_iou=iou,vehicle_alpha_inside_mse=float(masked_mean((alpha-1).square(),mask)),vehicle_alpha_outside_mse=float(masked_mean(alpha.square(),~mask)),vehicle_alpha_outside_observable_mse=float(masked_mean(alpha.square(),self.targets[raw]['negative_alpha_valid'])),vehicle_alpha_negative_valid_pixels=int(self.targets[raw]['negative_alpha_valid'].sum()),vehicle_alpha_occlusion_ignored_pixels=int(((~mask)&~self.targets[raw]['negative_alpha_valid']).sum()),vehicle_alpha_inside_mean=float(masked_mean(alpha,mask)),vehicle_alpha_outside_mean=float(masked_mean(alpha,~mask)),road_teacher_mae=float(l1(deleted['rgb'],self.domains[raw]['teacher'],legal)),road_HF_sigma1=rh,road_alpha_below_095=float((road['alpha'][legal]<.95).float().mean()),road_alpha_below_05=float((road['alpha'][legal]<.5).float().mean()),road_black_fraction=float((road['rgb'][:,legal].amax(0)<.02).float().mean()),full_rgb_mae=float((keep['rgb']-rgb).abs().mean()),hidden_vehicle_GT_available=False,hidden_vehicle_rgb_mae=None,delete_visible_vehicle_pixels=int((deleted['vehicle_alpha']>.5).sum()),keep_visible_vehicle_pixels=int((alpha>.5).sum()),isolated_vehicle_alpha_pixels=int((vehicle['alpha']>.5).sum()))
    @torch.no_grad()
    def refresh_metrics(self):
        self.metrics=[]
        for stage in ['native_before','canonical_init','canonical_prior_init','vehicle_fit','joint_fit']:
            for raw in FRAMES:
                p=ROOT/'delivery'/stage/f'{raw:03d}'/'render_float.npz'
                if not p.exists():continue
                a=load(p);k=dict(rgb=ten(a['joint_keep']),vehicle_alpha=ten(a['vehicle_alpha_keep']));d=dict(rgb=ten(a['joint_delete']),vehicle_alpha=ten(a['vehicle_alpha_delete']));r=dict(rgb=ten(a['road_rgb']),alpha=ten(a['road_alpha']));v=dict(rgb=ten(a['vehicle_rgb']),alpha=ten(a['vehicle_alpha_isolated']))
                row=self.metric(raw,stage,k,d,r,v)
                mask=self.targets[raw]['mask'];valid=mask|self.targets[raw]['negative_alpha_valid'];pred=k['vehicle_alpha']>.5;row['vehicle_alpha_iou_valid']=float((pred&mask&valid).sum()/((pred|mask)&valid).sum().clamp_min(1));row['vehicle_alpha_iou_full_image_diagnostic']=row['vehicle_alpha_iou']
                if not self.domains[raw]['teacher_available']:row.update(road_teacher_mae=None,road_HF_sigma1=None)
                self.metrics.append(row)
        write(ROOT/'delivery/metrics.json',dict(rows=self.metrics,stages=list(dict.fromkeys(x['stage'] for x in self.metrics)),TRAIN_FRAMES=TRAIN,diagnostic_not_blind=True))
    @torch.no_grad()
    def export(self,stage,native=False):
        status('exporting',stage=stage)
        rows=[]
        for raw in FRAMES:
            k=self.render(raw,'keep',native=native);d=self.render(raw,'delete',native=native);v=self.render(raw,branch='vehicle',native=native);r=self.render(raw,branch='road',native=native)
            p=ROOT/'delivery'/stage/f'{raw:03d}'
            for name,a in [('joint_keep',k['rgb']),('joint_delete',d['rgb']),('vehicle_only',v['rgb']),('road_only',r['rgb']),('vehicle_alpha_keep',k['vehicle_alpha']),('vehicle_alpha_delete',d['vehicle_alpha']),('vehicle_alpha_isolated',v['alpha']),('road_alpha',r['alpha'])]:png(p/(name+'.png'),a)
            # Exact floating output retained for independent QA and meaningful comparisons.
            np.savez_compressed(p/'render_float.npz',joint_keep=cpu(k['rgb']).numpy(),joint_delete=cpu(d['rgb']).numpy(),vehicle_rgb=cpu(v['rgb']).numpy(),road_rgb=cpu(r['rgb']).numpy(),vehicle_alpha_keep=cpu(k['vehicle_alpha']).numpy(),vehicle_alpha_delete=cpu(d['vehicle_alpha']).numpy(),vehicle_alpha_isolated=cpu(v['alpha']).numpy(),road_alpha=cpu(r['alpha']).numpy())
            row=self.metric(raw,stage,k,d,r,v)
            mask=self.targets[raw]['mask'];valid=mask|self.targets[raw]['negative_alpha_valid'];pred=k['vehicle_alpha']>.5;row['vehicle_alpha_iou_valid']=float((pred&mask&valid).sum()/((pred|mask)&valid).sum().clamp_min(1));row['vehicle_alpha_iou_full_image_diagnostic']=row['vehicle_alpha_iou']
            if not self.domains[raw]['teacher_available']:row.update(road_teacher_mae=None,road_HF_sigma1=None)
            rows.append(row)
            print('METRIC',stage,raw,row['vehicle_rgb_mae'],row['vehicle_alpha_iou'],row['road_teacher_mae'],flush=True)
        self.metrics=[x for x in self.metrics if x['stage']!=stage]+rows;write(ROOT/'delivery/metrics.json',dict(rows=self.metrics,stages=list(dict.fromkeys(x['stage'] for x in self.metrics)),TRAIN_FRAMES=TRAIN,diagnostic_not_blind=True));return rows
    def optimizer(self,stage):
        self.vehicle.opacity_logits.requires_grad_(stage=='vehicle');self.vehicle.log_scale_delta.requires_grad_(stage=='vehicle');self.road_color.requires_grad_(stage=='joint')
        groups=[dict(params=[self.vehicle.dc],lr=.01)]
        if stage=='vehicle':groups+=[dict(params=[self.vehicle.opacity_logits],lr=.01),dict(params=[self.vehicle.log_scale_delta],lr=.002)]
        else:groups+=[dict(params=[self.road_color],lr=.005)]
        return torch.optim.Adam(groups,betas=(.9,.999),eps=1e-8)
    def step(self,stage,step,opt,role):
        raw=TRAIN[step%len(TRAIN)];opt.zero_grad(set_to_none=True);target=self.targets[raw];mask=target['mask']
        keep=self.render(raw);alpha=keep['vehicle_alpha']
        rgb=l1(keep['rgb'],target['rgb'],mask);ssim=(1-masked_mean(ssim_map(keep['rgb'],target['rgb']),mask))/2
        ain=masked_mean((alpha-1).square(),mask);aout=masked_mean(alpha.square(),target['negative_alpha_valid'])
        reg_scale=self.vehicle.log_scale_delta.square().mean();loss=.8*rgb+.2*ssim+.1*ain+.1*aout+.001*reg_scale
        road_loss=loss*0
        if stage=='joint':
            deleted=self.render(raw,'delete');legal=self.domains[raw]['legal'];teacher=self.domains[raw]['teacher']
            road_loss=l1(deleted['rgb'],teacher,legal)+.1*edge(deleted['rgb'],teacher,legal);loss=loss+road_loss
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        loss.backward();self.backward_calls+=1
        grads={name:None if p.grad is None else float(p.grad.abs().max()) for name,p in [('vehicle_dc',self.vehicle.dc),('vehicle_opacity',self.vehicle.opacity_logits),('vehicle_scale',self.vehicle.log_scale_delta),('road_color',self.road_color)]}
        if not grads['vehicle_dc'] or (stage=='joint' and not grads['road_color']):raise RuntimeError('renderer gradient disconnected or exactly zero '+str(grads))
        before={name:p.detach().clone() for name,p in [('vehicle_dc',self.vehicle.dc),('road_color',self.road_color)]}
        opt.step()
        with torch.no_grad():
            self.vehicle.log_scale_delta.clamp_(np.log(.75),np.log(1.5));self.road_color.clamp_(0,1)
        updates={name:float((p-before[name]).abs().max()) for name,p in [('vehicle_dc',self.vehicle.dc),('road_color',self.road_color)]}
        record=dict(role=role,stage=stage,step=step+1,raw=raw,loss=float(loss),vehicle_rgb=float(rgb),vehicle_dssim=float(ssim),alpha_inside=float(ain),alpha_outside=float(aout),road=float(road_loss),gradient_max=grads,update_max=updates,elapsed=time.monotonic()-self.started)
        if stage=='joint' and not all(updates[x]>0 for x in ['vehicle_dc','road_color']):raise RuntimeError('joint branch failed to update')
        self.records.append(record)
        return record
    def snapshot(self,stage,step,opt,role):return dict(schema='vehicle_road_joint.fullstate.v1',stage=stage,step=step,role=role,vehicle=cpu(self.vehicle.state_dict()),road_color=cpu(self.road_color),optimizer=cpu(opt.state_dict()),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
    def restore(self,s,opt):
        self.vehicle.load_state_dict(s['vehicle'])
        with torch.no_grad():self.road_color.copy_(s['road_color'].cuda())
        opt.load_state_dict(copy.deepcopy(s['optimizer']));torch.set_rng_state(s['torch_rng']);torch.cuda.set_rng_state_all(s['cuda_rng']);np.random.set_state(s['numpy_rng']);random.setstate(s['python_rng'])
    def asset(self,stage):
        p=ROOT/'assets'/stage;p.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(p/'vehicle.npz',**self.vehicle.export(),source_row=self.initial['source_row'],source_branch=self.initial['source_branch'],source_raw=self.initial['source_raw'])
        road={k:v.copy() for k,v in self.road.items()};road['reference_srgb'][self.road_train_ids]=cpu(self.road_color).numpy();np.savez_compressed(p/'road.npz',**road)
        np.savez_compressed(p/'poses.npz',raw=np.array(FRAMES),model_from_actor=np.stack([cpu(self.poses[r]).numpy() for r in FRAMES]))
        save_persistent_road_handoff(ROAD/'persistent/scene.json',ROAD/'persistent/proof.json',p/'road.npz',p/'persistent',training_record=dict(task='vehicle_road_joint_20261121_r1',stage=stage,train_frames=TRAIN,geometry_and_coverage_frozen=True))
        write(p/'scene_binding.json',dict(schema='object_rigid_joint_asset.v1',vehicle=str(p/'vehicle.npz'),poses=str(p/'poses.npz'),target_actor=TARGET,target_track=TRACK,delete_actor=DELETE,source_road_manifest=str(p/'persistent/scene.json'),source_road_proof=str(p/'persistent/proof.json'),loader=str(ROOT/'code/run_joint.py'),colors_only_road_updates=True))
        return p
    def freeze_proof(self):
        self.scene.assert_persistent_unchanged()
        return dict(native_scene_unchanged=True,local_xyz_equal=bool(torch.equal(self.vehicle.base_xyz,ten(self.initial['xyz']))),local_rotation_equal=bool(torch.equal(self.vehicle.base_rotation,ten(self.initial['rotation']))),trajectory_equal=True,road_frozen_fields=['model_xyz','gaussian_sigma_m','normal_world','road_opacity','active','minimum_model_depth'],road_trainable_color_rows=len(self.road_train_ids),source_road_intact=True)
    def smoke(self):
        status('smoke');initial=cpu(self.vehicle.state_dict());road_initial=cpu(self.road_color)
        opt=self.optimizer('vehicle')
        for i in range(4):self.step('vehicle',i,opt,'smoke')
        s=self.snapshot('vehicle',4,opt,'smoke');save(ROOT/'smoke/checkpoint.pt',s)
        r1=self.step('vehicle',4,opt,'smoke_replay');state1=cpu(self.vehicle.state_dict());opt1=cpu(opt.state_dict())
        self.restore(torch.load(ROOT/'smoke/checkpoint.pt',map_location='cpu',weights_only=False),opt)
        restored_state_difference=differences(s,self.snapshot('vehicle',4,opt,'smoke'))
        if restored_state_difference:raise RuntimeError('checkpoint full state not restored '+str(restored_state_difference))
        r2=self.step('vehicle',4,opt,'smoke_replay');state2=cpu(self.vehicle.state_dict());opt2=cpu(opt.state_dict())
        per_parameter={k:float((state1[k]-state2[k]).abs().max()) for k in state1};maxdiff=max(per_parameter.values());lossdiff=abs(r1['loss']-r2['loss'])
        continuation=dict(saved_state_restore_bit_exact=True,next_step_parameter_max=maxdiff,next_step_parameter_differences=per_parameter,next_step_loss_abs=lossdiff,next_step_optimizer_bit_exact=not differences(opt1,opt2),parameter_abs_tolerance=2e-5,loss_abs_tolerance=1e-6,tolerance_basis='same FP32 native CUDA atomic accumulation; inherited R2 smoke uses 2e-5 state tolerance; checkpoint itself must restore bit exactly')
        write(ROOT/'smoke/continuation_numeric.json',continuation)
        if maxdiff>2e-5 or lossdiff>1e-6:raise RuntimeError('smoke numerical continuation failed '+str((maxdiff,lossdiff)))
        self.export('smoke_vehicle')
        optj=self.optimizer('joint')
        for i in range(3):self.step('joint',i,optj,'smoke')
        save(ROOT/'smoke/joint_checkpoint.pt',self.snapshot('joint',3,optj,'smoke'))
        before=self.render(134,'delete')['rgb'].detach();a=self.vehicle.export();reload_vehicle=Vehicle(a).cuda();previous=self.vehicle;self.vehicle=reload_vehicle
        after=self.render(134,'delete')['rgb'].detach();self.vehicle=previous;reload_diff=float((before-after).abs().max())
        if reload_diff>1e-5:raise RuntimeError('persistent vehicle save-load render mismatch '+str(reload_diff))
        proof=dict(passed=True,real_cuda=True,forward_backward=True,checkpoint_saved=True,optimizer_and_sampler_rng_restore=True,saved_state_restore_bit_exact=True,continuation=continuation,evaluation=True,joint_next_stage=True,two_branch_renderer_gradients=True,persistent_vehicle_render_reload_max=reload_diff,actual_smoke_optimizer_updates=9,formal_updates=0,freeze=self.freeze_proof())
        write(ROOT/'smoke/proof.json',proof)
        self.vehicle.load_state_dict(initial)
        with torch.no_grad():self.road_color.copy_(road_initial.cuda())
        print('SMOKE_PASSED',proof,flush=True);return proof
    def formal(self,vehicle_steps,joint_steps):
        torch.manual_seed(101);torch.cuda.manual_seed_all(101);np.random.seed(101);random.seed(101)
        v0=cpu(self.vehicle.state_dict());r0=cpu(self.road_color)
        for stage,budget,label in [('vehicle',vehicle_steps,'vehicle_fit'),('joint',joint_steps,'joint_fit')]:
            status('formal',stage=stage,budget=budget);opt=self.optimizer(stage)
            for i in range(budget):
                record=self.step(stage,i,opt,'formal')
                if i%10==0:print(json.dumps(record),flush=True)
                if (i+1)%50==0 or i+1==budget:
                    save(ROOT/'checkpoints'/f'{stage}_step{i+1:05d}.pt',self.snapshot(stage,i+1,opt,'formal'));write(ROOT/'training_records.json',dict(rows=self.records));status('formal',stage=stage,step=i+1,budget=budget)
            self.export(label);self.asset(label)
        formal=[x for x in self.records if x['role']=='formal'];joint=[x for x in formal if x['stage']=='joint']
        proof=self.freeze_proof();proof.update(vehicle_dc_max_update=float((self.vehicle.dc.detach().cpu()-v0['dc']).abs().max()),vehicle_opacity_max_update=float((self.vehicle.opacity_logits.detach().cpu()-v0['opacity_logits']).abs().max()),vehicle_scale_max_update=float((self.vehicle.log_scale_delta.detach().cpu()-v0['log_scale_delta']).abs().max()),road_color_max_update=float((self.road_color.detach().cpu()-r0).abs().max()),joint_real_renderer_two_branch_nonzero_all_steps=all(x['gradient_max']['vehicle_dc']>0 and x['gradient_max']['road_color']>0 and x['update_max']['vehicle_dc']>0 and x['update_max']['road_color']>0 for x in joint),formal_vehicle_updates=vehicle_steps,formal_joint_updates=joint_steps,native_renderer_calls=self.native_calls,backward_calls=self.backward_calls,power_operations=0,seconds=time.monotonic()-self.started)
        self.refresh_metrics()
        write(ROOT/'delivery/proofs.json',proof);write(ROOT/'resource_ledger.json',dict(native_calls=self.native_calls,backward_calls=self.backward_calls,smoke_updates=9,formal_vehicle_updates=vehicle_steps,formal_joint_updates=joint_steps,extra_replay_updates=2,seconds=time.monotonic()-self.started,GPU_keep_on=True,power_operations=0));write(ROOT/'status.json',dict(state='complete',phase='complete',complete=True,PID=os.getpid(),updated_at=time.time()));print('COMPLETE',proof,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['preview','smoke','formal','all'],default='all');ap.add_argument('--vehicle-steps',type=int,default=200);ap.add_argument('--joint-steps',type=int,default=200);ap.add_argument('--with-lower-prior',action='store_true');ap.add_argument('--render-asset',type=Path);args=ap.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    contract=dict(task='SCENE065-A9-DELETE-A20-COMPLETE-JOINT-20261121',target=dict(alias='a20',actor_id=TARGET,track_id=TRACK),delete=dict(alias='a9',actor_id=DELETE,track_id='NTjq-OhZ_97OMJ4xwJAxXg'),road_source=str(ROAD),train_frames=TRAIN,diagnostic_frames=[134,179,198],resolution=[1152,768],few_frame_adaptation_of_allframe_pretrained_asset=True,stages=['native_before','canonical_init','vehicle_fit','joint_fit'],vehicle_steps=args.vehicle_steps,joint_steps=args.joint_steps,vehicle_trainable=['canonical SH DC','opacity','per-axis scales bounded0.75-1.5'],joint_trainable=['vehicle SH DC','1191 approved road colors'],frozen=['vehicle local XYZ/rotation','trajectory','cameras','all native models','environment','static gates','road geometry/opacity/scales/active'],loss=dict(vehicle_rgb_l1=.8,vehicle_DSSIM=.2,alpha_inside_mse=.1,alpha_outside_mse=.1,scale_log_delta_l2=.001,joint_road_rgb_l1=1.,joint_road_edge_l1=.1),alpha_normalization='inside and outside normalized by respective full-image domain pixel counts; raw a20 composited owner contribution, no multiplication by mask',mask_content='real a20 body-only accepted mask, no shadow',road_domain='DELETE composite; allowed A|C|S including a9 hole, excluding a20 projected full OBB and retained actors/K',hidden_vehicle_supervision='only inherited native geometry and cross-view true visible RGB; never road teacher; no hidden RGB ground truth',initialization='original 30K actor21 static/dynamic Gaussians lifted into one canonical object at raw86/110; local SH rotation',seed=101,smoke='short real GPU forward/backward, save/full optimizer-RNG restore, eval, joint entry, persistent vehicle reload',new_diffusion_updates=0,TRELLIS=False,PVD=False,GPU_keep_on=True,power_operations=False)
    contract.update(initialization='exact effective native packet at reference raw110, current temporal opacity/covariance/colors; shader RGB baked as canonical SH0',alpha_normalization='inside visible body mask and observable outside separately normalized; composited target alpha; original occluding labels inside full projected OBB dilated9px excluded from negatives',road_domain='A original observed RGB, C/S frozen generated target, DELETE composite; a9 hole included outside target full projected OBB and retained labels/K; fixed actual roadalpha>.05 support',stages=['native_before','canonical_init','canonical_prior_init','vehicle_fit','joint_fit'],lower_shape_prior=args.with_lower_prior,lower_prior_scope='explicit box-conditioned panels/bumper/tire shapes at unobserved lower body, constrained by visible width, all5 occluder masks and annotated base; not measured wheel geometry',prior_geometry_and_hidden_opacity_scale_frozen=True,visible_native_support='actual unified-raster feature VJP over real visible masks')
    if args.render_asset:
        binding=read(args.render_asset/'scene_binding.json');e=Experiment(args.render_asset);a=load(args.render_asset/'vehicle.npz');e.initial=a;e.vehicle=Vehicle(a).cuda();poses=load(args.render_asset/'poses.npz');e.poses={int(r):ten(p) for r,p in zip(poses['raw'],poses['model_from_actor'])};e.export('joint_fit_reloaded');rows=[]
        for raw in FRAMES:
            a=load(ROOT/'delivery/joint_fit'/f'{raw:03d}'/'render_float.npz');b=load(ROOT/'delivery/joint_fit_reloaded'/f'{raw:03d}'/'render_float.npz');rows.append(dict(raw=raw,maximum={k:float(np.abs(a[k]-b[k]).max()) for k in a}))
        passed=all(v<2e-5 for r in rows for v in r['maximum'].values());write(ROOT/'disk_reload_proof.json',dict(passed=passed,fresh_process=True,asset=str(args.render_asset),binding=binding,rows=rows))
        if not passed:raise RuntimeError('disk asset roundtrip rendered mismatch')
        write(ROOT/'status.json',dict(state='complete',phase='disk_reload_complete',complete=True,PID=os.getpid()));return
    write(ROOT/'contract.json',contract);e=Experiment()
    if args.with_lower_prior:
        import lower_prior
        lower_prior.install(e,sys.modules[__name__])
    if args.phase in ['preview','all']:
        if not args.with_lower_prior:e.export('native_before',native=True)
        e.export('canonical_prior_init' if args.with_lower_prior else 'canonical_init')
        if args.phase=='preview':write(ROOT/'status.json',dict(state='preview_complete',phase='preview_complete',complete=False,PID=os.getpid()));return
    if args.phase in ['smoke','all']:e.smoke()
    if args.phase in ['formal','all']:e.formal(args.vehicle_steps,args.joint_steps)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(ROOT/'status.json',dict(state='failed',error=str(exc),traceback=traceback.format_exc(),PID=os.getpid(),updated_at=time.time()));traceback.print_exc();raise
