"""Whole two-sided native-Gaussian re-expression with a saved physical basis."""
import hashlib
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
import torch

RUN=Path(__file__).resolve().parent.parent
SOURCE=RUN.parent/'scene008_a6_depth_consistency_20261105'
sys.path.insert(0,str(SOURCE/'source'))
import consistency as previous
previous.rescue.volume.RUN=RUN
previous.rescue.volume.depth_order.takeover.calibrate.RUN=RUN
from consistency import Consistency,Trace,RAWS,BUILD,HELD,read,write,save
from depth_order import merge_rows,select_rows,tag
from fuse_a6 import build_rotation
from halfspace_moments import split_moments

def digest_arrays(arrays):
    h=hashlib.sha256()
    for value in arrays:
        a=np.ascontiguousarray(value);h.update(str((a.shape,str(a.dtype))).encode());h.update(a.tobytes())
    return h.hexdigest()

def parity(first,second,state):
    values={k:float((first[k]-second[k]).abs().max()) for k in ('rgb','alpha','depth_sum')}
    width=2 if state=='black' else 4
    values['extra']=float((first['extra'][:width]-second['extra'][:width]).abs().max())
    return values

class FullBasis(Consistency):
    def __init__(self):
        super().__init__()
        with np.load(SOURCE/'preflight_ledger.npz',allow_pickle=False) as f:
            assert np.array_equal(f['native_row_id'],self.tr.native_ids.cpu().numpy())
            self.eligible=f['eligible'].copy();self.normals=f['normal'].copy();self.cuts=f['cut'].copy()
        assert int(self.eligible.sum())==1502
        self.packet_dir=RUN/'basis_frames';self.packet_dir.mkdir(exist_ok=True)

    def original_actor(self,raw):return self.actor(raw,self.weight,self.extra)

    def inputs(self,raw,actor):
        lineage=self.actor_lineage[raw]
        candidate=self.zero.new_tensor(self.eligible,dtype=torch.bool)[lineage]
        take=torch.where(candidate&actor['mask']&(actor['opacities'].flatten()>0))[0]
        uid=lineage[take].cpu().numpy()
        mu=actor['means3D'][take].double().cpu().numpy()
        rot=build_rotation(actor['rotations'][take]).double().cpu().numpy()
        scales=actor['scales'][take].double().cpu().numpy()
        factor=rot*scales[:,None,:]
        opacity=actor['opacities'][take].flatten().double().cpu().numpy()
        feature=actor['features'][take].double().cpu().numpy()
        d=self.tr.raw(raw);pose=d['pose'].double().cpu().numpy()
        n=self.normals[uid]@pose[:3,:3].T/self.tr.scale
        b=n@pose[:3,3]+self.tr.scale*self.cuts[uid]
        W=d['cam'].world_view_transform[:3,:3].double().cpu().numpy()
        camera=d['cam'].camera_center.double().cpu().numpy()
        old_n=feature[:,16:19]@W.T
        old_b=feature[:,19]+old_n@camera
        clipped=feature[:,6]>.5
        values=dict(index=take.cpu().numpy(),uid=uid,mu=mu,factor=factor,opacity=opacity,
            normal=n,cut=b,old_normal=old_n,old_cut=old_b,clipped=clipped,feature=feature,
            white_normal=np.einsum('nji,nj->ni',factor,n),white_cut=b-np.einsum('ni,ni->n',n,mu),
            white_old_normal=np.einsum('nji,nj->ni',factor,old_n),
            white_old_cut=old_b-np.einsum('ni,ni->n',old_n,mu))
        values['sha']=np.asarray(digest_arrays([values[k] for k in ('index','uid','mu','factor','opacity','feature')]))
        return values

    def prepare(self,raw):
        actor=self.original_actor(raw);v=self.inputs(raw,actor)
        parents=[];sides=[];means=[];sizes=[];quats=[];alphas=[];prob=[];multipliers=[];tau=[]
        replaced=[];errors=[];unsplit=[];diagnostics=[];clip_cov=[];clip_mu=[]
        for j,index in enumerate(v['index']):
            try:
                a=float(v['opacity'][j])
                if not 0<a<=1:raise ValueError('Parent opacity outside (0,1]')
                inherited={}
                if v['clipped'][j]:
                    inherited=dict(inherited_normal=v['white_old_normal'][j],inherited_cut=v['white_old_cut'][j])
                    ft=v['feature'][j]
                    P=np.array([[ft[7],ft[8],ft[9]],[ft[8],ft[10],ft[11]],[ft[9],ft[11],ft[12]]])
                    cam=self.tr.raw(raw)['cam'];W=cam.world_view_transform[:3,:3].double().cpu().numpy()
                    camera=cam.camera_center.double().cpu().numpy()
                    cov_cam=W.T@v['factor'][j]@v['factor'][j].T@W
                    clip_cov.append(float(np.linalg.norm(np.linalg.inv(P)-cov_cam)/max(np.linalg.norm(cov_cam),1e-300)))
                    clip_mu.append(float(np.linalg.norm(np.linalg.solve(P,ft[13:16])-(v['mu'][j]-camera)@W)/self.tr.scale))
                moment=split_moments(np.zeros(3),np.eye(3),v['white_normal'][j],v['white_cut'][j],**inherited)
                children=moment['children']
                if any(c['status']!='OK' for c in children.values()):
                    unsplit.append(dict(index=int(index),uid=int(v['uid'][j]),status=moment['status'],
                        sides={k:c['status'] for k,c in children.items()}));continue
                local=[]
                for side,name in enumerate(('keep','replace')):
                    c=children[name];L=np.linalg.cholesky(c['cov'])
                    U,s,_=np.linalg.svd(v['factor'][j]@L)
                    if np.linalg.det(U)<0:U[:,-1]*=-1
                    if not np.isfinite(s).all() or not (s>0).all():raise ValueError('Invalid derived scale')
                    quaternion=Rotation.from_matrix(U).as_quat()[[3,0,1,2]]
                    sign,logdet=np.linalg.slogdet(c['cov'])
                    if sign<=0:raise ValueError('Non-positive whitened child covariance')
                    m=float(c['prob']*np.exp(-.5*logdet))
                    t=np.inf if a==1 else -np.log1p(-a)*m
                    alpha=1. if a==1 else -np.expm1(-t)
                    if not np.isfinite(m) or not 0<=alpha<=1:raise ValueError('Invalid optical-volume initialization')
                    local.append((int(index),side,v['mu'][j]+v['factor'][j]@c['mean'],s,quaternion,
                                  alpha,c['prob'],m,t))
                replaced.append(int(index));diagnostics.append(moment['diagnostics'])
                for index,side,mean,scale,quat,alpha,z,m,t in local:
                    parents.append(index);sides.append(side);means.append(mean);sizes.append(scale);quats.append(quat)
                    alphas.append(alpha);prob.append(z);multipliers.append(m);tau.append(t)
            except (ArithmeticError,ValueError,np.linalg.LinAlgError) as error:
                errors.append(dict(index=int(index),uid=int(v['uid'][j]),error=type(error).__name__,message=str(error)))
        packet=dict(source_sha=v['sha'],replaced=np.asarray(replaced,np.int64),parents=np.asarray(parents,np.int64),
            side=np.asarray(sides,np.int8),means=np.asarray(means,np.float32).reshape(-1,3),
            scales=np.asarray(sizes,np.float32).reshape(-1,3),quaternions=np.asarray(quats,np.float32).reshape(-1,4),
            opacities=np.asarray(alphas,np.float32),probability=np.asarray(prob),multiplier=np.asarray(multipliers),
            optical_tau=np.asarray(tau),raw=np.int16(raw),native_row_id=self.tr.native_ids.cpu().numpy())
        np.savez_compressed(self.packet_dir/f'{raw:03d}.npz',**packet)
        np.savez_compressed(self.packet_dir/f'{raw:03d}_inputs.npz',**v)
        parent_alpha=v['opacity'][np.searchsorted(v['index'],packet['parents'])] if len(parents) else np.empty(0)
        stats=dict(raw=raw,candidate_primitives=len(v['index']),clipped_candidate_primitives=int(v['clipped'].sum()),
            rewritten_primitives=len(replaced),children=len(parents),source_alpha_one=int((v['opacity']==1).sum()),
            derived_alpha_one=int((packet['opacities']==1).sum()),native_alpha_cap_entries=int((packet['opacities']>=.99).sum()),
            linear_volume_initialization_above_one=int((parent_alpha*packet['multiplier']>1).sum()),
            numerical_errors=errors,unsplit=unsplit,
            max_moment_errors={k:max((abs(d[k]) for d in diagnostics),default=0.) for k in
                ('relative_probability_error','mean_recovery_maxabs','second_recovery_maxabs')},
            clip_covariance_relative_gap_max=max(clip_cov,default=0.),clip_precision_mean_gap_m_max=max(clip_mu,default=0.))
        write(self.packet_dir/f'{raw:03d}.json',stats)
        return packet,stats

    def load(self,raw):
        with np.load(self.packet_dir/f'{raw:03d}.npz',allow_pickle=False) as f:return {k:f[k] for k in f.files}

    def rewrite_actor(self,raw,packet,replace_retire=0.,all_retire=0.):
        actor=self.original_actor(raw);v=self.inputs(raw,actor)
        assert str(packet['source_sha'])==str(v['sha']) and np.array_equal(packet['native_row_id'],self.tr.native_ids.cpu().numpy())
        parents=torch.as_tensor(packet['parents'],device='cuda')
        if not len(parents):return actor
        old= dict(actor,mask=actor['mask'].clone());old['mask'][torch.as_tensor(packet['replaced'],device='cuda')]=False
        child={k:None if value is None else value[parents].clone() for k,value in actor.items()}
        assert child['cov3D_precomp'] is None,'Explicit scale/quaternion representation required'
        for k,name in (('means3D','means'),('scales','scales'),('rotations','quaternions')):
            child[k]=actor[k].new_tensor(packet[name])
        side=actor['features'].new_tensor(packet['side'])
        child['opacities']=actor['opacities'].new_tensor(packet['opacities'])[:,None]*(1-all_retire)*(1-replace_retire*(side==1)[:,None])
        child['means2D']=torch.zeros_like(child['means3D'])
        child['features'][:,2:]=0
        view=self.tr.raw(raw)['cam'].world_view_transform[:,2]
        child['features'][:,0]=child['means3D']@view[:3]+view[3]
        child['features'][:,4]=(side==0).float();child['features'][:,5]=(side==1).float()
        return merge_rows(old,child)

    def trace_actor(self,raw,state,actor):
        n=self.tr.n
        if state=='black':
            rows=merge_rows(actor,self.cache[raw][3])
            kind=torch.full((len(rows['features']),),3,device='cuda',dtype=torch.int32)
            kind[:len(actor['features'])]=torch.where(actor['features'][:,6]>.5,1,2)
            return Trace(self,raw,rows,kind,self.tr.black,4)
        scene=self.assemble(raw,state=='delete');self.colourize(scene,self.road_colour)
        rows,_,env=scene;first=len(n.static.get_xyz);count=len(self.cache[raw][0]['features'])
        marks=rows['features'][:,2]==1
        assert int(marks.sum())==count and bool(marks[first:first+count].all())
        tagged=tag(dict(actor,features=actor['features'][:,:2]),0)
        tagged['features'][:,6:]=actor['features'][:,6:]
        rows=merge_rows(merge_rows(select_rows(rows,slice(0,first)),tagged),select_rows(rows,slice(first+count,None)))
        feat=rows['features'];kind=torch.zeros(len(feat),device='cuda',dtype=torch.int32)
        kind[feat[:,2]==1]=2;kind[(kind==2)&(feat[:,6]>.5)]=1;kind[feat[:,5]==1]=3
        return Trace(self,raw,rows,kind,n.bg,4,env)

    def render_basis(self,raw,state,packet=None,replace_retire=0.,all_retire=0.):
        actor=self.original_actor(raw) if packet is None else self.rewrite_actor(raw,packet,replace_retire,all_retire)
        return self.trace_actor(raw,state,actor).base()
