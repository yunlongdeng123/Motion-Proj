"""Colour-only, world-stable calibration on the accepted Scene008 road scene."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import binary_dilation,binary_erosion
from scipy.optimize import lsq_linear
import torch

RUN=Path(__file__).resolve().parent.parent
PARENT=RUN.parent/'scene008_a6_ray_switch_20261105'
ROAD=RUN.parent/'scene008_native_road1k_20261105_r1'
sys.path.insert(0,str(PARENT/'source'))
import switch
from switch import Trial
from ray_support import (RAWS,PILOTS,private,metric,save,read,write,prepare_rasterization,
                         road_kwargs,select_rows,merge_rows)
from street_switch import tag
from render_pvd import OBS

HOLDOUT=(89,95,101,105,113,121,125,130,137)
GRID=(6,5)
MAX_DELTA=.09
MIN_WEIGHT=.02
DC=.28209479177387814


class Experiment:
    def __init__(self):
        switch.RUN=RUN
        self.trial=Trial()
        self.trial.parents()
        self.previous=read(PARENT/'street.json')
        with np.load(PARENT/'switch.npz',allow_pickle=False) as f:
            assert np.array_equal(f['native_row_id'],self.trial.native_ids.cpu().numpy())
            self.selected=torch.as_tensor(f['selected'],device='cuda')
        road=self.trial.exp.repair.road
        self.road=road
        self.colour=road.colors().detach().clone()
        meta=self.trial.exp.meta
        inverse=np.linalg.inv(np.asarray(meta['world_to_model'])@np.asarray(meta['coordinate_anchor']))
        xyz=road.xyz.cpu().numpy()@inverse[:3,:3].T+inverse[:3,3]
        self.basis,self.knots=world_basis(xyz[:,:2])
        self.basis=torch.as_tensor(self.basis,device='cuda')
        assert torch.allclose(self.basis.sum(1),torch.ones(len(xyz),device='cuda'),atol=2e-6)

    def assemble(self,raw,delete):
        tr=self.trial
        n=tr.n
        cam=tr.raw(raw)['cam']
        repair=tr.exp.repair
        static=prepare_rasterization(cam,n.static,n.args)
        static['means2D']=torch.zeros_like(static['means3D'])
        ids=repair.ids
        static['opacities']=static['opacities'].index_copy(0,ids,static['opacities'][ids]*repair.gate_raw.sigmoid()[:,None])
        actor=tag(tr.switch_rows(raw,self.selected),0)
        q,parent,code=tr.code[raw]
        active=self.selected[parent]
        actor['features'][q[active],6:]=code[active]
        joint=merge_rows(tag(static,None),actor)
        if not delete:
            dyn=n.actor_kwargs(cam,delete_a37=False)
            joint=merge_rows(joint,tag(select_rows(dyn,~n.keep_a6),1))
        channels=joint['shs'].shape[1]
        road=tag(road_kwargs(cam,repair.road,n.args,channels,precomputed_color=False),2)
        start=len(joint['features'])
        joint=merge_rows(joint,road)
        end=len(joint['features'])
        patch=select_rows(tr.patch(raw,static),tr.lower)
        joint=merge_rows(joint,tag(patch,3))
        directions=cam.get_world_directions(False).permute(1,2,0)
        env=n.env(directions).permute(2,0,1)
        return joint,slice(start,end),env

    def render(self,raw,scene):
        rows,_,env=scene
        out=private(self.trial,raw,rows,background=self.trial.n.bg,channels=4)
        out['rgb']=out['rgb']+(1-out['alpha'])*env
        return out

    def colourize(self,scene,col):
        rows,where,_=scene
        rows['shs'][where,0]=(col-.5)/DC

    def evidence(self,raw,delete):
        d=self.trial.raw(raw)
        with np.load(OBS/f'{raw:03d}.npz',allow_pickle=False) as f:
            actual_road=f['road'].copy()
            target=f['road_edit_rgb'].copy() if delete else None
            road=(f['road_edit']&(self.trial.fixed[raw-89,3]>0)) if delete else actual_road
            protect=f['protect'] & ~road
            a37=f['visible'][1].copy()
        if not delete:
            with np.load(ROAD/'inputs/views'/f'{raw:03d}.npz',allow_pickle=False) as f:
                shadow=(f['shadow_proxy']|f['new_shadow_proxy'])
            fit=road&~shadow
        else:
            fit=road
        cv=lambda x:torch.as_tensor(x,device='cuda')
        rgb=d['rgb'] if target is None else cv(target).permute(2,0,1).float()
        if rgb.max()>1:
            rgb/=255
        return dict(road=cv(road),fit=cv(fit),protect=cv(protect),a6=d['obs'],a37=cv(a37),
                    true_rgb=d['rgb'],target=rgb)

    def check_parent(self,raw,state,base,evidence):
        exp=self.previous['rows'][raw-89][state]
        q=self.trial.evidence[raw]['mask']
        checks={'lower_after':metric(base['extra'][3],q),
                'road_after':metric((base['rgb']-evidence['target']).abs().mean(0),evidence['road'])}
        for name,m in checks.items():
            assert m['count']==exp[name]['count'] and abs(m['sum']-exp[name]['sum'])<.002,(raw,state,name,m,exp[name])

    def cache_design(self):
        dest=RUN/'design'
        dest.mkdir()
        checks=[]
        for raw in RAWS:
            if raw in HOLDOUT:
                continue
            for state in ('keep','delete'):
                delete=state=='delete'
                e=self.evidence(raw,delete)
                scene=self.assemble(raw,delete)
                base=self.render(raw,scene)
                self.check_parent(raw,state,base,e)
                weight=base['extra'][2]
                active=weight>=MIN_WEIGHT
                fit=e['fit']&active
                protection=e['protect']&active
                select=fit|protection
                # Render the exact alpha*T spatial basis. Scene geometry, depth
                # and clipping data in channels 6:24 never change.
                y,x=select.nonzero(as_tuple=True)
                design=self.basis.new_zeros((len(y),self.basis.shape[1]))
                rows,where,_=scene
                rows['features'][:,2:6]=0
                for index in range(0,self.basis.shape[1],4):
                    width=min(4,self.basis.shape[1]-index)
                    rows['features'][where,2:6]=0
                    rows['features'][where,2:2+width]=self.basis[:,index:index+width]
                    features=self.render(raw,scene)['extra']
                    design[:,index:index+width]=features[:width,y,x].T
                error=float((design.sum(1)-weight[y,x]).abs().max()) if len(y) else 0.
                assert error<1.1e-5,error
                is_fit=fit[y,x]
                residual=(e['target']-base['rgb'])[:,y,x].T
                residual[~is_fit]=0
                np.savez_compressed(dest/f'{raw:03d}_{state}.npz',basis=design.cpu().numpy(),
                                    residual=residual.cpu().numpy(),fit=is_fit.cpu().numpy())
                checks.append(dict(raw=raw,state=state,fit=int(is_fit.sum()),protect=int((~is_fit).sum()),basis_maxabs=error))
            if raw%10==0:
                print('DESIGN',raw,flush=True)
        write(RUN/'design.json',dict(holdout=HOLDOUT,knots=self.knots,checks=checks,
                                    frozen='Native scene, a6/PVD, road opacities and all ordering.',
                                    design='Exact native alpha*T from feature accumulators.'))

    def solve(self):
        arrays={'keep':[],'delete':[]}
        for name in arrays:
            for path in sorted((RUN/'design').glob(f'*_{name}.npz')):
                with np.load(path,allow_pickle=False) as data:
                    arrays[name].append((data['basis'],data['residual'],data['fit']))
        b=[];r=[];weights=[];fit_flags=[]
        for name,items in arrays.items():
            fit_n=sum(int(item[2].sum()) for item in items)
            protect_n=sum(int((~item[2]).sum()) for item in items)
            for basis,residual,fit in items:
                b.append(basis)
                r.append(residual)
                weights.append(np.where(fit,1/max(1,fit_n),.40/max(1,protect_n)))
                fit_flags.append(fit)
        b=np.concatenate(b).astype(np.float64)
        r=np.concatenate(r).astype(np.float64)
        weight=np.concatenate(weights)
        fit=np.concatenate(fit_flags)
        constraints=grid_reg()
        output={}
        for name in ('global','world'):
            design=b.sum(1,keepdims=True) if name=='global' else b
            reg=np.eye(1)*.03 if name=='global' else constraints
            fitted=[]
            for channel in range(3):
                x=np.zeros(design.shape[1])
                for _ in range(6):
                    error=design@x-r[:,channel]
                    robust=np.where(fit,1/np.sqrt(error*error+.012**2),1/.025)
                    w=weight*robust
                    matrix=design.T@(w[:,None]*design)+reg
                    right=design.T@(w*r[:,channel])
                    # Bound nodes rather than clipping a least-squares result.
                    chol=np.linalg.cholesky(matrix)
                    x=lsq_linear(chol.T,np.linalg.solve(chol,right),bounds=(-MAX_DELTA,MAX_DELTA),
                                  method='bvls',tol=1e-10).x
                fitted.append(x)
            coeff=np.stack(fitted,1)
            output[name]=coeff
        np.savez_compressed(RUN/'coefficients.npz',**output,colour_baseline=self.colour.cpu().numpy())
        return output

    def evaluate(self,coefs):
        dest=RUN/'render'
        dest.mkdir()
        rows=[]
        colours={'parent':self.colour}
        for name,coef in coefs.items():
            delta=self.basis.new_tensor(coef)
            shift=delta if name=='global' else self.basis@delta
            colours[name]=(self.colour+shift).clamp(0,1)
        checks=[]
        for raw in RAWS:
            rec=dict(raw=raw,states={},holdout=raw in HOLDOUT)
            for state in ('keep','delete'):
                delete=state=='delete'
                scene=self.assemble(raw,delete)
                e=self.evidence(raw,delete)
                old=self.render(raw,scene)
                self.check_parent(raw,state,old,e)
                w=old['extra'][2]
                boundary=(w>.15).cpu().numpy()
                seam=torch.as_tensor(binary_dilation(boundary,iterations=2)&~binary_erosion(boundary,iterations=2),device='cuda')
                affected=w>=MIN_WEIGHT
                q=self.trial.evidence[raw]['mask']
                rec['states'][state]={}
                for name,col in colours.items():
                    if name=='parent':
                        out=old
                    else:
                        self.colourize(scene,col)
                        out=self.render(raw,scene)
                    # Same signed alpha and ordering through both edit states.
                    parity=max(float((out[key]-old[key]).abs().max()) for key in ('alpha','depth_sum','extra'))
                    assert parity==0.,(name,raw,state,parity)
                    checks.append(parity)
                    residual=(out['rgb']-e['target']).abs().mean(0)
                    diff=(out['rgb']-old['rgb']).abs().mean(0)
                    err=(out['rgb']-e['true_rgb']).abs().mean(0)
                    rec['states'][state][name]=dict(
                       road=metric(residual,e['road']),affected=metric(residual,e['road']&affected),
                       seam=metric(residual,e['road']&seam),a6=metric(err,e['a6']),
                       a37=metric(err,torch.zeros_like(e['a37']) if delete else e['a37']),
                       full=metric(err,torch.ones_like(q)),protected_delta=metric(diff,e['protect']),
                       a6_delta=metric(diff,e['a6']),a37_delta=metric(diff,torch.zeros_like(e['a37']) if delete else e['a37']),
                       reveal_pvd=metric(out['extra'][3],q))
                    if raw in (*PILOTS,89,137):
                        save(dest/f'{raw:03d}_{state}_{name}.png',out['rgb'])
                oracle=(old['rgb']-e['target']).abs()-w[None]*MAX_DELTA
                rec['states'][state]['colour_oracle']={'road':metric(oracle.clamp_min(0).mean(0),e['road'])}
            rows.append(rec)
        assert self.trial.exp.unchanged()
        write(RUN/'result.json',dict(status='PASS',rows=rows,holdout=HOLDOUT,geometry_parity=0.,
                                     radiance_parameters=len(coefs['world'])*3,solver='Six robust linear solves, two model capacities',
                                     rgb_bound=MAX_DELTA,reference_role='Same-scene measured RGB and reobserved road; fit holdout excludes listed views.'))


def world_basis(xy):
    # Uniform world-local knots. Bilinear nonnegative weights partition unity.
    low=xy.min(axis=0);high=xy.max(axis=0)
    scaled=np.clip((xy-low)/np.maximum(high-low,1e-8)*(np.asarray(GRID)-1),0,np.asarray(GRID)-1)
    idx=np.minimum(scaled.astype(int),np.asarray(GRID)-2)
    fraction=scaled-idx
    b=np.zeros((len(xy),GRID[0]*GRID[1]),np.float32)
    for ox in (0,1):
        for oy in (0,1):
            coefficient=(fraction[:,0] if ox else 1-fraction[:,0])*(fraction[:,1] if oy else 1-fraction[:,1])
            b[np.arange(len(xy)),(idx[:,0]+ox)*GRID[1]+idx[:,1]+oy]=coefficient
    return b,dict(low_m=low.tolist(),high_m=high.tolist(),grid=GRID)


def grid_reg():
    n=GRID[0]*GRID[1]
    reg=np.eye(n)*.025
    for i in range(GRID[0]):
        for j in range(GRID[1]):
            a=i*GRID[1]+j
            for di,dj in ((1,0),(0,1)):
                if i+di>=GRID[0] or j+dj>=GRID[1]:
                    continue
                b=(i+di)*GRID[1]+j+dj
                reg[a,a]+=.08;reg[b,b]+=.08
                reg[a,b]-=.08;reg[b,a]-=.08
    return reg


if __name__=='__main__':
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser()
    parser.add_argument('--design-only',action='store_true')
    parser.add_argument('--reuse-design',action='store_true')
    args=parser.parse_args()
    with torch.no_grad():
        experiment=Experiment()
        if not args.reuse_design:
            experiment.cache_design()
        if not args.design_only:
            coefs=experiment.solve()
            experiment.evaluate(coefs)
