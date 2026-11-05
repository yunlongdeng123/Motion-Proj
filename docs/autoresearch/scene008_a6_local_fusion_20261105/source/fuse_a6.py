"""Zero-update a6 handoff between measured native rows and a fixed PVD proposal."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch

RUN = Path(__file__).resolve().parent.parent
PREVIOUS = RUN.parent/'scene008_native_road_pvd_compare_20261105'
sys.path.insert(0,str(PREVIOUS/'black_source_v2'))
from render_pvd import ROAD, OBS, REF, PVD, INSTANCES, Experiment, generated_layer, draw
from editor_common import prepare_rasterization, merge_rows, raster, save
from utils.general_utils import build_rotation

PILOTS = (105,113,130)
RAWS = range(89,138)
PARAMS = dict(lower_clearance_m=.03, surface_clearance_m=.12, pvd_support_sigma=3.,
              target_alpha=.25, target_lower_fraction=.80, native_depth_margin_m=.05,
              depth_bin_m=.20, pure_support_fraction=.80, max_protected_ratio=.02,
              min_blocker_mass=.05, mixed_trigger_fraction=.50,
              rgb_guard=.003, depth_guard_m=.03, evidence_guard_fraction=.99)


def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def put_features(rows,features):
    return dict(rows,features=torch.cat((rows['features'][:,:2],features),1))


def select_rows(rows,mask):
    return {key: None if value is None else value[mask].contiguous() for key,value in rows.items()}


def metric(value,mask):
    v=value[mask]
    if not torch.isfinite(v).all():
        raise RuntimeError('Nonfinite fixed-domain readout')
    return dict(count=v.numel(),sum=float(v.double().sum()))


class Trial:
    def __init__(self):
        self.exp=Experiment()
        self.n=self.exp.n
        road=ROAD/'formal/checkpoint001000.pt'
        self.road_checkpoint=road
        self.saved=torch.load(road,map_location='cpu',weights_only=False)
        assert self.saved['step']==1000 and self.saved['parent_identity']==self.n.before
        self.exp.repair.load_state_dict(self.saved['repair'],strict=True)
        self.exp.repair.requires_grad_(False)
        self.pvd=generated_layer()
        self.native_ids=torch.where(self.n.keep_a6)[0]
        self.black=self.n.bg.new_zeros(3)
        with np.load(INSTANCES/'observations/a6.npz',allow_pickle=False) as asset:
            self.anchors=asset['xyz_local_m'].copy()
            self.box=asset['box_lwh_m'].copy()
        self.zcut=float(self.anchors[:,2].min())-PARAMS['lower_clearance_m']
        self.measured_tree=cKDTree(self.anchors)
        point=self.pvd.anchor.cpu().numpy()
        sigma=self.pvd.sigma.cpu().numpy()
        distance=self.measured_tree.query(point,workers=1)[0]
        lower=(point[:,0]<0)&(point[:,2]<=self.zcut)
        lower &= distance>np.maximum(PARAMS['surface_clearance_m'],PARAMS['pvd_support_sigma']*sigma)
        self.lower=torch.as_tensor(lower,device='cuda')
        assert self.lower.any()
        with np.load(REF,allow_pickle=False) as data:
            self.fixed=data['depth_m'].copy()
        self.data={}
        self.evidence={}
        self.scale=self.exp.meta['meter_to_model']

    def raw(self,raw):
        if raw not in self.data:
            c=self.exp.camera(raw)
            with np.load(OBS/f'{raw:03d}.npz',allow_pickle=False) as d:
                assert abs(float(d['model_time'])-c.timestamp)<1e-6
                assert np.allclose(d['model_to_camera'].T,c.world_view_transform.cpu().numpy(),atol=1e-6)
                pose=torch.as_tensor(d['model_from_local'][0],device='cuda',dtype=torch.float32)
                value={k:d[k].copy() for k in ('rgb','visible','visible_reviewed','depth_valid','depth_m')}
            obs=torch.as_tensor(value['visible'][0],device='cuda')
            depth=torch.as_tensor(value['depth_valid'][0],device='cuda')
            protect=binary_dilation(value['visible_reviewed'][0],iterations=1)
            rgb=torch.as_tensor(value['rgb'],device='cuda',dtype=torch.float32).permute(2,0,1)
            if rgb.max()>1:
                rgb/=255.
            self.data[raw]=dict(cam=c,pose=pose,obs=obs,protect=torch.as_tensor(protect,device='cuda'),
                depth_valid=depth,depth=torch.as_tensor(value['depth_m'][0],device='cuda'),rgb=rgb)
        return self.data[raw]

    def native(self,raw):
        d=self.raw(raw)
        rows=prepare_rasterization(d['cam'],self.n.dynamic,self.n.args,dynamic=True)
        rows['means2D']=torch.zeros_like(rows['means3D'])
        return select_rows(rows,self.n.keep_a6)

    def patch(self,raw,native=None):
        d=self.raw(raw)
        channels=native['shs'].shape[1] if native is not None and native['shs'] is not None else 1
        return self.pvd.raster_rows(d['pose'],d['cam'],self.n.args,channels,
                                   precomputed=native is None or native['colors_precomp'] is not None)

    def render(self,raw,rows,degree=None):
        return raster(self.raw(raw)['cam'],self.n.args,
                      self.n.static.active_sh_degree if degree is None else degree,rows,self.black)

    def composite(self,raw,variant,native=None,gate=None,split=None):
        if native is None and variant not in ('pvd','pvd_lower'):
            native=self.native(raw)
        if variant=='native':
            tags=native['features'].new_zeros((len(native['features']),2))
            return self.render(raw,put_features(native,tags))
        patch=self.patch(raw,native)
        tags=torch.stack((torch.ones_like(self.lower),self.lower),1).to(patch['features'].dtype)
        patch=put_features(patch,tags)
        if variant in ('pvd_lower','lower_only','fusion','split'):
            patch=select_rows(patch,self.lower)
        if native is None:
            return self.render(raw,patch,degree=0)
        if variant in ('fusion','split'):
            native=dict(native,opacities=native['opacities']*gate[:,None])
        if variant=='split':
            native=self.clipped(raw,native,split)
        tags=native['features'].new_zeros((len(native['features']),2))
        return self.render(raw,merge_rows(put_features(native,tags),patch))

    def establish_evidence(self):
        out=RUN/'evidence'
        out.mkdir()
        rows=[]
        for raw in RAWS:
            d=self.raw(raw)
            whole=self.composite(raw,'pvd')
            lower=self.composite(raw,'pvd_lower')
            lower_alpha=lower['alpha'][0]
            whole_alpha=whole['alpha'][0]
            source=torch.as_tensor(self.fixed[raw-89,2]>0,device='cuda')
            mask=source & ~d['protect'] & ~d['depth_valid']
            mask &= (lower_alpha>=PARAMS['target_alpha'])
            mask &= whole['extra'][1] >= PARAMS['target_lower_fraction']*whole_alpha
            cutoff=lower['depth_sum'][0]/lower_alpha.clamp_min(1e-6)/self.scale
            step=PARAMS['depth_bin_m']
            bins=((cutoff-PARAMS['native_depth_margin_m'])/step).floor().int()
            count=int(mask.sum())
            self.evidence[raw]=dict(mask=mask,cutoff_bins=bins,lower_alpha=lower_alpha)
            np.savez_compressed(out/f'{raw:03d}.npz',lower=mask.cpu().numpy(),
                                cutoff_bin=np.where(mask.cpu().numpy(),bins.cpu().numpy(),-1).astype(np.int16),
                                isolated_lower_alpha=lower_alpha.cpu().numpy().astype(np.float32))
            rows.append(dict(raw=raw,target_pixels=count,known_pixels=int(d['obs'].sum()),
                             lidar_pixels=int(d['depth_valid'].sum())))
        assert all(rows[r-89]['target_pixels']>0 for r in PILOTS),rows
        contract=dict(lower_rows=int(self.lower.sum()),all_pvd_rows=len(self.lower),
            measured_anchors=len(self.anchors),z_cut_m=self.zcut,policy=PARAMS,frames=rows,
            target_pixels=sum(r['target_pixels'] for r in rows),
            fixed_before_native_readout=True,
            lower_domain='rear half x<0; local z below lowest fused LiDAR anchor minus clearance; 3D clearance from measured anchors; fixed hidden proxy; remove reviewed a6 silhouette + 1px and measured depths; require isolated lower alpha and lower PVD fraction',
            blocker_depth='isolated lower alpha-weighted camera z minus margin, floor-binned; native raster centers in front of cutoff',
            measured_geometry_is_frozen=True)
        write(out/'contract.json',contract)
        print('FIXED_EVIDENCE', {k:contract[k] for k in ('lower_rows','z_cut_m','target_pixels')},flush=True)

    def row_weights(self,raw,native,mask):
        # Derivative wrt a row's constant feature is the exact sum of alpha*T
        # on the selected pixels, independent of RGB and valid on opaque prefixes.
        if not bool(mask.any()):
            return native['features'].new_zeros(len(native['features']))
        with torch.enable_grad():
            marker=torch.zeros_like(native['features'][:,:1],requires_grad=True)
            result=self.render(raw,put_features(native,marker))
            return torch.autograd.grad(result['extra'][0,mask].sum(),marker)[0][:,0].detach()

    def attribute(self):
        n=len(self.native_ids)
        total=self.black.new_zeros(n)
        protected=total.clone()
        lower_weight=total.clone()
        geometry_guard=total.clone()
        for raw in RAWS:
            ev=self.evidence[raw]
            d=self.raw(raw)
            native=self.native(raw)
            protected+=self.row_weights(raw,native,d['obs']|d['depth_valid'])
            weights=total.new_zeros(n)
            active_bins=ev['cutoff_bins'][ev['mask']].unique().tolist()
            for b in active_bins:
                sel=ev['mask']&(ev['cutoff_bins']==b)
                cutoff=b*PARAMS['depth_bin_m']*self.scale
                part=dict(native,mask=native['mask']&(native['features'][:,0]<cutoff))
                weights+=self.row_weights(raw,part,sel)
            total+=weights
            hit=weights>1e-7
            if not bool(hit.any()):
                continue
            p=d['pose']
            local=torch.linalg.solve(p[:3,:3],(native['means3D'][hit]-p[:3,3]).T).T
            inside=(local[:,0]<.1)&(local[:,2]<self.zcut)&(local[:,2]>-self.box[2]*.5-.28)
            box=torch.as_tensor(self.box,device='cuda')
            inside&=(local.abs()<box*.5+.28).all(1)
            lower_weight[hit]+=weights[hit]*inside
            if native['scales'] is None:
                raise RuntimeError('Current parent requires the native scale/rotation contract')
            size=native['scales'][hit].max(1).values/self.scale
            distance=self.measured_tree.query(local.cpu().numpy(),workers=1)[0]
            distance=torch.as_tensor(distance,device='cuda')
            close=distance<=torch.maximum(size*2,size.new_full((),.10))
            geometry_guard[hit]+=weights[hit]*close
        active=total>=PARAMS['min_blocker_mass']
        safe_protection=protected<=PARAMS['max_protected_ratio']*total
        lower=lower_weight>=PARAMS['pure_support_fraction']*total
        near_measured=geometry_guard>.10*total
        pure=active&safe_protection&lower&~near_measured
        mixed=active&(~safe_protection|near_measured)
        unassigned=active&~pure&~mixed
        gate=torch.ones_like(total)
        gate[pure]=0
        all_mass=total.sum().item()
        mixed_fraction=float(total[mixed].sum())/max(all_mass,1e-12)
        receipt=dict(native_rows=n,eligible=int(active.sum()),pure=int(pure.sum()),mixed=int(mixed.sum()),
                     unassigned=int(unassigned.sum()),target_blocker_mass=all_mass,
                     pure_mass=float(total[pure].sum()),mixed_mass=float(total[mixed].sum()),
                     unassigned_mass=float(total[unassigned].sum()),mixed_fraction=mixed_fraction,
                     mixed_split_triggered=mixed_fraction>=PARAMS['mixed_trigger_fraction'],
                     row_index='Persistent index in frozen packed native checkpoint; same gates at every raw frame')
        np.savez_compressed(RUN/'ledger.npz',native_row_id=self.native_ids.cpu().numpy(),
                            blocker_mass=total.cpu().numpy(),protected_mass=protected.cpu().numpy(),
                            lower_mass=lower_weight.cpu().numpy(),near_measured_mass=geometry_guard.cpu().numpy(),
                            pure=pure.cpu().numpy(),mixed=mixed.cpu().numpy(),gate=gate.cpu().numpy())
        write(RUN/'ledger.json',receipt)
        print('BLOCKERS',receipt,flush=True)
        return gate,mixed,receipt

    def clipped(self,raw,rows,mixed):
        # Approximate the retained z >= z_cut side by a conditional Gaussian.
        # Native SH/material and peak opacity remain on that measured side.
        # The plane and candidate identities are shared in the car's local frame.
        pose=self.raw(raw)['pose']
        n=pose[:3,2]/pose[:3,2].norm()
        plane=pose[:3,3]@n+self.zcut*self.scale
        index=torch.where(mixed & rows['mask'])[0]
        if not len(index):
            return rows
        mu=rows['means3D'][index]
        rot=build_rotation(rows['rotations'][index])
        basis=rot*rows['scales'][index,None,:]
        cov=basis@basis.transpose(1,2)
        direction=cov@n
        deviation=(direction@n).clamp_min(1e-14).sqrt()
        a=(plane-mu@n)/deviation
        # A whole-row suppression is not a split. Leave severely one-sided
        # or wholly protected kernels intact in this candidate.
        use=(a> -2.5)&(a<1.30)
        if not bool(use.any()):
            return rows
        index=index[use]
        mu,cov,direction,deviation,a=(x[use] for x in (mu,cov,direction,deviation,a))
        q=.5*torch.erfc(a/np.sqrt(2))
        mills=torch.exp(-.5*a.square())/np.sqrt(2*np.pi)/q
        shift=direction/deviation[:,None]
        centre=mu+shift*mills[:,None]
        shape=cov+(a*mills-mills.square())[:,None,None]*(shift[:,:,None]*shift[:,None,:])
        eigen,axes=torch.linalg.eigh(shape)
        axes[:,:,0]*=torch.linalg.det(axes).sign()[:,None]
        quat=Rotation.from_matrix(axes.cpu().numpy()).as_quat()[:,[3,0,1,2]]
        result=dict(rows)
        result['means3D']=rows['means3D'].index_copy(0,index,centre)
        result['rotations']=rows['rotations'].index_copy(0,index,centre.new_tensor(quat))
        result['scales']=rows['scales'].index_copy(0,index,eigen.clamp_min(1e-12).sqrt())
        # The native raster derives camera depth from the modified centres; its
        # auxiliary depth channel must follow the same centres.
        feat=rows['features'].clone()
        w=self.raw(raw)['cam'].world_view_transform[:,2]
        feat[index,0]=centre@w[:3]+w[3]
        result['features']=feat
        return result

    def measured_metrics(self,raw,pkg,native):
        d=self.raw(raw)
        mask=self.evidence[raw]['mask']
        obs=d['obs']
        lidar=d['depth_valid']
        norm=lambda p:p['depth_sum'][0]/p['alpha'][0].clamp_min(1e-5)/self.scale
        return dict(lower_pvd=metric(pkg['extra'][1],mask),
                    isolated_lower_alpha=metric(self.evidence[raw]['lower_alpha'],mask),
                    pvd_on_known=metric(pkg['extra'][0],obs),
                    known_alpha=metric(pkg['alpha'][0],obs),
                    known_rgb=metric((pkg['rgb']-d['rgb']).abs().mean(0),obs),
                    known_rgb_change=metric((pkg['rgb']-native['rgb']).abs().mean(0),obs),
                    measured_depth=metric((norm(pkg)-d['depth']).abs(),lidar))

    def run(self):
        self.establish_evidence()
        # Pilot renders happen before the all-frame attribution/fusion.
        pilot=RUN/'pilots'
        pilot.mkdir()
        for raw in PILOTS:
            base=self.composite(raw,'native')
            for variant in ('native','pvd','direct'):
                pkg=base if variant=='native' else self.composite(raw,variant)
                save(pilot/f'{raw:03d}_{variant}.png',pkg['rgb'])
            print('PILOT',raw,int(self.evidence[raw]['mask'].sum()),flush=True)
        gate,mixed,receipt=self.attribute()
        variants=['native','pvd','direct','lower_only','fusion']
        if receipt['mixed_split_triggered']:
            variants.append('split')
        render=RUN/'render'
        render.mkdir()
        rows=[]
        for raw in RAWS:
            kw=self.native(raw)
            native=self.composite(raw,'native',kw)
            rec=dict(raw=raw,variants={})
            for variant in variants:
                pkg=native if variant=='native' else self.composite(raw,variant,
                    None if variant=='pvd' else kw,gate,mixed)
                assert float((pkg['extra'][0]-pkg['alpha'][0]).max())<3e-6
                rec['variants'][variant]=self.measured_metrics(raw,pkg,native)
                if variant!='lower_only':
                    save(render/f'{raw:03d}_{variant}.png',pkg['rgb'])
                    contribution=pkg['alpha'] if variant=='native' else pkg['extra'][0:1]
                    save(render/f'{raw:03d}_{variant}_pvd.png',contribution.expand(3,-1,-1))
                if variant in ('direct','fusion','split'):
                    evidence=self.evidence[raw]
                    deficit=(evidence['lower_alpha']-pkg['extra'][1]).clamp_min(0)*evidence['mask']
                    vis=torch.stack((deficit, .20*evidence['mask'], .08*evidence['mask']))
                    save(render/f'{raw:03d}_{variant}_blocked.png',vis)
            rows.append(rec)
        assert self.exp.unchanged()
        result=dict(status='PASS_RENDER',training_updates=0,rows=rows,variants=variants,
                    ledger=receipt,pvd_source=str(PVD),road_source=str(self.road_checkpoint),
                    render_policy='All main comparisons contain a6 only. Direct uses all fixed PVD rows. Fusion uses the fixed lower subset; all retained PVD geometry, color and alpha values are unchanged.')
        write(RUN/'results.json',result)

    def street_diagnostics(self):
        rows=[]
        for raw in PILOTS:
            d=self.raw(raw)
            kw=self.native(raw)
            cut=self.evidence[raw]['mask']
            local=self.composite(raw,'direct',kw)
            # Use one joint sort. The original helper's final channel here is
            # full-PVD contribution, so compare the same full-PVD proxy in all states.
            rec=dict(raw=raw,a6_only=metric(local['extra'][0],cut))
            for name,delete,hide in [('keep',False,False),('delete',True,False),('hide_native_a6',True,True)]:
                pkg=draw(self.exp,d['cam'],self.pvd,d['pose'],delete,hide_native_a6=hide)
                rec[name]=metric(pkg['extra'][3],cut)
                save(RUN/'pilots'/f'{raw:03d}_street_{name}.png',pkg['rgb'])
            rows.append(rec)
        write(RUN/'street_diagnostic.json',dict(rows=rows,role='Full-PVD weights only on the fixed lower domain; diagnostic scenes separate from the black a6 comparison.'))


if __name__=='__main__':
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser()
    parser.add_argument('--without-street',action='store_true')
    opts=parser.parse_args()
    with torch.no_grad():
        trial=Trial()
        trial.run()
        if not opts.without_street:
            trial.street_diagnostics()
