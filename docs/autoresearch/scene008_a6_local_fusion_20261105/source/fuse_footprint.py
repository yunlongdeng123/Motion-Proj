"""Footprint attribution and object-local one-sided split for mixed a6 blockers."""
import argparse
import json

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from fuse_a6 import (RUN, RAWS, PILOTS, PARAMS, Trial, build_rotation, put_features,
                     metric, save, write)

FOOTPRINT_MIN=.60
SPLIT_SEPARATION_M=.025
RAW_FIT=PILOTS


class FootprintTrial(Trial):
    def read_evidence(self):
        for raw in RAWS:
            with np.load(RUN/'evidence'/f'{raw:03d}.npz',allow_pickle=False) as d:
                self.evidence[raw]=dict(mask=torch.as_tensor(d['lower'],device='cuda'),
                    cutoff_bins=torch.as_tensor(d['cutoff_bin'],device='cuda'),
                    lower_alpha=torch.as_tensor(d['isolated_lower_alpha'],device='cuda'))

    def coordinate_weights(self,raw,native,mask):
        if not bool(mask.any()):
            return native['features'].new_zeros((len(native['features']),3))
        with torch.enable_grad():
            marker=native['features'].new_zeros((len(native['features']),3),requires_grad=True)
            out=self.render(raw,put_features(native,marker))['extra'][:,mask]
            y,x=mask.nonzero(as_tuple=True)
            coeff=torch.stack((torch.ones_like(x),x,y)).to(out.dtype)
            return torch.autograd.grad((out*coeff).sum(),marker)[0].detach()

    def local_point_sums(self,raw,native,marks):
        # Map a weighted pixel centroid to the native row's camera-z plane.
        # Transform back through the exact native camera and a6 pose.
        d=self.raw(raw)
        with np.load(RUN.parent/'scene008_instances_20261105_r1/observations/frames'/f'{raw:03d}.npz',
                     allow_pickle=False) as data:
            k=d['pose'].new_tensor(data['K'])
        weight,x,y=marks.unbind(1)
        depth=native['features'][:,0]
        camera=torch.stack(((x+(.5-k[0,2])*weight)*depth/k[0,0],
                            (y+(.5-k[1,2])*weight)*depth/k[1,1],depth*weight,weight),1)
        camera_to_local=torch.linalg.inv(d['pose'])@torch.linalg.inv(d['cam'].world_view_transform.T)
        return (camera@camera_to_local.T)[:,:3]

    def register(self, prefix='footprint', broader=False):
        n=len(self.native_ids)
        sums=self.black.new_zeros((3,n,3))
        mass=self.black.new_zeros((3,n))
        lidar_mass=self.black.new_zeros(n)
        support_pixels=0
        for raw in RAWS:
            ev=self.evidence[raw]
            d=self.raw(raw)
            native=self.native(raw)
            target=sums[0].new_zeros((n,3))
            mask,bins=ev['mask'],ev['cutoff_bins']
            if broader:
                isolated=self.composite(raw,'pvd_lower')
                whole=self.composite(raw,'pvd')
                low=isolated['alpha'][0]
                mask=(low>=PARAMS['target_alpha']) & ~d['protect'] & ~d['depth_valid']
                mask &= whole['extra'][1]>=PARAMS['target_lower_fraction']*whole['alpha'][0]
                z=isolated['depth_sum'][0]/low.clamp_min(1e-6)/self.scale
                bins=((z-PARAMS['native_depth_margin_m'])/PARAMS['depth_bin_m']).floor().int()
                lidar_mass+=self.coordinate_weights(raw,native,d['depth_valid'])[:,0]
            support_pixels+=int(mask.sum())
            for b in bins[mask].unique().tolist():
                selected=mask&(bins==b)
                cutoff=b*PARAMS['depth_bin_m']*self.scale
                prefix_rows=dict(native,mask=native['mask']&(native['features'][:,0]<cutoff))
                target+=self.coordinate_weights(raw,prefix_rows,selected)
            protected=self.coordinate_weights(raw,native,d['obs']|d['depth_valid'])
            other=self.coordinate_weights(raw,native,~mask)
            for i,marks in enumerate((target,protected,other)):
                mass[i]+=marks[:,0]
                sums[i]+=self.local_point_sums(raw,native,marks)
        with np.load(RUN/'ledger.npz',allow_pickle=False) as d:
            original=torch.as_tensor(d['pure'],device='cuda')
            close=torch.as_tensor(d['near_measured_mass'],device='cuda')
            earlier=torch.as_tensor(d['blocker_mass'],device='cuda')
        if not broader:
            assert torch.allclose(mass[0],earlier,atol=5e-4,rtol=5e-5)
        target,protected,other=mass
        enough=target>=PARAMS['min_blocker_mass']
        dominance=target/(target+other).clamp_min(1e-8)
        pure=enough & (dominance>=FOOTPRINT_MIN) & (protected<=PARAMS['max_protected_ratio']*target)
        pure &= close<=.10*target
        pure |= original
        if broader:
            pure &= lidar_mass<.10
        have_observed=protected>PARAMS['max_protected_ratio']*target
        reference=torch.where(have_observed[:,None],sums[1],sums[2])
        ref_mass=torch.where(have_observed,protected,other)
        keep=reference/ref_mass[:,None].clamp_min(1e-8)
        missing=sums[0]/target[:,None].clamp_min(1e-8)
        delta=keep-missing
        distance=delta.norm(dim=1)
        normal=delta/distance[:,None].clamp_min(1e-8)
        split=enough & ~pure & (ref_mass>.05) & (distance>=SPLIT_SEPARATION_M)
        if broader:
            split &= lidar_mass<.10
        # A conservative plane near the lower/blocked witness leaves a buffer
        # around the centroid of measured support.
        plane=((.70*missing+.30*keep)*normal).sum(1)
        gate=torch.ones_like(target)
        gate[pure]=0
        result=dict(candidates=int(enough.sum()),pure_rows=int(pure.sum()),split_rows=int(split.sum()),
                    split_measured_rows=int((split&have_observed).sum()),
                    pure_mass_fraction=float(target[pure].sum()/target.sum()),
                    split_mass_fraction=float(target[split].sum()/target.sum()),
                    support_pixels=support_pixels,broader_registration=broader,
                    lidar_protect_alpha_mass=.10 if broader else None,
                    dominance_min=FOOTPRINT_MIN,centroid_distance_min_m=SPLIT_SEPARATION_M,
                    plane='canonical per row; keep side points from target centroid toward protected centroid; plane 30% along their separation',
                    attribution='Native alpha*T at each row depth; canonical centroids from measured or remaining support and fixed lower rays.')
        write(RUN/f'{prefix}_ledger.json',result)
        np.savez_compressed(RUN/f'{prefix}_ledger.npz', native_row_id=self.native_ids.cpu().numpy(),
                            gate=gate.cpu().numpy(),split=split.cpu().numpy(),normal=normal.cpu().numpy(),
                            plane_m=plane.cpu().numpy(),centroid_distance=distance.cpu().numpy(),
                            masses=mass.cpu().numpy(),lidar_mass=lidar_mass.cpu().numpy())
        print('FOOTPRINT_LEDGER',result,flush=True)
        return gate,dict(use=split,normal=normal,plane=plane)

    def clipped(self,raw,rows,split):
        pose=self.raw(raw)['pose']
        index=torch.where(split['use']&rows['mask'])[0]
        if not len(index):
            return rows
        normal=split['normal'][index]@pose[:3,:3].T/self.scale
        plane=normal@pose[:3,3]+self.scale*split['plane'][index]
        mu=rows['means3D'][index]
        rot=build_rotation(rows['rotations'][index])
        basis=rot*rows['scales'][index,None,:]
        covariance=basis@basis.transpose(1,2)
        direction=(covariance@normal[:,:,None])[:,:,0]
        dev=(direction*normal).sum(1).clamp_min(1e-14).sqrt()
        a=(plane-(mu*normal).sum(1))/dev
        use=(a> -2.5)&(a<1.30)
        if not bool(use.any()):
            return rows
        index=index[use]
        mu,covariance,direction,dev,a=(x[use] for x in (mu,covariance,direction,dev,a))
        q=.5*torch.erfc(a/np.sqrt(2))
        mills=torch.exp(-.5*a.square())/np.sqrt(2*np.pi)/q
        shift=direction/dev[:,None]
        centre=mu+shift*mills[:,None]
        covariance=covariance+(a*mills-mills.square())[:,None,None]*(shift[:,:,None]*shift[:,None,:])
        eigen,axes=torch.linalg.eigh(covariance)
        axes[:,:,0]*=torch.linalg.det(axes).sign()[:,None]
        quat=Rotation.from_matrix(axes.cpu().numpy()).as_quat()[:,[3,0,1,2]]
        result=dict(rows)
        result['means3D']=rows['means3D'].index_copy(0,index,centre)
        result['rotations']=rows['rotations'].index_copy(0,index,centre.new_tensor(quat))
        result['scales']=rows['scales'].index_copy(0,index,eigen.clamp_min(1e-12).sqrt())
        feature=rows['features'].clone()
        view=self.raw(raw)['cam'].world_view_transform[:,2]
        feature[index,0]=centre@view[:3]+view[3]
        result['features']=feature
        self.clipped_counts[raw]=int(len(index))
        return result

    def output(self,raw,gate,split,folder):
        kw=self.native(raw)
        baseline=self.composite(raw,'native',kw)
        result={}
        for variant,name in [('fusion','footprint'),('split','footprint_split')]:
            pkg=self.composite(raw,variant,kw,gate,split)
            result[name]=self.measured_metrics(raw,pkg,baseline)
            save(folder/f'{raw:03d}_{name}.png',pkg['rgb'])
            save(folder/f'{raw:03d}_{name}_pvd.png',pkg['extra'][0:1].expand(3,-1,-1))
            e=self.evidence[raw]
            deficit=(e['lower_alpha']-pkg['extra'][1]).clamp_min(0)*e['mask']
            save(folder/f'{raw:03d}_{name}_blocked.png',torch.stack((deficit,.20*e['mask'],.08*e['mask'])))
        return result

    def run_footprint(self, resume=False, broader=False):
        self.clipped_counts={}
        self.read_evidence()
        prefix='lower_support' if broader else 'footprint'
        pilot=RUN/f'{prefix}_pilot'
        if resume:
            with np.load(RUN/'footprint_ledger.npz',allow_pickle=False) as ledger:
                assert np.array_equal(ledger['native_row_id'],self.native_ids.cpu().numpy())
                gate=torch.as_tensor(ledger['gate'],device='cuda')
                split={key:torch.as_tensor(ledger[field],device='cuda')
                       for key,field in [('use','split'),('normal','normal'),('plane','plane_m')]}
            pilots=json.loads((pilot/'metrics.json').read_text())
            for raw in RAW_FIT:
                self.clipped(raw,self.native(raw),split)
        else:
            gate,split=self.register(prefix,broader)
            pilot.mkdir()
            pilots=[dict(raw=raw,variants=self.output(raw,gate,split,pilot)) for raw in RAW_FIT]
            write(pilot/'metrics.json',pilots)
        print('PILOTS_DONE',flush=True)
        dest=RUN/('footprint_replayed' if resume else f'{prefix}_render')
        dest.mkdir()
        rows=[]
        for raw in RAWS:
            if raw in RAW_FIT:
                for path in pilot.glob(f'{raw:03d}_*.png'):
                    import os
                    os.link(path,dest/path.name)
                variants=next(row['variants'] for row in pilots if row['raw']==raw)
            else:
                variants=self.output(raw,gate,split,dest)
            rows.append(dict(raw=raw,variants=variants))
        assert self.exp.unchanged()
        write(RUN/f'{prefix}_results.json',dict(status='PASS',rows=rows,training_updates=0,
                                              render_dir=dest.name,
                                              split_counts=self.clipped_counts))


if __name__=='__main__':
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--resume',action='store_true')
    mode.add_argument('--broader-support',action='store_true')
    args=parser.parse_args()
    with torch.no_grad():
        FootprintTrial().run_footprint(resume=args.resume,broader=args.broader_support)
