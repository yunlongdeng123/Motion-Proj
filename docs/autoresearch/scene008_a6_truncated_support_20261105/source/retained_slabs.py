"""Constrained partial-support replacement on the frozen Scene008 a6 handoff."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

RUN=Path(__file__).resolve().parent.parent
LATEST=RUN.parent/'scene008_a6_residual_replay_20261105'
sys.path.insert(0,str(LATEST/'source'))
import replay_retirement
from replay_retirement import Trial as Replay, OTHER_RGB, OTHER_ALPHA, MIN_SCORE
from fuse_a6 import RAWS, PILOTS, PARAMS, build_rotation, metric, save, select_rows, write
from editor_common import merge_rows

REVEAL_DROP=.01
SLABS=3
SEPARATION=.025
PLANE_FRACTION=.5


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


class Trial(Replay):
    def load_parent(self):
        # Retain the original checkpoint and prior artifacts. Route this run's
        # inherited replay receipt to its own directory.
        replay_retirement.RUN=RUN
        self.start()
        self.parent=load(LATEST/'result.json')
        with np.load(LATEST/'shared_retirement.npz',allow_pickle=False) as data:
            assert np.array_equal(data['native_row_id'],self.native_ids.cpu().numpy())
            self.retired=torch.as_tensor(data['extra_retired'],device='cuda')
        self.raw38={raw:self.draw_cached(raw,self.zero) for raw in RAWS}
        self.current={raw:dict(self.split_cache[raw],mask=self.split_cache[raw]['mask']&~self.retired)
                      for raw in RAWS}
        self.before={raw:self.composite(raw,'lower_only',self.current[raw]) for raw in RAWS}
        for row in self.parent['result']:
            raw=row['raw']
            metrics=self.measured_metrics(raw,self.before[raw],self.protection[raw]['untouched'])
            for key,item in metrics.items():
                old=row['metrics'][key]
                assert item['count']==old['count'] and abs(item['sum']-old['sum'])<.001,(raw,key)
        write(RUN/'parent_replay.json',dict(frames=list(RAWS),max_metric_sum_tolerance=.001,
                                          retired=int(self.retired.sum()), inherited_split=True))

    def fit_planes(self):
        n=len(self.native_ids)
        total=self.black.new_zeros((4,n))
        xyz=self.black.new_zeros((3,n,3))
        for raw in RAWS:
            kw=self.current[raw]
            d=self.raw(raw)
            ev=self.evidence[raw]
            marks=xyz.new_zeros((n,3))
            for b in ev['cutoff_bins'][ev['mask']].unique().tolist():
                sel=ev['mask']&(ev['cutoff_bins']==b)
                cut=b*PARAMS['depth_bin_m']*self.scale
                prefix=dict(kw,mask=kw['mask']&(kw['features'][:,0]<cut))
                marks+=self.coordinate_weights(raw,prefix,sel)
            measured=self.coordinate_weights(raw,kw,d['obs']|d['depth_valid'])
            other=self.coordinate_weights(raw,kw,self.protection[raw]['outside'])
            for i,item in enumerate((marks,measured,other)):
                total[i]+=item[:,0]
                xyz[i]+=self.local_point_sums(raw,kw,item)
            total[3]+=self.row_weights(raw,kw,d['depth_valid'])
        target,known,outside,lidar=total
        support=(known+outside)
        saved=(xyz[1]+xyz[2])/support[:,None].clamp_min(1e-8)
        missing=xyz[0]/target[:,None].clamp_min(1e-8)
        delta=saved-missing
        separation=delta.norm(dim=1)
        normals=delta/separation[:,None].clamp_min(1e-8)
        cut=((missing+PLANE_FRACTION*delta)*normals).sum(1)
        useful=(target>=MIN_SCORE)&(support>=.05)&(separation>=SEPARATION)&~self.retired
        ids=torch.where(useful)[0]
        ranking=target/(known+outside+4*lidar+.01)
        order=ids[torch.argsort(ranking[ids],descending=True,stable=True)]
        np.savez_compressed(RUN/'planes.npz',native_row_id=self.native_ids.cpu().numpy(),
                            normal_local=normals.cpu().numpy(),cut_local_m=cut.cpu().numpy(),
                            order=order.cpu().numpy(),masses=total.cpu().numpy())
        receipt=dict(rows=len(order),residual_mass=float(target.sum()),represented_mass=float(target[order].sum()),
                     split_side='Retain side toward canonical protected centroid',
                     plane_fraction=PLANE_FRACTION,minimum_centroid_separation_m=SEPARATION,
                     standardized_window=[-1.75,1.],covariance_slabs=SLABS)
        write(RUN/'planes.json',receipt)
        print('PLANES',receipt,flush=True)
        return order,normals,cut

    def decomposed(self,raw,order,normal,cut,slabs):
        row=self.current[raw]
        active=order[row['mask'][order]]
        if not len(active):
            return active,None
        d=self.raw(raw)
        pose=d['pose']
        n=normal[active]@pose[:3,:3].T/self.scale
        b=n@pose[:3,3]+self.scale*cut[active]
        mu=row['means3D'][active]
        r=build_rotation(row['rotations'][active])
        basis=r*row['scales'][active,None,:]
        sigma=basis@basis.transpose(1,2)
        direction=(sigma@n[:,:,None])[:,:,0]
        dev=(direction*n).sum(1).clamp_min(1e-14).sqrt()
        a=(b-(mu*n).sum(1))/dev
        use=(a> -1.75)&(a<1.)
        if not bool(use.any()):
            return active[use],None
        active,mu,sigma,direction,dev,a=(value[use] for value in (active,mu,sigma,direction,dev,a))
        shift=direction/dev[:,None]
        outer=shift[:,:,None]*shift[:,None,:]
        # Conditional moments for standard-normal intervals on the retained
        # side. One child is the previous moment-match form. Three children
        # resolve the boundary and preserve tangential covariance separately.
        cdf=lambda x:.5*torch.erfc(-x/np.sqrt(2))
        density=lambda x:torch.exp(-.5*x.square())/np.sqrt(2*np.pi)
        total=1-cdf(a)
        centre_parts=[];scale_parts=[];quat_parts=[];opacity_parts=[]
        for j in range(slabs):
            start=cdf(a)+j*total/slabs
            end=cdf(a)+(j+1)*total/slabs
            low=a if j==0 else np.sqrt(2)*torch.erfinv((2*start-1).clamp(-.999999,.999999))
            high=np.sqrt(2)*torch.erfinv((2*end-1).clamp(-.999999,.999999))
            high_mass=density(high) if j+1<slabs else torch.zeros_like(a)
            q=(end-start).clamp_min(1e-7)
            mean=(density(low)-high_mass)/q
            var=(1+(low*density(low)-high*high_mass)/q-mean.square()).clamp_min(1e-3)
            centre=mu+shift*mean[:,None]
            cov=sigma+(var-1)[:,None,None]*outer
            eig,axes=torch.linalg.eigh(cov)
            axes[:,:,0]*=torch.linalg.det(axes).sign()[:,None]
            quat=Rotation.from_matrix(axes.cpu().numpy()).as_quat()[:,[3,0,1,2]]
            multiplier=(q/var.sqrt()).clamp_max(1.) if slabs>1 else torch.ones_like(q)
            centre_parts.append(centre)
            scale_parts.append(eig.clamp_min(1e-12).sqrt())
            quat_parts.append(mu.new_tensor(quat))
            opacity_parts.append(row['opacities'][active]*multiplier[:,None])
        parent=active.repeat(slabs)
        child={key:None if value is None else value[parent].contiguous() for key,value in row.items()}
        child['means3D']=torch.cat(centre_parts)
        child['scales']=torch.cat(scale_parts)
        child['rotations']=torch.cat(quat_parts)
        child['opacities']=torch.cat(opacity_parts)
        view=d['cam'].world_view_transform[:,2]
        child['features'][:,0]=child['means3D']@view[:3]+view[3]
        assert torch.isfinite(child['means3D']).all() and torch.isfinite(child['scales']).all()
        return active,(parent,child)

    def composite_rows(self,raw,selected,prepared):
        native=self.current[raw]
        active,data=prepared[raw]
        if data is None:
            return native
        parent,child=data
        hit=active[selected[active]]
        if not len(hit):
            return native
        mask=native['mask'].clone()
        mask[hit]=False
        return merge_rows(dict(native,mask=mask),select_rows(child,selected[parent]))

    def audit_candidate(self,selected,prepared,early=True):
        rows=[];fails=[]
        for raw in (*PILOTS,*[r for r in RAWS if r not in PILOTS]):
            native=self.composite_rows(raw,selected,prepared)
            out=self.composite(raw,'lower_only',native)
            result,violations=self.assess(raw,out,self.raw38[raw])
            previous=self.parent['result'][raw-89]['metrics']
            denominator=previous['isolated_lower_alpha']['sum']
            if denominator and result['lower_pvd']['sum']/denominator < previous['lower_pvd']['sum']/denominator-REVEAL_DROP:
                violations.append('lower_reveal')
            rows.append(dict(raw=raw,metrics=result,violations=violations))
            if violations:
                fails.append(dict(raw=raw,metrics=violations))
                if early:
                    break
        return sorted(rows,key=lambda r:r['raw']),fails

    def search_replacements(self,order,prepared):
        selected=self.zero.clone()
        log=[]
        # Try increasingly broad fixed groups. Subdivide rejected groups once;
        # this gives a bounded, reproducible search without parameter training.
        batches=torch.tensor_split(order,12)
        for group,batch in enumerate(batches):
            pending=[(0,batch)]
            while pending:
                depth,ids=pending.pop(0)
                if not len(ids):
                    continue
                candidate=selected.clone()
                candidate[ids]=True
                rows,rejects=self.audit_candidate(candidate,prepared)
                ratio=self.fraction(rows) if len(rows)==49 else None
                log.append(dict(group=group,depth=depth,added=len(ids),pass_gate=not rejects,
                                failures=rejects,ratio=ratio,frames=len(rows)))
                if not rejects and ratio>=self.current_ratio-1e-7:
                    selected=candidate
                    self.current_ratio=ratio
                elif depth==0:
                    pending.extend((1,p) for p in torch.tensor_split(ids,2))
        rows,failed=self.audit_candidate(selected,prepared,early=False)
        assert not failed
        return selected,rows,log

    def render_outputs(self,prepared,selected):
        folder=RUN/'render'
        folder.mkdir()
        for raw in RAWS:
            after=self.composite(raw,'lower_only',self.composite_rows(raw,selected,prepared))
            for name,out in [('previous',self.before[raw]),('new',after)]:
                save(folder/f'{raw:03d}_{name}.png',out['rgb'])
                save(folder/f'{raw:03d}_{name}_pvd.png',out['extra'][:1].expand(3,-1,-1))
                ev=self.evidence[raw]
                deficit=(ev['lower_alpha']-out['extra'][1]).clamp_min(0)*ev['mask']
                save(folder/f'{raw:03d}_{name}_blocked.png',torch.stack((deficit,.20*ev['mask'],.08*ev['mask'])))

    def run(self):
        self.load_parent()
        order,normals,cut=self.fit_planes()
        assert len(order)>0
        before=self.parent['result']
        baseline=self.fraction(before)
        outcomes={}
        best=baseline
        chosen='previous'
        selected=self.zero.clone()
        for label,slabs in [('one_child',1),('three_slabs',SLABS)]:
            self.current_ratio=baseline
            prepared={raw:self.decomposed(raw,order,normals,cut,slabs) for raw in RAWS}
            all_ids=self.zero.clone()
            all_ids[order]=True
            pilot=self.audit_candidate(all_ids,prepared,early=True)
            print('PILOT',label,pilot[1],flush=True)
            current,rows,log=self.search_replacements(order,prepared)
            ratio=self.fraction(rows)
            active_counts=[int(current[a].sum()) for a,_ in prepared.values()]
            outcomes[label]=dict(ratio=ratio,selected=int(current.sum()),rows=rows,search=log,
                                 min_active=min(active_counts),max_active=max(active_counts),
                                 pilot_all_candidates_failures=pilot[1])
            if ratio>best+1e-7:
                best=ratio
                chosen=label
                selected=current
            print('ARM',label,ratio,int(current.sum()),flush=True)
        slabs=SLABS if chosen=='three_slabs' else 1
        prepared={raw:self.decomposed(raw,order,normals,cut,slabs) for raw in RAWS}
        self.render_outputs(prepared,selected)
        np.savez_compressed(RUN/'accepted.npz',native_row_id=self.native_ids.cpu().numpy(),
                            selected=selected.cpu().numpy(),slabs=slabs)
        write(RUN/'result.json',dict(status='PASS',training_updates=0,baseline=baseline,chosen=chosen,best=best,
                                    outcomes=outcomes,frames=49,per_frame_reveal_drop_limit=REVEAL_DROP,
                                    outside_reference='Original guarded 38.3% fusion; applies across rounds to bound cumulative collateral change.'))
        assert self.exp.unchanged()


if __name__=='__main__':
    torch.set_num_threads(1)
    with torch.no_grad():
        Trial().run()
