"""Forward-only 3D half-space retention with the native depth ordering."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path
import sys

import numpy as np
import torch

RUN=Path(__file__).resolve().parent.parent
PARENT=RUN.parent/'scene008_a6_truncated_support_20261105'
sys.path[:0]=[str(PARENT/'source'),str(RUN/'private_cuda')]
import retained_slabs
from retained_slabs import Trial as Retained
from replay_retirement import MIN_SCORE
from fuse_a6 import RAWS,PILOTS,PARAMS,build_rotation,put_features,select_rows,metric,save,write
from editor_common import prepare_rasterization,merge_rows,road_kwargs
from a6_clip_rasterization import GaussianRasterizer
from render_pvd import OBS

SLACK=.01


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def padded(rows,width=24):
    source=rows['features']
    assert source.shape[1]<=6
    feature=source.new_zeros((len(source),width))
    feature[:,:source.shape[1]]=source
    return feature


def private(trial,raw,rows,background=None,channels=2):
    from gaussian_renderer import GaussianRasterizationSettings
    import math
    d=trial.raw(raw)
    c=d['cam']
    args=trial.n.args
    bg=trial.black if background is None else background
    spec=GaussianRasterizationSettings(
        image_height=c.image_height,image_width=c.image_width,
        tanfovx=math.tan(-.5 if args.neg_fov else c.FoVx*.5),
        tanfovy=math.tan(-.5 if args.neg_fov else c.FoVy*.5),
        bg=bg,scale_modifier=1.,viewmatrix=c.world_view_transform,projmatrix=c.full_proj_transform,
        sh_degree=trial.n.static.active_sh_degree,campos=c.camera_center,prefiltered=False,debug=False)
    assert rows['features'].shape[1]==24 and not torch.is_grad_enabled()
    _,rgb,feat,radii=GaussianRasterizer(raster_settings=spec)(**rows)
    return dict(rgb=rgb,depth_sum=feat[:1],alpha=feat[1:2],extra=feat[2:2+channels],radii=radii)


def intrinsics(camera):
    # Invert the actual pinhole projection at raster pixel centers.
    p=camera.projection_matrix
    x=2/camera.image_width
    y=2/camera.image_height
    w=p[2,3]+p[3,3]
    return torch.stack((x*w/p[0,0],((x*.5-1)*w-p[2,0]-p[3,0])/p[0,0],
                        y*w/p[1,1],((y*.5-1)*w-p[2,1]-p[3,1])/p[1,1]))


class Trial(Retained):
    def parents(self):
        retained_slabs.RUN=RUN
        self.load_parent()
        self.latest=read(PARENT/'result.json')
        data=np.load(PARENT/'accepted.npz',allow_pickle=False)
        assert np.array_equal(data['native_row_id'],self.native_ids.cpu().numpy())
        selected=torch.as_tensor(data['selected'],device='cuda')
        slabs=int(data['slabs'])
        data.close()
        with np.load(PARENT/'planes.npz',allow_pickle=False) as f:
            assert np.array_equal(f['native_row_id'],self.native_ids.cpu().numpy())
            order=torch.as_tensor(f['order'],device='cuda')
            normal=torch.as_tensor(f['normal_local'],device='cuda')
            cut=torch.as_tensor(f['cut_local_m'],device='cuda')
        self.latest_rows={}
        self.lineage={}
        self.original_pair={}
        base=torch.arange(len(self.native_ids),device='cuda')
        self.base_options=dict(order=order,normal=normal,cut=cut,selected=selected)
        self.native_parity=0.
        for raw in RAWS:
            prepared={raw:self.decomposed(raw,order,normal,cut,slabs)}
            native=self.composite_rows(raw,selected,prepared)
            _,desc=prepared[raw]
            parent=desc[0] if desc is not None else base[:0]
            self.lineage[raw]=torch.cat((base,parent[selected[parent]]))
            assert len(self.lineage[raw])==len(native['mask'])
            self.latest_rows[raw]=native
            original=self.composite(raw,'lower_only',native)
            self.original_pair[raw]=original
            expectation=self.latest['outcomes'][self.latest['chosen']]['rows'][raw-89]['metrics']
            replay=self.measured_metrics(raw,original,self.protection[raw]['untouched'])
            assert all(item['count']==expectation[k]['count'] and abs(item['sum']-expectation[k]['sum'])<.001
                       for k,item in replay.items()),raw
            full=self.compose_clip(raw,native)
            for key in ('rgb','alpha','depth_sum','extra'):
                self.native_parity=max(self.native_parity,float((full[key]-original[key]).abs().max()))
        assert self.native_parity<1e-6,self.native_parity
        self.analytic_check()
        write(RUN/'shader_validation.json',dict(noop_maxabs=self.native_parity,analytic=self.analytic,frames=49,
                                               previous_selected_rows=int(selected.sum()),active_parent=self.latest['best']))
        print('SHADER_PASS',self.native_parity,self.analytic,flush=True)

    def compose_clip(self,raw,native,metadata=None,selected=None):
        features=native['features']
        staged=padded(dict(native,features=features[:,:2]))
        if metadata is not None:
            positions,parents,terms=metadata
            take=selected[parents]
            staged[positions[take],6:]=terms[take]
        native=dict(native,features=staged)
        patch=self.patch(raw,native)
        tags=patch['features'].new_ones((len(self.lower),2))
        patch=select_rows(put_features(patch,tags),self.lower)
        patch=dict(patch,features=padded(patch))
        return private(self,raw,merge_rows(native,patch))

    def analytic_check(self):
        raw=113
        c=self.raw(raw)['cam']
        kw=select_rows(self.native(raw),torch.arange(len(self.native_ids),device='cuda')==0)
        mu=c.camera_center+c.world_view_transform[:3,:3]@self.black.new_tensor([0.,0.,1.8])
        kw['means3D']=mu[None]
        kw['mask'].fill_(True)
        kw['scales'].copy_(kw['scales'].new_tensor([[.08,.08,.08]]))
        kw['rotations'].copy_(kw['rotations'].new_tensor([[1.,0.,0.,0.]]))
        kw['opacities'].fill_(.7)
        kw['features'].copy_(kw['features'].new_tensor([[1.8,1.]]))
        blank=padded(kw)
        base=private(self,raw,dict(kw,features=blank))['alpha'][0]
        assert float(base.max())>.5
        code=blank.clone()
        p=1/.08**2
        intr=intrinsics(c)
        code[0,6:]=code.new_tensor([1,p,0,0,p,0,p,0,0,1.8*p,0.,1.,0.,.02,*intr.tolist()])
        clipped=private(self,raw,dict(kw,features=code))['alpha'][0]
        y,x=torch.meshgrid(torch.arange(base.shape[0],device='cuda',dtype=base.dtype),
                           torch.arange(base.shape[1],device='cuda',dtype=base.dtype),indexing='ij')
        dx=intr[0]*x+intr[1];dy=intr[2]*y+intr[3]
        denom=p*(dx.square()+dy.square()+1)
        t=1.8*p/denom
        spread=dy.abs()/denom.sqrt()
        side=dy*t-.02
        keep=torch.where(spread>1e-10,.5*torch.erfc(-side/(spread.clamp_min(1e-10)*np.sqrt(2))),
                         (side>=0).to(base.dtype))
        oracle=base*keep
        oracle=torch.where(oracle>=1/255,oracle,0)
        error=float((oracle-clipped).abs().max())
        assert error<3e-5,error
        one=code.clone();one[0,19]=-100
        zero=code.clone();zero[0,19]=100
        assert torch.equal(private(self,raw,dict(kw,features=one))['alpha'][0],base)
        assert private(self,raw,dict(kw,features=zero))['alpha'].abs().max().item()==0
        self.analytic=dict(pixel_alpha_error=error,full_retention_exact=True,full_removal_exact=True)

    def register(self):
        size=len(self.native_ids)
        totals=self.black.new_zeros((4,size))
        points=self.black.new_zeros((3,size,3))
        for raw in RAWS:
            kw=self.latest_rows[raw]
            ids=self.lineage[raw]
            ev=self.evidence[raw]
            info=self.raw(raw)
            target=totals.new_zeros((len(ids),3))
            for b in ev['cutoff_bins'][ev['mask']].unique().tolist():
                plane=b*PARAMS['depth_bin_m']*self.scale
                prefix=dict(kw,mask=kw['mask']&(kw['features'][:,0]<plane))
                target+=self.coordinate_weights(raw,prefix,ev['mask']&(ev['cutoff_bins']==b))
            known=self.coordinate_weights(raw,kw,info['obs']|info['depth_valid'])
            outside=self.coordinate_weights(raw,kw,self.protection[raw]['outside'])
            for i,value in enumerate((target,known,outside)):
                totals[i].index_add_(0,ids,value[:,0])
                points[i].index_add_(0,ids,self.local_point_sums(raw,kw,value))
            totals[3].index_add_(0,ids,self.row_weights(raw,kw,info['depth_valid']))
        target,known,outside,lidar=totals
        protect=(points[1]+points[2])/(known+outside).clamp_min(1e-8)[:,None]
        hidden=points[0]/target.clamp_min(1e-8)[:,None]
        gap=protect-hidden
        separation=gap.norm(dim=1)
        normal=gap/separation.clamp_min(1e-8)[:,None]
        cut=((hidden+.5*gap)*normal).sum(1)
        ids=torch.where((target>=MIN_SCORE)&(known+outside>.05)&(separation>=.025))[0]
        score=target/(known+outside+4*lidar+.01)
        order=ids[torch.argsort(score[ids],descending=True,stable=True)]
        assert len(order)>0
        np.savez_compressed(RUN/'shared_planes.npz',native_row_id=self.native_ids.cpu().numpy(),
                            order=order.cpu().numpy(),normal_local=normal.cpu().numpy(),
                            cut_local_m=cut.cpu().numpy(),masses=totals.cpu().numpy())
        write(RUN/'candidate.json',dict(rows=len(order),target_mass=float(target.sum()),registered_mass=float(target[order].sum()),
                                       cutoff='Midpoint of fixed lower blocking and protected object-local footprint centroids'))
        print('CANDIDATES',len(order),flush=True)
        return order,normal,cut

    def camera_code(self,raw,native,lineage,eligible,normal,cut):
        q=torch.where(eligible[lineage]&native['mask'])[0]
        parent=lineage[q]
        d=self.raw(raw)
        c=d['cam']
        pose=d['pose']
        w=c.world_view_transform[:3,:3]
        world_n=normal[parent]@pose[:3,:3].T/self.scale
        plane=world_n@(pose[:3,3]-c.camera_center)+self.scale*cut[parent]
        n=world_n@w
        rot=build_rotation(native['rotations'][q])
        r=w.T@rot
        variance=native['scales'][q].square()
        variance=torch.maximum(variance,variance.max(1,keepdim=True).values*1e-5).clamp_min(1e-12)
        precision=(r/variance[:,None,:])@r.transpose(1,2)
        mu=(native['means3D'][q]-c.camera_center)@w
        pm=(precision@mu[:,:,None])[:,:,0]
        tri=precision[:,[0,0,0,1,1,2],[0,1,2,1,2,2]]
        code=torch.cat((plane.new_ones((len(q),1)),tri,pm,n,plane[:,None],
                        intrinsics(c)[None].expand(len(q),-1)),1)
        assert code.shape[1]==18 and torch.isfinite(code).all()
        return q,parent,code

    def measure(self,raw,out):
        values,failed=self.assess(raw,out,self.raw38[raw])
        old=self.latest['outcomes'][self.latest['chosen']]['rows'][raw-89]['metrics']
        den=old['isolated_lower_alpha']['sum']
        if den and (old['lower_pvd']['sum']-values['lower_pvd']['sum'])/den>SLACK:
            failed.append('lower_reveal')
        return dict(raw=raw,metrics=values,violations=failed)

    def check(self,selected,source,code,early=True):
        rows=[]
        for raw in (*PILOTS,*[r for r in RAWS if r not in PILOTS]):
            one=self.measure(raw,self.compose_clip(raw,source[raw],code[raw],selected))
            rows.append(one)
            if early and one['violations']:
                break
        return sorted(rows,key=lambda r:r['raw'])

    def search(self,order,code):
        chosen=self.zero.clone()
        baseline=self.latest['best']
        score=baseline
        log=[]
        for group,ids in enumerate(torch.tensor_split(order,12)):
            queue=[(0,ids)]
            while queue:
                depth,batch=queue.pop(0)
                if not len(batch):
                    continue
                proposal=chosen.clone()
                proposal[batch]=True
                rows=self.check(proposal,self.latest_rows,code)
                passed=len(rows)==49 and not any(r['violations'] for r in rows)
                ratio=self.fraction(rows) if len(rows)==49 else None
                log.append(dict(group=group,depth=depth,n=len(batch),passes=passed,ratio=ratio,
                               violations=[dict(raw=r['raw'],metrics=r['violations']) for r in rows if r['violations']]))
                if passed and ratio>=score-1e-7:
                    chosen,score=proposal,ratio
                elif depth<2:
                    queue.extend((depth+1,x) for x in torch.tensor_split(batch,2))
        rows=self.check(chosen,self.latest_rows,code,early=False)
        assert not any(r['violations'] for r in rows)
        return chosen,rows,log

    def run(self):
        self.parents()
        order,normal,cut=self.register()
        allowed=self.zero.clone()
        allowed[order]=True
        code={raw:self.camera_code(raw,self.latest_rows[raw],self.lineage[raw],allowed,normal,cut) for raw in RAWS}
        all_cut=self.check(allowed,self.latest_rows,code,early=False)
        print('ALL_CUT',self.fraction(all_cut),sum(bool(r['violations']) for r in all_cut),flush=True)
        # Matched-row diagnostic replaces the last experiment's one-Gaussian
        # approximations with ray retention, starting from their shared parent.
        prior=self.base_options
        origin={raw:self.current[raw] for raw in RAWS}
        lineage=torch.arange(len(self.native_ids),device='cuda')
        comparison={raw:self.camera_code(raw,origin[raw],lineage,prior['selected'],prior['normal'],prior['cut']) for raw in RAWS}
        same_rows=self.check(prior['selected'],origin,comparison,early=False)
        print('SAME_ROWS',self.fraction(same_rows),sum(bool(r['violations']) for r in same_rows),flush=True)
        chosen,rows,log=self.search(order,code)
        ratio=self.fraction(rows)
        np.savez_compressed(RUN/'accepted.npz',native_row_id=self.native_ids.cpu().numpy(),selected=chosen.cpu().numpy())
        write(RUN/'result.json',dict(status='PASS',training_updates=0,frames=49,baseline=self.latest['best'],
                guarded=ratio,selected=int(chosen.sum()),rows=rows,all_cut=all_cut,same_rows=same_rows,
                all_cut_ratio=self.fraction(all_cut),same_rows_ratio=self.fraction(same_rows),search=log,
                max_per_frame_reveal_loss=SLACK,
                depth_policy='Original native Gaussian center sorting and depth auxiliary retained. Camera ray only scales alpha.'))
        print('GUARDED',ratio,int(chosen.sum()),flush=True)
        dest=RUN/'render'
        dest.mkdir()
        for raw in RAWS:
            out=self.compose_clip(raw,self.latest_rows[raw],code[raw],chosen)
            for label,pkg in [('previous',self.original_pair[raw]),('new',out)]:
                save(dest/f'{raw:03d}_{label}.png',pkg['rgb'])
                save(dest/f'{raw:03d}_{label}_pvd.png',pkg['extra'][:1].expand(3,-1,-1))
                e=self.evidence[raw]
                deficit=(e['lower_alpha']-pkg['extra'][1]).clamp_min(0)*e['mask']
                save(dest/f'{raw:03d}_{label}_blocked.png',torch.stack((deficit,.20*e['mask'],.08*e['mask'])))
        assert self.exp.unchanged()
        write(RUN/'receipt.json',dict(status='COMPLETE',training_updates=0,shader='private forward-only',
                                    native_renderer_unchanged=True,frames=49))


if __name__=='__main__':
    torch.set_num_threads(1)
    with torch.no_grad():
        Trial().run()
