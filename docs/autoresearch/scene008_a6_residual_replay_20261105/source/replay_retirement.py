"""Replay shared row retirement against the accepted a6 fusion and frozen protection domains."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import json
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.ndimage import binary_dilation

RUN=Path(__file__).resolve().parent.parent
PREVIOUS=RUN.parent/'scene008_a6_local_fusion_20261105'
sys.path.insert(0,str(PREVIOUS/'source'))
from fuse_footprint import FootprintTrial
from fuse_a6 import RAWS, PILOTS, PARAMS, put_features, metric, save, write

OTHER_RGB=.005
OTHER_ALPHA=.01
MIN_SCORE=.05


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


class Trial(FootprintTrial):
    def start(self):
        self.clipped_counts={}
        self.read_evidence()
        self.previous=read(PREVIOUS/'delivery/summary.json')
        assert self.previous['chosen']=='support_split'
        self.protection={}
        self.native_cache={}
        self.split_cache={}
        self.zero=self.black.new_zeros(len(self.native_ids),dtype=torch.bool)
        with np.load(PREVIOUS/'lower_support_ledger.npz',allow_pickle=False) as data:
            assert np.array_equal(data['native_row_id'],self.native_ids.cpu().numpy())
            self.inherited=torch.as_tensor(data['gate'],device='cuda')
            self.split={key:torch.as_tensor(data[field],device='cuda') for key,field in
                        [('use','split'),('normal','normal'),('plane','plane_m')]}
        replay=[]
        for raw in RAWS:
            source=self.native(raw)
            native=self.composite(raw,'native',source)
            held=dict(source,opacities=source['opacities']*self.inherited[:,None])
            self.split_cache[raw]=self.clipped(raw,held,self.split)
            d=self.raw(raw)
            ev=self.evidence[raw]
            lower=self.composite(raw,'pvd_lower')
            full=self.composite(raw,'pvd')
            action=(lower['alpha'][0]>=PARAMS['target_alpha'])
            action&=(full['extra'][1]>=PARAMS['target_lower_fraction']*full['alpha'][0])
            action&=~d['protect']&~d['depth_valid']
            assert (ev['mask']&~action).sum().item()==0
            collar=binary_dilation(action.cpu().numpy(),iterations=1)
            outside=~torch.as_tensor(collar,device='cuda') & (native['alpha'][0]>.05)
            self.protection[raw]=dict(untouched=native, outside=outside, action=action)
            prior=self.draw_cached(raw,self.zero)
            local=self.measured_metrics(raw,prior,native)
            expected=self.previous['rows'][raw-89]['variants']['support_split']
            assert all(local[key]['count']==expected[key]['count'] and
                       abs(local[key]['sum']-expected[key]['sum'])<.001 for key in expected), raw
            replay.append(raw)
        write(RUN/'replay.json',dict(frames=replay,previous_chosen='support_split',maximum_sum_tolerance=.001,
                                    invariant='same a6 native temporal state, inherited gate and split, identical fixed lower domain'))

    def draw_cached(self,raw,remove):
        kw=self.split_cache[raw]
        if bool(remove.any()):
            kw=dict(kw,mask=kw['mask']&~remove)
        return self.composite(raw,'lower_only',kw)

    def estimate(self):
        target=self.black.new_zeros(len(self.native_ids))
        known=target.clone()
        measured=target.clone()
        outside=target.clone()
        for raw in RAWS:
            n=self.split_cache[raw]
            e=self.evidence[raw]
            d=self.raw(raw)
            for b in e['cutoff_bins'][e['mask']].unique().tolist():
                cut=b*PARAMS['depth_bin_m']*self.scale
                prefix=dict(n,mask=n['mask']&(n['features'][:,0]<cut))
                target+=self.row_weights(raw,prefix,e['mask']&(e['cutoff_bins']==b))
            known+=self.row_weights(raw,n,d['obs'])
            measured+=self.row_weights(raw,n,d['depth_valid'])
            outside+=self.row_weights(raw,n,self.protection[raw]['outside'])
        candidates=(target>=MIN_SCORE)&~self.zero&(self.inherited>0)
        # Normalize measured LiDAR support to the RGB-domain exposure.
        rank=target/(known+4*measured+outside+.01)
        ids=torch.where(candidates)[0]
        order=ids[torch.argsort(rank[ids],descending=True,stable=True)]
        info=dict(candidates=len(order),remaining_prefix_mass=float(target.sum()),
                  candidates_prefix_mass=float(target[order].sum()),
                  measured_overlap_prefix_mass=float(target[order][measured[order]>.10].sum()),
                  outside_overlap_prefix_mass=float(target[order][outside[order]>.10].sum()),
                  ordering='fixed lower prefix alpha*T / (known alpha*T + 4*LiDAR alpha*T + non-target alpha*T + 0.01)',
                  minimum_prefix_mass=MIN_SCORE)
        write(RUN/'candidate_summary.json',info)
        np.savez_compressed(RUN/'candidates.npz',native_row_id=self.native_ids.cpu().numpy(),order=order.cpu().numpy(),
                            target=target.cpu().numpy(),known=known.cpu().numpy(),
                            measured=measured.cpu().numpy(),outside=outside.cpu().numpy())
        print('CANDIDATES',info,flush=True)
        return order

    @staticmethod
    def mean(m):
        return m['sum']/m['count'] if m['count'] else 0.

    def assess(self,raw,out,old):
        protect=self.protection[raw]
        native=protect['untouched']
        m=self.measured_metrics(raw,out,native)
        native_m=self.previous['rows'][raw-89]['variants']['native']
        issues=[]
        if self.mean(m['known_rgb'])-self.mean(native_m['known_rgb'])>PARAMS['rgb_guard']:
            issues.append('known_rgb')
        if self.mean(m['known_rgb_change'])>PARAMS['rgb_guard']:
            issues.append('known_rgb_change')
        if self.mean(m['measured_depth'])-self.mean(native_m['measured_depth'])>PARAMS['depth_guard_m']:
            issues.append('measured_depth')
        if self.mean(m['known_alpha'])<PARAMS['evidence_guard_fraction']*self.mean(native_m['known_alpha']):
            issues.append('known_alpha')
        mask=protect['outside']
        m['outside_rgb_delta']=metric((out['rgb']-old['rgb']).abs().mean(0),mask)
        m['outside_alpha_delta']=metric((out['alpha'][0]-old['alpha'][0]).abs(),mask)
        if self.mean(m['outside_rgb_delta'])>OTHER_RGB:
            issues.append('outside_rgb')
        if self.mean(m['outside_alpha_delta'])>OTHER_ALPHA:
            issues.append('outside_alpha')
        return m,issues

    def audit(self,remove,guarded=True,early=True):
        frames=(*PILOTS,*[raw for raw in RAWS if raw not in PILOTS])
        rows=[]
        rejects=[]
        for raw in frames:
            old=self.before[raw]
            new=self.draw_cached(raw,remove)
            m,issues=self.assess(raw,new,old)
            fail=issues if guarded else [key for key in issues if not key.startswith('outside_')]
            rows.append(dict(raw=raw,metrics=m,violations=issues))
            if fail:
                rejects.append(dict(raw=raw,violations=fail))
                if early:
                    break
        return sorted(rows,key=lambda r:r['raw']),rejects

    @staticmethod
    def fraction(rows):
        total=sum(row['metrics']['lower_pvd']['sum'] for row in rows)
        denominator=sum(row['metrics']['isolated_lower_alpha']['sum'] for row in rows)
        return total/max(denominator,1e-8)

    def search(self,order,guarded):
        lo,hi=0,len(order)+1
        log=[]
        while hi-lo>1:
            mid=(hi+lo)//2
            remove=self.zero.clone()
            remove[order[:mid]]=True
            rows,reject=self.audit(remove,guarded)
            log.append(dict(prefix=mid,pass_gate=not reject,frames_read=len(rows),
                            failures=reject,ratio=self.fraction(rows) if len(rows)==49 else None))
            if reject:
                hi=mid
            else:
                lo=mid
        chosen=self.zero.clone()
        chosen[order[:lo]]=True
        rows,rejects=self.audit(chosen,guarded,early=False)
        assert not rejects
        # Isolated rejection can block a contiguous ranking. For the full
        # protection case, check a small fixed set of later groups separately.
        if guarded:
            blocks=torch.tensor_split(order[lo:],12)
            for i,block in enumerate(blocks):
                if not len(block):
                    continue
                take=chosen.clone()
                take[block]=True
                trial,bad=self.audit(take,guarded)
                log.append(dict(extra_group=i,group_size=len(block),pass_gate=not bad,
                                frames_read=len(trial),failures=bad,
                                ratio=self.fraction(trial) if len(trial)==49 else None))
                if not bad:
                    chosen=take
                    rows=trial
        return chosen,rows,log

    def run(self):
        self.start()
        self.before={raw:self.draw_cached(raw,self.zero) for raw in RAWS}
        order=self.estimate()
        all_rows=self.zero.clone()
        all_rows[order]=True
        before,bad=self.audit(self.zero,early=False)
        assert not bad
        ceiling,violations=self.audit(all_rows,early=False)
        only,measured_only,log1=self.search(order,False)
        final,guarded,log2=self.search(order,True)
        print('RESULTS',dict(previous=self.fraction(before),full_mask=self.fraction(ceiling),
                             measured_only=self.fraction(measured_only),guarded=self.fraction(guarded),
                             retired=int(final.sum())),flush=True)
        write(RUN/'result.json',dict(status='PASS',frames=len(guarded),training_updates=0,
            protection=dict(rgb_guard=PARAMS['rgb_guard'],depth_guard_m=PARAMS['depth_guard_m'],
                            outside_rgb=OTHER_RGB,outside_alpha=OTHER_ALPHA),
            baselines=dict(previous=before,all_candidate_retirement=ceiling,measured_only=measured_only),
            result=guarded,controls=dict(all_retired=int(all_rows.sum()),all_violations=violations,
                                        measured_only_retired=int(only.sum()),guarded_retired=int(final.sum())),
            reveal=dict(previous=self.fraction(before),all_mask=self.fraction(ceiling),
                        measured_only=self.fraction(measured_only),guarded=self.fraction(guarded)),
            search=dict(measured_only=log1,guarded=log2),
            validity='Same-scene strong replay. Prefix search + later fixed groups is bounded and may miss feasible subsets. Full removal is a destructive diagnostic.',
            render_policy='Selection uses fixed identities. Added opacity zero persists across views and raw time. PVD and inherited split values remain fixed.'))
        np.savez_compressed(RUN/'shared_retirement.npz',native_row_id=self.native_ids.cpu().numpy(),
                            extra_retired=final.cpu().numpy(),measured_only=only.cpu().numpy())
        output=RUN/'render'
        output.mkdir()
        for raw in RAWS:
            old=self.before[raw]
            new=self.draw_cached(raw,final)
            for label,pkg in [('previous',old),('guarded',new)]:
                save(output/f'{raw:03d}_{label}.png',pkg['rgb'])
                save(output/f'{raw:03d}_{label}_pvd.png',pkg['extra'][0:1].expand(3,-1,-1))
                e=self.evidence[raw]
                residual=(e['lower_alpha']-pkg['extra'][1]).clamp_min(0)*e['mask']
                save(output/f'{raw:03d}_{label}_blocked.png',torch.stack((residual,.20*e['mask'],.08*e['mask'])))
            if raw in PILOTS:
                for label,mask in [('measured_only',only),('all_candidates',all_rows)]:
                    pkg=self.draw_cached(raw,mask)
                    save(output/f'{raw:03d}_{label}.png',pkg['rgb'])
                    save(output/f'{raw:03d}_{label}_pvd.png',pkg['extra'][0:1].expand(3,-1,-1))
        assert self.exp.unchanged()
        write(RUN/'receipt.json',dict(status='COMPLETE',training_updates=0,guarded_frames=49,
                                     scene_unchanged=True,checkpoint_replay_verified=True))


if __name__=='__main__':
    torch.set_num_threads(1)
    with torch.no_grad():
        Trial().run()
