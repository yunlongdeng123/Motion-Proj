"""Replace compatible prior approximations with exact ray retention on the same rows."""
import json
from pathlib import Path
import sys
import numpy as np
import torch

RUN=Path(__file__).resolve().parent.parent
RAY=RUN.parent/'scene008_a6_ray_support_20261105'
sys.path.insert(0,str(RAY/'source'))
import ray_support
from ray_support import Trial as Ray,RAWS,PILOTS,read,write,save


class Trial(Ray):
    def parents(self):
        ray_support.RUN=RUN
        super().parents()
        self.baseline=self.latest['best']
        self.code={}
        n=len(self.native_ids)
        identity=torch.arange(n,device='cuda')
        choices=self.base_options['selected']
        for raw in RAWS:
            self.code[raw]=self.camera_code(raw,self.current[raw],identity,choices,
                                          self.base_options['normal'],self.base_options['cut'])

    def switch_rows(self,raw,selected):
        parent=self.latest_rows[raw]
        previous=self.current[raw]
        mask=parent['mask'].clone()
        n=len(self.native_ids)
        mask[:n]=torch.where(selected,previous['mask'],mask[:n])
        mask[n:]&=~selected[self.lineage[raw][n:]]
        return dict(parent,mask=mask)

    def readout(self,raw,selected):
        return self.compose_clip(raw,self.switch_rows(raw,selected),self.code[raw],selected)

    def check_switch(self,selection,early=True):
        metrics=[]
        for raw in (*PILOTS,*[r for r in RAWS if r not in PILOTS]):
            row=self.measure(raw,self.readout(raw,selection))
            metrics.append(row)
            if early and row['violations']:
                break
        return sorted(metrics,key=lambda r:r['raw'])

    def run(self):
        self.parents()
        all_options=self.base_options['selected']
        full=self.check_switch(all_options,early=False)
        source=read(RAY/'result.json')
        for new,old in zip(full,source['same_rows']):
            assert new['raw']==old['raw']
            assert all(abs(v['sum']-old['metrics'][k]['sum'])<.002 for k,v in new['metrics'].items()),new['raw']
        order=self.base_options['order']
        order=order[all_options[order]]
        chosen=self.zero.clone()
        score=self.baseline
        trace=[]
        for group,ids in enumerate(torch.tensor_split(order,12)):
            queue=[(0,ids)]
            while queue:
                depth,batch=queue.pop(0)
                if not len(batch):
                    continue
                proposed=chosen.clone()
                proposed[batch]=True
                rows=self.check_switch(proposed)
                failures=[dict(raw=r['raw'],metrics=r['violations']) for r in rows if r['violations']]
                value=self.fraction(rows) if len(rows)==49 else None
                passed=not failures and len(rows)==49 and value>=score-1e-7
                trace.append(dict(group=group,depth=depth,n=len(batch),passes=passed,ratio=value,failures=failures))
                if passed:
                    chosen,score=proposed,value
                elif depth<3:
                    queue.extend((depth+1,q) for q in torch.tensor_split(batch,2))
        final=self.check_switch(chosen,early=False)
        assert len(final)==49 and not any(r['violations'] for r in final)
        ratio=self.fraction(final)
        out=RUN/'render'
        out.mkdir()
        for raw in RAWS:
            new=self.readout(raw,chosen)
            for label,pkg in [('previous',self.original_pair[raw]),('new',new)]:
                save(out/f'{raw:03d}_{label}.png',pkg['rgb'])
                save(out/f'{raw:03d}_{label}_pvd.png',pkg['extra'][:1].expand(3,-1,-1))
                e=self.evidence[raw]
                gap=(e['lower_alpha']-pkg['extra'][1]).clamp_min(0)*e['mask']
                save(out/f'{raw:03d}_{label}_blocked.png',torch.stack((gap,.20*e['mask'],.08*e['mask'])))
        assert self.exp.unchanged()
        np.savez_compressed(RUN/'switch.npz',native_row_id=self.native_ids.cpu().numpy(),
                            selected=chosen.cpu().numpy())
        write(RUN/'result.json',dict(status='PASS',frames=49,training_updates=0,previous=self.baseline,
                 ratio=ratio,switched=int(chosen.sum()),rows=final,all_switched_ratio=self.fraction(full),
                 all_switched_failures=sum(bool(r['violations']) for r in full),trace=trace,
                 policy='Replace a subset of prior one-Gaussian re-expressions by ray retention on their same earlier parent and plane. Shared across time and edit.'))
        print('SWITCH',ratio,int(chosen.sum()),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    with torch.no_grad():
        Trial().run()
