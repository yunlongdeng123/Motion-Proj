"""Parent road1K parity independent of the saved fine road colour field."""
import json
import torch
from full_pvd import RUN
from scene_runtime import Runtime

@torch.no_grad()
def main():
    rt=Runtime(fine_road=False);rows=[]
    for raw in (105,113,130):
        for keep in (True,False):
            old=rt.exp.render(raw,keep=keep)
            new=rt.render_scene(raw,baseline=True,delete=not keep)
            error=float((new['rgb']-old['rgb']).abs().max())
            assert error<5e-6,(raw,keep,error)
            rows.append(dict(raw=raw,keep_a37=keep,rgb_maxabs=error))
    assert rt.modelunchanged()
    (RUN/'baseline_parity.json').write_text(json.dumps(dict(status='PASS',fine_road=False,
       target='Original road1K Experiment.render, no saved fine color field',rows=rows),indent=2))
    print(json.dumps(rows),flush=True)

if __name__=='__main__':main()
