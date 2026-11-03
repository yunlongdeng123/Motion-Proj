import torch,json
from pathlib import Path
from local_runtime import Experiment,RUN
from editor_common import save
def main():
    e=Experiment();old=RUN.with_name('shadow_quality_20261103_r7')
    ck=torch.load(old/'formal/checkpoint001000.pt',map_location='cuda',weights_only=False)
    assert torch.equal(ck['repair']['ids'],e.repair.ids)
    with torch.no_grad():
        e.repair.gate_raw.copy_(ck['repair']['gate_raw'])
        out=RUN/'detail_probe';out.mkdir(exist_ok=False)
        for raw in [101,105,113]:save(out/('%03d.png'%raw),e.render(raw)['rgb'])
        e.evaluate(out,[101,105,113],'metrics')
    (out/'receipt.json').write_text(json.dumps(dict(updates=0,same_R7_gate=True,parent_unchanged=e.unchanged())))
if __name__=='__main__':main()
