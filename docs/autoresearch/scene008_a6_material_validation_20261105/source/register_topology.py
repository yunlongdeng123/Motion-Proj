"""Register the ancestry topology control BEFORE any pseudo material fit."""
import json,hashlib,shutil
from pathlib import Path
RUN=Path(__file__).resolve().parent.parent
assert not (RUN/'smoke').exists() and not (RUN/'formal').exists()
path=RUN/'cases/contract.json';prep=RUN/'cases/preparation.json'
shutil.copyfile(path,path.with_name('contract_before_topology_control.json'))
shutil.copyfile(prep,prep.with_name('preparation_before_topology_control.json'))
c=json.loads(path.read_text());c['methods']=['median','child_nearest','child_graph','parent_nearest','parent_graph']
c['topology_control']=dict(registration='Added before any pseudo RGB fit or outcome',
    motivation='Geometry-only child8NN graph has1659 components; subdivision ancestry may retain mother-surface adjacency.',
    child='Original geometry-only mutual8NN graph unchanged',
    parent='Same graph formula on2048 exact source mothers; lift to existing child0 centres and verified within-mother tangent star edges',
    seed_colours='Identical nonzero TRAIN fine-point values; no parent RGB averaging or refitting',
    physical_scale='Spacing, edge radius and plane tolerance evaluated on each discretization; not identical metre thresholds or a topology guarantee',
    forbidden='No extra primitives, no geometry/opacity change, no colour-dependent graph, no case-specific best-method switching',
    formal_colour_steps_unchanged=6000)
path.write_text(json.dumps(c,indent=2)+'\n')
p=json.loads(prep.read_text());p['contract_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();prep.write_text(json.dumps(p,indent=2)+'\n')
print(p['contract_sha256'])
