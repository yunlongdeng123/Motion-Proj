from pathlib import Path
import json
p=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1/gpu_20261119/source_filtered_training/prospective_contract.json')
assert not(p.parent/'formal_r1/started.json').exists()
d=json.loads(p.read_text())
d['regularization_semantics']='Same objective on effective rendered field: original regularizer definitions, coefficients and fixed denominators now evaluate filtered mH*H because filtered H is the new material parameterization; no extra prior loss or unfiltered-H penalty.'
d['filter_radius_uv']=[2,1]
d['radius_convention']='scipy ndimage truncate3 radius=int(3*sigma+0.5), v convolution then u, replicate edge'
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
