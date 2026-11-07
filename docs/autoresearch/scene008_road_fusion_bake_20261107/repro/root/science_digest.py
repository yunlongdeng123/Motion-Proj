from pathlib import Path
import json
RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_road_fusion_bake_20261107_r1')
def read(p):return json.loads(Path(p).read_text())
AB=read(RUN/'red_review/full49_review_summary_v1.json')
C=read(RUN/'red_review/full49_edgeblend_summary_v1.json')
primary=[]
for d in (AB,C):
    for r in d['raw128_primary']:
        primary.append(dict(state=r['state'],method=r['method'],
            road_MAE_baseline=r['original_road']['baseline']['mean'],road_MAE=r['original_road']['candidate']['mean'],
            near_baseline=r['render_near']['baseline']['near_x']['equal_bin_abs_residual_jump_mean'],
            near=r['render_near']['candidate']['near_x']['equal_bin_abs_residual_jump_mean'],
            counts=r['target_composition']))
seams=[]
for d in (AB,C):
    for r in d['all49_world_seam']:
        seams.append({k:v for k,v in r.items() if k not in ('worst_bins','delta_distribution')})
native=[]
for name in ('full49','edgeblend49'):
    d=read(RUN/f'results/{name}/receipt.json')
    native+=d['pooled']
negative=[]
for r in C['all49_world_seam']:
    negative.append({k:r[k] for k in ('state','edge','worst_bins')})
result=dict(raw128_primary=primary,all49seams=seams,native_pooled=native,C_worst_bins=negative,
    frozen_checks_AB=AB['frozen_checks'],frozen_checks_C=C['frozen_checks'],fixed_red_C=C['fixed_red_rays'])
out=RUN/'root/science_digest_v1.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(primary=primary,seams=seams,C_frozen=C['frozen_checks']),ensure_ascii=False))
