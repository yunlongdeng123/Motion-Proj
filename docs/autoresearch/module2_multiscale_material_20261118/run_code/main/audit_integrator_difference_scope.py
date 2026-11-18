from pathlib import Path
import numpy as np,json
P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
with np.load(P/'trellis2/midpoint_full_r1/row_queries.npz') as z:d={k:z[k].copy() for k in ['filtered_rgb_A0','baseline_A0_GH15','common_valid_mass','copied_same_A0','active_world_rows']}
with np.load(P/'method_scope_r1/scope.npz') as z:s={k:z[k].copy() for k in z.files}
assert np.array_equal(d['active_world_rows'],s['active_world_rows'])
e=np.abs(d['filtered_rgb_A0']-d['baseline_A0_GH15']).max(1)
groups={'all':np.ones(len(e),bool),'affected':~d['copied_same_A0'],'write':s['eligible'],
 'write_affected':s['eligible']&~d['copied_same_A0'],'write_S':s['eligible']&(s['actual_S_mass']>0),
 'write_affected_mass_ge_point9':s['eligible']&~d['copied_same_A0']&(d['common_valid_mass']>=.9)}
r={}
for name,m in groups.items():
    r[name]=dict(rows=int(m.sum()),p99=float(np.quantile(e[m],.99)),max=float(e[m].max()),above4code=int((e[m]>4/255).sum()),above1code=int((e[m]>1/255).sum()))
bad=np.flatnonzero(s['eligible']&(e>4/255))
out=dict(schema='integrator_A0_difference_scope_audit.v1',GPU_executed=False,optimizer_updates=0,groups=r,
 outlier_world_rows=d['active_world_rows'][bad].tolist(),outlier_max_error=e[bad].tolist(),
 outlier_S_mass=s['actual_S_mass'][bad].tolist(),outlier_kernel_mass=d['common_valid_mass'][bad].tolist(),
 scope='new denser-integrator sensitivity, not hidden-road GT or a universal bound; original representative qualification retained')
(P/'qa/integrator_difference_scope.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
