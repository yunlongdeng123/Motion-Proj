from pathlib import Path
import json,time

P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
p=P/'matched_rgb_control_contract.json'
if p.exists():raise FileExistsError('保留登记')
d=dict(schema='module2_matched_independent_RGB_control.v1',task_id='MODULE2-MULTISCALE-MATERIAL-20261118',
 registered_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
 reason='historical I0 used oldphoto-S2222; a fair current material comparison needs same input, current3131 teacher mask, starting shader, row eligibility and800-step sampler',
 initial='zero-delta filteredA0 derived from cpu_exact_r1 authoritative world field; same readonly nonqualified legacy outputs',
 scope='same final method_scope_r1,84266 rows, original-known C-only protection; same geometry/alpha/scales/active/static/vehicles/cameras',
 parameterization='independent ci = clamp(c0_i + delta_i,0,1); train delta_i only, initial0, constrain delta to[-1,1]',
 loss=dict(photo='same native premult A+0.5C+S, independent domain normalization, q=1 inside frozen validity masks',
   independent_effective_delta_squared=.001,
   comparison='regularizers differ by method: current shared fields add registered spatial/H constraints; not a pure parameter-count-only ablation'),
 optimizer=dict(name='Adam',lr=.01,betas=[.9,.999],eps=1e-8,weight_decay=0.,seed=101),
 formal_updates=800,sampler='same800 raw86/110 trace as shared matched methods',
 engineer_updates_limit=8,engineer='actual gradient/save/evaluate/full-state restore/next update/native K and public reload; no warm-start formal',
 GPU_launch='root grants exclusive lease after shared-field600s smoke; no concurrent native task',
 no_automatic_extension=True,
 accounting='800 additional independentRGB formal updates; shared-method physical1400 and both800 effective budgets unchanged; all engineering separate',
 GPU_keep_on=True,production_promoted=False,human_verdict=None)
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
q=dict(schema='module2_material_quality_pre_registered.v1',task_id=d['task_id'],
 baseline='same filteredA0 zero-update and same matched independentRGB800',
 stripe=dict(frames=[134,179,198],masks='same previously frozen diagnostic stencils; never select after current outputs',
   score='same historical vertical second-difference proxy, report horizontal counterpart and native images',
   excess='score_method - score_filteredA0',
   criterion='if matchedI0 excess>0, ABC excess <=0.5*matchedI0 excess in each diagnostic frame; nonpositive I0 excess=N/A, not auto win'),
 fit=dict(ABC_minus_matchedI0_MAE_per_view_domain_max=1/255,domains=['A','C','S'],frames=[86,110],
   real_visible_ABC_minus_filteredA0_MAE_max=1/255,
   labels='A is visible RGB; C/S frozen2D pseudo, no hidden-road GT'),
 matched_low='report ABC-L800 differences, high effective energy/DC/leakage/phase drift; A200/B500 are stage diagnostics not equal-compute H controls',
 marking='only frozen real-observed proxy where available; no ideal12cm or predicted dash phase relabeled as real structural GT',
 coverage='fixed alpha exact, fullS3187/old2222/new909/low56 and D38416 kept',
 keep='all5 frames77266 joint floatRGB exact0',
 no_automatic_human_verdict=True,no_default_promotion=True,
 negative_capacity_certificate=False,TRELLIS='numerical material sampling qualification separate; failed integration is not model quality evidence')
(P/'quality_contract.json').write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n')
print('Matched independentRGB800 and quality gates registered before formal')
