from pathlib import Path
from datetime import datetime,timezone,timedelta
import json

C=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_direct_teacher_20261116_r1')
target=C/'protocol_locked.json'
if target.exists():raise FileExistsError('锁定协议不可覆盖')
evidence=['qa/supervision_r3_independent/acceptance.json',
    'qa/lifecycle_r4_independent/snapshot_and_pairing.json','qa/lifecycle_r4_independent/render_binding.json']
for name in evidence:
    value=json.loads((C/name).read_text())
    if value.get('status') not in ('pass','passed') and value.get('passed') is not True:
        raise RuntimeError('独立门未通过: '+name)
    if isinstance(value.get('checks'),dict) and not all(value['checks'].values()):
        raise RuntimeError('检查项未全过: '+name)
protocol=json.loads((C/'protocol_draft.json').read_text())
trainer=json.loads((C/'trainer_contract_draft.json').read_text())
trainer['status']='locked_before_formal'
trainer['opacity_constraint']='effective native loader initial probability hard lower; upper 1-1e-5; frozen raw values preserved'
trainer['C_pairing']='photo_C common F0-valid fixed mask; coverage_C full geometry-qualified Cteacher independent of F0 color; both arms share identical masks and q'
trainer['loss_weights']['C_edge_relative']=.1
trainer['loss_weights']['edge']=.1
trainer['stage_losses']['color']='L_A + 0.5 L_C + 0.1 edge_A + 0.01 edge_C + outside_RGB + 4 retained_RGB + 0.001 prior'
protocol.update(status='locked_before_formal',formal_authorized=True,
    locked_at_Asia_Singapore=datetime.now(timezone(timedelta(hours=8))).isoformat(),
    trainer_contract=trainer,hyperparameters_status='locked; actual source and supervision bindings recorded in each frozen_contract.json',
    independent_phase0_evidence=[str(C/name) for name in evidence],
    engineering_updates_before_formal=9,matched_control='same initialized S2 and scopes; only selected_C_srgb target differs',
    final_coverage_C='full legal Cteacher geometry including F0-invalid color; common photo C masks unchanged',
    retained_guard='actual installed native CUDA maximum-sigma radii, 16x16 tile-expanded bounds plus rounding margin, five-frame full K union; 38982 active rows guarded',
    numerical_replay='full saved state restores exactly; next update nonbitwise opacity logits max7.45e-9 and Adam max3.55e-15, below preregistered1e-6; no bitwise claim')
target.write_text(json.dumps(protocol,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(target)
