from pathlib import Path
import json

P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
assert json.loads((P/'gpu_lease.json').read_text())['active'] is False
c=dict(schema='module2_native_packet_stage0.v1',task_id='MODULE2-MULTISCALE-MATERIAL-20261118',
 input_asset='/root/autodl-tmp/drivingeditor_repro/campaign/road_activation_20261118_r1/B32_beam95_native_r1/asset_guarded',
 permitted='current fixed packet five-frame K/near, actual A/C/S contribution, alpha and color-leaf gradients; no learned field or coverage updates',
 training_source_frames=[86,110],K_frames=[86,110,134,179,198],S_photo=3131,S_full=3187,low_S=56,K_pixel_frames=77266,
 source='frozen V4 A/C/S with supported S refreshed by repaired native alpha; source/material filtering qualification remains independent',
 field_gradient_status='not proved by color-leaf probe; later actual shared-grid/native gradient required',
 optimizer_updates=0,native_call_limit=20,
 prohibited=['any alpha/scale/geometry/active updates','GH7 formal field fitting','new SD or TRELLIS call'],
 GPU_keep_on=True,production_promoted=False)
(P/'stage0_native_contract.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n')
(P/'gpu_lease.json').write_text(json.dumps(dict(gpu_authorized=True,active=True,owner='road_asset',keep_GPU_on=True,
 scope='stage0 packet/source qualification only, max20native0updates'),indent=2)+'\n')
print('stage0 exclusive lease: road_asset, <=20 native calls, 0 updates')
