import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v41_20261023/arms/B/run30k')
ap=argparse.ArgumentParser();ap.add_argument('--record',action='store_true');a=ap.parse_args()
current={str(B/f'{prefix}30000.pth'):dict(bytes=(B/f'{prefix}30000.pth').stat().st_size,mtime_ns=(B/f'{prefix}30000.pth').stat().st_mtime_ns)
    for prefix in ['chkpnt','dynamic_chkpnt','env_light_chkpnt','training_state']}
p=ROOT/'parent_before_formal.json'
if a.record:
    assert not p.exists();p.write_text(json.dumps(current,indent=2)+'\n')
else:
    assert current==json.loads(p.read_text());(ROOT/'parent_unchanged.json').write_text(json.dumps(dict(original_B_four_components_unchanged=True,files=current),indent=2)+'\n')
print(json.dumps(dict(status='RECORDED_PARENT' if a.record else 'PASS_PARENT_UNCHANGED')))
