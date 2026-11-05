"""Freeze every qualified BUILD surface group before fitting."""
import numpy as np
from coupled import RUN,source_asset,jsonable,write
from surface_groups import build_group

asset=source_asset()
first=build_group(asset)
count=int(first['qualified_component_count'])
records=[]
assert count>0,first
for rank in range(count):
    r=build_group(asset,selection_rank=rank)
    assert r['status']=='PASS' and r['qualified_component_count']==count
    ledger=r.pop('ledger')
    for key in ('center','basis','bounds'):
        ledger[key]=np.asarray(r[key],np.float32)
    assert min(ledger['rgb_frame_count'])>=2
    np.savez_compressed(RUN/f'group_ledger_{rank}.npz',**ledger)
    write(RUN/f'group_audit_{rank}.json',jsonable(r))
    records.append(dict(rank=rank,proxies=len(ledger['xyz']),component_size=r['audit']['selected_component_anchor_count']))
write(RUN/'groups.json',dict(status='PASS',qualified_count=count,all_qualified_groups=records))
print(records,flush=True)
