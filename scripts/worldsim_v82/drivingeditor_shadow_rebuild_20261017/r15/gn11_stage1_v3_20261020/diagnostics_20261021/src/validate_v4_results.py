"""检查实际V4导出是否完整、可解码且与30K渲染核验一致。"""
import argparse,base64,json,zlib
from pathlib import Path
import numpy as np
from PIL import Image

def read(path):
    value=json.loads(path.read_text(encoding='utf-8').split(']=',1)[1].rstrip(';'))
    if 'data' not in value:return value
    blob=zlib.decompress(base64.b64decode(value['data']))
    return {k:np.frombuffer(blob,dtype=r['dtype'],count=r['bytes']//np.dtype(r['dtype']).itemsize,offset=r['offset']).reshape(r['shape']) for k,r in value['arrays'].items()}

def validate(root,arm,ready=None):
    key='v4_'+arm;ready=ready or json.loads((root/f'data/{key}/ready.json').read_text());state=read(root/f'data/{key}/road_state.js');cloud=read(root/f'data/{key}/cloud.js');geometry=read(root/f'data/{key}/geometry_index.js')
    assert ready['status']=='V4_DIAGNOSTICS_READY' and len(ready['checks'])==49
    assert max(c['native_png_replay_max_abs'] for c in ready['checks'])<=.5/255+2e-5
    assert max(c['same_raster_group_replay_max_abs'] for c in ready['checks'])<=2e-5
    assert len(state['unit_ids'])==14034 and state['output_mask'].sum()==ready['road']['output']
    assert (cloud['static_role']==1).sum()==state['output_mask'].sum()
    assert np.array_equal(np.sort(cloud['static_road_id'][cloud['static_role']==1]),state['unit_ids'][state['output_mask'].astype(bool)])
    assert np.isfinite(state['xyz']).all() and np.isfinite(state['opacity_value']).all()
    assert ready['geometry_fixed_verified'] and ready['training_updates']==0
    for name,b in geometry['branches'].items():
        assert sum(chunk['count'] for chunk in b['chunks'])==b['ordinary_total']
        for chunk in b['chunks']:
            a=read(root/chunk['path']);assert len(a['row'])==chunk['count'];assert np.isfinite(a['xyz']).all()
    for raw in range(89,138):
        maps=read(root/f'data/{key}/{raw}.js');surfaces=read(root/f'data/inspection/{key}_{raw}.js');units=read(root/f'data/{key}/units_{raw}.js')
        assert all(a.shape==(320,480) for a in maps.values())
        assert np.isfinite(surfaces['ordinary_depth_m']).all() and np.isfinite(surfaces['road_depth_m']).all()
        assert len(units['in_view'])==26088
        assert (maps['road_bg'].astype(int)<=maps['static_alpha'].astype(int)+1).all()
        assert (maps['road_joint'].astype(int)<=maps['joint_alpha'].astype(int)+1).all()
        # 相同光栅的道路组贡献不会大于道路独立alpha；1/65535量化误差单列。
        assert (maps['road_bg'].astype(int)<=maps['road_alpha'].astype(int)+2).all()
        for ch in ('joint','deleted_five','dynamic_raw','road_raw','road_surface_color'):
            with Image.open(root/f'assets/{key}/{ch}/{raw:03d}.webp') as im:assert im.size==(480,320)
        members=read(root/f'data/geometry/{key}/members_{raw}.js')
        for name in ('static','dynamic'):assert len(members[name+'_projection'])==geometry['branches'][name]['ordinary_total']
    return dict(status='PASSED',arm=key,frames=49,images=245,road_created=14034,road_output=int(state['output_mask'].sum()),
                native_png_replay_max_abs=max(c['native_png_replay_max_abs'] for c in ready['checks']),training_updates=0,
                checks=['49 full-resolution native CUDA images/maps','same-raster road contribution bounds','fixed geometry and output mask','all sampling chunks/49 vehicle regions','independent ordinary/road depth markers'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--arm',default='R',choices=['R','RP']);a=p.parse_args();print(json.dumps(validate(a.root,a.arm),indent=2))
