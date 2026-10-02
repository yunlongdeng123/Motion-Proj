"""实际来源匹配、地图分派、旧Boston几何及零地面返回的工程回归。"""
from pathlib import Path
import sys,json,copy
sys.path.insert(0,str(Path(__file__).parents[1]))
from geometry_factory import Geometry,read,dump
from shapely.geometry import Polygon
from shapely.ops import unary_union
O=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929/r37')


def main():
    root=O/'factory';g=Geometry(root,include_parking=True)
    assert set(g.roads)=={c['location'] for c in g.sources.values()}
    assert all(g.road_for(sid) is g.roads[c['location']] for sid,c in g.sources.items())
    assert len(g.map_notes['singapore-hollandvillage']['explicit_null_polygon_records'])==2
    assert not g.map_notes['boston-seaport']['explicit_null_polygon_records']
    assert not g.map_notes['singapore-onenorth']['explicit_null_polygon_records']
    try:g.road
    except ValueError:pass
    else:raise AssertionError('多地图不能静默回退Boston')
    raw=read(root/'maps/expansion/boston-seaport.json');nodes={n['token']:(n['x'],n['y']) for n in raw['node']}
    polys={p['token']:Polygon([nodes[t] for t in p['exterior_node_tokens']],[[nodes[t] for t in h['node_tokens']] for h in p['holes']]) for p in raw['polygon']}
    tokens={t for r in raw['drivable_area'] for t in r['polygon_tokens']}|{r['polygon_token'] for r in raw.get('carpark_area',[])}
    old=unary_union([polys[t] for t in tokens]).buffer(.02)
    diff=old.symmetric_difference(g.roads['boston-seaport'].context).area;assert diff==0,diff
    selection=read(root/'source_selection.json');old_scenes=set(selection['excluded_old_scenes']);split={}
    for c in g.sources.values():
        assert c['scene'] not in old_scenes
        assert c['scene'] not in split;split[c['scene']]=c['source_split']
        stamps=[f['timestamp'] for f in c['frames']]
        assert len(c['frames'])==len(set(stamps))==30
        assert all(0<b-a<=180000 for a,b in zip(stamps,stamps[1:]))
        assert max(f['delta_ms'] for f in c['frames'])<=55
    # 在隔离夹具中以同一真实相机/上下文模拟无LiDAR返回，必须拒绝而非造平面。
    fixture=O/'checks_empty_lidar';fixture.mkdir(exist_ok=True)
    c=copy.deepcopy(next(iter(g.sources.values())))
    for fr in c['frames']:fr.pop('_w2c',None)
    ctx=copy.deepcopy(g.context[c['source_id']])
    dump(fixture/'source_manifest.json',{'clips':[c]});dump(fixture/'source_context.json',{'clips':[ctx]})
    if not (fixture/'maps').exists():(fixture/'maps').symlink_to(root/'maps',target_is_directory=True)
    for i in [0,3,6]:
        path=fixture/'rgb'/ctx['frames'][i]['sensors']['LIDAR_TOP']['filename'];path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():path.write_bytes(b'')
        assert path.stat().st_size==0
    zero=Geometry(fixture);result=zero.prepare(c['source_id'])
    assert result['pass'] is False and result['low_grid_points']==0 and 'plane' not in result
    report={'actual_sources':len(g.sources),'actual_frames':30*len(g.sources),'ordered_unique_exposure_contract':True,
            'scene_split_isolated':True,'map_dispatch':sorted(g.roads),'map_records':g.map_notes,'mixed_map_legacy_access_rejected':True,
            'Boston_parking_road_symmetric_difference_area':diff,'empty_LiDAR_rejected_without_plane':True,
            'training_steps':0,'human_verdict':None}
    dump(O/'infra_validation.json',report);print(report,flush=True)


if __name__=='__main__':main()
