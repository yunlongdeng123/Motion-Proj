"""验证九例实际产物和DELETE状态差异；不替代人眼验收。"""
from nine_common import *
import copy,numpy as np
from PIL import Image
reg=read(ROOT/'registration.json');assert len(reg['scenes'])==9
assert read(ROOT/'mask_state.json')['state']=='complete'
assert read(ROOT/'drive_state.json')['state']=='complete'
assert read(ROOT/'omega_state.json')['state']=='complete'
assert len(read(ROOT/'omega_state.json')['completed'])==270
results=[]
for s in reg['scenes']:
    base=ROOT/s['name'];summary=read(ROOT/'review'/s['name']/'summary.json');assert summary['frames']==30 and all(v==30 for v in summary['video_frames'].values())
    factual=read(base/'scene_factual.json');deleted=read(base/'scene_delete.json');expected=copy.deepcopy(factual);expected['actors'][0]['visible']=False
    # apply_delete会记录独立编辑命令；结构化状态其余字段应完全相同。
    for k in expected:assert deleted[k]==expected[k],k
    checks=0
    for v in s['streams']:
        stream=base/f"cam{v['camera']}"
        for f in range(30):
            rgb=np.array(Image.open(stream/'rgb'/f'{f:05}.png'));bg=np.array(Image.open(stream/'background'/f'{f:05}.png'))
            if v['active']:
                m=np.array(Image.open(stream/'model_mask'/f'{f:05}.png'))>0;p=np.array(Image.open(stream/'protect'/f'{f:05}.png'))>0;w=np.array(Image.open(stream/'write_mask'/f'{f:05}.png'))>0;raw=np.array(Image.open(stream/'native'/f'{f:05}.png'))
                assert np.array_equal(rgb[~m],bg[~m]) and np.array_equal(rgb[p],bg[p]) and np.array_equal(raw[w],bg[w])
            else:assert np.array_equal(rgb,bg)
            checks+=1
    layers=read(base/'actor_layers/render_summary.json');placements=read(base/'actor_layers/placement_checks.json')
    expected_layers=sum(sum(x>0 for x in v['projected_areas']) for v in s['streams'])
    assert len(layers['rows'])==expected_layers
    assert max(x['projection_max_error_px'] for x in layers['rows'])<.02
    assert max(x['center_error_m'] for x in placements)<.002 and max(x['size_error_m'] for x in placements)<.002
    result=dict(scene=s['name'],rgb_contracts=checks,layer_renders=expected_layers,projection_max_px=max(x['projection_max_error_px'] for x in layers['rows']),query_only_visibility_change=True,video_counts=summary['video_frames'],human_verdict=None)
    results.append(result)
    print('SCENE_VALIDATED',s['name'],checks,expected_layers,flush=True)
dump(ROOT/'validation.json',dict(task_id=reg['task_id'],run_id='r1',scenes=results,total_scenes=9,total_rgb_contracts=sum(r['rgb_contracts'] for r in results),omega_forwards=270,layer_renders=sum(r['layer_renders'] for r in results),training_steps=0,strict_generalization=False,human_verdict=None))
