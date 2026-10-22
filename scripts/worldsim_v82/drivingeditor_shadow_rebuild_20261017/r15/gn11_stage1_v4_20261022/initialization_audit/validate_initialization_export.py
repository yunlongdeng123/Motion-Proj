"""导出的全量公共网格、49帧浮点统计及显示编码核验。"""
import argparse,base64,json,zlib
from pathlib import Path
import numpy as np


def payload(path):
    text=path.read_text(encoding='utf-8')
    if path.name=='data.js':return json.loads(text.split('window.V4AUDIT=',1)[1][:-1])
    return json.loads(text.split(']=',1)[1][:-1])


def unpack(p):
    b=zlib.decompress(base64.b64decode(p['data']))
    return {k:np.frombuffer(b,dtype=v['dtype'],offset=v['offset'],count=v['bytes']//np.dtype(v['dtype']).itemsize).reshape(v['shape']) for k,v in p['arrays'].items()}


def validate(root):
    p=payload(root/'data.js');d=unpack(p['data']);s=p['summary'];receipt=json.loads((root/'receipt.json').read_text())
    assert receipt['status']=='COMPLETE' and receipt['frame_count']==49
    assert s['V4_native_initial']['ground_count']==71120 and s['V4_native_initial']['total_static']==585943
    assert abs(s['V4_native_initial']['initial_sigma_t']-np.sqrt(.096))<1e-12
    assert np.array_equal(s['V4_native_initial']['actual_loader_time_duration'],[-.48,.48])
    np.testing.assert_allclose(d['xy'],(d['keys']+.5)*.2,atol=1e-5)
    assert len(set(map(tuple,d['keys'])))==len(d['keys'])
    assert d['supported'].sum()==14034 and (d['group']==1).sum()==1919 and (d['group']==2).sum()==11082
    assert d['outside_original_grid'].sum()==189
    assert ((d['used_k8_count']>=3)|~d['supported'].astype(bool)).all()
    assert not (d['supported'].astype(bool)&d['outside_original_grid'].astype(bool)).any()
    core=s['regions'];assert [core[k]['core']['native_centers'] for k in ['5','37','6']]==[838,866,4]
    assert [core[k]['core']['road_created'] for k in ['5','37','6']]==[210,200,134]
    assert all(np.isfinite(v).all() for v in d.values())
    checked=0;worst=0.
    for raw in range(89,138):
        packed=payload(root/f'frame_{raw}.js');maps=unpack(packed)
        for actor in ['5','37','6']:
            for scope in ['core','context']:
                mask=maps[f'mask_{actor}_{scope}'].astype(bool)
                for model in ['native','all','v3','R','RP']:
                    stats=packed['stats'][actor][scope]['models'][model]
                    assert int(mask.sum())==stats['pixels']
                    if stats['pixels']:
                        mean=float((maps['alpha_'+model][mask].astype(float)/65535).mean());err=abs(mean-stats['alpha_mean']);worst=max(worst,err);assert err<=1/65535+2e-7,(raw,actor,model,err)
                    checked+=1
        assert all(np.isfinite(v).all() for v in maps.values())
    result=dict(status='PASSED',frames=49,quantized_vs_full_float_statistic_checks=checked,max_alpha_mean_display_quantization_error=worst,full_grid_cells=len(d['keys']),outside_cells_kept=189,training_updates=0,GPU_used=False,browser_ui_verified=False)
    (root/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);validate(p.parse_args().root)
