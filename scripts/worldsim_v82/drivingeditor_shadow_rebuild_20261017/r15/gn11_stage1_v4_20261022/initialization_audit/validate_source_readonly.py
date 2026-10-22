"""已有运行也可使用的只读源码核对；不实例化Scene、RoadTraining或优化器。"""
import argparse,json,sys
from pathlib import Path


def validate(campaign):
    plan=json.loads((campaign/'plan.json').read_text());source=Path(plan['V1_source'])
    sys.path.insert(0,str(campaign/'tools'))
    from v4_train_hooks import strip_hooks
    assert strip_hooks((campaign/'engine/train.py').read_text())==(source/'train.py').read_text()
    n=0
    for p in source.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.pyc' or p.name=='train.py':continue
        assert p.read_bytes()==(campaign/'engine'/p.relative_to(source)).read_bytes(),str(p.relative_to(source));n+=1
    baseline=json.loads(Path(plan['baseline_config']).read_text())
    for arm in ['B','R','RP']:
        expected=dict(baseline,model_path=str(campaign/'arms'/arm/'run30k'),v4_arm=arm,v4_bundle=plan['road_bundle'])
        assert json.loads((campaign/'configs'/f'{arm}.json').read_text())==expected
    return dict(status='PASSED_READONLY_SOURCE_AND_CONFIG',unchanged_native_files=n,V1_train_equal_after_removing_two_hooks=True,B_R_RP_common_fresh_configuration=True,
        existing_run_directories=[str(p) for p in (campaign/'arms').glob('*/run30k')],existing_runs_modified=False,training_or_road_updates=0,
        note='original fresh-only validator refuses existing run directories;this check only reads source/config and does not instantiate training objects')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);o=p.parse_args();r=validate(o.campaign);o.output.write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r))
