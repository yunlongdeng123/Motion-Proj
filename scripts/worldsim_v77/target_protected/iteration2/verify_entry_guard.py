"""真实历史坏例不能绕过新渲染入口；全部写到r2隔离目录。"""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from render_pairs import render

def main(root,old):
    src=old/'native_window_factory';probe=root/'entry_guard_probe';probe.mkdir(exist_ok=True)
    for name in ['source_manifest.json','segmented','mask_review_secondary']:
        p=probe/name
        if not p.exists():p.symlink_to((src/name).resolve())
    pair=json.loads((src/'synthetic/W029/pair_manifest.json').read_text());manifest=probe/'candidate.json';manifest.write_text(json.dumps({'selected':[pair]}))
    try:render(probe,probe/'review',manifest)
    except ValueError as e:
        assert 'donor admission failed' in str(e) and 'human_reported_source_blocked' in str(e)
        assert not list((probe/'synthetic/W029').rglob('*.png'))
        result={'historical_candidate':'W029','renderer_refused_before_pixel_writes':True,'error':str(e),'old_artifacts_untouched':True}
        (root/'entry_guard_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(result)
    else:raise AssertionError('blocked donor bypassed renderer')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--old',type=Path,required=True);a=p.parse_args();main(a.root,a.old)
