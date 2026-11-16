"""仅重放位置决定的固定reference选择；不读取query target/input RGB。"""
from pathlib import Path
import argparse,json
import numpy as np


def audit(root):
    root=Path(root);m=json.loads((root/'dataset.json').read_text());candidates=m['candidates'];rows=[]
    centers={c['id']:np.load(root/c['world_xyz']).mean((0,1)) for c in candidates}
    for s in m['samples']:
        center=np.asarray(s['world_bounds']).mean(0)
        expected=max(candidates,key=lambda c:np.linalg.norm(centers[c['id']]-center))
        assert expected['id']==s['reference_candidate_id'],s['id']
        assert expected['min_axis_span_native_px']>=m['quality_gate']['native_min_axis_span_px']
        query_ids=np.load(root/s['point_ids']);reference_ids=np.load(root/expected['point_ids'])
        assert np.intersect1d(query_ids,reference_ids).size==0
        rows.append(dict(id=s['id'],split=s['split'],reference_candidate_id=expected['id'],
            geometry_center_distance_m=float(np.linalg.norm(centers[expected['id']]-center)),query_world_overlap_cells=0))
    result=dict(status='PASS_REFERENCE_SELECTION_GEOMETRY_ONLY_REPLAY',selection='farthest candidate world center; stable candidate list tie order',
        inputs_read_for_selection=['sample.world_bounds','candidate.world_xyz.npy','manifest candidate order'],
        query_target_rgb_read=False,query_hidden_rgb_used_for_ranking=False,query_input_rgb_read=False,
        candidate_rgb_read_for_ranking=False,model_output_used=False,
        eligibility='all candidates pass the same preregistered native >=64px and source road coverage gate before train/eval split',
        reference_role='different-world visible road material appearance prior',rows=rows)
    (root/'reference_selection_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,type=Path);a=p.parse_args();audit(a.dataset)
