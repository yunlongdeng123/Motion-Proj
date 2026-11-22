"""Read saved candidate rejection bits; no source qualification is relaxed."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    receipt=json.loads((a.run/'receipt.json').read_text());names=receipt['reason_bit_names'];assert len(names)==9
    order=[0,6,8,7,1,2,5,4,3];rows=[];total=np.zeros(len(names),np.int64);diagnostic_union=None
    for raw in receipt['examined_frames']:
        path=a.run/f'{raw:03d}_all_candidates.npz'
        with np.load(path) as z:bits=z['rejection_bits'];qualified=z['qualified_proxy_certificate']
        remaining=np.ones(len(bits),bool);funnel=[]
        for b in order:
            rejected=remaining&((bits&(1<<b))!=0);remaining&=~rejected
            funnel.append(dict(reason=names[b],rejected=int(rejected.sum()),remaining=int(remaining.sum())))
            total[b]+=rejected.sum()
        assert np.array_equal(remaining,qualified)
        # This describes the conservative proxy bottleneck, not a new acceptance rule.
        ignore=(1<<2)|(1<<5)
        other_checks=(bits&np.uint16(65535^ignore))==0
        diagnostic_union=other_checks.copy() if diagnostic_union is None else diagnostic_union|other_checks
        rows.append(dict(raw=raw,ordered_rejections=funnel,strict_qualified=int(qualified.sum()),
            diagnostic_other_checks_pass_without_dynamic_rectangle_and_OBB_proxies=int(other_checks.sum()),
            diagnostic_is_not_recertification=True,candidate_file_sha256=sha(path)))
    out=dict(schema='R10.whole_asset.fresh_search.fixed_partition.v1',complete=True,CPU_only=True,nativeF=0,nativeB=0,updates=0,
        source_receipt_sha256=sha(a.run/'receipt.json'),candidate_records=receipt['candidate_records'],asset_vertices=receipt['actual_asset_vertex_addresses'],
        actual_asset_cells=receipt['actual_asset_cells'],actual_asset_world_surface_area_m2=receipt['asset_world_surface_area_m2'],
        strict_fresh_vertices=receipt['fresh_known'],strict_fresh_all4_world_area=receipt['fresh_all4_world_area'],
        exclusive_priority=[names[b] for b in order],exclusive_totals={names[b]:int(total[b]) for b in order},
        diagnostic_without_two_geometric_dynamic_proxies_unique_vertices=int(diagnostic_union.sum()),
        reason_nonexclusive_and_priority_exclusive_both_retained=True,
        interpretation='All candidates remain stored; source qualification is unchanged. GT OBB ray intersection and dynamic rectangles are conservative potential-occlusion proxies, not actual surfaces or alpha truth. No fresh certificate is not a proof that all photos are physically invisible. No field is written.',
        frames=rows)
    (a.run/'fixed_rejection_partition.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ['candidate_records','actual_asset_world_surface_area_m2','strict_fresh_vertices','exclusive_totals','diagnostic_without_two_geometric_dynamic_proxies_unique_vertices']}))
if __name__=='__main__':main()
