"""在独立目录中快照实际运行轻量正清单；不调用 Git，不接触训练封包。"""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/vehicle_road_joint_20261121_r1')
ARCHIVE = 'docs/autoresearch/vehicle_road_joint_20261121_r1'
CODE = ['run_joint.py','lower_prior.py','post_numeric_probe.py','restore_optimizer.py',
        'fix_a9_handoff.py','guard_a9_handoff.py','finalize_assets.py','qa_final_cpu.py',
        'joint_detail_metrics.py','road_detail_metrics.py','independent_detail_bench.py',
        'run_detail_bench.py','audit_teacher_overlap.py']
META = ['contract.json','contract_final.json','asset_index.json','canonical_source_r2.json',
        'lower_prior.json','supervision_domains.json','resource_ledger.json','post_numeric_probe.json',
        'asset_cpu_freeze_proof.json','final_optimizer_restore_proof.json','disk_reload_proof.json',
        'handoff_selection.json','handoff_same_asset_proof.json','handoff_guarded_admission.json',
        'handoff_guarded_proof.json','handoff_fixed_guarded_disk_reload_proof.json','qa_final_cpu.json',
        'qa_canonical_r2_comparison.json','qa_r2_baseline_comparison.json','qa_supervision_contract_r2.json',
        'qa_teacher_render_overlap.json','implementation_delivery_receipt.json','takeback_manifest.json',
        'smoke/proof.json','smoke/continuation_numeric.json','delivery/metrics.json','delivery/proofs.json',
        'delivery/detail_metrics.json','delivery/independent_detail_metrics.json','takeback/RESTORE.md']
OPTIONAL = ['delivery/detail_metrics_full.json','delivery/independent_detail_metrics_full.json']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main(output):
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(ROOT / 'publication'):
        raise ValueError('必须使用本轮publication下的新目录')
    selected = [('run_code/' + r, 'code/' + r, 'run_source') for r in CODE]
    selected += [('run_code/frozen_formal_r2/' + r, 'code/frozen_formal_r2/' + r, 'run_source') for r in ['run_joint.py','lower_prior.py']]
    selected += [(r,r,'contract' if r.startswith('contract') else 'record') for r in META]
    selected += [(r,r,'metrics') for r in OPTIONAL if (ROOT/r).is_file()]
    for kind in ['joint_fit','handoff_fixed','handoff_fixed_guarded']:
        for name in ['scene_binding.json','poses_full.json','actor_track_map.json']:
            rel = 'assets/' + kind + '/' + name
            if (ROOT/rel).is_file(): selected.append((rel,rel,'provenance'))
    sources=[]
    for dest,rel,role in selected:
        source=ROOT/rel
        if not source.is_file(): raise FileNotFoundError(source)
        cap=8_000_000 if source.suffix=='.json' else 2_000_000
        if source.stat().st_size>cap: raise ValueError('过大: '+rel)
        data=source.read_bytes();data.decode('utf-8-sig')
        if source.suffix=='.json': json.loads(data)
        target=output/'tree'/dest
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        sources.append(dict(source=str(target),destination=ARCHIVE+'/'+dest,role=role,
                            original_source=str(source),sha256=sha(data),bytes=len(data)))
    catalog=dict(schema='explicit_remote_source_catalog.v1',publication_ready=False,
                 root_report_complete=False,formal_stage=False,commit_created=False,pushed=False,
                 source_root=str(ROOT),sources=sources,total_bytes=sum(x['bytes'] for x in sources))
    (output/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(catalog=str(output/'catalog.json'),files=len(sources),bytes=catalog['total_bytes'],formal_stage=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
