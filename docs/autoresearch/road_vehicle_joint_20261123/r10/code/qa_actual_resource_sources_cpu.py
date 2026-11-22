"""Validate resources-first accounting with smoke replay, failure-only state and standalone verify."""
from pathlib import Path
import argparse,hashlib,json,tempfile
from summarize_actual_resources_cpu import actual_run

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True);root=a.output.parent.resolve()
    with tempfile.TemporaryDirectory(prefix='resource_source_controls_',dir=root) as tmp:
        base=Path(tmp);assert base.resolve().is_relative_to(root)
        def make(name,receipt=None,resources=None):
            d=base/name;d.mkdir()
            if receipt is not None:(d/'receipt.json').write_text(json.dumps(receipt))
            if resources is not None:(d/'resources.json').write_text(json.dumps(resources))
            return actual_run(d)
        smoke,error=make('smoke6',dict(complete=True,mode='smoke',actual_optimizer_updates=7,logical_steps=6,nativeF=999,nativeB=999),
            dict(complete=True,error=None,mode='smoke',nativeF=12,nativeB=7,actual_optimizer_updates=7,formal_updates=0,smoke_updates=7,logical_steps_completed=6))
        assert error is None and (smoke['actual_nativeF'],smoke['actual_nativeB'])==(12,7)
        assert smoke['formal_updates']==0 and smoke['smoke_updates']==7 and smoke['actual_optimizer_updates']==7
        assert smoke['logical_steps_completed']==6 and smoke['counter_source_kind']=='resources.json'
        assert smoke['receipt_resource_counter_differences']['nativeF']==dict(resources=12,receipt=999)
        failed,error=make('fit_failed',resources=dict(complete=False,error=dict(type='RuntimeError',message='synthetic control'),mode='fit',nativeF=3,nativeB=2,actual_optimizer_updates=2,formal_updates=2,smoke_updates=0))
        assert error is None and failed['receipt'] is None and not failed['complete'] and failed['error']
        assert failed['formal_updates']==2 and failed['actual_nativeF']==3
        verify,error=make('fresh_verify',receipt=dict(complete=True,decision='PASS',nativeF=5,nativeB=0,updates=0))
        assert error is None and verify['counter_source_kind']=='receipt.json' and verify['actual_nativeF']==5
        pending,error=make('fit_receipt_before_finally',receipt=dict(complete=True,mode='fit',actual_optimizer_updates=4,logical_steps=4))
        assert pending is None and error is not None
        rows=[smoke,failed,verify]
        assert sum(x['actual_nativeF'] for x in rows)==20 and sum(x['actual_nativeB'] for x in rows)==9
        assert sum(x['actual_optimizer_updates'] for x in rows)==9
        assert sum(x['formal_updates'] for x in rows)==2 and sum(x['smoke_updates'] for x in rows)==7
    producer=Path(__file__).with_name('summarize_actual_resources_cpu.py')
    out=dict(schema='R10.resources_priority_accounting_CPU_QA.v1',complete=True,decision='PASS',producer_sha256=hashlib.sha256(producer.read_bytes()).hexdigest(),
        controls=4,resources_over_receipt_no_double_count=True,smoke6_plus1_replay_actual_updates_7=True,
        formal_smoke_authoritative_split=True,resource_only_failure_counted_with_exception=True,standalone_verify_receipt_counted=True,
        result_without_resources_and_native_counters_reported_unresolved=True,expected_synthetic_totals=dict(nativeF=20,nativeB=9,actual_optimizer_updates=9,formal_updates=2,smoke_updates=7),
        original_receipts_modified=False,GPU_calls=0)
    assert not a.output.exists();a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
