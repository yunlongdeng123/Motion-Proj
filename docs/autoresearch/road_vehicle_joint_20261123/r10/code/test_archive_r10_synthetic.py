"""Synthetic-only archive safety test; never packs the real R10 run."""
import argparse,copy,importlib.util,json,tempfile,tarfile,io
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--helper',type=Path,required=True);p.add_argument('--test-work',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    spec=importlib.util.spec_from_file_location('r10_archive_fixture',a.helper);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    a.test_work.mkdir(parents=True,exist_ok=True);tests=[]
    with tempfile.TemporaryDirectory(prefix='synthetic_archive_',dir=a.test_work) as temp:
        # Verify the recursive temporary cleanup target is our owned workspace.
        base=Path(temp).resolve();base.relative_to(a.test_work.resolve());run=base/h.RUN;run.mkdir()
        for value in h.CRITICAL:
            path=run/value;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture-payload-only')
        for value in ['report/index.html','publication/large.npz','archives/old.tar.gz','code/__pycache__/x.pyc','pending.npz.part','index.html']:
            path=run/value;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('excluded')
        (run/'failed_v0').mkdir();(run/'failed_v0/traceback.log').write_text('preserved original failure')
        final=run/'footprint/receipt.json';h.write(final,dict(complete=True,nativeF=5,nativeB=5))
        deps=run/'provenance/previous_stage_archive_reference.json';h.write(deps,dict(schema='R10.previous_stage_external_dependencies.v1',complete=True,large_weights_copied_into_R10=False,untrusted_reference='/remote/not-present/model.pth'))
        signal=run/'snapshot.json';ready=dict(schema='R10.method_snapshot_ready.v1',run=h.RUN,complete=True,method_stage_frozen=True,overall_run_complete=False,
            whole_scene_restoration_accepted=False,GPU_stays_on=True,report_HTML_included=False,pending_next_footprint_experiment=False,
            final_footprint_receipt=dict(path='footprint/receipt.json',sha256=h.sha(final)),required_complete_receipts=[],critical_members=[])
        h.write(signal,ready);estimate=h.estimate(run);assert not estimate['missing_critical_files']
        files,skip=h.inventory(run);assert any(name.endswith('failed_v0/traceback.log') for _,name,_ in files)
        assert not any('/report/' in name or name.endswith(('.part','.pyc','.html')) for _,name,_ in files);tests.append('live estimate and exclusions, failures retained')
        archive=run/'archives/synthetic.tar.gz';manifest=run/'archives/synthetic.manifest.json';h.create(run,archive,manifest,signal,deps)
        verify=h.verify(archive,manifest);assert verify['all_member_SHA_verified'] and not verify['remote_reference_paths_dereferenced'];tests.append('create and stream all members without executing or extracting remote references')
        md=h.read(manifest);bad=run/'archives/tampered.tar.gz'
        with tarfile.open(archive,'r:gz') as source,tarfile.open(bad,'w:gz') as target:
            for index,member in enumerate(source):
                data=source.extractfile(member).read()
                if index==0:data=bytes([data[0]^1])+data[1:]
                target.addfile(member,io.BytesIO(data))
        altered=copy.deepcopy(md);altered['archive']=dict(name=bad.name,bytes=bad.stat().st_size,sha256=h.sha(bad));badmanifest=run/'archives/tampered.json';h.write(badmanifest,altered)
        try:h.verify(bad,badmanifest)
        except AssertionError:tests.append('member corruption rejected even with recomputed outer archive SHA')
        else:raise AssertionError('Missed tampered member')
        for name in [h.RUN+'/../escape','C:/escape',h.RUN+'\\escape',h.RUN+'//escape',h.RUN+'/report/escape']:
            try:h.member_name(name)
            except AssertionError:continue
            raise AssertionError('Unsafe member accepted: '+name)
        tests.append('traversal, drive, backslash, empty path and excluded-tree names rejected')
        blocked=copy.deepcopy(ready);blocked['method_stage_frozen']=False;h.write(signal,blocked)
        try:h.create(run,run/'archives/blocked.tar.gz',run/'archives/blocked.json',signal,deps)
        except AssertionError:tests.append('unfrozen signal rejected before packing')
        else:raise AssertionError('Unfrozen source was packed')
        assert not (run/'archives/blocked.tar.gz').exists()
    result=dict(schema='R10.archive_helper_synthetic_safety_test.v1',complete=True,decision='PASS',tests=tests,
        real_R10_payload_packed=False,real_method_stage_frozen=False,GPU_calls=0)
    h.write(a.receipt,result);print(json.dumps(result))


if __name__=='__main__':main()
