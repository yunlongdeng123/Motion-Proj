"""登记通用v44保持/覆盖修复，固定原30K、归属、道路与1000步预算。"""
from pathlib import Path
import json,os,shutil,subprocess,time
import numpy as np
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign');R=C/'waymo_val065_delete30k_20261109_r1';O=C/'waymo_val065_handoff_fix_20261111_r1'
PUBLIC=C/'waymo_allframes_20261109_r1/public_api';PY='/root/autodl-tmp/envs/drivingeditor/bin/python';TRACK='NTjq-OhZ_97OMJ4xwJAxXg';KEYS=[60,119,150,198]
def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def status(stage,**kw):write(O/'status.json',dict(stage=stage,unix=time.time(),controller_pid=os.getpid(),automatic_shutdown=False,**kw))
def run(name,module,*args,cpu=False):
    command=[PY,'-B','-m',module,*map(str,args)];env=os.environ.copy();env.update(PYTHONPATH=str(O/'runtime_api'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');env['PATH']=str(Path(PY).parent)+os.pathsep+env.get('PATH','')
    if cpu:env['CUDA_VISIBLE_DEVICES']=''
    start=time.monotonic()
    with (O/(name+'.log')).open('x') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=O);status(name,child_pid=child.pid,command=command);code=child.wait()
    write(O/(name+'_command.json'),dict(command=command,seconds=time.monotonic()-start,exit_code=code))
    if code:raise RuntimeError(name+' failed; '+str(O/(name+'.log')))
def manifest(name,value):
    p=O/(name+'.json');write(p,value);return p
def render(name,m,delete=False,keys=None):
    q=dict(focus_actor_ids=[10],diagnostic_actor_ids=[10,8,21,20,24],diagnostic_branch_alpha=True)
    if delete:q['delete_track_ids']=[TRACK]
    qp=manifest(name+'_query',q);args=['render','--manifest',m,'--query',qp,'--output',O/name]
    if keys is not None:args+=['--frames-json',json.dumps(keys)]
    run(name,'drivingeditor_dynamic_edit.v44_cli',*args)
def compare_ref(new,old):
    a=np.load(new);b=np.load(old);assert set(a.files)==set(b.files)
    differences={}
    for k in a.files:
        assert a[k].shape==b[k].shape,(k,a[k].shape,b[k].shape)
        if not np.array_equal(a[k],b[k]):differences[k]=float(np.max(abs(a[k]-b[k])))
    assert not differences,differences
    return dict(all_fields_identical=True,fields=len(a.files),road_rows=int(a['active'].sum()),reference_or_geometry_added=False)
def main():
    O.mkdir(exist_ok=False);(O/'source').mkdir();shutil.copy2('/tmp/v44_handoff_fix_run.py',O/'source/run.py')
    shutil.copytree(PUBLIC,O/'runtime_api')
    pkg=O/'runtime_api/drivingeditor_dynamic_edit'
    for name in ['v44_gate_fit.py','v44_road.py','v44_handoff_policy.py','test_v44_handoff_policy.py']:shutil.copy2(Path('/tmp/v44_handoff_fix_source')/name,pkg/name)
    for name in ['v44_gate_fit.py','v44_road.py','v44_handoff_policy.py','test_v44_handoff_policy.py']:shutil.copy2(pkg/name,O/'source'/name)
    model=Path(read(R/'registration.json')['source_model']);sizes={p.name:dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in model.glob('*30000.pth')}
    write(O/'registration.json',dict(task='WAYMO-V44-HANDOFF-FIX-20261111',scene='065',context='15021599536622641101_556_150_576_150',target_track=TRACK,target_actor=10,
        source_model=str(model),iteration=30000,frames=199,heldout=0,seed=0,native_gaussian_updates=0,formal_gate_updates=1000,
        lifecycle='32 formal steps save, native CLI initialization/pilot renders, exact full state restore, remaining968; no extra optimization updates',
        method='generic full target preservation exclusion and candidate support, matching input/baseline identity; unchanged reference photo and loss weights',
        unknown_target_reference_added=False,initial_gate_logit=2.,initial_multiplier=.880797,road_geometry_colour_fixed=True,
        short_job=True,long_training_smoke_applicable=False,reason='short1000-step opacity handoff, no hour-scale model training',
        baseline_run=str(R),native_checkpoint_files=sizes,automatic_shutdown=False,failure_ledger_refs=['V82-F06']))
    run('policy_tests','unittest','discover','-s',pkg,'-p','test_v44_handoff_policy.py','-v',cpu=True)
    base=read(R/'road_manifest.json')
    assert [f['key'] for f in base['frames']]==list(range(199));assert dict((x['track_id'],x['actor_id']) for x in base['actors'])[TRACK]==10
    job=next(x for x in read(C/'waymo_allframes_20261109_r1/edit_jobs.json')['jobs'] if x['scene']=='065')
    native=read(R/'native_manifest.json');v1=model.parent/'v1_30000'
    run('world_road','drivingeditor_dynamic_edit.v44_road','--bank',job['bank'],'--masks',job['masks'],'--semantics',job['semantics'],
        '--native-eval',v1,'--transform',native['world_to_model'],'--output',O/'world_road','--track-id',TRACK,'--actor-id',10,cpu=True)
    write(O/'reference_equivalence.json',compare_ref(O/'world_road/reference.npz',R/'world_road/reference.npz'))
    rows=[dict(key=r['frame_key'],view=f"{r['frame_key']:03d}.npz",baseline=r['files']['float_npz']) for r in read(R/'semantic_delete/provenance.json')['frames']]
    write(O/'world_road/views/manifest.json',dict(frames=rows))
    base['road']=str(O/'world_road/reference.npz');m=manifest('road_manifest',base)
    args=['--manifest',m,'--views',O/'world_road/views','--baseline',R/'semantic_delete','--delete-track-id',TRACK]
    run('gate32','drivingeditor_dynamic_edit.v44_gate_fit',*args,'--output',O/'gate32','--steps',32)
    cov=read(O/'gate32/coverage.json');assert len(cov['rows'])==199
    assert all(r['target_keep_pixels']==r['target_left_pixels']==r['protected_target_conflicts']==0 for r in cov['rows'])
    assert all(r['target_candidate_pixels']==r['target_pixels'] for r in cov['rows'])
    assert cov['optimization_frames']==read(R/'gate1000/receipt.json')['fit_frames']
    assert (O/'gate32/checkpoint000000.pt').is_file() and (O/'gate32/checkpoint000032.pt').is_file()
    init=dict(base,static_gates=dict(indices=str(O/'gate32/contributors.npz'),checkpoint=str(O/'gate32/checkpoint000000.pt')))
    im=manifest('initial_manifest',init);render('initial_keep',im,keys=KEYS);render('initial_delete',im,True,KEYS)
    pilot=dict(base,static_gates=dict(indices=str(O/'gate32/contributors.npz'),checkpoint=str(O/'gate32/checkpoint000032.pt')))
    pm=manifest('pilot32_manifest',pilot);render('pilot32_keep',pm,keys=KEYS);render('pilot32_delete',pm,True,KEYS)
    run('gate1000','drivingeditor_dynamic_edit.v44_gate_fit',*args,'--output',O/'gate1000','--steps',1000,'--resume',O/'gate32/checkpoint000032.pt')
    assert read(O/'gate1000/restore.json')['exact_full_state'];assert read(O/'gate1000/receipt.json')['updates']==968
    final=dict(base,static_gates=dict(indices=str(O/'gate1000/contributors.npz'),checkpoint=str(O/'gate1000/checkpoint001000.pt')))
    fm=manifest('final_manifest',final);render('final_keep',fm);render('delete_white',fm,True)
    q=manifest('regression_queries',[dict(delete_track_ids=[TRACK])])
    run('roundtrip','drivingeditor_dynamic_edit.v44_regression','--manifest',fm,'--queries',q,'--frames-json',json.dumps(KEYS),'--output',O/'roundtrip.json')
    for name in ['final_keep','delete_white']:
        prov=read(O/name/'provenance.json');assert prov['status']=='COMPLETE' and prov['iteration']==30000
        assert [f['frame_key'] for f in prov['frames']]==list(range(199));assert prov['persistent_check_before'] and prov['persistent_check_after']
    after={p.name:dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in model.glob('*30000.pth')};assert after==sizes
    write(O/'completed.json',dict(status='HANDOFF_FIX_RENDERED_PENDING_VISUAL_REVIEW',iteration=30000,frames=199,formal_gate_updates=1000,native_gaussian_updates=0,
        source_native_checkpoint_unchanged=True,final_manifest=str(fm),baseline_run=str(R),road_reference_unchanged=True,human_verdict=None,automatic_shutdown=False))
    status('RENDER_COMPLETE_PENDING_RETRIEVAL_HTML_REVIEW_GIT')
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if O.exists():status('FAILED',error=repr(e))
        raise
