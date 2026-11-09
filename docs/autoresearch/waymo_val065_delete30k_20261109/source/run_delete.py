"""用现有共享 v44 通路执行已登记目标的30K删除；不训练高斯或修改原资产。"""
from pathlib import Path
import json
import os
import subprocess
import time
import numpy as np

BASE = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
SOURCE = BASE/'waymo_val065_allframes10k_20261109_r1'
OUT = BASE/'waymo_val065_delete30k_20261109_r1'
PUBLIC = BASE/'waymo_allframes_20261109_r1/public_api'
PY = '/root/autodl-tmp/envs/drivingeditor/bin/python'
TRACK = 'NTjq-OhZ_97OMJ4xwJAxXg'
ACTOR = 10
KEYS = [60, 119, 150, 198]


def read(p): return json.loads(Path(p).read_text())
def write(p,d): Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def state(stage,**kw): write(OUT/'status.json',dict(stage=stage,unix=time.time(),controller_pid=os.getpid(),automatic_shutdown=False,**kw))
def manifest(name,data,**kw):
    value=dict(data);value.update(kw);path=OUT/(name+'.json');write(path,value);return path


def run(name,module,*args,cpu=False):
    command=[PY,'-B','-m','drivingeditor_dynamic_edit.'+module,*map(str,args)]
    env=os.environ.copy();env.update(PYTHONPATH=str(PUBLIC),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    env['PATH']=str(Path(PY).parent)+os.pathsep+env.get('PATH','')
    if cpu: env['CUDA_VISIBLE_DEVICES']=''
    start=time.monotonic()
    with (OUT/(name+'.log')).open('x') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=OUT)
        state(name,child_pid=child.pid,command=command)
        code=child.wait()
    write(OUT/(name+'_command.json'),dict(command=command,seconds=time.monotonic()-start,exit_code=code))
    if code: raise RuntimeError(name+' exit='+str(code)+'; log='+str(OUT/(name+'.log')))


def render(name,m,query,keys=None):
    q=OUT/(name+'_query.json');write(q,query)
    args=['render','--manifest',m,'--query',q,'--output',OUT/name]
    if keys is not None:args+=['--frames-json',json.dumps(keys)]
    run(name,'v44_cli',*args)


def main():
    OUT.mkdir(exist_ok=False)
    native=read(SOURCE/'shared_native_manifest.json')
    assert {r['track_id']:r['actor_id'] for r in native['actors']}[TRACK]==ACTOR
    assert [r['key'] for r in native['frames']]==list(range(199))
    model=SOURCE/'continuation30k_20261109_r1/formal30k'
    returned=read(model/'returned.json')
    assert returned['iteration']==30000 and returned['actual_additional_iterations']==20000
    sizes={p.name:dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in model.glob('*30000.pth')}
    assert len(sizes)==4
    write(OUT/'registration.json',dict(task='WAYMO-065-DELETE-WHITE30K-20261109',scene='065',
          source_model=str(model),iteration=30000,training_frames=199,heldout=0,
          target=dict(track_id=TRACK,native_actor_id=ACTOR,alias='a9',description='前方白色轿车'),
          native_reconstruction_verdict='用户确认30K质量通过',other_scene_verdict='用户确认029仍未通过，暂不推进',
          method='existing shared v44 visible-mass ownership + measured world-road recovery +1000 static-opacity handoff steps',
          native_gaussian_updates=0,gate_updates=1000,source_code=str(PUBLIC),
          native_checkpoint_files=sizes,automatic_shutdown=False,failure_ledger_refs=['V82-F06']))
    native['model']=str(model);native['iteration']=30000
    m=manifest('native_manifest',native)
    query=dict(focus_actor_ids=[ACTOR],diagnostic_actor_ids=[ACTOR,8,21,20,24],diagnostic_branch_alpha=True)
    delete=dict(query,delete_track_ids=[TRACK])
    render('native_parity',m,query,KEYS)
    records=[]
    for row in read(OUT/'native_parity/provenance.json')['frames']:
        raw=row['frame_key']
        current=np.load(OUT/'native_parity'/row['files']['float_npz'])['rgb']
        original=np.load(SOURCE/'continuation30k_20261109_r1/v1_30000'/f'{raw:03d}'/'keep.npz')['rgb']
        err=float(np.max(np.abs(current-original)))
        assert err<3e-6, (raw,err)
        records.append(dict(raw=raw,maxabs=err))
    write(OUT/'native_parity_check.json',dict(status='PASS_ACTUAL_NATIVE30K_EQUIVALENCE',rows=records))
    run('ownership','v44_calibrate','--manifest',m,'--output',OUT/'ownership')
    semantic=dict(native,ownership=str(OUT/'ownership/semantic_ownership.npz'),zero_semantic_policy='source_fallback')
    sm=manifest('semantic_manifest',semantic)
    render('semantic_delete',sm,delete)
    old=read(BASE/'waymo_allframes_20261109_r1/edit_jobs.json')['jobs']
    job=next(x for x in old if x['scene']=='065')
    run('world_road','v44_road','--bank',job['bank'],'--masks',job['masks'],'--semantics',job['semantics'],
        '--native-eval',SOURCE/'continuation30k_20261109_r1/v1_30000',
        '--transform',SOURCE/'inputs/native/transform_matrix.txt','--output',OUT/'world_road',
        '--track-id',TRACK,'--actor-id',ACTOR,cpu=True)
    road=dict(semantic,road=str(OUT/'world_road/reference.npz'))
    rm=manifest('road_manifest',road)
    rows=[dict(key=r['frame_key'],view=f"{r['frame_key']:03d}.npz",baseline=r['files']['float_npz'])
          for r in read(OUT/'semantic_delete/provenance.json')['frames']]
    write(OUT/'world_road/views/manifest.json',dict(frames=rows))
    args=['--manifest',rm,'--views',OUT/'world_road/views','--baseline',OUT/'semantic_delete','--delete-track-id',TRACK]
    run('gate32','v44_gate_fit',*args,'--output',OUT/'gate32','--steps',32)
    if (OUT/'gate32/checkpoint000032.pt').exists():
        run('gate1000','v44_gate_fit',*args,'--output',OUT/'gate1000','--steps',1000,'--resume',OUT/'gate32/checkpoint000032.pt')
        assert read(OUT/'gate1000/restore.json')['exact_full_state'] is True
        road['static_gates']=dict(indices=str(OUT/'gate1000/contributors.npz'),checkpoint=str(OUT/'gate1000/checkpoint001000.pt'))
    fm=manifest('final_manifest',road)
    render('final_keep',fm,query)
    render('delete_white',fm,delete)
    q=OUT/'regression_queries.json';write(q,[dict(delete_track_ids=[TRACK])])
    run('roundtrip','v44_regression','--manifest',fm,'--queries',q,'--frames-json',json.dumps(KEYS),'--output',OUT/'roundtrip.json')
    for name in ['final_keep','delete_white']:
        d=read(OUT/name/'provenance.json')
        assert d['status']=='COMPLETE' and d['iteration']==30000
        assert [r['frame_key'] for r in d['frames']]==list(range(199))
        assert d['persistent_check_before'] and d['persistent_check_after']
    after={p.name:dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in model.glob('*30000.pth')}
    assert sizes==after
    write(OUT/'completed.json',dict(status='DELETE30K_RENDERED_PENDING_VISUAL_ACCEPTANCE',scene='065',iteration=30000,
          frames=199,target_track_id=TRACK,source_native_checkpoint_unchanged=True,native_gaussian_updates=0,
          final_manifest=str(fm),read_only_query_regression=read(OUT/'roundtrip.json')['status'],
          automatic_shutdown=False,human_verdict=None,failure_ledger_delta='none_pending_visual_review'))
    state('COMPLETE_PENDING_RETRIEVAL_HTML_VISUAL_REVIEW_GIT')


if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():state('FAILED',error=repr(e))
        raise
