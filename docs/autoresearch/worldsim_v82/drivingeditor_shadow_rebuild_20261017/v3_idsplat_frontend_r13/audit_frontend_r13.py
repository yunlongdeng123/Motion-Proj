from pathlib import Path
import numpy as np,json,sys,subprocess,os,fcntl,time
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');S=R/'ownership_controls_r12_sam3';D=R/'idsplat_frontend_r13';targets=[5,6,12,18,25,26,37]
A=np.load(S/'data_A_box/seed.npz');N=np.load(D/'data_IDSplat/seed.npz');tr=np.load(D/'data_IDSplat/trajectory.npz');m=json.loads((D/'data_IDSplat/manifest.json').read_text());checks=[]
assert set(A.files)==set(N.files)
assert (S/'data_A_box/trajectory.npz').read_bytes()==(D/'data_IDSplat/trajectory.npz').read_bytes();checks.append('GT trajectory bytes unchanged')
for a in sorted(set(A['owner'].tolist())-set(targets)):
    ai=np.flatnonzero(A['owner']==a);ni=np.flatnonzero(N['owner']==a)
    assert len(ai)==len(ni)
    for key in A.files:
        if key=='neighbor':assert np.array_equal(A[key][ai]-ai[0],N[key][ni]-ni[0]),(a,key)
        else:assert np.array_equal(A[key][ai],N[key][ni]),(a,key)
checks.append('all static + other36 Actor seed fields exact, neighbors reindexed only')
for row in m['frames']:
    for key in ['rgb','supervision']:
        assert (S/'data_A_box'/row[key]).read_bytes()==(D/'data_IDSplat'/row[key]).read_bytes()
checks.append('all49 RGB and full supervision files exact')
for a in targets:
    take=N['owner']==a;fi=N['source_frame'][take];raw=N['source_raw_row'][take];world=N['source_world'][take];xyz=N['xyz'][take]
    assert np.all(raw%5!=0)
    for f in np.unique(fi):
        assert m['frames'][int(f)]['split']=='train'
        i=fi==f;matrix=tr['poses'][f,a];pred=xyz[i]@matrix[:3,:3].T+matrix[:3,3]
        assert np.max(np.abs(pred-world[i]))<2e-5
        actual=np.load(S/'raw_measurements'/(m['frames'][int(f)]['frame_id']+'.npz'))
        assert np.all(actual['BUILD'][raw[i]]) and np.array_equal(actual['world'][raw[i]],world[i])
    source=np.load(D/'pointsets'/f'a{a:02d}_all_candidates.npz');assert np.all(source['accepted']==(source['evidence_reason']==5))
checks.append('all7 Actor seeds sourced from exact BUILD LiDAR rows, correct GT world/local coordinates, uncertain sources retained')
live=[];suspect=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        state=(p/'stat').read_text().split()[2];cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    except (OSError,UnicodeError):continue
    if state=='Z' or int(p.name)==os.getpid():continue
    if str(R) in cmd:live.append(dict(pid=int(p.name),cmd=cmd))
    elif any(t in cmd for t in ['/root/autodl-tmp/','flock ']) and any(t in cmd for t in ['python','train','render','eval','controller','flock','rsync','scp']):suspect.append(dict(pid=int(p.name),cmd=cmd))
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
with open('/root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock','a') as lock:
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);free=True;fcntl.flock(lock,fcntl.LOCK_UN)
    except BlockingIOError:free=False
v=dict(status='ACTUAL_DATA_AUDIT_PASSED',checks=checks,live_owned=live,other_potential_work=suspect,GPU=gpu,GPU_slot_free=free,epoch=time.time(),scene_training_updates=0,shutdown_command_exists=Path('/usr/bin/shutdown').is_file())
(D/'actual_data_audit.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
