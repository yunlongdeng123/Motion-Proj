from pathlib import Path
import os, signal, time, json, subprocess, datetime
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
D=R/'ownership_controls_r12_sam3'
def processes():
    out={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            fields=(p/'stat').read_text().split()
            if fields[2]=='Z': continue
            out[int(p.name)]={'ppid':int(fields[3]),'cmd':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()}
        except (OSError,UnicodeError): pass
    return out
allp=processes()
roots={pid for pid,v in allp.items() if str(R) in v['cmd'] and 'three_control_controller_r12.py' in v['cmd']}
targets=set(roots)
while True:
    more={pid for pid,v in allp.items() if v['ppid'] in targets}
    if more<=targets: break
    targets|=more
request={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'USER_STOP_BC_REPLACE_WITH_IDSPLAT_FRONTEND_POINT_GATE','processes':[{**allp[p],'pid':p} for p in sorted(targets)],'automatic_resume_forbidden':True}
(D/'USER_STOP_BC.json').write_text(json.dumps(request,indent=2))
for pid in sorted(roots):
    try: os.kill(pid,signal.SIGSTOP)
    except ProcessLookupError: pass
for pid in targets-roots:
    try: os.kill(pid,signal.SIGSTOP)
    except ProcessLookupError: pass
for sig in [signal.SIGTERM,signal.SIGCONT]:
    for pid in sorted(targets,reverse=True):
        try: os.kill(pid,sig)
        except ProcessLookupError: pass
time.sleep(3)
remaining=processes()
for pid in targets:
    if pid in remaining and remaining[pid]['cmd']==allp[pid]['cmd']:
        try: os.kill(pid,signal.SIGKILL)
        except ProcessLookupError: pass
time.sleep(1)
after=processes()
request['remaining_owned']=[{**v,'pid':p} for p,v in after.items() if str(R) in v['cmd'] and any(t in v['cmd'] for t in ['three_control_controller_r12.py','run.py train','r12_geometry_review.py','C_split_r12.py','C_finalize_owner_r12.py'])]
request['arms']={}
for name in ['A_box','B_semantic_motion','C_correctable']:
    d=D/name; p=d/'progress.json'
    request['arms'][name]={'progress':json.loads(p.read_text()) if p.exists() else None,'complete':(d/'complete.json').exists(),'checkpoints':[{'name':x.name,'bytes':x.stat().st_size,'mtime':x.stat().st_mtime} for x in d.glob('*.pt')]}
request['GPU']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
(D/'pipeline_state.json').write_text(json.dumps({'status':'STOPPED_BY_USER_REPLACED_WITH_IDSPLAT_POINT_GATE','updated_utc':request['at'],'no_resume_BC':True},indent=2))
(D/'USER_STOP_BC_receipt.json').write_text(json.dumps(request,indent=2))
print(json.dumps(request))
assert not request['remaining_owned'], request['remaining_owned']
