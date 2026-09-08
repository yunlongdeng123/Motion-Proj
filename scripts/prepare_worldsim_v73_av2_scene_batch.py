"""Apply the old-development AV2 scene construction to every registered external log."""
import argparse,json,os,resource,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--actor-data',type=Path,required=True)
    parser.add_argument('--sensor-root',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False); started=time.monotonic()
    index=json.loads((args.actor_data/'index.json').read_text()); logs=index['logs']
    def save(name,value): (args.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    save('manifest.json',{'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'actor_data':str(args.actor_data),'sensor_root':str(args.sensor_root),'log_ids':[r['log_id'] for r in logs],
        'protocol':'all registered external identities and Actor cohorts; original four build/two heldout sweeps; no quality selection',
        'background':'fixed old-development .06m PCA surface, boxes+0.1m removed; build-only whole triangle carving before first return minus0.2m',
        'time':'reference-compensated endpoints once, per-return sensor origins/ownership and read-only Actor trajectories',
        'evaluation_boundary':'raw heldout beams formatted separately; no model inference or heldout quality scores; build carving residuals are construction diagnostics',
        'controller':'one CPU scene constructor at a time; failure retained and stops batch, no skipped/replaced identity',
        'optimizer_updates':0})
    save('status.json',{'status':'running','logs_done':0}); scenes=[]; execution=[]
    try:
        for row in logs:
            log=row['log_id']; output=args.output/log
            command=[sys.executable,str(ROOT/'scripts/prepare_worldsim_v73_av2_scene_geometry.py'),
                     '--log',str(args.sensor_root/log),'--actor-data',str(args.actor_data),'--output',str(output)]
            save('status.json',{'status':'running','current_log':log,'logs_done':len(execution),'elapsed_s':time.monotonic()-started})
            with (args.output/(log+'.log')).open('w') as stream:
                subprocess.run(command,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2',
                    'OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},stdout=stream,stderr=subprocess.STDOUT,check=True)
            child=json.loads((output/'index.json').read_text())
            for scene in child['scenes']:
                (args.output/scene['scene']).symlink_to(Path(log)/scene['scene'],target_is_directory=True)
                scenes.append(scene)
            execution.append({'log_id':log,'wall_s':child['wall_s'],'peak_rss_gib':child['peak_rss_gib']})
            print(json.dumps({'log_id':log,'logs_done':len(execution),'status':'constructed'}),flush=True)
        result={'status':'done','scenes':scenes,'execution':execution,'wall_s':time.monotonic()-started,
                'peak_parent_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                'max_constructor_rss_gib':max(r['peak_rss_gib'] for r in execution),
                'model_quality_computed':False,'optimizer_updates':0,'external_confirmation_values_processed':True}
        save('index.json',result); save('status.json',{'status':'done','logs_done':len(execution)})
        print(json.dumps({k:v for k,v in result.items() if k not in ['scenes','execution']}),flush=True)
    except Exception as exc:
        save('status.json',{'status':'failed','logs_done':len(execution),'exception':type(exc).__name__,'message':str(exc)})
        (args.output/'traceback.txt').write_text(traceback.format_exc()); raise

if __name__=='__main__': main()
