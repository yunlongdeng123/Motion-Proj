"""Export the already registered AV2 confirmation cohort without model scoring."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
import torch
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--dataset-root',type=Path,default=Path('/root/autodl-tmp/data/av2/sensor/train'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2); started=time.monotonic()
    def save(name,value):
        (args.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    plan=json.loads(args.plan.read_text()); identities=plan['selected_log_ids']
    save('manifest.json',{'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'plan':str(args.plan),'selected_log_ids':identities,'dataset_root':str(args.dataset_root),
        'role':'external_confirmation','identity_selection':'reuse preregistered list unchanged; no quality or coverage selection',
        'operation':'CPU coordinate/Actor/ray export using same old-development implementation',
        'external_confirmation_values_processed':True,'model_quality_computed':False,'optimizer_updates':0,
        'heldout_boundary':'stored as heldout rays only, never build support or image correspondence',
        'model_selection':'this export does not choose a model or report reconstruction quality'})
    save('status.json',{'status':'running','phase':'export','logs_done':0,'logs_total':len(identities)})
    try:
        cases=[]; scenes=[]; records=[]
        environment={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
        for identity in identities:
            target=args.output/identity
            with (args.output/(identity+'.log')).open('w') as log:
                command=[sys.executable,str(ROOT/'scripts/prepare_worldsim_v73_av2_actors.py'),
                         '--log',str(args.dataset_root/identity),'--output',str(target),
                         '--window-plan',str(args.plan),'--role','external_confirmation']
                completed=subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f'AV2 export failed for registered log {identity}; exit={completed.returncode}; preserve all logs, do not drop identity')
            index=json.loads((target/'index.json').read_text())
            cases.extend({**entry,'file':identity+'/'+entry['file']} for entry in index['cases'])
            scenes.extend(index['scenes'])
            records.append({'log_id':identity,'actors':len(index['cases']),'input_views':index['scenes'][0]['input_views'],
                            'wall_s':index['wall_s'],'peak_rss_gib':index['peak_rss_gib'],'scans':index['scans']})
            save('status.json',{'status':'running','phase':'export','logs_done':len(records),'logs_total':len(identities),
                                'last_log':identity,'elapsed_s':time.monotonic()-started})
            print(json.dumps({'log_id':identity,'logs_done':len(records),'status':'exported'}),flush=True)
        save('status.json',{'status':'running','phase':'combine_build_observations','logs_done':len(records),'logs_total':len(identities)})
        observations=[]
        for identity in identities:
            observations.extend(torch.load(args.output/identity/'build_observations.pt',map_location='cpu',weights_only=False,mmap=True))
        torch.save(observations,args.output/'build_observations.pt')
        result={'status':'done','cases':cases,'scenes':scenes,'logs':records,'selection':'all_window_rigid',
            'selection_boundary':'preregistered 20 external logs; all known rigid build-window tracks, no sensor/model-quality selection',
            'wall_s':time.monotonic()-started,'peak_parent_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            'max_exporter_rss_gib':max((r['peak_rss_gib'] for r in records),default=0.),
            'model_quality_computed':False,'optimizer_updates':0,'external_confirmation_values_processed':True}
        save('index.json',result); save('status.json',{'status':'done','logs_done':len(records),'logs_total':len(identities)})
        print(json.dumps({k:v for k,v in result.items() if k not in ['cases','scenes','logs']}),flush=True)
    except Exception as exc:
        save('status.json',{'status':'failed','exception':type(exc).__name__,'message':str(exc)})
        (args.output/'traceback.txt').write_text(traceback.format_exc()); raise


if __name__=='__main__': main()
