"""独立保存fit全轨迹真实几何/首回波标签，绝不替换build模型输入。"""
import argparse,json
from pathlib import Path
import resource,subprocess,sys,time,traceback
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v73.actor_rays import ActorRayDataset


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--native-run',type=Path,required=True)
    parser.add_argument('--actor-data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=False); started=time.monotonic()
    def save(name,value):
        tmp=args.output/(name+'.tmp')
        tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        tmp.replace(args.output/name)
    save('manifest.json',{'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'native_run':str(args.native_run),'actor_data':str(args.actor_data),
        'boundary':'fit-only full known trajectory labels; no change to build RGB/points, metric scales, or query seeds; no development labels generated',
        'unknown_surface':'temporal accumulation expands observed support, does not produce complete surface GT',
        'sensor_boundary':'original first returns, known Actor pose at LiDAR scan time, overlap-box ownership proxy, no per-point timestamps'})
    save('status.json',{'status':'running','phase':'load_metadata'})
    try:
        scenes={s['scene_id']:s for s in torch.load(args.native_run/'build_observations.pt',map_location='cpu',weights_only=False) if s['role']=='fit'}
        entries=[r for r in json.loads((args.actor_data/'index.json').read_text())['cases'] if r['role']=='fit']
        sensor=ActorRayDataset('/root/autodl-tmp/data/worldsim_v4/drivestudio_raw_trainval',set(scenes))
        rows=[]
        for name,scene in scenes.items():
            scan_cache={}
            for entry in [r for r in entries if r['scene']==name]:
                rays=sensor.actor(name,entry['owner'],{v['sample_id'] for v in scene['views']},
                    include_track=True,scan_cache=scan_cache)
                parts=[r['points_actor_m'][r['positive_actor']] for r in rays]
                points=torch.unique(torch.cat(parts),dim=0) if parts else torch.empty(0,3)
                row={**entry,'target_frames':len(rays),'target_unique_points':len(points),
                    'target_owned_returns':sum(r['owned_points'] for r in rays),
                    'target_raw_near_box_rays':sum(r['near_box_rays'] for r in rays),
                    'label_only_frames':sum(r['role']=='fit_label_time' for r in rays)}
                filename=name+'__'+entry['owner']+'.pt'; row['target_file']=filename
                torch.save({'metadata':row,'target_points_actor_m':points,'target_rays':rays},args.output/filename)
                rows.append(row)
                save('status.json',{'status':'running','phase':'fit_track_targets','actors_done':len(rows),
                    'scene':name,'target_unique_points':len(points),'cached_scans':len(scan_cache),
                    'elapsed_s':time.monotonic()-started})
                print(json.dumps({'scene':name,'owner':entry['owner'],'target_unique_points':len(points),
                    'target_frames':len(rays),'input_build_points':entry['build_points']}),flush=True)
            del scan_cache
        save('index.json',{'status':'done','cases':rows,'wall_s':time.monotonic()-started,
            'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            'boundary':'fit labels only; same original input cohort including missing inputs; no automatic training or input substitution'})
        save('status.json',{'status':'done','actors':len(rows)})
    except Exception as exc:
        save('status.json',{'status':'blocked','exception':type(exc).__name__,'message':str(exc)})
        (args.output/'traceback.txt').write_text(traceback.format_exc()); raise


if __name__=='__main__': main()
