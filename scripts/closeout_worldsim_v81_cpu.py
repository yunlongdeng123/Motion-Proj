"""冻结CPU事实、模型前样本规则和启动命令；不运行GPU或定时器。"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,collections,statistics,subprocess,importlib.metadata as md,shutil,datetime
from pathlib import Path
import numpy as np
import pyarrow as pa,pyarrow.parquet as pq

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run);base=run.parent/'20260913-cpu-r1';repo=Path('/root/autodl-tmp/motion_proj')
    rows=[json.loads(l) for l in (run/'v81_roi_registry.jsonl').read_text().splitlines()];byid={r['roi_id']:r for r in rows}
    high=json.loads((run/'high_overlap_spotcheck_index.json').read_text());reviews=[]
    # 按实际显示过的46张缩略图记录观察，不代填human verdict。
    reasons={0:'wet road / reflection; photometric confound',1:'wet road / reflection; photometric confound',2:'wall plus frame/edge; restrict planar interior',3:'wall plus frame/edge; restrict planar interior',4:'wall plus frame/edge; restrict planar interior',5:'road patch with curb depth boundary',6:'wall panel with seams; potential clean interior',7:'wet/specular road and markings',8:'foreground mesh fence over background wall',9:'curb/grass/road mixture',10:'curb/grass/road mixture',11:'asphalt road; ground stratum only',12:'asphalt road; ground stratum only',13:'wall/eave boundary',14:'foreground mesh fence over background',15:'wall/eave boundary',16:'foreground fence plus depth boundary',17:'foreground sign/fence over ground',18:'curb and pedestrian edge',19:'wall panel with markings; candidate planar interior',20:'same wall panel adjacent frame; not independent log',21:'trees/grass/curb mixture',22:'bicycle and pavement mixture',23:'ground near wall; boundary requires exclusion',24:'foreground guardrail over road',25:'foreground guardrail over road',26:'pavement pattern plus shadow; ground stratum',27:'pavement pattern plus shadow; ground stratum',28:'curb/steps/foreground poles',29:'foreground railing over walkway',30:'pavement/road and shadow boundary',31:'curb/grass/road mixture',32:'curb/grass/road mixture',33:'sign/fence over grass',34:'vegetation; not planar low-texture target',35:'grass/road boundary',36:'grass surface; excluded from hard-surface plane study',37:'grass surface; excluded from hard-surface plane study',38:'grass and pavement boundary',39:'grass surface; excluded from hard-surface plane study',40:'ropes/poles foreground and grass',41:'equipment and fence foreground',42:'vegetation; not planar target',43:'hedge/curb/road mixture',44:'tree/grass/curb mixture',45:'foreground tree/car over wall'}
    for i,r in enumerate(high):
        note=reasons.get(i,'manual image screen pending')
        category='PLANAR_INTERIOR_CANDIDATE' if i in [2,3,4,6,19,20] else 'GROUND_ONLY_CANDIDATE' if i in [11,12,26,27] else 'CONFOUND_EXCLUDE_MAIN'
        reviews.append({'roi_id':r['roi_id'],'source':f'high_overlap_spotcheck_{i//16}.png index {i}','reviewer':'assistant_visual_spotcheck','observation':note,'visual_screen':category,'human_verdict':None})
    reviews.extend([
      {'roi_id':'scene-0626_9a9c05fe_CAM_BACK_LEFT_13','reviewer':'assistant_visual_spotcheck','observation':'planar sign, clear texture; used only for synthetic intervention mask','visual_screen':'PLANAR_INTERIOR_CANDIDATE','human_verdict':None},
      {'roi_id':'scene-0632_fd5b6a5c_CAM_FRONT_RIGHT_02','reviewer':'assistant_visual_spotcheck','observation':'low-gradient wall/slatted panel with held-out planar support; boundary candidate','visual_screen':'PLANAR_INTERIOR_CANDIDATE','human_verdict':None},
      {'roi_id':'scene-0800_a4354e58_CAM_BACK_LEFT_13','reviewer':'assistant_visual_spotcheck','observation':'dark vertical panel; flag shadow/exposure as possible confound','visual_screen':'EXPOSURE_CONFOUND_CANDIDATE','human_verdict':None}])
    (run/'spot_checks.json').write_text(json.dumps(reviews,indent=2,ensure_ascii=False))
    for rev in reviews:
        if rev['roi_id'] in byid:byid[rev['roi_id']]['visual_screen']=rev['visual_screen'];byid[rev['roi_id']]['visual_note']=rev['observation']
    for r in rows:r.setdefault('visual_screen','UNREVIEWED');r['scientific_acceptance']='NOT_ESTABLISHED';r['human_verdict']=None
    (run/'v81_roi_registry.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));pq.write_table(pa.Table.from_pylist(rows),run/'texture_overlap_lidar_prior_stats.parquet')
    # 按log/粗语义/range bucket匹配，不能把它升级为因果结果。
    groups=collections.defaultdict(lambda:collections.defaultdict(list))
    for r in rows:
        if not r['cohort'] or r['visual_screen']=='CONFOUND_EXCLUDE_MAIN':continue
        bucket=int(np.digitize(r['range_m'],[10,20,40,80]))
        groups[(r['log'],r['semantic'],bucket)][r['cohort']].append(r)
    matches=[]
    for (log,semantic,bucket),cells in sorted(groups.items()):
        if all(c in cells for c in ['C00','C10','C01','C11']):
            chosen={c:sorted(rs,key=lambda r:r['roi_id'])[0]['roi_id'] for c,rs in cells.items()}
            matches.append({'log':log,'semantic':semantic,'range_bucket':bucket,'cases':chosen,'selection':'metadata order, same log/semantic/range bucket; no model outcomes','status':'GEOMETRICALLY_MATCHED_EXPLORATORY; unreviewed cells remain provisional'})
    (run/'matched_cohorts.json').write_text(json.dumps(matches,indent=2))
    arrays=list((run/'reference_geometry').glob('*.npz'));disjoint=0
    for path in arrays:
        with np.load(path) as d:
            if str(d['input_sample']) in d['reference_samples'].tolist():raise RuntimeError('input/reference sample leak: '+path.name)
            disjoint+=1
    # 模型前queue列出真实路径；默认dry，不自动执行。
    first=sorted((run/'input_manifests').glob('*.json'))[0]
    preview=[]
    py='/root/autodl-tmp/envs/worldsim-v81/bin/python'
    for method in ['dvgt','vggt','dggt']:
        dest=f'/root/autodl-tmp/runs/worldsim_v81/WS-V81-GPU-P2-01/{method}/{first.stem}/full6'
        preview.append({'method':method,'argv':[py,str(repo/'scripts/run_worldsim_v81_inference.py'),'--method',method,'--variant','full6','--manifest',str(first),'--out',dest], 'execute_requires':'explicit --execute after GPU is opened','gpu_phase':'first two primary; DGGT extension conditional'})
    (run/'gpu_pilot_commands.json').write_text(json.dumps(preview,indent=2))
    interventions=[]
    for rid in ['scene-0626_9a9c05fe_CAM_BACK_LEFT_13','scene-0632_fd5b6a5c_CAM_FRONT_RIGHT_02','scene-0139_7e27d5c0_CAM_FRONT_LEFT_10']:
        if rid not in byid:continue
        row=byid[rid]
        for method in ['dvgt','vggt']:
            for variant in ['full6','sparse3','sparse2','temporal18']:
                interventions.append({'method':method,'roi_id':rid,'window_id':row['window_id'],'variant':variant,'anchor_camera':row['camera'],'manifest':str(run/'input_manifests'/f"{row['window_id']}.json"),'status':'WAIT_GPU','selection':'visual/evidence-only before first model output'})
    (run/'gpu_interventions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in interventions))
    for name in ['official_assets.json','meta_dvgt.json','meta_vggt.json','meta_dggt.json']:
        shutil.copy2(base/name,run/name)
    versions={}
    for name in ['torch','torchvision','numpy','Pillow','huggingface_hub','einops','safetensors','iopath','open3d','gsplat','pyarrow','ijson','scipy','opencv-python']:
        try:versions[name]=md.version(name)
        except md.PackageNotFoundError:versions[name]=None
    (run/'runtime_versions.json').write_text(json.dumps(versions,indent=2))
    summary=json.loads((run/'atlas_summary.json').read_text());summary['matched_blocks']=len(matches);summary['matched_logs']=len({r['log'] for r in matches});summary['reference_disjoint_checks']=disjoint;summary['visual_spotcheck_count']=len(reviews);summary['visual_exclusions']=sum(r['visual_screen']=='CONFOUND_EXCLUDE_MAIN' for r in reviews);summary['cohort_semantics']={c:dict(collections.Counter(r['semantic'] for r in rows if r['cohort']==c)) for c in ['C00','C10','C01','C11']}
    summary['controls']={}
    for sem in ['ground_plane','non_ground_plane_candidate']:
        vals=[r['plane_control_mae_m'] for r in rows if r['semantic']==sem and r.get('plane_control_mae_m') is not None]
        summary['controls'][sem]={'valid':len(vals),'median_mae_m':statistics.median(vals)}
    (run/'atlas_summary.json').write_text(json.dumps(summary,indent=2))
    status={'task':'WS-V81-CPU-01','run':run.name,'status':'CPU_COMPLETE_WAIT_GPU','time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'failure_ledger_refs':['V74-H2-F11','V74-H2-F09','V74-H2-F10','V74-F01'],'failure_ledger_delta':['V81-F01','V81-F02'],'cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),'memory_max':Path('/sys/fs/cgroup/memory.max').read_text().strip(),'tests':'see validation.json','model_parameter_checks':'DVGT/VGGT/DGGT strict meta load pass','sota_inference_count':0,'human_verdict':None,'auto_resume':False,'v82_decision':'NO_GO_PENDING_EVIDENCE','gpu_suggestion':'2x48GB for independent inference, or 1x80GB sequential; RAM>=64GB, CPU>=8'}
    (run/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
