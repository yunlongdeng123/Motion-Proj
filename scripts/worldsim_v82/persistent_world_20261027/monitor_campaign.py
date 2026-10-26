"""One read-only campaign health snapshot; never starts or stops a worker."""
import datetime,json,os,shutil,subprocess
from pathlib import Path
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')

def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None

def identity(record):
    if not record:return None
    try:
        fields=Path(f'/proc/{record["pid"]}/stat').read_text().split()
        return dict(pid=record['pid'],expected_start_ticks=str(record['start_ticks']),actual_start_ticks=fields[21],state=fields[2],alive=fields[21]==str(record['start_ticks']) and fields[2]!='Z')
    except FileNotFoundError:return dict(pid=record['pid'],expected_start_ticks=str(record['start_ticks']),alive=False)

def main():
    result=dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),quota=read(ROOT/'quota_state.json'),controllers={})
    for family in ['native_scene_training_v1','motion_suite_execution_v1','benchmark_execution_v1','ego_speed_clock_control_v1']:
        records={}
        for path in (ROOT/family).glob('*/status.json'):
            state=read(path)
            if not state:continue
            item={key:state.get(key) for key in ['phase','completed_native_cases','completed_backend_cases','completed_native_families','completed_fixed_backend_families','error']}
            item.update(parent=identity(state),child=identity(state.get('child')))
            if family=='native_scene_training_v1':
                run=path.parent/'formal30k';item.update(latest_complete_node=read(run/'latest_complete_node.json'),returned=read(run/'returned.json'),cuda_allocator_peak=read(run/'cuda_memory_peak.json'))
                steps=run/'paired_steps.jsonl'
                if steps.exists():
                    with steps.open('rb') as stream:
                        stream.seek(max(0,steps.stat().st_size-32768));lines=stream.read().splitlines()
                    for line in reversed(lines):
                        try:row=json.loads(line);break
                        except json.JSONDecodeError:continue
                    else:row={}
                    item['last_step']=dict(iteration=row.get('iteration'),loss_hybrid_total=row.get('losses',{}).get('loss_hybrid_total'))
            records[path.parent.name]=item
        result['controllers'][family]=records
    mask=read(ROOT/'benchmark_mask_preparation_v1/status.json');result['mask_preparation']=dict(state=mask,parent=identity(mask),child=identity(mask.get('child'))) if mask else None
    peak=read(ROOT/'gpu_peak.json');result['NVML_1s_peak']=peak;result['gpu_monitor_identity']=identity(peak)
    result['disk_bytes']=dict(zip(['total','used','free'],shutil.disk_usage(ROOT)))
    result['GPU_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv'],text=True)
    path=ROOT/'monitor_state.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2));os.replace(temp,path)
    with (ROOT/'monitor_history.jsonl').open('a') as stream:stream.write(json.dumps(result)+'\n')
    print(json.dumps(dict(checked_at_utc=result['checked_at_utc'],disk_free_GiB=result['disk_bytes']['free']/2**30,GPU_processes=result['GPU_processes'],NVML_peak_MiB=peak.get('peak_used_mib') if peak else None)))

if __name__=='__main__':main()
