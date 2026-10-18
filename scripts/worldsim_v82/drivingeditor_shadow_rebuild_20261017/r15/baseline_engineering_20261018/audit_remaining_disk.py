"""Read-only forecast for the added5K durability nodes; preserves all assets."""
from pathlib import Path
import json,sys,shutil,datetime
E=Path(__file__).resolve().parent;BASE=Path('/root/autodl-tmp/drivingeditor_repro')
sys.path.insert(0,str(BASE/'campaign/followup_scene_prep_20261013/tools/followup'))
from contracts import disk_forecast,build_config,scene_spec
def main():
 run=BASE/'campaign/r15_baseline_owner5_20261018/runs/scene0008_089_137_hybrid_30000_seed0_GN11_resume20k'
 cfg=json.loads((run/'effective_config.json').read_text());live=json.loads((run/'status.json').read_text())
 counts=(int(live['static_points']),int(live['memory']['extra']['render_dynamic_points']))
 remaining=sum(s>live['iteration'] for s in [25000,30000])
 a=disk_forecast(cfg,actual_counts=counts,remaining_nodes=remaining)
 M=E/'all_vehicle_pipeline/metadata/scene0008_089_137'
 bcfg=build_config(json.loads((M/'reference_GN11.json').read_text()),scene_spec(json.loads((M/'scene_spec.json').read_text())),BASE/'campaign/r15_baseline_allvehicles_20261018')
 b=disk_forecast(bcfg,remaining_nodes=6)
 # Rigid checkpoint schema has no PVG t/velocity parameters. Use actual latest
 # point counts when available, with all unsaved durability nodes accounted.
 D=E/'direct_instance30k';row={}
 if (D/'steps.jsonl').exists():
  with (D/'steps.jsonl').open('rb') as f:
   f.seek(max(0,(D/'steps.jsonl').stat().st_size-4096));tail=f.read().splitlines()
  row=json.loads(tail[-1])
 saved=[s for s in [1,5000,10000,15000,20000,25000,30000] if (D/f'checkpoint_{s:06d}.pt').is_file()]
 instance_counts=(int(row.get('background_points',counts[0])),int(row.get('instance_points',149072)))
 instance=disk_forecast(cfg,actual_counts=instance_counts,remaining_nodes=7-len(saved))
 required=a['required_free_bytes']+b['required_free_bytes']+instance['required_free_bytes']
 free=shutil.disk_usage(BASE).free
 out=dict(status='READONLY_COMBINED_REMAINING_DISK_FORECAST',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          current_free_bytes=free,estimated_remaining_required_bytes=required,estimated_headroom_bytes=free-required,
          fits_this_forecast=free>=required,A_native_remaining=a,A_instance_estimate=instance,B_native_six_nodes=b,
          instance_durable_steps=saved,instance_latest_logged_counts=instance_counts,
          limitations='Forecast only. A instance can densify beyond current counts; recheck real counts at saves and before B. No assets removed.',training_updates=0)
 (E/'remaining_disk_forecast.json').write_text(json.dumps(out,indent=2))
 print(json.dumps({k:v for k,v in out.items() if k not in ['A_native_remaining','A_instance_estimate','B_native_six_nodes']}))
if __name__=='__main__':main()
