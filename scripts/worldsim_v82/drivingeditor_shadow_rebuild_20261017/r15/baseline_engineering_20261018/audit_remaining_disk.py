"""Read-only forecast for the added5K durability nodes; preserves all assets."""
from pathlib import Path
import json,sys,shutil,datetime,math
E=Path(__file__).resolve().parent;BASE=Path('/root/autodl-tmp/drivingeditor_repro')
sys.path.insert(0,str(BASE/'campaign/followup_scene_prep_20261013/tools/followup'))
from contracts import disk_forecast,build_config,scene_spec
def main():
 run=BASE/'campaign/r15_baseline_owner5_20261018/runs/scene0008_089_137_hybrid_30000_seed0_GN11_resume20k'
 cfg=json.loads((run/'effective_config.json').read_text());live=json.loads((run/'status.json').read_text())
 counts=(int(live['static_points']),int(live['memory']['extra']['render_dynamic_points']))
 remaining=sum(s>live['iteration'] for s in [25000,30000])
 assert remaining==0 and live['status']=='complete','this shared-phase forecast requires the completed A first stage'
 M=E/'all_vehicle_pipeline/metadata/scene0008_089_137'
 B=BASE/'campaign/r15_baseline_allvehicles_20261018/runs/scene0008_089_137_hybrid_30000_seed0_GN11'
 bcfg=json.loads((B/'effective_config.json').read_text());blive=json.loads((B/'status.json').read_text())
 D=Path(json.loads((E/'direct_active_run.json').read_text())['output']);row=json.loads((D/'status.json').read_text())
 saved=[s for s in range(5000,30001,5000) if (D/f'checkpoint_{s:06d}.pt').is_file()]
 bsaved=[s for s in range(5000,30001,5000) if all((B/name.format(s)).is_file() for name in ['chkpnt{}.pth','dynamic_chkpnt{}.pth','env_light_chkpnt{}.pth','training_state{}.pth'])]
 # A live status may already say5000 while its atomic checkpoint is still writing.
 # Account for every unsaved scheduled node; do not drop an in-progress node or
 # call a file absent during a save a completed-training failure.
 anodes=[s for s in range(5000,30001,5000) if s not in saved]
 bnodes=[s for s in range(5000,30001,5000) if s not in bsaved]
 instance_counts=(int(row['background_points']),int(row['instance_points']))
 bcounts=(int(blive['static_points']),int(blive['memory']['extra']['render_dynamic_points']))
 caps=(math.ceil(max(int(cfg['num_pts']),int(cfg['densify_until_num_points']))*1.1),math.ceil(int(cfg['dynamic_densify_until_num_points'])*1.1))
 bcaps=(math.ceil(max(int(bcfg['num_pts']),int(bcfg['densify_until_num_points']))*1.1),math.ceil(int(bcfg['dynamic_densify_until_num_points'])*1.1))
 def forecast(acounts,bcounts):
  ai=disk_forecast(cfg,actual_counts=acounts,remaining_nodes=len(anodes));bi=disk_forecast(bcfg,actual_counts=bcounts,remaining_nodes=len(bnodes))
  # A produces only small preview PNGs at save nodes, and one final32-image/frame export.
  # Its checkpoint is monolithic, so account for a full temporary bundle rather than one native component.
  afinal=49*480*320*72*2 if row.get('state')!='complete' or not (E/'direct_full49/receipt.json').exists() else 0
  asub=ai['checkpoint_bytes']+afinal+ai['logs_bytes']+(ai['forecast_bundle_bytes'] if anodes else 0)
  ai.update(evaluation_and_viewer_bytes=afinal,atomic_component_temporary_bytes=ai['forecast_bundle_bytes'] if anodes else 0,forecast_headroom_bytes=math.ceil(asub*.1),required_free_bytes=math.ceil(asub*1.1),remaining_save_steps=anodes)
  bfinal=49*480*320*17*2 if not (E/'all_vehicle_full49/receipt.json').exists() else 0
  bi['additional_final_png_and_transfer_allowance_bytes']=bfinal;bi['required_free_bytes']+=bfinal;bi['remaining_save_steps']=bnodes
  return ai,bi,ai['required_free_bytes']+bi['required_free_bytes']
 instance,b,required=forecast(caps,bcaps);current_instance,current_b,current_required=forecast(instance_counts,bcounts)
 free=shutil.disk_usage(BASE).free
 out=dict(status='READONLY_COMBINED_REMAINING_DISK_FORECAST',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          current_free_bytes=free,estimated_remaining_required_bytes=required,estimated_headroom_bytes=free-required,
          fits_this_forecast=free>=required,A_native_remaining=dict(remaining_nodes=0,required_free_bytes=0,reason='complete; no future native saves/evaluations pending'),A_instance_estimate=instance,B_native_remaining=b,
          current_counts_required_bytes=current_required,current_counts_headroom_bytes=free-current_required,current_counts_A=current_instance,current_counts_B=current_b,
          instance_durable_steps=saved,instance_latest_logged_counts=instance_counts,
          B_durable_steps=bsaved,B_latest_logged_counts=bcounts,configured_caps_with10percent_overshoot=dict(A=caps,B=bcaps),
          due_unpersisted_nodes=dict(A=[s for s in anodes if s<=row['step']],B=[s for s in bnodes if s<=blive['iteration']]),
          corrections='Completed A native requires0 future nodes; resumed A does not recreate step1; B5K already saved; A checkpoints do not each trigger native49 FP32 export. Full monolithic A temporary save included.',
          limitations='Forecast only; configured point caps have generic-topology overshoot and are not strict maxima. Recheck actual counts and remaining nodes at saves. All assets preserved.',training_updates=0)
 (E/'remaining_disk_forecast.json').write_text(json.dumps(out,indent=2))
 print(json.dumps({k:v for k,v in out.items() if k not in ['A_native_remaining','A_instance_estimate','B_native_remaining','current_counts_A','current_counts_B']}))
if __name__=='__main__':main()
