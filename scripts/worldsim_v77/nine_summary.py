"""汇总已完成本轮的实际分母/资源；不计算自动视觉通过率。"""
from nine_common import *
import numpy as np

def main():
    reg=read(ROOT/'effective_registration.json');drive=read(ROOT/'drive_state.json');omega=read(ROOT/'omega_state.json');sequence=read(ROOT/'sequence_state.json')
    assert all(x['state']=='complete' for x in [drive,omega,sequence])
    cases=[]
    for s in reg['scenes']:
        b=ROOT/s['name'];q=read(ROOT/'review'/s['name']/'summary.json');ori=read(b/'orientation.json');src=read(b/'asset/registration.json')['source']
        dr=[x for x in drive['completed'] if x['scene']==s['name']];om=[x for x in omega['completed'] if x['scene']==s['name']];rows=q['rows']
        a=[r for r in rows if r['asset_pixels']>0];visible=[r['visible_asset_pixels']/r['asset_pixels'] for r in a]
        cov=[v['geometry_coverage'] for x in om for v in x['views']];mcov=[v['deletion_mask_geometry_coverage'] for x in om for v in x['views'] if v['deletion_mask_geometry_coverage'] is not None]
        cases.append(dict(scene=s['name'],actor=s['actor'],primary_camera=q['primary_camera'],reference=src,selected_yaw=ori['selected_yaw_deg'],orientation_ambiguous=ori['ambiguous'],
            visible_gt_viewtimes=q['gt_visible_view_times'],empty_mask_visible_viewtimes=q['mask_missing_when_gt_visible'],drive_generated_windows=sum(r['generated'] for r in dr),drive_empty_windows=sum(not r['generated'] for r in dr),
            omega_forwards=len(om),actor_layers=q['actual_layer_renders'],geometry_coverage_median=float(np.median(cov)),mask_geometry_coverage_median=float(np.median(mcov)) if mcov else None,
            asset_depth_pass_median=float(np.median(visible)),asset_depth_pass_min=float(min(visible)),video_files=len(q['video_frames']),video_frames=sum(q['video_frames'].values()),human_verdict=None))
    out=dict(task_id=reg['task_id'],run_id='r1',scenes=cases,counts=dict(scenes=9,source_rgb_viewtimes=1620,sam_streams=len(read(ROOT/'mask_state.json')['completed']),
        drive_window_records=len(drive['completed']),drive_generated_windows=sum(x['generated'] for x in drive['completed']),drive_empty_windows=sum(not x['generated'] for x in drive['completed']),
        omega_forwards=len(omega['completed']),actor_layers=sum(x['actor_layers'] for x in cases),orientation_controls=18,hunyuan_shape_calls=9,hunyuan_paint_calls=9,video_files=sum(x['video_files'] for x in cases),video_frames=sum(x['video_frames'] for x in cases),training_steps=0),
        resources=dict(gpu='RTX3090 24GiB',sequence_jobs=sequence['completed'],drive_seconds=drive['seconds'],drive_peak_gib=max(x['peak_gib'] for x in drive['completed']),omega_seconds=omega['seconds'],omega_peak_gib=max(x['peak_gib'] for x in omega['completed'])),
        role='all nine previously exposed development cases; not independent generalization',human_verdict=None,failure_ledger_refs=['V77-F02'])
    dump(ROOT/'actual_summary.json',out);print('ACTUAL_SUMMARY',out['counts'],flush=True)
if __name__=='__main__':main()
