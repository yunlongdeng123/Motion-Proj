"""Original GT/ego metadata compared with fixed road-point match differences.

No pose correction, model fitting, point/frame exclusion, or GPU execution.
"""
from pathlib import Path
import argparse,csv,hashlib,json,time
import numpy as np
from scipy.spatial.transform import Rotation


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def csvwrite(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.root;assert not a.output.exists();start=time.time()
    B=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs');frames=[60,70,110,145,165];paths=[B/f'observations/{x:03d}.json' for x in frames];observations=[read(x) for x in paths]
    registry=read(B/'track_registry.json');aliases={t:v.get('alias',t) for t,v in registry['actors'].items()};A={z['alias']:t for t,z in registry['actors'].items() if z.get('alias') in ['a9','a20']}
    cache=r/'audit/ground_contact_prior_raw_GT_v2/GT_bottom_and_road_evidence.npz'
    with np.load(cache) as z:
        bottom_primary={(raw,alias):z[f'{raw:03d}_{alias}_bottom_world'].mean(0) for raw in frames for alias in ['a9','a20']}
    times=np.asarray([o['timestamp_micros'] for o in observations],np.float64)/1e6;ego=[]
    for raw,o in zip(frames,observations):
        T=np.asarray(o['ego_world']).reshape(4,4);rpy=Rotation.from_matrix(T[:3,:3]).as_euler('xyz',degrees=True)
        ego.append(dict(raw=raw,timestamp_micros=int(o['timestamp_micros']),X_world=float(T[0,3]),Y_world=float(T[1,3]),Z_world=float(T[2,3]),roll_deg=float(rpy[0]),pitch_deg=float(rpy[1]),yaw_deg=float(rpy[2])))
    dictionaries=[{b['track']:b for b in o['boxes']} for o in observations];common=sorted(set.intersection(*(set(d) for d in dictionaries)))
    # Fixed proximity to the pair8 road location; no use of Z change for selecting tracks.
    anchor=np.array([2322.58898561005,1368.3545221066024]);selection=[]
    for track in common:
        distance=float(np.linalg.norm(np.asarray(dictionaries[2][track]['center_world'])[:2]-anchor))
        if distance<=30 or track in A.values():selection.append((distance,track))
    selection.sort();records=[];summaries=[];details={}
    for distance,track in selection:
        rows=[];centre=[];bottom=[];primary_error=[]
        for raw,o,d in zip(frames,observations,dictionaries):
            b=d[track];T=np.asarray(b['world_from_actor'],np.float64).reshape(4,4);dimensions=np.asarray(b['dimensions']);point=T[:3,3]-T[:3,2]*dimensions[2]/2
            alias=aliases.get(track,track);using_primary=alias in ['a9','a20']
            error=float(abs(point-bottom_primary[(raw,alias)]).max()) if using_primary else None
            if using_primary:point=bottom_primary[(raw,alias)]
            centre.append(T[:3,3]);bottom.append(point);primary_error.append(error)
            row=dict(raw=raw,track=track,alias=alias,timestamp_micros=int(o['timestamp_micros']),centre_X=float(T[0,3]),centre_Y=float(T[1,3]),centre_Z=float(T[2,3]),
                bottom_X=float(point[0]),bottom_Y=float(point[1]),bottom_Z=float(point[2]),bottom_primary_raw_GT_cache=using_primary,obs_bottom_vs_rawGT_max_abs_m=error,
                length_m=float(dimensions[0]),width_m=float(dimensions[1]),height_m=float(dimensions[2]),distance_XY_to_pair8_anchor_at110_m=distance)
            rows.append(row);records.append(row)
        centre=np.asarray(centre);bottom=np.asarray(bottom);speed=np.linalg.norm(np.diff(centre[:,:2],axis=0),axis=1)/np.diff(times)
        maxdisp=float(np.linalg.norm(centre[:,None,:2]-centre[None,:,:2],axis=-1).max());dZ={f'{t:03d}_minus_{s:03d}':float(bottom[frames.index(t),2]-bottom[frames.index(s),2]) for t in [60,70] for s in [110,145,165]}
        summary=dict(track=track,alias=aliases.get(track,track),distance_at110_XY_m=distance,observed_centre_max_XY_displacement_m=maxdisp,
            interval_XY_speed_mps=speed.tolist(),max_observed_interval_XY_speed_mps=float(speed.max()),bottom_Z_range_m=float(np.ptp(bottom[:,2])),
            centre_XY_stationary_proxy=bool(maxdisp<=.30 and speed.max()<=.20),stationary_proxy_not_physical_proof=True,
            bottom_Z_target_minus_reference=dZ,obs_bottom_vs_primary_GT_max_abs_m=max(x for x in primary_error if x is not None) if any(x is not None for x in primary_error) else None)
        summaries.append(summary);details[track]=dict(centres=centre,bottoms=bottom,rows=rows)
    pair_summary=read(r/'audit/height_pair_Brown_cpu_v1/summary.json');comparisons=[]
    for pstats in pair_summary['stats']:
        if pstats['XY_radius_m']!=.10:continue
        t,s=pstats['target_raw'],pstats['reference_raw'];i,j=frames.index(t),frames.index(s)
        for track,vals in details.items():
            centre=vals['centres'];bottom=vals['bottoms'];dt=abs(times[i]-times[j]);dz=float(bottom[i,2]-bottom[j,2]);road=pstats['target_minus_reference_Z_m']['median']
            comparisons.append(dict(target_raw=t,reference_raw=s,track=track,alias=aliases.get(track,track),road_matched_pairs=pstats['matches'],road_delta_Z_median_m=road,
                GT_bottom_delta_Z_m=dz,road_minus_GT_bottom_delta_m=road-dz,GT_centre_delta_X_m=float(centre[i,0]-centre[j,0]),GT_centre_delta_Y_m=float(centre[i,1]-centre[j,1]),
                GT_centre_XY_distance_m=float(np.linalg.norm(centre[i,:2]-centre[j,:2])),interval_seconds=float(dt),GT_XY_mean_speed_mps=float(np.linalg.norm(centre[i,:2]-centre[j,:2])/dt),
                ego_delta_Z_m=ego[i]['Z_world']-ego[j]['Z_world']))
    a.output.mkdir(parents=True);csvwrite(a.output/'ego_pose_metadata.csv',ego);csvwrite(a.output/'GT_track_frame_metadata.csv',records);csvwrite(a.output/'road_vs_GT_bottom_pair_differences.csv',comparisons)
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        pairkeys=[(t,s) for t in [60,70] for s in [110,145,165]];fig,ax=plt.subplots(figsize=(11,4),layout='constrained');x=np.arange(6)
        vals=np.array([next(q['road_delta_Z_median_m'] for q in comparisons if q['target_raw']==t and q['reference_raw']==s) for t,s in pairkeys]);ax.bar(x-.25,100*vals,width=.24,label='road matched median')
        for shift,alias in [(0,'a9'),(.25,'a20')]:
            v=[next(q['GT_bottom_delta_Z_m'] for q in comparisons if q['target_raw']==t and q['reference_raw']==s and q['alias']==alias) for t,s in pairkeys];ax.bar(x+shift,np.asarray(v)*100,width=.24,label=alias+' original GT bottom')
        ax.set(xticks=x,xticklabels=[f'{t:03d}-{s:03d}' for t,s in pairkeys],ylabel='target-reference world Z (cm)',title='Unmodified metadata and point differences; no pose correction');ax.grid(axis='y',alpha=.2);ax.legend();fig.savefig(a.output/'road_vs_original_GT_bottom.png',dpi=160);plt.close(fig)
    except ImportError:pass
    report=dict(schema='R10.original_GT_ego_height_stability_CPU.v1',complete=True,CPU_only=True,nativeF=0,nativeB=0,updates=0,frames=frames,
        input_observations=[pin(p) for p in paths],registry=pin(B/'track_registry.json'),original_GT_bottom_cache=pin(cache),road_pairs=pin(r/'audit/height_pair_Brown_cpu_v1/summary.json'),
        proximity_selection='all GT tracks present in all five frames with centre within30m of the fixed pair8 XY at raw110; A9/A20 included explicitly. No Z/residual selection.',
        stationary_proxy_rule='Observed5frame XY maxdisplacement<=.30m AND every interval mean XYspeed<=.20m/s; a descriptor, not an input filter or proof of physical stationarity.',
        ego_metadata=ego,track_summaries=summaries,pair_comparisons=comparisons,
        interpretation='Common world-height changes in nearly stationary XY tracks support investigating registration/metadata or annotation consistency. Ego height can also change because the vehicle moves on a slope. GT bottom is not dense road ground truth. A box-local normal height is not identical to world-vertical height; rotations are retained here.',
        cause_not_uniquely_identified=True,pose_adjusted=False,frames_deleted=False,GT_or_assets_modified=False,geometry_approved=False,seconds=time.time()-start)
    (a.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(complete=True,summary=pin(a.output/'summary.json'),ego=ego,tracks=summaries),ensure_ascii=False))


if __name__=='__main__':main()
