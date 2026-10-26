"""按操作从完整轨迹提出窗口，避免最大框窗口不适合运动编辑。

只读输入标注；新窗口须重新做图像/标定/资产/空间QA，不自动进入正式评测。
"""
import json
from pathlib import Path
import numpy as np

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')


def contiguous(ids):
    longest=current=0;last=None
    for i in ids:
        current=current+1 if last is not None and i==last+1 else 1;longest=max(longest,current);last=i
    return longest


def main():
    manifest=json.loads((ROOT/'evaluation_set_v1/manifest.json').read_text());results=[]
    for scene in manifest['scenes']:
        data=json.loads((ROOT/'evaluation_set_v1/scenes'/scene['context']/'scene.json').read_text())
        tracks={}
        for frame in data['raw_annotations']:
            for obj in frame['front_vehicles']:
                tracks.setdefault(obj['track'],[]).append(dict(raw=frame['raw'],timestamp=frame['timestamp_micros']/1e6,**obj))
        best_motion=None;best_speed=None;best_current=None
        for track,all_rows in tracks.items():
            if len(all_rows)<40:continue
            for start in range(max(0,min(r['raw'] for r in all_rows)-8),min(scene['frames']-49,max(r['raw'] for r in all_rows))+1):
                rows=[r for r in all_rows if start<=r['raw']<start+49];ids=[r['raw'] for r in rows]
                if len(ids)<40 or contiguous(ids)<35 or max(np.diff(ids)-1,default=0)>2:continue
                area=float(np.median([r['area']/16 for r in rows]))
                if area<300:continue
                world=[r for r in rows if r.get('center_world') is not None]
                speed=0.
                if len(world)>=20:
                    dt=np.diff([r['timestamp'] for r in world]);speed=float(np.median(np.linalg.norm(np.diff([r['center_world'] for r in world],axis=0),axis=1)/dt))
                visible=[]
                border=[]
                for r in rows:
                    x,y,w,h=np.array(r['xywh'])/4
                    visible.append(max(0,min(480,x+w/2)-max(0,x-w/2))*max(0,min(320,y+h/2)-max(0,y-h/2))/max(w*h,1))
                    border.append(x-w/2<=2 or x+w/2>=478 or y-h/2<=2 or y+h/2>=318)
                if np.mean(border)>.1:continue
                score=(len(rows),contiguous(ids),float(np.mean(np.array(visible)>=.95)),min(area,5000))
                proposal=dict(target_track=track,window=list(range(start,start+49)),presence=len(rows),longest_contiguous=contiguous(ids),median_area_at480=area,
                              median_world_speed_mps=speed,fully_in_image_fraction=float(np.mean(np.array(visible)>=.95)),
                              border_touch_fraction=float(np.mean(border)),
                              status='input_proposal_requires_window_rescreen_and_asset_geometry_QA',formal_ready=False)
                if best_motion is None or score>best_motion[0]:best_motion=(score,proposal)
                if speed>=1 and (best_speed is None or score>best_speed[0]):best_speed=(score,proposal)
                if speed>=1 and proposal['window']==scene['selected_window'] and (best_current is None or score>best_current[0]):best_current=(score,proposal)
        results.append(dict(context=scene['context'],split=scene['split'],motion_window=best_motion[1] if best_motion else None,
                            moving_target_speed_window=best_speed[1] if best_speed else None,
                            current_window_moving_target=best_current[1] if best_current else None,
                            current_largest_box_window=scene['selected_window'],current_selected_track=scene['selected_track']))
    sequence=[
        dict(request='native reconstruction',module_to_audit=[1,2,3],expected='main copy identity preserved'),
        dict(request='selected vehicle speed factor0.8',module_to_audit=[1],expected='object trajectory changes; background/other objects unchanged'),
        dict(request='selected vehicle lateral1m',module_to_audit=[1,2],expected='old footprint exposed; destination support/collision checked'),
        dict(request='selected vehicle yaw+10deg',module_to_audit=[1],expected='asset identity and surface coverage tracked; unseen surface marked uncertain'),
        dict(request='camera lateral0.5m then return',module_to_audit=[1,2,3],expected='correct disocclusion and cross-view reuse; no forced invented GT'),
        dict(request='delete selected vehicle then restore original',module_to_audit=[1,2,3],expected='persistent request sequence; final source identity and retained regions verified')]
    result=dict(version=1,does_not_use_edit_outputs=True,proposals=results,
                counts=dict(contexts=36,motion_window_proposed=sum(x['motion_window'] is not None for x in results),
                            moving_target_speed_window_proposed=sum(x['moving_target_speed_window'] is not None for x in results),formal_ready=0),
                sequence_template=sequence,failure_ownership={'instance_trajectory_asset':1,'background_missing_or_contaminated':2,'generation_overwrite':3},
                horizonforge_comparison=dict(status='reference_task_design_only',matched_condition_runs=0,superiority_claim=False),
                limitations=['input proposals are not completed cases','turning asset orientation coverage remains required','speed map must remain inside learned asset time support',
                             'camera and destination require actual measured spatial support','new windows require rescreen; old window checks do not transfer automatically'])
    (ROOT/'evaluation_admission_v2/operation_window_proposals.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['counts']))


if __name__=='__main__':main()
