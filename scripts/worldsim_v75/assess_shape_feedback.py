"""逐帧复现两任务形状对照的实际动作，完整评价固定参考并生成真实视频对比。"""
from pathlib import Path
import json,os,time
import av
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from following_geometry import scene
from raster_ground import RasterGround
from rgb_idm_policy import RGBIDMPolicy
from run_approach_state_control import rollout

ROOT=Path('/root/autodl-tmp/runs/worldsim_v75')
RUN=ROOT/'WS-V75-SHAPE-FEEDBACK-01/20260920-conditioning-r2'
ARMS=['dvgt_class_prior','dvgt_visible_extent']


def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    queue=json.loads((RUN/'queue_result.json').read_text());assert queue['status']=='complete' and len(queue['runs'])==4
    out=RUN/'review';assert not out.exists();out.mkdir()
    began=time.monotonic();cases=[]
    for folder in sorted(p for p in RUN.iterdir() if p.is_dir() and (p/'frozen_state_protocol.json').exists()):
        protocol=json.loads((folder/'frozen_state_protocol.json').read_text());parent=Path(protocol['source_generated_run'])
        base=Path(protocol['base']);reference=scene(base);tr=np.load(base/'trajectory.npz')
        target=next(t for t in reference['tracks'] if t['id']==protocol['target'] and 0 in t['frames'])
        extrinsic=np.linalg.inv(tr['ego_world'][0])@tr['camera_world'][0]
        policy=RGBIDMPolicy(tr['ego_world'][:,:3,3],tr['K'],extrinsic,[0,0,0],detector=False)
        terrain=RasterGround(base).mesh_for_route(tr['ego_world'][:,:2,3])
        baseline_rows=json.loads((parent/'gt_clean/decisions.json').read_text())
        baseline_progress=(policy.route.project([baseline_rows[-1]['ego_state']['x_m'],baseline_rows[-1]['ego_state']['y_m']])[0]
                           -policy.route.project(tr['ego_world'][0,:2,3])[0])
        audit=json.loads(Path(protocol['audit_case_result']).read_text())
        rows=[];all_decisions={}
        for arm in ['gt_clean','oracle_shape_dvgt']+ARMS:
            directory=parent/('dvgt_metric' if arm=='oracle_shape_dvgt' else arm) if arm not in ARMS else folder/arm
            result=json.loads((directory/'result.json').read_text());decisions=json.loads((directory/'decisions.json').read_text())
            assert result['status']=='complete' and len(decisions)==15
            assert decisions[0]['command']==baseline_rows[0]['command']
            condition=json.loads((directory/'condition_scene.json').read_text())
            trajectory,cameras=rollout(base,tr,policy,terrain,condition,reference,target,decisions,True)
            error=float(np.max(abs(cameras-np.load(directory/'camera_trajectory.npz')['camera_world'])))
            assert error<=1e-7,error
            physical_diffs=np.array([r['acceleration_difference_mps2'] for r in decisions])
            actual=np.array([r['policy']['acceleration_mps2'] for r in decisions])
            gt=np.array([r['policy']['acceleration_mps2'] for r in baseline_rows])
            value={'arm':arm,'progress_m':trajectory['progress_m'],
                   'progress_change_vs_gt_m':trajectory['progress_m']-baseline_progress,
                   'mean_abs_action_change_vs_gt_mps2':float(np.mean(abs(actual-gt))),
                   'median_abs_action_error_at_own_ego_mps2':float(np.median(abs(physical_diffs))),
                   'mean_abs_action_error_at_own_ego_mps2':float(np.mean(abs(physical_diffs))),
                   'max_underbraking_mps2':float(np.maximum(physical_diffs,0).max()),
                   'max_overbraking_mps2':float(np.maximum(-physical_diffs,0).max()),
                   'final_target_reference_clearance_m':trajectory['final_target_reference_clearance_m'],
                   'final_speed_mps':trajectory['final_speed_mps'],'overlap_frames':trajectory['overlap_frames'],
                   'camera_replay_max_abs':error,'camera_replay_frames':117,
                   'max_route_lateral_m':max(abs(x['route_progress_lateral'][1]) for x in trajectory['frames']),
                   'source':str(directory)}
            if arm in ARMS:
                original=next(x for x in audit['rows'] if x['arm']==arm)
                value.update(direct_control=original['direct_control'],dimensions_m=original['dimensions_m'],
                             center_error_m=original['center_error_m'],near_gap_error_m=original['near_gap_error_m'])
                save(directory/'dense_reference_result.json',{'status':'passed' if not trajectory['overlap_frames'] and value['max_route_lateral_m']<=1 else 'failed',
                     'frames':117,'replay_max_abs':error,'overlap_frames':trajectory['overlap_frames'],'max_route_lateral_m':value['max_route_lateral_m'],'human_verdict':None})
                save(directory/'dense_replay.json',trajectory)
            all_decisions[arm]=decisions;rows.append(value)
        # 真实完整视频并排展示；只在实际观察帧绘制对应检测框。
        videos=[np.load(folder/arm/'generated.npy',mmap_mode='r') for arm in ARMS]
        assert all(v.shape==(117,704,1280,3) for v in videos)
        destination=out/f'{protocol["source_log"][:8]}-shape-pair.mp4'
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
        with av.open(str(destination),'w') as writer:
            stream=writer.add_stream('libx264',rate=30);stream.width=1280;stream.height=420;stream.pix_fmt='yuv420p';stream.options={'crf':'18'}
            for f in range(117):
                sheet=Image.new('RGB',(1280,420),'#152536');d=ImageDraw.Draw(sheet)
                for i,arm in enumerate(ARMS):
                    latest=max((r for r in all_decisions[arm] if r['observation_frame']<=f),key=lambda r:r['observation_frame'])
                    rgb=Image.fromarray(videos[i][f]);lead=latest['policy']['lead']
                    if latest['observation_frame']==f and lead:ImageDraw.Draw(rgb).rectangle(lead['box'],outline='#31d8ba',width=3)
                    d.text((i*640+8,5),['Fixed class shape','Visible-extent fit'][i]+f' | {protocol["source_log"][:8]} t={f/30:.2f}s',font=font,fill='white')
                    d.text((i*640+8,28),f"action={latest['policy']['acceleration_mps2']:+.2f} m/s2 | last RGB observation f={latest['observation_frame']}",font=font,fill='white')
                    sheet.paste(rgb.resize((640,352)),(i*640,64))
                for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(sheet),format='rgb24')):writer.mux(packet)
            for packet in stream.encode():writer.mux(packet)
        with av.open(str(destination)) as reader:decoded=sum(1 for _ in reader.decode(video=0))
        assert decoded==117
        snapshots=Image.new('RGB',(1280,4*400),'#152536');d=ImageDraw.Draw(snapshots)
        for index,f in enumerate([4,36,76,108]):
            for j,arm in enumerate(ARMS):
                row=next(x for x in all_decisions[arm] if x['observation_frame']==f)
                rgb=Image.fromarray(videos[j][f]);lead=row['policy']['lead']
                if lead:ImageDraw.Draw(rgb).rectangle(lead['box'],outline='#31d8ba',width=3)
                d.text((j*640+8,index*400+4),f"{arm} | t={f/30:.2f}s | a={row['policy']['acceleration_mps2']:+.2f}",font=font,fill='white')
                snapshots.paste(rgb.resize((640,352)),(j*640,index*400+40))
        snapshots.save(out/f'{protocol["source_log"][:8]}-observations.jpg',quality=94)
        case={'log_id':protocol['source_log'],'rows':rows,'decoded_pair_frames':decoded,
              'near_gap_difference_m':protocol['near_gap_difference_m'],'direct_progress_difference_m':protocol['direct_progress_difference_m'],
              'generated_progress_difference_m':rows[-1]['progress_m']-rows[-2]['progress_m'],
              'human_verdict':None}
        cases.append(case);print(json.dumps(case),flush=True)
    assert not torch.cuda.is_initialized()
    result={'status':'complete','cases':cases,'new_generated_frames':468,'replay_verified_frames':936,
            'new_gpu_calls':0,'cuda_initialized':False,'wall_s':time.monotonic()-began,'human_verdict':None,'failure_ledger_delta':'none'}
    save(out/'comparison.json',result)


if __name__=='__main__':main()
