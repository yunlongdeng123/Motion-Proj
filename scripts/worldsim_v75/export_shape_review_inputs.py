"""导出真实RGB、全时序和统一选帧依据；不把策略速度当作生成世界真值。"""
from pathlib import Path
import json,shutil
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-SHAPE-FEEDBACK-01/20260920-conditioning-r2')
ARMS=['dvgt_class_prior','dvgt_visible_extent']


def main():
    review=ROOT/'review';comparison=json.loads((review/'comparison.json').read_text());assert comparison['status']=='complete'
    for case in comparison['cases']:
        folder=ROOT/case['log_id'];frozen=json.loads((folder/'frozen_state_protocol.json').read_text())
        out=review/case['log_id'][:8];out.mkdir(exist_ok=True)
        audit=json.loads(Path(frozen['audit_case_result']).read_text())
        base=Path(frozen['base']);shutil.copy2(base/'initial_rgb.png',out/'initial.png')
        rows=[json.loads((folder/arm/'decisions.json').read_text()) for arm in ARMS]
        # 两例采用相同规则；完整15步曲线同时保留，单帧不作为总体排名。
        index=max(range(1,15),key=lambda i:abs(rows[1][i]['policy']['acceleration_mps2']-rows[0][i]['policy']['acceleration_mps2']))
        f=rows[0][index]['observation_frame'];inputs=[]
        for arm,decisions in zip(ARMS,rows):
            image=Image.fromarray(np.load(folder/arm/'generated.npy',mmap_mode='r')[f])
            lead=decisions[index]['policy']['lead']
            if lead:ImageDraw.Draw(image).rectangle(lead['box'],outline='#20d7ac',width=4)
            image.save(out/f'{arm}.jpg',quality=96)
            dense=json.loads((folder/arm/'dense_replay.json').read_text())
            inputs.append({'arm':arm,'readout':next(x for x in audit['rows'] if x['arm']==arm),
                'selected_observation':decisions[index],
                'time_s':[r['observation_frame']/30 for r in decisions],
                'actual_acceleration_mps2':[r['policy']['acceleration_mps2'] for r in decisions],
                'reference_acceleration_mps2':[r['oracle_acceleration_mps2'] for r in decisions],
                'dense_time_s':[r['time_s'] for r in dense['frames']],
                'dense_target_clearance_m':[r['target_reference_clearance_m'] for r in dense['frames']],
                'dense_progress_m':[r['route_progress_lateral'][0] for r in dense['frames']]})
        item={'log_id':case['log_id'],'frame_selection':'largest paired applied-action difference after shared startup; both complete time series shown',
              'selected_frame':f,'selected_time_s':f/30,'arms':inputs,'summary':case,
              'camera_warning':'green boxes are actual policy detections; no GT overlay or verified generated-camera assumption',
              'human_verdict':None}
        (out/'figure_inputs.json').write_text(json.dumps(item,indent=2)+'\n')
        print(json.dumps({'log':case['log_id'],'selected_frame':f,'action_class':rows[0][index]['policy']['acceleration_mps2'],'action_extent':rows[1][index]['policy']['acceleration_mps2']}),flush=True)


if __name__=='__main__':main()
