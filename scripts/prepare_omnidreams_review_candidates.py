"""导出用户待审的6秒原始视频和反事实提案；绝不启动模型。"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_proj.cfbench.geometry import project_box
from prepare_omnidreams_cfbench import pose_at
from add_cfbench_original_videos import make_original

CAMERAS = [('CAM_FRONT','前视'), ('CAM_FRONT_LEFT','左前视'), ('CAM_FRONT_RIGHT','右前视'),
           ('CAM_BACK_LEFT','左后视'), ('CAM_BACK_RIGHT','右后视'), ('CAM_BACK','后视')]
REVIEWS = [
    {'case_id':'CFB-SPEED-EGO-01', 'reviewer':'user', 'scores':None,
     'observations':'factual建筑扭曲、相邻车道车辆有重叠；counterfactual能看出减速，但有开一段、停一下、再开的卡顿感。'},
    {'case_id':'CFB-SPEED-EGO-02', 'reviewer':'user', 'scores':None,
     'observations':'自车原本静止，乘加速倍率不适用，需换case或指令；生成出现车厢幻觉，原始卡车的空车厢被填补。',
     'branch_note':'车厢幻觉所指分支在口述中为“cycle”，未确认；不自行认定存在cycle分支。'},
    {'case_id':'CFB-SPEED-EGO-03', 'reviewer':'user', 'scores':None,
     'observations':'生成建筑歪斜，树枝叶与建筑交叠处模糊；视频太短，看不清减速是否发生。'},
    {'case_id':'CFB-SPEED-ACTOR-01', 'reviewer':'user', 'scores':None,
     'observations':'原始画面看起来像快速倒车，需要明确相机视角和ego身份；factual鬼影严重，counterfactual像速度慢一些的鬼影。'}
]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--refresh',action='store_true',help='仅允许刷新仍在待审的提案，不覆盖推理或已批准配置')
    args=p.parse_args()
    if args.refresh:
        previous=json.loads((args.output/'candidates.json').read_text())
        assert previous['status']=='awaiting_user_review_no_inference'
        assert all(row['approval']['status']=='pending' for row in previous['cases'])
        assert not list(args.output.glob('cases/*/factual.mp4'))
        assert not list(args.output.glob('cases/*/counterfactual.mp4'))
    args.output.mkdir(parents=True,exist_ok=args.refresh)
    old=json.loads((args.inputs/'index.json').read_text())['cases'][:4]
    candidates=[]
    for i,row in enumerate(old):
        case=copy.deepcopy(row['case'])
        parent=case['case_id']
        case['case_id']=parent+'-R2-6S'
        case['status']='awaiting_user_review_no_inference'
        case['parent_case_id']=parent
        case['anchor'].update(rollout_frames=56,clip_frames=61)
        reason='同一场景、目标和事件，窗口由2.3秒延长至6秒。'
        if i==1:
            case['dataset']=copy.deepcopy(old[0]['case']['dataset'])
            reason='原scene-0242自车近乎静止，改用scene-0230运动自车；与第1例共用原始片段、分别测试加速/减速，不算独立场景。'
        case['qualification']['human_verdict']=None
        camera=row['camera_index']
        root=Path(case['dataset']['root'])/case['dataset']['scene_id']
        event=case['anchor']['event_frame']; start=event-5; stop=event+55
        scale=case['intervention']['counterfactual']['speed_scale']
        if case['target']['role']=='ego':
            poses={int(path.stem):np.loadtxt(path) for path in (root/'lidar_pose').glob('*.txt')}
            noun='自车（搭载相机的采集车）'
        else:
            ann=json.loads((root/'instances/instances_info.json').read_text())[case['target']['actor_key']]['frame_annotations']
            poses={int(f):np.array(pose) for f,pose in zip(ann['frame_idx'],ann['obj_to_world'])}
            size=np.median(np.array(ann['box_size']),axis=0)
            noun='其他汽车 #'+case['target']['actor_key']+'（不是自车）'
        times=np.arange(start,stop+1)
        actual=[pose_at(poses,f) for f in times]
        cf=[pose_at(poses,f if f<event else event+(f-event)*scale) for f in times]
        assert all(pose is not None for pose in actual+cf), '轨迹不足，不能外推补齐'
        positions=np.stack([pose[:3,3] for pose in actual])
        pathlength=float(np.linalg.norm(np.diff(positions[5:],axis=0),axis=1).sum())
        delta=float(np.linalg.norm(actual[-1][:3,3]-cf[-1][:3,3]))
        assert pathlength>1 and delta>1, '任务变化不足1米，不自动进入待审提案'
        camera_name,view=CAMERAS[camera]
        scene={'179':'scene-0230','191':'scene-0242','204':'scene-0255'}[case['dataset']['scene_id']]
        action='减速' if scale<1 else '加速'
        target_short='自车' if case['target']['role']=='ego' else '汽车 #'+case['target']['actor_key']
        description=f'从0.5秒开始，将{target_short}沿原轨迹的推进速度设为原来的{scale*100:g}%（{scale:g}×，{action}）；其余对象状态不改。'
        entry={'case_id':case['case_id'],'parent_case_id':parent,'case':case,'scene_name':scene,
               'camera_index':camera,'camera_name':camera_name,'view':view,'target_description':noun,
               'counterfactual_description':description,'revision_reason':reason,
               'source_frame_range':[start,stop],'event_s':.5,'last_frame_s':6.0,
               'planned_generated_frames':181,'planned_fps':30,'tail_padding':0,
               'source_role':'development_candidate_not_independent_test',
               'input_check':{'factual_postevent_path_length_m':pathlength,'target_endpoint_delta_m':delta,
                              'trajectory_available_without_extrapolation':True,'full_collision_road_gate':'not_claimed'},
               'approval':{'status':'pending','reviewer':'user','inference_allowed':False},
               'prior_user_review':REVIEWS[i]}
        out=args.output/'cases'/case['case_id']
        make_original(case,camera,out)
        if case['target']['role']!='ego':
            visible={}
            for name,branch in [('factual',actual),('counterfactual',cf)]:
                boxes=[project_box(root,int(f),camera,pose,size) for f,pose in zip(times,branch)]
                visible[name]={'visible_frames':sum(box is not None for box in boxes),'total_frames':len(boxes),
                               'last_visible_s':max([(n/10) for n,box in enumerate(boxes) if box is not None],default=None)}
            entry['input_check']['target_projection_visibility']=visible
            im=Image.open(root/f'images/{event:03d}_{camera}.jpg').convert('RGB')
            box=project_box(root,event,camera,poses[event],size)
            assert box is not None
            draw=ImageDraw.Draw(im)
            draw.rectangle(box['bbox_xyxy'],outline='yellow',width=6)
            x,y=box['bbox_xyxy'][:2]
            draw.rectangle([x,max(0,y-30),x+170,y],fill='black')
            draw.text((x+3,max(0,y-25)),'TARGET: car #8',fill='yellow')
            im.save(out/'target-reference.jpg',quality=92)
        save(out/'proposal.json',entry)
        candidates.append(entry)
    manifest={'task_id':'WS-V75-OMNI-REVIEW-02','run_id':'20260923-proposal-r1',
              'created_utc':datetime.now(timezone.utc).isoformat(),'status':'awaiting_user_review_no_inference',
              'model_calls':0,'new_ai_video_reviews':0,'case_count':len(candidates),'cases':candidates}
    save(args.output/'candidates.json',manifest)
    save(args.output/'human-review-original-four.json',{'source':'用户2026-09-23聊天口述，保留定性意见，不转换成数值分','reviews':REVIEWS})
    body=['<h1>OmniDreams · 4 个待审 case</h1><p class="notice">只导出6秒原始视频和干预提案，尚未跑新版推理。你确认后才执行。原始参考不是反事实结果。</p>',
          '<p>ego = 搭载相机的采集车；画面中标出的其他车辆不是ego。全部视频正常10Hz播放，未减速、循环或插帧。</p>',
          '<div class="architecture">nuScenes 原始片段 → 指定对象与反事实 → <strong>你的人工确认</strong> → 同seed factual / counterfactual（待运行）</div>',
          '<p><a href="candidates.json">候选参数</a> · <a href="human-review-original-four.json">你对旧版的人工 review</a></p>']
    for row in candidates:
        cid=row['case_id']; base='cases/'+cid
        body += [f'<article id="{cid}"><h2>{cid}</h2>',
                 f'<p>{row["scene_name"]} · {row["camera_name"]}（{row["view"]}）· 目标：{row["target_description"]}</p>',
                 f'<p class="intent"><strong>反事实：</strong>{row["counterfactual_description"]}</p>',
                 '<p class="note">'+row['revision_reason']+'</p>',
                 '<div class="videos"><div><h3>原始 nuScenes · 6秒</h3>'+f'<video controls preload="metadata" playsinline src="{base}/original-nuscenes.mp4"></video></div>'+
                 '<div><h3>factual</h3><div class="placeholder">待你确认后推理</div></div><div><h3>counterfactual</h3><div class="placeholder">待你确认后推理</div></div></div>',
                 f'<p class="note">视频0–0.5秒为公共前缀；之后5.5秒施加干预。事实分支按原始状态继续。源帧{row["source_frame_range"]}。</p>',
                 f'<p>输入轨迹检查：干预后事实行进{row["input_check"]["factual_postevent_path_length_m"]:.1f}米；计划CF终点相对事实差{row["input_check"]["target_endpoint_delta_m"]:.1f}米。这是输入差，不是模型效果。</p>']
        if 'target_projection_visibility' in row['input_check']:
            visible=row['input_check']['target_projection_visibility']
            body += ['<p>后视画面不能直接理解为自车倒车。黄色框仅标明原始参考中的目标车。</p>',
                     f'<details><summary>查看目标汽车 #8（原始0.5秒帧）</summary><img src="{base}/target-reference.jpg" alt="原始后视画面，黄框标出目标汽车8"></details>',
                     '<p class="note">几何投影可见帧：事实 '+str(visible['factual']['visible_frames'])+'/61；计划CF '+str(visible['counterfactual']['visible_frames'])+'/61。投影在画内不保证无遮挡。</p>']
        body += ['<p>新版结果评价：<span class="empty">—</span>（未推理，待人工 review；不做 AI 视频评分）</p>',
                 '<details><summary>你对旧版短片的 review（不代表新版结果）</summary><p class="review">'+html.escape(row['prior_user_review']['observations'])+'</p></details>',
                 '<p><strong>待你确认：保留 / 修改 / 不采用。</strong>未收到明确确认，不运行推理。</p></article>']
    css="body{font:16px/1.65 system-ui,'Microsoft YaHei',sans-serif;background:#f5f7fa;color:#213044;margin:0}main{max-width:1320px;margin:auto;padding:24px}article{background:white;padding:24px;margin:25px 0;border:1px solid #dae2eb;border-radius:8px}h2{font-size:22px}h3{font-size:16px}.notice,.intent{background:#edf6fa;padding:12px}.note{color:#586776;font-size:14px}.videos{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}video{width:100%;aspect-ratio:16/9;background:#111}.placeholder{aspect-ratio:16/9;display:flex;align-items:center;justify-content:center;border:1px dashed #adb7c2;background:#f5f7fa;color:#788491;box-sizing:border-box}.empty{color:#788491}img{width:100%}.review{border-left:3px solid #9c8d5b;padding-left:12px}.architecture{padding:18px;border:1px solid #9bb8c6;background:white}a{color:#086684}@media(max-width:800px){.videos{grid-template-columns:1fr}main{padding:12px}article{padding:16px}}"
    (args.output/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OmniDreams 6秒候选·待人工确认</title><style>'+css+'</style><main>'+''.join(body)+'</main></html>',encoding='utf-8')
    print(json.dumps([{'case':row['case_id'],'input_check':row['input_check']} for row in candidates],ensure_ascii=False))


if __name__=='__main__':
    main()
