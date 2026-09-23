"""导出24个10秒原始视频和反事实提案；只做CPU检查，绝不启动模型。"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_proj.cfbench.geometry import project_box, shifted_pose
from prepare_omnidreams_cfbench import pose_at
from add_cfbench_original_videos import make_original

CAMERAS = [('CAM_FRONT','前视'), ('CAM_FRONT_LEFT','左前视'), ('CAM_FRONT_RIGHT','右前视'),
           ('CAM_BACK_LEFT','左后视'), ('CAM_BACK_RIGHT','右后视'), ('CAM_BACK','后视')]
SCENE_NAMES = {str(i): scene['name'] for i, scene in enumerate(json.loads(
    Path('/root/autodl-tmp/data/worldsim_v75_downstream_bench/map_metadata/scene.json').read_text()))}
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


def planned_pose(case, poses, frame):
    event=case['anchor']['event_frame']
    family=case['intervention']['family']
    spec=case['intervention']['counterfactual']
    after=frame>=event
    if family=='actor_removal' and after:
        return None
    if family=='actor_insertion' and not after:
        return None
    query=event+(frame-event)*spec['speed_scale'] if after and family=='actor_speed_change' else frame
    pose=pose_at(poses,query)
    if pose is None:
        return None
    if after and family=='actor_lateral_relocation':
        # 延长观察，不把原1.8秒横移过程偷偷拉长到整段视频。
        alpha=float(np.clip((frame-event)/18,0,1))
        signed_left=spec['lateral_offset_m']*alpha**2*(3-2*alpha)
        # DriveStudio ego lidar: +Y前进、+X向右；nuScenes actor: +X前进、+Y向左。
        # intervention统一规定正号=行驶方向左侧，负号=右侧。
        local_offset=[-signed_left,0,0] if case['target']['role']=='ego' else [0,signed_left,0]
        pose=shifted_pose(pose,local_offset)
    elif after and family=='actor_insertion':
        pose=shifted_pose(pose,case['target']['proposal_offset_actor_frame_m'])
    return pose


def describe(case, noun):
    family=case['intervention']['family']; spec=case['intervention']['counterfactual']
    begin=case['anchor']['pre_frames']/10
    if family=='actor_speed_change':
        scale=spec['speed_scale']; action='减速' if scale<1 else '加速'
        return f'从{begin:g}秒开始，将{noun}沿原轨迹的推进速度设为原来的{scale*100:g}%（{scale:g}×，{action}）；其余对象状态不改。'
    if family=='actor_lateral_relocation':
        offset=spec['lateral_offset_m']
        direction='向左' if offset>0 else '向右'
        return f'从{begin:g}秒开始，让{noun}在1.8秒内沿行驶方向{direction}横移{abs(offset):g}米（有符号值{offset:+g}米；正号=左，负号=右），随后保持该偏移；其余对象状态不改。'
    if family=='actor_removal':
        return f'从{begin:g}秒开始移除{noun}，其他车辆、自车轨迹和地图保持不变。'
    if family=='actor_insertion':
        x,y,z=case['target']['proposal_offset_actor_frame_m']
        return f'从{begin:g}秒开始新增一辆汽车：以{noun}为尺寸/轨迹供体，局部坐标偏移X {x:+g}米、Y {y:+g}米、Z {z:+g}米；供体原车保留。'
    raise ValueError(family)


def build_candidate(row, first, out_root, revision='R3', reuse_original_from=None):
    case=copy.deepcopy(row['case']); parent=case['case_id']
    case.update(case_id=parent+f'-{revision}-10S',parent_case_id=parent,status='awaiting_user_review_no_inference')
    reasons=['延长至10秒，保留原对象、倍率/偏移和源事件。']
    if parent=='CFB-SPEED-EGO-02' and case['dataset']['scene_id']=='191':
        case['dataset']=copy.deepcopy(first['case']['dataset'])
        reasons.append('沿用上一提案：静止scene-0242改为运动scene-0230，与第1例共源，不算独立场景。')
    root=Path(case['dataset']['root'])/case['dataset']['scene_id']
    camera=row['camera_index']; camera_name,view=CAMERAS[camera]
    event=case['anchor']['event_frame']; pre=case['anchor']['pre_frames']
    family=case['intervention']['family']; target=case['target']; ego=target['role']=='ego'
    classes={'vehicle.car':'汽车','vehicle.truck':'卡车','vehicle.bus.rigid':'巴士','vehicle.trailer':'挂车',
             'vehicle.construction':'施工车辆','vehicle.bicycle':'自行车/骑行者','vehicle.motorcycle':'摩托车/骑行者'}
    if ego:
        poses={int(path.stem):np.loadtxt(path) for path in (root/'lidar_pose').glob('*.txt')}
        noun='自车'; size=None; key=None
    else:
        key=target.get('actor_key',target.get('source_asset_actor_key'))
        ann=json.loads((root/'instances/instances_info.json').read_text())[key]['frame_annotations']
        poses={int(f):np.array(p) for f,p in zip(ann['frame_idx'],ann['obj_to_world'])}
        size=np.median(np.array(ann['box_size']),axis=0)
        noun=classes.get(target.get('class_name'),target.get('class_name','对象'))+' #'+key
    if family=='actor_speed_change' and case['intervention']['counterfactual']['speed_scale']>1:
        scale=case['intervention']['counterfactual']['speed_scale']
        # 固定事件/倍率，只增加必要的真实前缀，避免加速查询超出已有轨迹。
        required=int(np.ceil(100-(max(poses)-event)/scale))
        if required>pre and event-required>=min(poses):
            pre=required
            reasons.append(f'为覆盖10秒且不外推加速轨迹，前缀从0.5秒增至{pre/10:g}秒；源事件不变。')
    start=event-pre; state_end=start+100; stop=state_end-1
    assert start>=0 and all((root/f'images/{f:03d}_{camera}.jpg').is_file() for f in range(start,stop+1))
    case['anchor'].update(pre_frames=pre,rollout_frames=100-pre,clip_frames=100,video_duration_s=10.0)
    case['qualification']['human_verdict']=None
    if family=='actor_lateral_relocation':
        case['intervention']['transition_duration_s']=1.8
    times=np.arange(start,state_end+1)
    actual=[pose_at(poses,float(f)) for f in times]
    cf=[planned_pose(case,poses,float(f)) for f in times]
    required_cf=[i for i,f in enumerate(times) if not (family=='actor_removal' and f>=event) and not (family=='actor_insertion' and f<event)]
    missing_actual=[int(times[i]) for i,p in enumerate(actual) if p is None]
    missing_cf=[int(times[i]) for i in required_cf if cf[i] is None]
    pathlength=sum(float(np.linalg.norm(b[:3,3]-a[:3,3])) for a,b in zip(actual[pre:-1],actual[pre+1:]) if a is not None and b is not None)
    delta=float(np.linalg.norm(actual[-1][:3,3]-cf[-1][:3,3])) if actual[-1] is not None and cf[-1] is not None and family not in ['actor_insertion','actor_removal'] else None
    warnings=[]
    if missing_actual or missing_cf:
        warnings.append(f'10秒内存在目标轨迹缺口（事实{len(missing_actual)}个状态采样、计划CF{len(missing_cf)}个）；不补造运动，推理前需处理。')
    if family=='actor_speed_change' and (delta is None or delta<1):
        warnings.append('目标速度干预不足以形成清楚的终点差；不把该候选自动视为有效速度测试。')
    if family=='actor_lateral_relocation' and ego and pathlength<1:
        warnings.append('自车原本近乎静止，本例是横向重定位，不应称为正常行驶换道。')
    if family=='actor_lateral_relocation':
        warnings.append('左右按车辆行驶方向定义，不是画面左右；实际道路合法性仍待检查。')
    elif family=='actor_insertion':
        warnings.append('插入偏移采用供体对象自身局部坐标，不是屏幕左右；需人工确认位置是否合理。')
    prior_path=Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-DOWNSTREAM-FULL-01/20260923-r1/omnidreams-r2')/parent/'input-audit.json'
    prior_audit=json.loads(prior_path.read_text()) if prior_path.exists() else {}
    prior_road_fraction=prior_audit.get('counterfactual_center_in_drivable_fraction')
    if prior_road_fraction is not None and prior_road_fraction<1:
        warnings.append('旧短窗检查已发现计划位置的目标中心部分不在可行驶区域内；本页未替你更改偏移，待你审位置。')
    entry={'case_id':case['case_id'],'parent_case_id':parent,'case':case,
           'scene_name':SCENE_NAMES[str(case['dataset']['scene_id'])],
           'camera_index':camera,'camera_name':camera_name,'view':view,
           'target_description':noun+('（搭载相机的采集车）' if ego else '（供体原车保留）' if family=='actor_insertion' else '（不是自车）'),
           'counterfactual_description':describe(case,noun),'revision_reason':' '.join(reasons),
           'source_frame_range':[start,stop],'source_state_endpoint_frame':state_end,'event_s':pre/10,
           'video_duration_s':10.0,'last_reference_frame_s':9.9,'planned_generated_frames':300,
           'planned_native_frames':301,'planned_fps':30,'native_tail_trim_frames':1,'tail_padding':0,
           'input_check':{'factual_postevent_path_length_m':pathlength,'target_endpoint_delta_m':delta,
                          'missing_factual_state_frames':missing_actual,'missing_counterfactual_state_frames':missing_cf,
                          'trajectory_available_without_extrapolation':not missing_actual and not missing_cf,
                          'full_collision_road_gate':'not_claimed'},
           'warnings':warnings,'approval':{'status':'pending','reviewer':'user','inference_allowed':False},
           'prior_short_window_input_audit':prior_audit,
           'prior_user_review':next((r for r in REVIEWS if r['case_id']==parent),None)}
    out=out_root/'cases'/case['case_id']
    if reuse_original_from is not None:
        prior=json.loads((reuse_original_from/'original-nuscenes.json').read_text())
        assert prior['source_root']==str(root)
        assert prior['source_frames']==list(range(start,stop+1))
        assert prior['camera_index']==camera and prior['frame_count']==100
        assert prior['intervention_time_s']==pre/10
        out.mkdir(parents=True,exist_ok=False)
        os.link(reuse_original_from/'original-nuscenes.mp4',out/'original-nuscenes.mp4')
        prior['case_id']=case['case_id']
        save(out/'original-nuscenes.json',prior)
    make_original(case,camera,out)
    if not ego:
        visible={}
        for name,branch in [('factual',actual),('counterfactual',cf)]:
            boxes=[project_box(root,int(f),camera,p,size) if p is not None else None for f,p in zip(times[:-1],branch[:-1])]
            eligible=[i for i,f in enumerate(times[:-1]) if not (name=='counterfactual' and family=='actor_removal' and f>=event) and not (name=='counterfactual' and family=='actor_insertion' and f<event)]
            seen=[i for i in eligible if boxes[i] is not None]
            visible[name]={'visible_frames':len(seen),'eligible_frames':len(eligible),'total_video_frames':100,
                           'last_visible_s':max([i/10 for i in seen],default=None)}
            if len(eligible)>0 and len(seen)<len(eligible)*.5:
                warnings.append(f'{"事实目标" if name=="factual" else "计划CF目标"}仅{len(seen)}/{len(eligible)}个应存在帧投影在画内，延长视频不等于延长有效目标观察。')
        entry['input_check']['target_projection_visibility']=visible
        # 延长后不重选对象；保留原始关键帧并标注目标/供体与计划位置。
        mark=row.get('review_reference_frame')
        if mark is None:
            mark=min(event+18,stop) if family=='actor_lateral_relocation' else event
        assert start <= mark <= stop
        im=Image.open(root/f'images/{mark:03d}_{camera}.jpg').convert('RGB'); draw=ImageDraw.Draw(im)
        for name,pose,color in [('SOURCE #'+key,pose_at(poses,mark),'yellow'),('PLANNED',planned_pose(case,poses,mark),'lime')]:
            if pose is None or (name=='PLANNED' and family not in ['actor_lateral_relocation','actor_insertion']):
                continue
            box=project_box(root,mark,camera,pose,size)
            if box:
                draw.rectangle(box['bbox_xyxy'],outline=color,width=5)
                x,y=box['bbox_xyxy'][:2]; draw.rectangle([x,max(0,y-25),x+150,max(25,y)],fill='black')
                draw.text((x+3,max(0,y-22)),name,fill=color)
        im.save(out/'target-reference.jpg',quality=90)
        entry['target_reference_s']=(mark-start)/10
    save(out/'proposal.json',entry)
    print(json.dumps({'prepared':entry['case_id'],'event_s':pre/10,'warnings':warnings},ensure_ascii=False),flush=True)
    return entry


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
    old=json.loads((args.inputs/'index.json').read_text())['cases']
    assert len(old)==24
    candidates=[build_candidate(row,old[0],args.output) for row in old]
    manifest={'task_id':'WS-V75-OMNI-REVIEW-02','run_id':'20260923-proposal-r2-10s',
              'created_utc':datetime.now(timezone.utc).isoformat(),'status':'awaiting_user_review_no_inference',
              'prior_first_four_feedback':'用户总体无大问题，要求延长10秒；本版窗口/参数仍待最终确认，未授权推理',
              'model_calls':0,'new_ai_video_reviews':0,'case_count':len(candidates),'cases':candidates}
    save(args.output/'candidates.json',manifest)
    save(args.output/'human-review-original-four.json',{'source':'用户2026-09-23聊天口述，保留定性意见，不转换成数值分','reviews':REVIEWS})
    write_report(args.output,manifest)
    print(json.dumps({'cases':len(candidates),'seconds':10,'model_calls':0,'status':manifest['status']}))


def write_report(output, manifest):
    candidates=manifest['cases']
    summary=manifest.get('revision_summary','24段10秒原始视频与干预提案，推理结果与评价留空。等你确认后再跑。')
    body=['<h1>OmniDreams · 24 个 case · 10 秒待审版</h1><p class="notice">'+html.escape(summary)+'</p>',
          '<p>ego = 搭载相机的采集车；画面中标出的其他车辆不是ego。全部视频正常10Hz播放，未减速、循环或插帧。</p>',
          '<p>横移方向按车辆行驶方向：<strong>+ 向左，− 向右</strong>；后视画面中的屏幕左右可能正好相反。</p>',
          '<div class="architecture">nuScenes / DriveStudio 10Hz → 原24例；修订例经 DriverQ 场景/运动/投影查询 → 10秒片段 + 指定反事实 → <strong>你的人工确认</strong> → factual / counterfactual（待运行）</div>',
          '<p><a href="candidates.json">候选参数</a> · <a href="human-review-original-four.json">你对旧版的人工 review</a></p>',
          '<nav>'+''.join(f'<a href="#{r["case_id"]}">{r["parent_case_id"].replace("CFB-","")}</a>' for r in candidates)+'</nav>']
    for row in candidates:
        cid=row['case_id']; base='cases/'+cid
        approved=row.get('approval',{}).get('status')=='approved'
        placeholder='本例已批准，待整组审核与GPU恢复后推理' if approved else '待你确认后推理'
        body += [f'<article id="{cid}"><h2>{cid}</h2>',
                 f'<p>{row["scene_name"]} · {row["camera_name"]}（{row["view"]}）· 目标：{row["target_description"]}</p>',
                 '<p class="note"><strong>审核状态：</strong>'+('用户已批准' if approved else '待用户审核')+'</p>',
                 f'<p class="intent"><strong>反事实：</strong>{row["counterfactual_description"]}</p>',
                 '<p class="note">'+row['revision_reason']+'</p>',
                 '<div class="videos"><div><h3>原始 nuScenes · 10秒</h3>'+f'<video controls preload="metadata" playsinline src="{base}/original-nuscenes.mp4"></video></div>'+
                 '<div><h3>factual</h3><div class="placeholder">'+placeholder+'</div></div><div><h3>counterfactual</h3><div class="placeholder">'+placeholder+'</div></div></div>',
                 f'<p class="note">从{row["event_s"]:g}秒开始编辑；此前为公共前缀。事实分支保留原状态。ego是相机所在的采集车。</p>']
        if row['warnings']:
            body.append('<p class="warning">注意：'+' '.join(html.escape(w) for w in row['warnings'])+'</p>')
        if row.get('selection_evidence'):
            body.append('<p class="note">筛选依据：'+html.escape(row['selection_evidence']['summary'])+'</p>')
        if row.get('target_reference_s') is not None:
            opened=' open' if row.get('show_target_reference') else ''
            if row.get('target_reference_kind')=='ego_bev':
                title=f'查看自车俯视落点（原始{row["target_reference_s"]:g}秒状态）'
                caption='黄色框是事实自车，绿色框是计划自车；俯视图而非生成结果。自车位于前视相机后方，不能在自己的画面中画出真实车身框。灰色地图不能代表锥桶，仍需结合原视频人工审查。'
            else:
                title=f'查看目标/计划位置（原始{row["target_reference_s"]:g}秒帧）'
                caption='黄色是原目标/供体；绿色是计划位置，仅几何框，不是生成结果。投影在画内不保证无遮挡。'
            body += [f'<details{opened}><summary>{title}</summary><img loading="lazy" src="{base}/target-reference.jpg" alt="事实目标黄色框与计划目标绿色框"><p class="note">{caption}</p></details>']
        body += ['<p>新版结果评价：<span class="empty">—</span>（未推理，待人工 review；不做 AI 视频评分）</p>']
        if row['prior_user_review']:
            body += ['<details><summary>你对旧版短片的 review（不代表新版结果）</summary><p class="review">'+html.escape(row['prior_user_review']['observations'])+'</p></details>']
        body += ['<p><strong>'+('本例已批准；未运行推理。' if approved else '待你确认：保留 / 修改 / 不采用。')+'</strong></p></article>']
    css="body{font:16px/1.65 system-ui,'Microsoft YaHei',sans-serif;background:#f5f7fa;color:#213044;margin:0}main{max-width:1320px;margin:auto;padding:24px}article{background:white;padding:24px;margin:25px 0;border:1px solid #dae2eb;border-radius:8px}h2{font-size:22px}h3{font-size:16px}.notice,.intent{background:#edf6fa;padding:12px}.note{color:#586776;font-size:14px}.videos{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}video{width:100%;aspect-ratio:16/9;background:#111}.placeholder{aspect-ratio:16/9;display:flex;align-items:center;justify-content:center;border:1px dashed #adb7c2;background:#f5f7fa;color:#788491;box-sizing:border-box}.empty{color:#788491}img{width:100%}.review{border-left:3px solid #9c8d5b;padding-left:12px}.architecture{padding:18px;border:1px solid #9bb8c6;background:white}a{color:#086684}@media(max-width:800px){.videos{grid-template-columns:1fr}main{padding:12px}article{padding:16px}}"
    css+='nav{display:flex;gap:8px 16px;flex-wrap:wrap;padding:12px;background:white}nav a{font-size:13px}.warning{background:#fff3dc;border-left:3px solid #c58c25;padding:10px;font-size:14px}'
    (output/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OmniDreams 24-case 10秒·待人工确认</title><style>'+css+'</style><main>'+''.join(body)+'</main></html>',encoding='utf-8')


if __name__=='__main__':
    main()
