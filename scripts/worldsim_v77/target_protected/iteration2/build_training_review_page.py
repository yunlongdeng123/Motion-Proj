"""复用旧视频，按训练输入新判据建立独立人工记录空间；不覆写旧评分。"""
import argparse,json
from pathlib import Path
from collections import Counter

def read(p):return json.loads(Path(p).read_text())
def main(root,old,template):
    admission=read(root/'training_admission.json');mapping={r['case_id']:r for r in admission['cases']};source=read(old/'synthetic_delivery/review_manifest.json');clips=[]
    for c in source['clips']:
        r=mapping[c['case_id']]
        if r['original_ai_score']!=1:continue
        status={'train_usable_pending_human':'pass','uncertain':'uncertain','reject':'reject'}[r['training_admission']];v=r['visual_reassessment'];t=r['engineering_audit']
        c=c|{'review':{'synthetic_status':status,'reviewed_frames':v['reviewed_frames'],'issues':v.get('issues',[]),
            'note':'训练输入重评：'+v['reason']+'；工程flags='+str(t['engineering_flags'])+'。RGB外观不单独评分，人工全检尚未完成。'},
            'human_verdict':None,'training_ready':False,'original_ai_score':1,'training_admission':r['training_admission']}
        prefix='../v77-target-protected-synthetic/'
        c['videos']={k:prefix+p for k,p in c['videos'].items()};c['contacts']=[prefix+p for p in c['contacts']]
        c['preview_frames']=[f|{k:prefix+f[k] for k in ['gt','input','labels','condition']} for f in c['preview_frames']];clips.append(c)
    data=source|{'run_id':'r2','human_scoring_scope':'training_input_five_checks_v2','clips':clips,'counts':dict(Counter(c['review']['synthetic_status'] for c in clips)),
        'scene_count':len({c['scene'] for c in clips}),'passed_scene_count':len({c['scene'] for c in clips if c['review']['synthetic_status']=='pass'}),
        'type_counts':dict(Counter(c['type'] for c in clips)),'training_ready':0,'legacy_RGB_scores_preserved':True}
    text=template.read_text(encoding='utf-8').replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/'))
    text=text.replace("KEY='v77-target-protected-synthetic-frames-v1'","KEY='v77-training-input-frames-r2'")
    text=text.replace('合成数据逐帧全检','训练输入逐帧全检').replace('20260929 / r1','20260929 / r2')
    text=text.replace('可人工全检：独立通过','技术可用：等待人工全检').replace('独立质检','训练输入复核').replace('独立通过','技术检查通过')
    text=text.replace('已产出 ${DATA.clips.length} 例','本页复核旧 ${DATA.clips.length} 例')
    text=text.replace('V77_Target_Protected_frame_review.json','V77_training_input_r2_frame_review.json')
    text=text.replace('本帧问题：身份、接地、遮挡、边缘、清晰度、闪烁等','本帧问题：洞形、尺度位置/压ego、轨迹连续、遮挡次序、合成像素漏出等')
    banner='<div class="card warning"><b>新判据：只评价训练输入五项。</b>洞像真实车辆；尺度/位置合理且不压ego；时间连续；A在B前并保留B的合法可见部分；所有合成影响被最终H清掉。车身颜色、材质、受光、贴片感本身不扣分。真实空间悬浮/异常洞形仍不合格。37个旧1分例已由一个Sol xhigh抽帧复核及CPU全帧检查；这页的人工评分独立保存，旧评分不会被覆盖。<a href="index.html">回到本轮报告</a>。</div>'
    text=text.replace('<main>','<main>'+banner,1)
    (root/'review/data_review.html').write_text(text,encoding='utf-8');(root/'review/training_review_manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    link=root/'v77-target-protected-synthetic'
    if not link.exists():link.symlink_to((old/'synthetic_delivery').resolve())
    print('TRAINING_REVIEW_PAGE',len(clips),data['counts'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--old',type=Path,required=True);p.add_argument('--template',type=Path,required=True);a=p.parse_args();main(a.root,a.old,a.template)
