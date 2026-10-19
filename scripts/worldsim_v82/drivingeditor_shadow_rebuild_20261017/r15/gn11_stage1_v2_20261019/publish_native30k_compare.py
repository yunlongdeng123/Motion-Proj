"""精确发布已完成两组的原生49视频；不运行训练或独立GPU评估。"""
from pathlib import Path
import datetime,json,shutil
from publish_ablation_nodes import W,O,REPO,DOC,CODE,long,copy,write,read

def main():
    r=read(O/'native30k_media.json');v=read(O/'native30k_validation.json')
    if r['status']!='READY_TWO_COMPLETED_30K_NATIVE49_VIDEO_COMPARISON' or len(r['videos'])!=8 or v['all_decoded_frames']!=392:
        raise ValueError('complete actual pair and video validation required')
    if r['GPU_used_for_export'] or r['additional_training_updates'] or r['independent_delete_evaluation_completed']:
        raise ValueError('native existing-channel export scope differs')
    card=REPO/'docs/research_failures/entries/V82-F06.md';old=card.read_text(encoding='utf-8')
    heading='## 2026-10-19：已完成两组30K原生完整视频交付'
    if heading in old:raise RuntimeError('native pair publication already exists')
    label='已发布视频：完整版与real_only各30K的原图、完整重建、静态、动态四通道，8个完整480×320/49帧/10fps全I帧视频，支持同步播放、逐帧定位、切换通道与速度。'
    current=REPO/'docs/RESEARCH_STATUS.md';text=current.read_text(encoding='utf-8')
    marker='## 已完成的原A/B基线与历史证据';top,history=text.split(marker,1)
    if 'native30k_compare.html' in top:raise RuntimeError('existing video link requires explicit update workflow')
    top+='\n\n'+label+'[完整同步对比]('+DOC.relative_to('docs').as_posix()+'/native30k_compare.html)。来自已有原生节点通道，本轮远端只读PNG、不下载checkpoint/NPZ、不新增GPU或训练更新；独立删车/alpha评估和四组最终质量结论仍待完成。\n\n'
    write(current,top+marker+history)
    entry='\n\n'+heading+'\n\n'+label+'[对比页面](../../'+DOC.relative_to('docs').as_posix()+'/native30k_compare.html)。392张原生PNG逐文件hash、49帧身份及两组原图hash一致性已核验；取回76,755,996B仅PNG归档，远端未写文件。8个视频本地CPU编码共'+str(sum(x['bytes'] for x in r['videos'].values()))+'B。\n\n全部392帧实际解码并对照对应原生PNG顺序；实际生成JS的DOM/media fixture覆盖4通道×49位置、关键帧、逐帧、速度、漂移纠正、循环和播放失败提示，非浏览器UI验收。主报告与消融报告顶部均有入口。raw129静态完整画面已实际查看，但本次不替代四组全49最终质量复核，human_verdict=null；训练仍按既有controller继续，不能提前关机。\n'
    write(card,old+entry)
    record=read(O/'task_record.json')
    record['native30k_video_comparison']={'status':'READY_NATIVE_CHECKPOINT_CHANNELS','page':'native30k_compare.html',
        'arms':r['arms'],'videos':8,'frames_each':49,'full_frame_wh':[480,320],
        'independent_delete_evaluation_completed':False,'quality_accepted':False}
    write(O/'task_record.json',json.dumps(record,indent=2,ensure_ascii=False))
    files=['native30k_compare.html','native30k_compare.svg','native30k_media.json','native30k_sources.json','native30k_validation.json',
        'index.html','candidate_source.zip','local_delivery_checks.json','task_record.json','ablations/index.html','ablations/local_delivery_checks.json']
    for x in r['videos'].values():files.extend([x['path'],x['poster']])
    paths=['docs/RESEARCH_STATUS.md','docs/research_failures/entries/V82-F06.md']
    for name in files:
        rel=DOC/name;copy(O/name,REPO/rel);paths.append(rel.as_posix())
    for name in ['collect_native30k_pair.py','build_native30k_compare.py','verify_native30k_compare.py',
                 'check_native30k_player.js','native30k_compare_template.html','publish_native30k_compare.py',
                 'build_report.py','build_ablation_report.py']:
        rel=CODE/name;copy(W/name,REPO/rel);paths.append(rel.as_posix())
    (W/'publish_native30k_pair_paths.json').write_text(json.dumps(paths,indent=2),encoding='utf-8')
    print(json.dumps({'paths':len(paths),'video_files':len(r['videos']),'additional_training_updates':0}))

if __name__=='__main__':main()
