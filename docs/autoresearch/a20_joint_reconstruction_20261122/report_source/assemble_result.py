"""由可追加阶段登记与真实本地收据组装报告；不导入runtime/Torch，不读取数组。"""
from pathlib import Path
import json,os
from urllib.parse import quote

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parents[1]/'outputs/a20_joint_reconstruction_20261122'
REGISTRY=ROOT/'report_groups.json'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def url(path):return quote(os.path.relpath(Path(path).resolve(),OUT.resolve()).replace('\\','/'),safe='/._-:')
def image(path):return url(path) if Path(path).is_file() else None

def bootstrap():
    if REGISTRY.exists():return
    write(REGISTRY,dict(schema='a20.joint_report_groups.v1',groups=[
        dict(id='initial',label='初始化 · 419005行',state='exported',role='initial',planned_steps=None,
            metrics='takeback/execution_initial/initial/evaluation/fresh_process/metrics.json',
            eval_root='takeback/execution_initial/initial/evaluation/fresh_process',
            reload='takeback/execution_initial/initial/fresh_process_reload_proof.json',summary='实际初始化导出；0参数更新'),
        dict(id='r1_G40',label='r1 · G40（工程诊断）',state='complete',role='engineering_diagnostic',stage='geometry',planned_steps=40,
            completion='takeback/execution_r1/geometry/completion.json',metrics='takeback/execution_r1/geometry/evaluation/geometry_final/metrics.json',
            eval_root='takeback/execution_r1/geometry/evaluation/geometry_final',smoke='takeback/execution_r1/smoke/smoke_proof.json',
            reload='takeback/execution_r1/geometry/fresh_reload_proof.json',root_engineering_gate=False,
            summary='测距门满足，shared V覆盖与保留实例仍失败'),
        dict(id='r2_G40',label='r2 · G40（待收据）',state='pending',role='engineering_replay',stage='geometry',planned_steps=40,
            summary='干净初始化；isolate-V监督与teacher外域alpha约束；LR和G40预算保持'),
        dict(id='r3_G40',label='r3 · G40（待root通知）',state='pending',role='engineering_replay',stage='geometry',planned_steps=40,
            summary='六新视图外域VJP清退；干净smoke＋G40；尚无实际结果'),
        dict(id='A80',label='A80（待灰模验收）',state='pending',role='appearance',stage='appearance',planned_steps=80,summary='尚未启动'),
        dict(id='J40',label='J40（待灰模验收）',state='pending',role='joint',stage='joint',planned_steps=40,summary='尚无实际收据')],
        history=[dict(label='历史车辆教师／24k与410k对照',href='../a20_sd15_sameview_20261122/index.html'),
                 dict(label='旧固定贡献／局部融合路线',href='../a20_native_gaussian_fusion_20261122/index.html')]))

def assemble():
    bootstrap();registry=read(REGISTRY);groups=[]
    for spec in registry['groups']:
        group=dict(spec,actual_steps=spec.get('root_reported_actual_steps'),smoke_counts=None,smoke_passed=None,gate_passed=None,
            scene_qualified=None,real=[],novel=[],views={},receipts=[],range_summary=None,reload_passed=None,
            engineering_gate=spec.get('root_engineering_gate'),teacher_bg_alpha_max=spec.get('root_reported_teacher_bg_alpha_max'))
        if spec['state'] not in ('complete','exported'):groups.append(group);continue
        required=['metrics','eval_root']+(['completion'] if spec['state']=='complete' else [])
        if any(not spec.get(key) for key in required):raise ValueError('完整阶段缺实际来源：'+spec['id'])
        completion=read(ROOT/spec['completion']) if spec.get('completion') else dict(steps=0)
        metrics=read(ROOT/spec['metrics']);evaluation=ROOT/spec['eval_root']
        if spec['state']=='complete' and completion.get('completed') is not True:raise ValueError('阶段没有完成收据：'+spec['id'])
        group.update(actual_steps=completion['steps'],gate_passed=metrics.get('gate',{}).get('passed'),
            scene_qualified=False if metrics.get('gate',{}).get('passed') is False else None,
            real=metrics.get('real',[]),novel=metrics.get('novel',[]),
            range_summary=dict(count=metrics.get('opaque_stable_count'),mae_m=metrics.get('opaque_mae_m'),p90_m=metrics.get('opaque_p90_m')),
            active_rows=completion.get('active_rows'),retired_rows=completion.get('retired_rows'))
        conditions=metrics.get('gate',{}).get('conditions',{})
        group['range_gate_passed']=all(conditions.get(key) is True for key in ['opaque_all246_stable','opaque_mae','opaque_p90'])
        group['extra_figures']=[dict(title=item['title'],src=image(ROOT/item['source'])) for item in spec.get('extra_figures',[]) if (ROOT/item['source']).is_file()]
        for key in ['completion','metrics','smoke','reload','receipt']:
            if spec.get(key):
                source=ROOT/spec[key]
                if not source.is_file():raise FileNotFoundError(source)
                group['receipts'].append(dict(label=key,href=url(source)))
        if spec.get('smoke'):
            smoke=read(ROOT/spec['smoke']);group['smoke_passed']=smoke.get('passed')
            group['smoke_counts']=dict(main=smoke.get('actual_geometry_smoke_updates'),replay=smoke.get('extra_replay_updates'),discarded=smoke.get('discarded_next_stage_updates'))
        if spec.get('reload'):group['reload_passed']=read(ROOT/spec['reload']).get('passed')
        for camera in ['raw086','raw110']+['novel_'+str(i).zfill(2) for i in range(6)]:
            base=evaluation/camera
            teacher=(ROOT/spec['teacher_root'] if spec.get('teacher_root') else evaluation.parent.parent/'teacher_cache')/camera
            group['views'][camera]=dict(
                isolated=dict(rgb=image(base/'actor/rgb.png'),alpha=image(base/'actor/actor_alpha.png'),grey=image(base/'grey/rgb.png'),
                    depth=image(base/'actor/expected_depth_grey.png'),normal=image(base/'actor/expected_depth_normal.png')),
                shared=dict(rgb=image(base/'shared/rgb.png'),alpha=image(base/'shared/actor_alpha.png'),grey=image(base/'shared_grey/rgb.png'),
                    depth=image(base/'shared/expected_depth_grey.png'),normal=image(base/'shared/expected_depth_normal.png')),
                teacher=dict(rgb=image(teacher/'rgb.png'),alpha=image(teacher/'alpha.png'),grey=None,
                    depth=image(teacher/'expected_depth_grey.png'),normal=image(teacher/'expected_depth_normal.png')))
        for camera,modes in spec.get('image_overrides',{}).items():
            for mode,channels in modes.items():
                for channel,path in channels.items():
                    if not (ROOT/path).is_file():raise FileNotFoundError(path)
                    group['views'].setdefault(camera,{}).setdefault(mode,{})[channel]=url(ROOT/path)
        group['grey_overview']=image(evaluation/'grey_six_views.png')
        if not any(v for camera in group['views'].values() for mode in camera.values() for v in mode.values()):
            raise ValueError('阶段尚无实际导出图：'+spec['id'])
        groups.append(group)
    complete=[x for x in groups if x['state']=='complete']
    final=read(ROOT/'final_iteration_receipt.json') if (ROOT/'final_iteration_receipt.json').is_file() else None
    asset_verification=read(ROOT/'assets/takeback_verification.json') if (ROOT/'assets/takeback_verification.json').is_file() else None
    final_links=[]
    if final:
        for label,path in [('最终候选（灰模未过）',ROOT/'assets/a20_geometry_candidate.npz'),('实际收尾账目',ROOT/'final_iteration_receipt.json'),('灰模审查REJECTED',ROOT/'root_geometry_review.json'),('本地资产验真',ROOT/'assets/takeback_verification.json')]:
            if path.is_file():final_links.append(dict(label=label,href=url(path)))
    result=dict(schema='a20.joint_reconstruction.report.v1',title='a20：统一高斯场联合重建',
        groups=groups,current_group=registry.get('default_group') or (complete[-1]['id'] if complete else groups[0]['id']),
        history=registry['history'],architecture='architecture.svg',initial_rows=419005,
        cameras=[dict(id='raw086',label='真实相机 · raw86'),dict(id='raw110',label='真实相机 · raw110')]+[
            dict(id='novel_'+str(i).zfill(2),label='新视图 · '+str(i+1)) for i in range(6)],
        final_iteration=final,asset_verification=asset_verification,final_links=final_links,
        scene_qualified=False if final else None,publication_ready=bool(final),root_report_complete=bool(final),
        disclaimer='渲染期望距离由native depth_sum/alpha换算ray range，离散多层可能混合，始终与灰模验收分开。期望深度法向由实际期望深度梯度计算。生成隐藏区域为先验；BUILD86/110与合成新视图分别统计，未读取封存帧。各40步组独立从共同初始化重放；smoke工程通过或孤立覆盖不能替代场景准入。')
    write(ROOT/'result.json',result)
    print(json.dumps(dict(groups=len(groups),complete_groups=len(complete),current_group=result['current_group'],runtime_imported=False)))

if __name__=='__main__':assemble()
