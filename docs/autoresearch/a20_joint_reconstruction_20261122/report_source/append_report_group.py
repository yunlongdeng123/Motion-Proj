"""追加已有真实收据/图片的一组；不执行runtime。"""
from pathlib import Path
import argparse,json
ROOT=Path(__file__).resolve().parent

def main(a):
    run=Path(a.execution_root)
    if not run.is_absolute():run=ROOT/run
    label=a.eval_label or a.stage+'_final'
    stage=run/a.stage
    source=lambda value,default: (ROOT/Path(value) if not Path(value).is_absolute() else Path(value)) if value else default
    completion=source(a.completion,stage/'completion.json');metrics=source(a.metrics,stage/'evaluation'/label/'metrics.json')
    evaluation=source(a.eval_root,metrics.parent)
    if not metrics.is_file():raise FileNotFoundError('先取回实际metrics，再追加')
    if not a.export_only:
        if not completion.is_file():raise FileNotFoundError(completion)
        data=json.loads(completion.read_text(encoding='utf-8'))
        if data.get('completed') is not True:raise ValueError('没有实际完成收据，不将计划阶段写成完成')
    elif not a.receipt or not source(a.receipt,ROOT).is_file():raise ValueError('0更新导出/消融必须绑定实际receipt')
    rel=lambda p:p.resolve().relative_to(ROOT.resolve()).as_posix()
    group=dict(id=a.id,label=a.label,state='exported' if a.export_only else 'complete',role=a.role,stage=a.stage,planned_steps=a.planned_steps,
        metrics=rel(metrics),eval_root=rel(evaluation),summary=a.summary)
    if not a.export_only:group['completion']=rel(completion)
    if a.receipt:group['receipt']=rel(source(a.receipt,ROOT))
    if a.engineering_failed:group['root_engineering_gate']=False
    if a.teacher_bg_alpha_max is not None:group['root_reported_teacher_bg_alpha_max']=a.teacher_bg_alpha_max
    smoke=run/'smoke/smoke_proof.json';reload=stage/'fresh_reload_proof.json'
    if smoke.is_file():group['smoke']=rel(smoke)
    if reload.is_file():group['reload']=rel(reload)
    path=ROOT/'report_groups.json';registry=json.loads(path.read_text(encoding='utf-8'))
    ids=[x['id'] for x in registry['groups']]
    if a.id in ids:registry['groups'][ids.index(a.id)]=group
    else:registry['groups'].append(group)
    if a.make_default:registry['default_group']=a.id
    path.write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Registered actual group: '+a.id+'; run assemble_result.py then build_report.py')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--label',required=True);p.add_argument('--execution-root',required=True)
    p.add_argument('--stage',choices=['geometry','appearance','joint'],default='geometry');p.add_argument('--eval-label');p.add_argument('--role',default='formal');p.add_argument('--planned-steps',type=int);p.add_argument('--summary',default='实际结果按收据展示')
    p.add_argument('--engineering-failed',action='store_true');p.add_argument('--teacher-bg-alpha-max',type=float)
    p.add_argument('--completion');p.add_argument('--metrics');p.add_argument('--eval-root');p.add_argument('--receipt')
    p.add_argument('--export-only',action='store_true');p.add_argument('--make-default',action='store_true');main(p.parse_args())
