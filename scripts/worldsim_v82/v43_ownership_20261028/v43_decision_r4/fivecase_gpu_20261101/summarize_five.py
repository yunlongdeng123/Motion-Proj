"""CPU逐项读取真实GPU输出；不构造反事实GT或事后最优标签。"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import csv,json,pathlib,shutil,tarfile,time
import numpy as np
from PIL import Image
ROOT=pathlib.Path(__file__).resolve().parent
CASES={'waymo-train-008_a37':'gpu','waymo-train-212_a0':'gpu_r2','waymo-train-212_a35':'gpu_r2',
       'waymo-train-564_a0':'gpu_564_replay','waymo-train-564_a22':'gpu_564_replay'}
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,x):pathlib.Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def mask_for(context,fid,w,h):
    row=next(r for r in context['frames'] if r['frame_id']==fid)
    return np.asarray(Image.open(row['mask_ref']).convert('L').resize((w,h),Image.Resampling.NEAREST))>0
def main():
    out=ROOT/'delivery';out.mkdir(exist_ok=False);assets=out/'review/assets';assets.mkdir(parents=True)
    evidence=out/'evidence';evidence.mkdir();metrics=[];summary=[];media=[]
    for case,folder in CASES.items():
        run=ROOT/folder/case;cpu=ROOT/'cpu'/case;context=read(cpu/'camera_track_context.json')
        audit=read(run/'audit49/audit.json')
        copied=evidence/case;copied.mkdir()
        for name in ['audit49/audit.json','parent_smoke/saved_state_receipt.json','refined_smoke/saved_state_receipt.json',
                     'parent_enum/restore_replay.json','refined_enum/restore_replay.json','refined_sequence/restore_replay.json']:
            shutil.copy2(run/name,copied/(name.replace('/','_')))
        for name in ['candidate_manifest.json','camera_track_context.json','dynamic_binding.json','frozen_H0.json']:
            shutil.copy2(cpu/name,copied/('parent_'+name));shutil.copy2(ROOT/'cpu_refined'/case/name,copied/('refined_'+name))
        exposed=ROOT/'gpu_exposed'/case
        broad_sequence=ROOT/'gpu_broad_sequence'/case/'broad_sequence'
        fullsource=ROOT/'gpu_fullsource'/case
        manifests={stage:read(run/stage/'gpu_manifest.json') for stage in ['parent_enum','refined_enum','refined_sequence']}
        for stage in ['exposed_enum','exposed_sequence']:manifests[stage]=read(exposed/stage/'gpu_manifest.json')
        manifests['broad_sequence']=read(broad_sequence/'gpu_manifest.json')
        for stage in ['source_enum','source_sequence']:manifests[stage]=read(fullsource/stage/'gpu_manifest.json')
        for stage in ['source_enum','source_sequence']:shutil.copy2(fullsource/stage/'restore_replay.json',copied/(stage+'_restore_replay.json'))
        shutil.copy2(ROOT/'cpu_fullsource'/case/'candidate_manifest.json',copied/'fullsource_candidate_manifest.json')
        shutil.copy2(broad_sequence/'restore_replay.json',copied/'broad_sequence_restore_replay.json')
        peel=read(exposed/'source_peel49/summary.json');shutil.copy2(exposed/'source_peel49/summary.json',copied/'source_peel49_summary.json')
        shutil.copy2(exposed/'source_peel49/pre_feature_contract.json',copied/'source_peel_contract.json')
        for stage in ['exposed_enum','exposed_sequence']:
            shutil.copy2(exposed/stage/'restore_replay.json',copied/(stage+'_restore_replay.json'))
        shutil.copy2(ROOT/'cpu_exposed'/case/'candidate_manifest.json',copied/'exposed_candidate_manifest.json')
        scene=dict(case=case,scene_key=context['scene_key'],actor=case.rsplit('_',1)[1],
            frames=context['frames'],windows=[context['frames'][0]['raw'],context['frames'][-1]['raw']],
            domains={},video_pairs=[],formal_eligibility=None,decision_gap=None,
            dynamic_binding_mode=audit['target_dynamic_binding']['mode'])
        spec=next(x for x in read(ROOT/'cpu/run_contract.json')['checkpoint_inputs'] if x['id']==case)
        event=read(spec['event']);camera_rows={x['img_name']:x for x in read(pathlib.Path(event['model_path'])/'cameras.json')}
        for frame in scene['frames']:
            camera=camera_rows[frame['frame_id']]
            frame['render_intrinsics']=[camera['fx']*480/camera['width'],camera['fy']*320/camera['height'],camera['cx']*480/camera['width'],camera['cy']*320/camera['height']]
        scene['render_wh']=[480,320]
        max_parity=max(m['allkeep_native_none_parity_max_abs'] for m in manifests.values())
        all_finite=True
        for stage,m in manifests.items():
            stage_root=fullsource if stage.startswith('source_') else exposed if stage.startswith('exposed') else run
            manifest_path=broad_sequence/'gpu_manifest.json' if stage=='broad_sequence' else stage_root/stage/'gpu_manifest.json'
            shutil.copy2(manifest_path,copied/(stage+'_manifest.json'))
            if stage.endswith('sequence'):continue
            domain='fullsource' if stage=='source_enum' else 'broad' if stage=='parent_enum' else 'exposed' if stage=='exposed_enum' else 'conservative'
            c=read((cpu if domain=='broad' else ROOT/('cpu_fullsource' if domain=='fullsource' else 'cpu_exposed' if domain=='exposed' else 'cpu_refined')/case)/'candidate_manifest.json')
            scene['domains'][domain]=dict(routes=m['action_count'],groups=[len(g['original_static_rows']) for g in c['groups']],
                anchors=m['evaluated_anchor_frames'],images=[])
            for row in m['rows']:
                fid=row['frame'];arrays=np.load(row['fp32']);baseline=np.load(run/'audit49/original_none'/(fid+'.npz'))
                a=arrays['final'];b=baseline['final'];h,w=a.shape[-2:];mask=mask_for(context,fid,w,h)
                finite=all(bool(np.isfinite(arrays[k]).all()) for k in arrays.files);all_finite &=finite
                rec=dict(case=case,domain=domain,route=row['action_id'],operation=row['edit'],frame=fid,body_pixels=int(mask.sum()),
                    body_RGB_change_vs_original_none=float(abs(a[:,mask]-b[:,mask]).mean()),
                    outside_source_body_RGB_change=float(abs(a[:,~mask]-b[:,~mask]).mean()),
                    body_alpha=float(arrays['alpha'][:,mask].mean()),
                    body_CG_abs_mean=float(abs(arrays['C_G'][:,mask]).mean()),body_fill_abs_mean=float(abs(arrays['fill'][:,mask]).mean()),
                    closure_max_abs=float(abs(arrays['final']-arrays['C_G']-arrays['fill']).max()),
                    finite=finite,native_render_seconds=row['gpu_render_and_readback_wall_seconds'])
                metrics.append(rec)
                relative=pathlib.Path('assets')/case/domain/row['action_id']/row['edit']/(fid+'.png')
                target=out/'review'/relative;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(pathlib.Path(row['fp32']).with_name(fid+'_final.png'),target)
                scene['domains'][domain]['images'].append(dict(route=row['action_id'],operation=row['edit'],frame=fid,path=relative.as_posix()))
        # 默认视频直接展示未删掉遮挡候选的宽域固定控制，不把空保守域冒充改进。
        seq=manifests['source_sequence'];right=seq['allrebind_action_id']
        fp32=out/'fp32_controls'/case
        for row in manifests['source_enum']['rows']:
            if row['action_id'] not in ['h000',manifests['source_enum']['allrebind_action_id']]:continue
            dst=fp32/row['action_id']/row['edit'];dst.mkdir(parents=True,exist_ok=True)
            shutil.copy2(row['fp32'],dst/(row['frame']+'.npz'))
        # 将49帧两条固定端点路由的PNG取回供本地编码；不将全部FP32装入网页。
        for op in ['none','delete','shift','rotate']:
            for side,hid in [('left','h000'),('right',right)]:
                dest=out/'video_frames'/case/op/side;dest.mkdir(parents=True)
                rows=[r for r in seq['rows'] if r['action_id']==hid and r['edit']==op]
                rows.sort(key=lambda r:r['frame'])
                if len(rows)!=49:raise ValueError('完整视频必须49帧:'+case+op)
                for i,row in enumerate(rows):shutil.copy2(pathlib.Path(row['fp32']).with_name(row['frame']+'_final.png'),dest/('%03d.png'%i))
            scene['video_pairs'].append(dict(operation=op,path='videos/'+case+'_'+op+'.mp4',boxed_path='videos/'+case+'_'+op+'_boxes.mp4',frame_count=49,fps=10))
        for fid in manifests['parent_enum']['evaluated_anchor_frames']:
            target=assets/case/'observed';target.mkdir(exist_ok=True)
            source=run/'audit49'/(fid+'_observed_RGB.png')
            if source.exists():shutil.copy2(source,target/(fid+'.png'))
        summary.append(dict(case=case,parent_candidate_rows=audit['parent_candidate_rows'],refined_candidate_rows=audit['refined_candidate_rows'],
            exposed_candidate_rows=peel['exposed_candidate_rows'],
            fullsource_candidate_rows=sum(scene['domains']['fullsource']['groups']),
            native_none_parity_max_abs=max_parity,main_anchor_outputs=sum(len(manifests[x]['rows']) for x in ['parent_enum','refined_enum','exposed_enum','source_enum']),
            full_sequence_outputs=sum(len(manifests[x]['rows']) for x in ['broad_sequence','refined_sequence','exposed_sequence','source_sequence']),review_video_sequence='full-source-domain fixed endpoints',
            original49_frames=audit['frames'],finite_all=all_finite,
            strict_contribution_mass_pass=audit.get('strict_contribution_mass_all_frames_passed'),
            contribution_mass_max_abs=max(max(r['body_conservation_abs'],r['keep_conservation_abs']) for r in audit['records']),
            contribution_mass_max_relative=max(max(r['body_conservation_abs']/max(r['body_alphaT'],1),r['keep_conservation_abs']/max(r['keep_alphaT'],1)) for r in audit['records']),
            target_dynamic_rows=len(audit['target_dynamic_binding']['target_dynamic_rows']),dynamic_binding_mode=scene['dynamic_binding_mode'],
            formal_P=None,decision_gap=None,reference_role='observed-real partial diagnostics; edited task GT unavailable',
            restore_replay_max_abs=max([read(run/s/'restore_replay.json')['max_abs'] for s in ['parent_enum','refined_enum','refined_sequence']]+
                [read(exposed/s/'restore_replay.json')['max_abs'] for s in ['exposed_enum','exposed_sequence']]+[read(broad_sequence/'restore_replay.json')['max_abs']]+
                [read(fullsource/s/'restore_replay.json')['max_abs'] for s in ['source_enum','source_sequence']])))
        media.append(scene)
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    write(out/'summary.json',dict(task_id='WS-V43-R4-FIVECASE-GPU-20261101-R1',cases=summary,
        independent_scene_ancestors=3,new_training_updates=0,formal_eligibility_unknown=20,
        interpretation='fixed representation shared-route experiments; no exact regret or edit-capability pass labels',
        main_anchor_outputs=sum(x['main_anchor_outputs'] for x in summary),full_sequence_outputs=sum(x['full_sequence_outputs'] for x in summary),
        human_verdict=None,root=str(ROOT),failure_ledger_refs=['V82-F09','V82-F10']))
    (out/'review/data.js').write_text('window.FIVE_DATA='+json.dumps(media,ensure_ascii=False)+';',encoding='utf-8')
    for name in ['launch.json','engineering_attempts.json','private_copy_deduplication.json','controller_final.json','controller_final_gpu_r2.json','controller_final_gpu_564_replay.json','controller_final_exposed.json','controller_final_controls.json','controller_final_fullsource.json','controller_events.jsonl','prepare.log','prepare_r2.log']:
        if (ROOT/name).exists():shutil.copy2(ROOT/name,evidence/name)
    write(out/'run_contract.json',read(ROOT/'cpu/run_contract.json'))
    source=out/'source';source.mkdir()
    shutil.copy2(ROOT/'cpu_fullsource/source_recall_control.json',evidence/'source_recall_control.json')
    p=ROOT/'logs/waymo-train-212_a35_refined_sequence_OOM_failed.log'
    if p.exists():shutil.copy2(p,evidence/p.name)
    for name in ['prepare_five.py','run_five.py','run_five_engineering_r1.py','run_five_engineering_r2.py','controller.py','controller_peel.py','controller_final_controls.py','prepare_fullsource.py','controller_fullsource.py','deduplicate_private_copies.py','summarize_five.py']:
        shutil.copy2(ROOT/name,source/name)
    # 逐阶段资源/调用，包括失败重放；不将并发wall相加冒充总耗时。
    resources=[]
    for p in ROOT.glob('gpu*/**/execution.json'):
        value=read(p);resources.append(dict(output=str(p.parent),**value))
    write(out/'execution_resources.json',dict(completed_stage_invocations=resources,failed_partial_invocations_in_events=str(ROOT/'controller_events.jsonl'),
        note='all engine attempts retained; per-process walls overlap; sum is not experiment elapsed wall'))
    with tarfile.open(ROOT/'delivery.tar.gz','w:gz') as t:t.add(out,arcname='delivery')
    print(json.dumps(dict(status='COMPLETE',cases=summary,archive_bytes=(ROOT/'delivery.tar.gz').stat().st_size)),flush=True)
if __name__=='__main__':main()
