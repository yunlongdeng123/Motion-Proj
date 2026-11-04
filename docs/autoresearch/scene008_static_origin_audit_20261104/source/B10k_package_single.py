"""10k身份验收、完整49帧交付与原生四件检查点取回包；不新增模型更新。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,subprocess,tarfile,shutil,hashlib,math,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
import torch
ROOT=Path(__file__).resolve().parent.parent

def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def add(t,p,arc):t.add(p,arcname=arc,filter=lambda x:None if '__pycache__' in x.name else x)

def main():
    assert read(ROOT/'queue_status.json')['stage']=='COMPLETE_TRAIN_AND_LAYER_EVALUATION'
    subprocess.run([sys.executable,str(ROOT/'source/parent_guard.py')],check=True)
    q=read(ROOT/'evaluation/summary.json');formal=read(ROOT/'formal10k/returned.json');smoke=read(ROOT/'smoke_summary.json')
    assert q['frame_count']==49 and q['native_png_replay_max_uint8_difference']<=1
    assert formal['actual_updates']==10000 and formal['learning_rate_horizon']==30000
    saved_state=torch.load(ROOT/'formal10k/training_state10000.pth',map_location='cpu',weights_only=False)
    assert saved_state['iteration']==10000 and saved_state['hybrid_sampler']['steps']==10000
    assert saved_state['config']['start_checkpoint'] is None
    iterations=[];curve=[];paired=[]
    original_log=ROOT.parent/'r15_gn11_stage1_v41_20261023/arms/B/run30k/paired_steps.jsonl'
    baseline=original_log.open()
    with (ROOT/'formal10k/paired_steps.jsonl').open() as f:
        for line in f:
            row=json.loads(line);iterations.append(row['iteration']);loss=float(row['losses']['loss_hybrid_total']);assert math.isfinite(loss)
            old=json.loads(next(baseline))
            for key in ['iteration','frame_id','time_shift','learning_rates']:
                assert row[key]==old[key],('original B paired schedule mismatch',row['iteration'],key)
            if row['iteration'] in [1,10000]:paired.append({k:row[k] for k in ['iteration','frame_id','time_shift','learning_rates']})
            if row['iteration']%100==0:curve.append(dict(iteration=row['iteration'],loss=loss))
    assert iterations==list(range(1,10001))
    baseline.close()
    dest=ROOT/'review';dest.mkdir(exist_ok=False);(dest/'frames').mkdir();(dest/'boards').mkdir();(dest/'source').mkdir()
    for p in (ROOT/'source').glob('*.py'):shutil.copy2(p,dest/'source'/p.name)
    for raw in range(89,138):shutil.copytree(ROOT/'evaluation'/f'{raw:03d}',dest/'frames'/f'{raw:03d}')
    columns=['gt','composite','dynamic_a37','delete_a37'];labels=['Observed RGB','Single a37 B @ 10k','Dynamic: a37 only','Delete a37 (no road add-on)']
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    for raw in range(89,138):
        im=Image.new('RGB',(960,704),'#f2f5f8');draw=ImageDraw.Draw(im)
        for i,(name,label) in enumerate(zip(columns,labels)):
            x=(i%2)*480;y=(i//2)*352
            draw.text((x+12,y+7),label,fill='#13283b',font=font)
            im.paste(Image.open(dest/'frames'/f'{raw:03d}'/(name+'.png')),(x,y+32))
        im.save(dest/'boards'/f'{raw:03d}.png')
    ff=imageio_ffmpeg.get_ffmpeg_exe();decoded={}
    for name,pattern in [('all49_layers','boards/%03d.png'),('dynamic_a37_all49','frames/%03d/dynamic_a37.png'),('delete_a37_all49','frames/%03d/delete_a37.png')]:
        subprocess.run([ff,'-hide_banner','-loglevel','error','-n','-framerate','10','-start_number','89','-i',str(dest/pattern),
            '-frames:v','49','-c:v','libx264','-crf','17','-pix_fmt','yuv420p',str(dest/(name+'.mp4'))],check=True)
        it=imageio_ffmpeg.read_frames(str(dest/(name+'.mp4')));next(it);decoded[name]=sum(1 for _ in it);assert decoded[name]==49
    shutil.copy2(dest/'boards/105.png',dest/'preview.png')
    config=read(ROOT/'config.json');event=read(ROOT/'formal10k/checkpoint_events/iteration_010000.json')
    native=read(Path(event['metrics']));paper=read(Path(event['paper_metrics']))
    summary=dict(task_id='SCENE008-A37-ONLY-B10K-20261103-R1',dynamic_aliases=[37],retained_aliases=[5,26,18,6],
        formal_updates=10000,smoke_updates=smoke['smoke_updates'],resume_extra_updates=smoke['resume_extra_updates'],
        total_new_updates=10000+smoke['smoke_updates']+smoke['resume_extra_updates'],prior_updates=11238,
        native_metrics=native,identity_metrics={k:v for k,v in q.items() if k!='frames'},
        formal=formal,smoke=smoke,learning_rate_horizon=30000,checkpoint_nodes_completed=[10000],
        original_B_unchanged=read(ROOT/'parent_unchanged.json')['original_B_four_components_unchanged'],
        native_video_frames=decoded,checkpoint_state_iteration=10000,saved_sampler_steps=10000,road_repair_updates=0,R4_or_a6_addon_loaded=False,
        no_subagents=True,GPU_shutdown=False,human_verdict=None,quality_accepted=False,
        input_role='49observedtrain BUILD; no independent test',failure_ledger_refs=['V82-F01','V82-F06'],failure_ledger_delta='none')
    summary['cumulative_updates']=summary['prior_updates']+summary['total_new_updates']
    write(ROOT/'result_summary.json',summary);write(dest/'result_summary.json',summary);write(dest/'identity_metrics_all49.json',q)
    write(dest/'loss_curve.json',curve);write(dest/'native_metrics.json',native);write(dest/'paper_metrics.json',paper)
    write(dest/'original_B_schedule_pairing.json',dict(status='PASS_EXACT_ALL10000',fields=['iteration','frame_id','time_shift','learning_rates'],endpoints=paired))
    for name in ['identity_contract.json','config.json','smoke_summary.json','parent_unchanged.json','static_seed_audit.json','engineering_replay.json']:
        shutil.copy2(ROOT/name,dest/name)
    shutil.copy2(ROOT/'smoke_resume/restore_audit.json',dest/'restore_audit.json')
    shutil.copy2(ROOT/'formal10k/actual_identity_initialization.json',dest/'actual_identity_initialization.json')
    # 四件原生状态单独打包。正式训练全部完成于10k；smoke全部现场留在远端。
    archive=ROOT/'a37_B10k_replay.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as t:
        for prefix in ['chkpnt','dynamic_chkpnt','env_light_chkpnt','training_state']:
            add(t,ROOT/'formal10k'/f'{prefix}10000.pth','formal10k/'+f'{prefix}10000.pth')
        for name in ['effective_config.json','cameras.json','actual_scene_binding.json','actual_identity_initialization.json','returned.json','paired_steps.jsonl']:
            add(t,ROOT/'formal10k'/name,'formal10k/'+name)
        for name in ['source','identity_contract.json','config.json','smoke_summary.json','result_summary.json','parent_unchanged.json','static_seed_audit.json','engineering_replay.json']:
            add(t,ROOT/name,name)
        add(t,ROOT/'smoke_resume/restore_audit.json','validation/restore_audit.json')
        add(t,ROOT/'formal10k/checkpoint_events/iteration_010000.json','formal10k/checkpoint_events/iteration_010000.json')
        scene=Path(config['source_path'])
        for name in ['dynamic_masks','dynamic_actor_assets','points3d_dynamic_0.ply','preparation_manifest.json','dynamic_multi_actor_preparation.json','transform_matrix.txt']:
            add(t,scene/name,'inputs/'+name)
    h=hashlib.sha256()
    with archive.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    receipt=dict(file=archive.name,bytes=archive.stat().st_size,sha256=h.hexdigest(),native_four_components_included=True,
        training_RGB_LiDAR_and_native_engine_external=True,remote_run=str(ROOT),formal_stop=10000)
    write(ROOT/'package_receipt.json',receipt);write(dest/'package_receipt.json',receipt)
    with tarfile.open(ROOT/'review_light.tar.gz','w:gz',compresslevel=1) as t:
        for p in dest.iterdir():
            if p.name!='boards':add(t,p,'review/'+p.name)
    print(json.dumps(dict(status='PACKAGED_NATIVE10K',native_metrics=native,identity=q['retained_weighted'],package=receipt)),flush=True)

if __name__=='__main__':main()
