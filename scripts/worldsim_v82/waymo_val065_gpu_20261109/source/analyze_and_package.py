"""验算既有输出，打包逐帧PNG和必要证据；没有训练或电源操作。"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,hashlib,tarfile,math,time
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent.parent
PREP=Path('/root/autodl-tmp/codex_tmp/waymo_val065_cpu_20261108_r1')
KEYS=[30,60,150,198]
def read(p):return json.loads(p.read_text())
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def main():
    assert read(ROOT/'edit_pipeline_status.json')['stage']=='GPU_EDITS_COMPLETE_PENDING_ANALYSIS_RETRIEVAL_REPORT_PUSH'
    out=ROOT/'delivery';out.mkdir(exist_ok=False);start=time.monotonic()
    stages={k:read(ROOT/k/'summary.json') for k in ['formal10k_eval','semantic_all_eval','final_all_eval','final_heldout_eval','native_heldout_delete']}
    base=stages['formal10k_eval']['rows'];final=stages['final_all_eval']['frames'];held=stages['final_heldout_eval']['frames'];aggregate={}
    for key in ['target_white','front_pickup','preserved_pickup','all_other_vehicles']:
        valid=[v['regions'][key] for v in final if v['regions'][key]['pixels']>0];den=sum(v['pixels'] for v in valid)
        aggregate[key]=dict(frames=len(valid),pixels=den,native_mae=sum(v['pixels']*v['native_keep_gt_mae'] for v in valid)/den if den else None,current_mae=sum(v['pixels']*v['keep_gt_mae'] for v in valid)/den if den else None,delete_keep_mae=sum(v['pixels']*v['delete_vs_keep_mae'] for v in valid)/den if den else None,frames_keep_error_decreased=sum(v['keep_gt_mae']<v['native_keep_gt_mae'] for v in valid),frames_keep_error_increased=sum(v['keep_gt_mae']>v['native_keep_gt_mae'] for v in valid))
    full=[];layers=[];proxy=[]
    for raw in range(199):
        hold=79<=raw<=118;current=ROOT/('final_heldout_eval' if hold else 'final_all_eval')/f'{raw:03d}';native=ROOT/('native_heldout_delete' if hold else 'formal10k_eval')/f'{raw:03d}'
        gt=np.array(Image.open(current/'observed.png'),np.float32).transpose(2,0,1)/255
        a=np.load(native/'keep.npz')['rgb'];b=np.load(current/'keep.npz')['rgb']
        full.append(dict(raw=raw,heldout=hold,native_mae=float(abs(a-gt).mean()),current_mae=float(abs(b-gt).mean()),native_psnr=float(-10*np.log10(np.maximum(((a-gt)**2).mean(),1e-20))),current_psnr=float(-10*np.log10(np.maximum(((b-gt)**2).mean(),1e-20)))))
        if not hold:
            view=np.load(ROOT/'world_road/views'/f'{raw:03d}.npz');delete=np.load(current/'delete.npz')['rgb'];lin=np.where(delete<=.04045,delete/12.92,((delete.clip(0,1)+.055)/1.055)**2.4)
            for role,mask in [('observed_road',view['real_domain']&view['observed_road']),('hidden_observed_chart',view['real_domain']&~view['observed_road']),('hidden_inferred_chart',view['inferred_domain']&~view['observed_road'])]:
                n=int(mask.sum());proxy.append(dict(raw=raw,role=role,pixels=n,linear_proxy_mae=float(abs(lin-view['reference_linear'])[:,mask].mean()) if n else None))
        if raw in KEYS:
            labels=np.array(Image.open(PREP/'masks_all_resolved_r2'/f'{raw:03d}/labels.png'));mask=labels==10
            for stage in ['semantic_all_eval','final_all_eval']:
                for state in ['keep','delete']:
                    z=np.load(ROOT/stage/f'{raw:03d}/{state}.npz');groups=z['group_alpha'];error=float(abs(groups.sum(0)-z['alpha'][0]).max());assert error<3e-6
                    layers.append(dict(raw=raw,stage=stage,state=state,target_mask_pixels=int(mask.sum()),mean_group_mass_on_target=groups[:,mask].mean(1).tolist(),conservation_maxabs=error))
        print('ANALYZE',raw,flush=True)
    groupstats={}
    for split in [False,True]:
        rows=[v for v in full if v['heldout']==split];groupstats['heldout' if split else 'build']=dict(frames=len(rows),**{k:float(np.mean([v[k] for v in rows])) for k in ['native_mae','current_mae','native_psnr','current_psnr']})
    native_groups=read(ROOT/'native_source_attribution/summary.json')
    summary=dict(task_id='WAYMO-VAL065-A9-GPU-20261109',status='EXECUTED_MECHANICAL_CHECKS_PASS_QUALITY_UNJUDGED',formal=read(ROOT/'formal10k/returned.json'),smoke=read(ROOT/'smoke_gate.json'),target_native_source_delete_unchanged_frames=sum(v['changed_pixels']==0 for v in base),native_source_target_zero_mass_frames=sum(v['target_keep_alpha_max']<.01 for v in base),native_frames=len(base),semantic_KEEP_native_maxabs=max(v['native_keep_parity_maxabs'] for v in stages['semantic_all_eval']['frames']),V1_maxabs=max(v['saved_native_maxabs'] for v in read(ROOT/'formal10k_v1/summary.json')['rows']),native_source_groups=native_groups,regions=aggregate,full_view=groupstats,full_view_frames=full,key_group_attribution=layers,road_proxy_regions=proxy,scope='shared ownership plus road handoff; no new vehicle surface-completion asset',vehicle_reveal_correspondences=0,max_road_correspondence_cells=11,quality_accepted=None,human_verdict=None,automatic_shutdown=False,failure_ledger_delta='none')
    dump(out/'metrics.json',summary)
    files=[]
    def add(src,arc):
        assert src.is_file(),src
        files.append((src,arc))
    for raw in range(199):
        hold=79<=raw<=118;c=ROOT/('final_heldout_eval' if hold else 'final_all_eval')/f'{raw:03d}';n=ROOT/('native_heldout_delete' if hold else 'formal10k_eval')/f'{raw:03d}';v=ROOT/('native_heldout_v1' if hold else 'formal10k_v1')/f'{raw:03d}'
        add(c/'observed.png',f'viewer/observed/{raw}.png');add(v/'keep.png',f'viewer/native/{raw}/keep.png');add(n/'delete.png',f'viewer/native/{raw}/delete.png')
        for state in ['keep','delete']:add(c/f'{state}.png',f'viewer/current/{raw}/{state}.png')
    for raw in KEYS:
        for state in ['keep','delete']:add(ROOT/'semantic_all_eval'/f'{raw:03d}/{state}.png',f'viewer/semantic/{raw}/{state}.png')
        for state in ['keep','delete']:add(ROOT/'final_all_eval'/f'{raw:03d}/{state}.npz',f'evidence/key_float/{raw}/{state}.npz')
        for state in ['keep','delete']:add(ROOT/'native_source_attribution'/f'{raw:03d}/{state}.npz',f'evidence/native_float/{raw}/{state}.npz')
        add(PREP/'masks_all_resolved_r2'/f'{raw:03d}/labels.png',f'evidence/masks/{raw}.png')
    for stage in ['formal10k_eval','semantic_all_eval','final_all_eval','final_heldout_eval','native_heldout_delete','native_heldout_v1','formal10k_v1','smoke_eval','smoke_v1']:
        add(ROOT/stage/'summary.json',f'evidence/{stage}_summary.json')
    for rel in ['config.json','identity_contract.json','preparation_receipt.json','arm_contract.json','all_cameras.json','smoke_gate.json','formal10k/returned.json','formal10k/effective_config.json','formal10k/runtime_source_manifest.json','formal10k/user_budget_and_diagnostic_cadence.json','smoke_r1/returned.json','smoke_restore/restore_audit.json','smoke_restore/resume_entry.json','semantic_all159/receipt.json','world_road/receipt.json','road_gate32/receipt.json','heldout/receipt.json','edit_pipeline_status.json']:
        add(ROOT/rel,'evidence/'+rel.replace('/','__'))
    for rel in ['road_gates/receipt.json','road_gates/restore.json']:
        if (ROOT/rel).is_file():add(ROOT/rel,'evidence/'+rel.replace('/','__'))
    for name in ['selected_case.json','inputs/receipt.json','inputs/verification.json','inputs/explicit_timeline.json','inputs/track_registry.json','pilot_qa.json','gpu_policy_20261109.json','verified_surface_correspondences.json','mask_pilot_anchored_r2/receipt.json','masks_all_resolved_r2/receipt.json','semantics/receipt.json']:
        add(PREP/name,'evidence/input__'+name.replace('/','__'))
    for p in (ROOT/'source').glob('*.py'):add(p,'evidence/source/'+p.name)
    for p in (PREP/'source').glob('*.py'):add(p,'evidence/input_source/'+p.name)
    for name in ['smoke_r1','smoke_restore','formal10k']:
        for filename in ['progress.json','runtime.jsonl']:
            p=ROOT/name/filename
            if p.exists():add(p,f'evidence/logs/{name}__{filename}')
    add(out/'metrics.json','metrics.json')
    manifest=[]
    archive=out/'scene065_results.tar'
    with tarfile.open(archive,'w') as tar:
        for src,arc in files:
            payload=src.read_bytes();manifest.append(dict(member=arc,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()));tar.add(src,arcname=arc)
    dump(out/'retrieval_manifest.json',dict(status='PACKAGED_COMPLETE_RESULTS',files=manifest,bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),weights_retained_remote=str(ROOT/'formal10k'),seconds=time.monotonic()-start,automatic_shutdown=False))
    print('PACKAGE_COMPLETE',archive.stat().st_size,flush=True)
if __name__=='__main__':main()
