"""Read one saved C49 phase on CPU; epoch loss is not a seam score.

--phase is deliberately one of smoke/formal; this tool never joins them.
Full epochs are exactly49 distinct raw89..137. Partial tails are separate.
Parameter comparison uses actual saved endpoints, not invented epoch snapshots.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def load_cpu(p):
    import torch
    assert not torch.cuda.is_initialized()
    try:return torch.load(str(p),map_location='cpu',mmap=True,weights_only=False)
    except (TypeError,RuntimeError,ValueError):return torch.load(str(p),map_location='cpu',weights_only=False)


def measure(rows):
    k=sum(float(r['keep_MSE']) for r in rows)/len(rows)
    d=sum(float(r['delete_MSE']) for r in rows)/len(rows)
    return dict(samples=len(rows),keep_MSE=k,delete_MSE=d,paired_MSE=.5*(k+d))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--phase',choices=['smoke','formal'],required=True);p.add_argument('--compare-checkpoint',type=Path)
    p.add_argument('--epoch-log',type=Path);p.add_argument('--fixed-review',type=Path);p.add_argument('--components-review',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
    files=[Path(__file__),a.checkpoint]+[v for v in [a.compare_checkpoint,a.epoch_log,a.fixed_review,a.components_review] if v is not None]
    before=[dict(path=str(v),sha256=sha(v),bytes=v.stat().st_size,mtime_ns=v.stat().st_mtime_ns) for v in files]
    state=load_cpu(a.checkpoint);history=state['history'];step=int(state['step']);assert history and len(history)==step
    assert [int(r['step']) for r in history]==list(range(1,step+1))
    assert all(int(r['epoch'])==(int(r['step'])-1)//49 for r in history)
    assert all(np.isfinite([r['keep_MSE'],r['delete_MSE']]).all() for r in history)
    assert state['config']['eligible_parents']==2119
    epochs=[];partial=[]
    for epoch in sorted({int(r['epoch']) for r in history}):
        rows=[r for r in history if int(r['epoch'])==epoch]
        complete=len(rows)==49 and sorted(int(r['raw']) for r in rows)==list(range(89,138))
        row=dict(epoch_zero_based=epoch,epoch_number=epoch+1,first_step=int(rows[0]['step']),last_step=int(rows[-1]['step']),complete=complete,**measure(rows))
        if complete:epochs.append(row)
        else:
            assert epoch==max(int(r['epoch']) for r in history) and len(rows)<49
            row['actual_raws']=[int(r['raw']) for r in rows];partial.append(row)
    window={'status':'INSUFFICIENT_TEN_COMPLETE_EPOCHS'}
    if len(epochs)>=10:
        prev,last=epochs[-10:-5],epochs[-5:]
        stats={}
        for key in ['keep_MSE','delete_MSE','paired_MSE']:
            b=sum(r[key] for r in prev)/5;c=sum(r[key] for r in last)/5
            stats[key]=dict(previous5=b,last5=c,absolute_delta=c-b,relative_delta=(c-b)/b if b else None)
        window=dict(status='COMPLETE_FIXED_FIVE_VS_PREVIOUS_FIVE',previous_epoch_numbers=[r['epoch_number'] for r in prev],last_epoch_numbers=[r['epoch_number'] for r in last],values=stats)
    epoch_log_check=None
    if a.epoch_log:
        records=[json.loads(line) for line in a.epoch_log.read_text().splitlines() if line.strip()]
        by={int(r['epoch']):r for r in records};checked=[]
        for e in epochs:
            row=by.get(e['epoch_zero_based'])
            if row:
                assert int(row['last_step'])==e['last_step']
                assert row['keep_MSE']==e['keep_MSE'] and row['delete_MSE']==e['delete_MSE']
                checked.append(e['epoch_number'])
        epoch_log_check={'checked_epoch_numbers':checked,'status':'PASS_EXACT_WHERE_PUBLISHED','missing_published_rows':len(epochs)-len(checked)}
    rgb=state['parent_RGB'].detach().cpu().numpy();assert rgb.shape==(2119,3) and np.isfinite(rgb).all() and rgb.min()>=0 and rgb.max()<=1
    material={'rows':2119,'scalars':6357,'min':float(rgb.min()),'max':float(rgb.max()),'endpoint_comparison':None,
              'last5_vs_previous5_RGB_window':'NOT_SAVED; cannot reconstruct parameter path from RGB min/max or training losses'}
    if a.compare_checkpoint:
        previous=load_cpu(a.compare_checkpoint)
        for key in ['eligible_parent_ids_sha256','initial_parent_RGB_sha256','qualification_manifest_sha256','training_contract_sha256','model_binding']:
            assert previous['config'][key]==state['config'][key],key
        old=previous['parent_RGB'].detach().cpu().numpy();assert old.shape==rgb.shape
        delta=rgb.astype(np.float64)-old.astype(np.float64)
        material['endpoint_comparison']={'from_step':int(previous['step']),'to_step':step,'mean_abs_RGB_change':float(abs(delta).mean()),
            'max_abs_RGB_change':float(abs(delta).max()),'RMS_RGB_change':float(np.sqrt((delta**2).mean())),
            'rows_changed_gt1e_6':int((abs(delta).max(1)>1e-6).sum()),
            'meaning':'Net endpoint displacement, not cumulative parameter travel or an epoch-window trace'}
    seam=None
    if a.fixed_review:
        review=json.loads(a.fixed_review.read_text());seam={'scope_raws':review['raw_ids'],'pooled_world_seams':review['pooled_world_seams'],
            'registered_failed_gates':review['failed_gates'],'unsupported_gates':review['unsupported_gates'],
            'role':'Saved-render fixed photometric/seam evaluation; separate from the training MSE axes and domains'}
    geometry=None
    if a.components_review:
        comp=json.loads(a.components_review.read_text());rows=[r['global_geometry_vs_initial'] for r in comp['records']]
        geometry={'actual_records':len(rows),'all_fields_present':all(row is not None and all('bitwise_equal' in v for v in row.values()) for row in rows),
            'all_saved_fields_bitwise_equal_to_Cwarm0':all(row is not None and all(v.get('bitwise_equal') is True for v in row.values()) for row in rows),
            'source_review_sha256':sha(a.components_review),'role':'Frozen geometry/visibility audit; cannot turn a known C2 occlusion failure into a pass'}
    with (a.output/(a.phase+'_epoch49_mse.csv')).open('w',newline='') as f:
        fields=['epoch_number','first_step','last_step','samples','keep_MSE','delete_MSE','paired_MSE'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in epochs:w.writerow({k:r[k] for k in fields})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4.5),constrained_layout=True)
    for key,label in [('keep_MSE','KEEP training MSE'),('delete_MSE','DELETE training MSE'),('paired_MSE','Paired objective MSE')]:
        ax.plot([r['last_step'] for r in epochs],[r[key] for r in epochs],label=label)
    ax.set(xlabel='Actual '+a.phase+' material updates (49 per complete epoch)',ylabel='Mean encoded RGB squared error on BUILD fit masks',title=a.phase.upper()+' only: complete49-view epochs; partial tail excluded')
    ax.grid(alpha=.25);ax.legend();fig.savefig(a.output/(a.phase+'_epoch49_mse.png'),dpi=150);fig.savefig(a.output/(a.phase+'_epoch49_mse.svg'));plt.close(fig)
    after=[dict(path=str(v),sha256=sha(v),bytes=v.stat().st_size,mtime_ns=v.stat().st_mtime_ns) for v in files];assert before==after
    result={'schema':'C49_EPOCH_AND_MATERIAL_ENDPOINT_CPU_V1','status':'PASS_SAVED_PHASE_TRACE_READOUT_NOT_QUALITY_ACCEPTANCE','phase':a.phase,'checkpoint_step':step,
        'complete_epochs':epochs,'complete_epoch_count':len(epochs),'partial_epoch_tail':partial,'window5_vs_previous5':window,'source_epoch_log_check':epoch_log_check,
        'parent_RGB':material,'independent_seam_evaluation':seam,'geometry_visibility_vs_Cwarm0':geometry,
        'loss_definition':'.5*(KEEP MSE+DELETE MSE), each state mean over that raw original legalfit_mask and3RGB channels; epoch is equal mean over49 frames, not fixed-road seam loss.',
        'new_CPU_GPU_forwards':0,'new_CPU_backwards':0,'new_CPU_optimizer_updates':0,'source_integrity':{'status':'PASS_SIZE_MTIME_SHA_UNCHANGED','files':before},
        'smoke_formal_concatenated':False,'quality_accepted':False,'human_verdict':None,
        'limits':['Training MSE decrease is not seam improvement. Report independent primary128/worldnear/lateral render metrics.',
          'Training parameters change within each49-view epoch; this is loss-at-visit-time, not a frozen-parameter49-view evaluation.',
          'All49 views are BUILD/development, not independent held-out generalisation.',
          'The frozen C geometry/real tire occlusion can persist at any RGB fit; inspect actual component/geometry evidence.',
          'Root decides whether2K suffices or a separately released5K/10K extension is worthwhile; no automatic adoption.']}
    out=a.output.parent/(a.output.name+'.json');out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(out),'phase':a.phase,'step':step,'complete_epochs':len(epochs),'partial_samples':sum(r['samples'] for r in partial),'quality_accepted':False}))


if __name__=='__main__':main()
