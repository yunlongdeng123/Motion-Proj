"""Independent full-state and standard-NPZ disk reload, with zero optimizer steps."""
from pathlib import Path
import argparse
import time
import numpy as np
from run_control import bank,bootstrap,control_actor,prove_frozen,read,ref,restore,sha,targets,write


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('contract','assets','output','formal-receipt','approval'):p.add_argument('--'+k,type=Path,required=True)
    args=p.parse_args();args.resume=None
    approval=read(args.approval);formal=read(args.formal_receipt)
    if not approval.get('approved') or approval['contract_sha256']!=sha(args.contract) or approval['assets_sha256']!=sha(args.assets/'assets_receipt.json') or approval['runtime_sha256']!=sha(Path(__file__).with_name('run_control.py')):
        raise ValueError('Formal control approval binding changed')
    if formal['steps']!=80 or formal['role']!='formal80':raise ValueError('Expected final native-support control')
    actor=args.formal_receipt.parent/'actor.npz';checkpoint=args.formal_receipt.parent/'checkpoint_final.pt'
    if sha(actor)!=formal['actor']['sha256'] or sha(checkpoint)!=formal['checkpoint']['sha256']:raise ValueError('Formal files changed')
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();c,b,r=bootstrap(args);tar=targets(args,r)
    t=r.torch;t.cuda.reset_peak_memory_stats()
    def real_metrics(out,key):
        V=r.real[key]['V'];gt=r.real[key]['rgb'];inner=r.F.avg_pool2d(V.float()[None,None],5,1,2)[0,0]>=1-1e-6
        return dict(key=key,RGB_L1=float(r.mean((out['rgb']-gt).abs().mean(0),V)),edge_L1=float(r.m.edge(out['rgb'],gt,V)),HF=r.m.hf(out['rgb'],gt,inner))
    with t.inference_mode():
        initial=[real_metrics(r.render(r.student,key,'eval'),key) for key in r.real]
        original=[real_metrics(tar[key],key) for key in r.real]
    opt=t.optim.Adam([r.student.observed_dc],lr=.01,betas=(.9,.999),eps=1e-8)
    state=t.load(checkpoint,map_location='cpu',weights_only=False);restore(args,r,opt,state)
    exported=bank(actor);observed=r.m.cpu(r.student.observed_ids).numpy()
    if not np.array_equal(r.m.cpu(r.student.observed_dc).numpy(),exported['coefficients'][observed,0]):raise RuntimeError('Checkpoint mutable state differs from standard NPZ')
    checkpoint_outputs={};comparison=[]
    with t.inference_mode():
        for key in r.cameras:
            out=r.render(r.student,key,'eval');grey=r.render(r.student,key,'eval',grey=True)
            checkpoint_outputs[key]={k:r.m.cpu(v).numpy() for k,v in dict(rgb=out['rgb_premult'],alpha=out['alpha'],grey=grey['rgb_premult']).items()}
            folder=args.output/'checkpoint_reload'/key
            for name,value in [('rgb',out['rgb_premult']),('alpha',out['alpha']),('grey',grey['rgb_premult'])]:r.m.png(folder/(name+'.png'),value)
    r.student=control_actor(r,exported)
    for parameter in r.student.parameters():parameter.requires_grad_(False)
    prove_frozen(r)
    with t.inference_mode():
        for key in r.cameras:
            out=r.render(r.student,key,'eval');grey=r.render(r.student,key,'eval',grey=True)
            for name,value,original_name in [('rgb',out['rgb_premult'],'student_rgb_premult.png'),('alpha',out['alpha'],'student_alpha.png'),('grey',grey['rgb_premult'],'student_grey.png')]:
                current=r.m.cpu(value).numpy();delta=np.abs(current-checkpoint_outputs[key][name])
                before=np.asarray(r.m.Image.open(args.formal_receipt.parent/'eval'/key/original_name)).astype(np.int16)
                q=np.round(np.clip(current,0,1)*255).astype(np.uint8)
                if q.ndim==3:q=q.transpose(1,2,0)
                quantized=np.abs(q.astype(np.int16)-before)
                comparison.append(dict(key=key,channel=name,checkpoint_vs_NPZ_max_float_difference=float(delta.max()),
                    NPZ_vs_resident_max_8bit_difference=int(quantized.max()),changed_8bit_values=int((quantized>0).sum())))
                r.m.png(args.output/'NPZ_reload'/key/(name+'.png'),value)
    t.cuda.synchronize()
    write(args.output/'receipt.json',dict(schema='a20.native_control.independent_disk_reload.v1',source=ref(__file__),
        formal_receipt=ref(args.formal_receipt),actor=ref(actor),checkpoint=ref(checkpoint),
        full_mutable_optimizer_and_RNG_restore=True,full_immutable_model_source_hash_verified=True,
        checkpoint_observed_DC_equals_exported_NPZ=True,geometry_and_unknown_DC_bitwise_equal=True,
        render_comparison=comparison,maximum_float_difference=max(x['checkpoint_vs_NPZ_max_float_difference'] for x in comparison),
        maximum_8bit_difference=max(x['NPZ_vs_resident_max_8bit_difference'] for x in comparison),
        original_teacher_real_V=original,clean_control_init_real_V=initial,formal80_real_V=formal['metrics']['real_V'],
        optimizer_steps=0,training_updates=0,native_calls=r.calls,heldout_opened=[],
        wall_seconds_including_load=time.monotonic()-started,cuda_peak_allocated_bytes=t.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=t.cuda.max_memory_reserved()))
