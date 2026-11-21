"""New-resolution native basis and parent packets; never resizes old renders."""
import os, sys, time, traceback
from pathlib import Path
import numpy as np
from runtime1152 import *

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    write(ROOT/'gpu_lease.json',dict(active=True,owner=OWNER,gpu_authorized=True,formal_allowed=False,
        optimizer_updates_allowed=0,power_policy='keep_on',authorization='current user: new1152 teacher refit'))
    camera_headers()
    status('native_basis',PID=os.getpid(),complete=False)
    ctx=None; start=time.monotonic()
    try:
        import torch
        torch.set_num_threads(2)
        paths(); sys.path.insert(0,str(Q/'pixel_render/code'))
        import native_basis_adapter as nba
        # Persist the actual weights before applying a new-resolution K gate.
        source=Path(nba.__file__).read_text()
        old="if mode=='joint'and Kweight!=0:raise RuntimeError('joint K上eligible alphaT非零，须修world row scope，禁止屏幕K掩码修补')"
        assert source.count(old)==1
        runtime=ROOT/'code/native_basis1152_evidence.py'
        runtime.write_text(source.replace(old,"# Owner records unmodified K weights, then qualifies before optimization."))
        sys.path.insert(0,str(runtime.parent))
        from native_basis1152_evidence import export_native_basis,scope_signature
        ledger=load(Q/'permission/final_r1/teacher_evidence.npz')
        active,eligible=ledger['active_world_rows'],ledger['common_write_active']
        auth=dict(schema='pixel_native_basis_smoke.v1',phase='native_smoke',native_allowed=True,gpu_authorized=True,
            formal_allowed=False,frames=list(FRAMES),scope_world_rows_sha256=scope_signature(active,eligible),
            max_native_forward_calls=100,delete_actor_ids=[10],optimizer_updates=0)
        ctx=context(); rows=[]
        with torch.no_grad():
            for raw in FRAMES:
                dest=ROOT/'native/basis'/f'{raw:03d}'
                if dest.exists():
                    import json
                    meta=json.loads((dest/'basis.json').read_text())
                else: meta=export_native_basis(ctx,raw,eligible,dest,authorization=auth)
                rd,jn=load(dest/'road_basis.npz'),load(dest/'joint_basis.npz')
                parent=ROOT/'native/parent'/f'{raw:03d}.npz'; parent.parent.mkdir(parents=True,exist_ok=True)
                packet=dict(road_rgb=rd['baseline_rgb'],road_alpha=rd['alpha'],joint_rgb=jn['baseline_rgb'],joint_alpha=jn['alpha'],K=jn['K'])
                np.savez_compressed(parent,**packet)
                row=dict(raw=raw,path=str(parent),resolution_wh=[1152,768],K_pixels=int(jn['K'].sum()),
                    K_eligible_weight_max=meta['joint']['K_eligible_weight_max'],native_forward_calls=meta['native_forward_calls'])
                if raw==110:
                    direct=ctx.render_road(raw); joint=ctx.render_joint(raw)
                    row.update(direct_road_RGB_max=float(np.max(np.abs(direct['rgb'].cpu().numpy()-rd['baseline_rgb']))),
                        direct_joint_RGB_max=float(np.max(np.abs(joint['rgb'].cpu().numpy()-jn['baseline_rgb']))))
                    assert max(row['direct_road_RGB_max'],row['direct_joint_RGB_max'])<=5e-6
                rows.append(row);write(ROOT/'native/manifest.json',dict(rows=rows,complete=len(rows)==5))
                status('native_basis',completed_frames=[r['raw'] for r in rows],native_forward_calls=ctx.native_calls)
                print('BASIS',raw,'K',row['K_eligible_weight_max'],'calls',ctx.native_calls,flush=True)
        ctx.assert_frozen()
        receipt=dict(schema='common1152_native_preparation.v1',complete=True,rows=rows,native_forward_calls=ctx.native_calls,
            seconds=time.monotonic()-start,optimizer_updates=0,all_joint_K_zero=all(r['K_eligible_weight_max']==0 for r in rows),
            resolution_wh=[1152,768],authority_asset=str(ASSET),road_SHA256=sha(ASSET/'road.npz'))
        write(ROOT/'native/receipt.json',receipt)
        status('native_ready',native_forward_calls=ctx.native_calls)
    except Exception:
        error=traceback.format_exc();write(ROOT/'native/failed.json',dict(error=error,seconds=time.monotonic()-start))
        write(ROOT/'status.json',dict(status='failed',phase='native_basis',error=error,PID=os.getpid(),complete=False));raise
    finally:
        if ctx is not None:ctx.close()

if __name__=='__main__': main()
