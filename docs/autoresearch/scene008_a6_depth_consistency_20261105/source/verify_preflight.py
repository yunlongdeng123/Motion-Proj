"""Independent raw-calibration coordinate check and BUILD preflight replay."""
import numpy as np
import torch
from consistency import RUN,Consistency,Window,read,write
from rescue import OBS
from reexpression_preflight import frame_roles

def main():
    source=read(RUN/'preflight.json');e=Consistency();records=[]
    with np.load(RUN/'preflight_ledger.npz',allow_pickle=False) as f:
        ledger={k:f[k] for k in f.files}
    assert np.array_equal(ledger['native_row_id'],e.tr.native_ids.cpu().numpy())
    totals=np.asarray([r['masses'] for r in source['frame_receipts']]).sum(0)
    assert np.allclose(totals,ledger['masses'].sum(1),rtol=0,atol=1e-8)
    for raw in (106,114,129):
        masses,_,receipt=frame_roles(e,raw)
        old=next(r for r in source['frame_receipts'] if r['raw']==raw)
        delta=float(np.abs(np.asarray(receipt['masses'])-old['masses']).max())
        assert delta<1e-8,(raw,delta)
        w=e.frame(raw,'black');t=w.trace;d=e.tr.raw(raw)
        selected=d['depth_valid'].flatten()[t.pixels.long()]
        pixels=t.pixels.long()[selected]
        ray=Window.directions(t)[selected]
        origin,direction=Window.in_car(e,t,ray)
        z=d['depth'].flatten()[pixels]
        actual=origin.double()+direction.double()*(z.double()*e.tr.scale)[:,None]
        with np.load(OBS/'frames'/f'{raw:03d}.npz',allow_pickle=False) as frame:
            K=frame['K'];inverse=np.linalg.inv(frame['world_to_camera']@frame['world_from_local'][0])
        px=(pixels%t.w).double()+.5;py=(pixels//t.w).double()+.5
        camera=torch.stack(((px-K[0,2])*z/K[0,0],(py-K[1,2])*z/K[1,1],z.double()),1)
        matrix=camera.new_tensor(inverse)
        expected=camera@matrix[:3,:3].T+matrix[:3,3]
        error=float((actual-expected).abs().max()) if len(actual) else 0.
        assert error<1e-4,(raw,error)
        records.append(dict(raw=raw,mass_replay_maxabs=delta,raw_calibration_local_xyz_maxabs_m=error,
                            measured_points=len(actual)))
    assert e.tr.exp.unchanged()
    write(RUN/'preflight_verification.json',dict(status='PASS',records=records,
        receipt_mass_ledger_totals_checked=True,independent_raw_coordinate_mapping=True,
        note='Three BUILD frames; children re-expression not implemented or tested'))

if __name__=='__main__':
    torch.set_num_threads(1)
    with torch.no_grad():main()
