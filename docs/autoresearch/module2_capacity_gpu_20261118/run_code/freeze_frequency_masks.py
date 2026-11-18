from pathlib import Path
import hashlib
import json
import numpy as np

C=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_capacity_cpu_20261117_r1')
G=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_capacity_gpu_20261118_r1')
dest=G/'check3/frozen_frequency_masks.npz'
if dest.exists():raise FileExistsError('不按结果重选频率掩码')
arrays={};rows=[]
for raw in (86,110,134,179,198):
    path=C/'ideal_targets_r2/geometry'/f'{raw:03d}.npz'
    with np.load(path,allow_pickle=False) as data:
        D=data['fixed_evaluation_domain'].astype(bool)
        alpha=data['fixed_alpha'].reshape(D.shape)
        P=data['baseline_forward_rgb_srgb__not_teacher']
    if P.shape[0]!=3:P=P.transpose(2,0,1)
    M=D&(alpha>.9)
    V=M[2:]&M[1:-1]&M[:-2]
    H=M[:,2:]&M[:,1:-1]&M[:,:-2]
    arrays[f'{raw:03d}_domain']=M
    arrays[f'{raw:03d}_vertical']=V;arrays[f'{raw:03d}_horizontal']=H
    P=P.astype(np.float64)
    v=np.abs(P[:,2:]-2*P[:,1:-1]+P[:,:-2])
    h=np.abs(P[:,:,2:]-2*P[:,:,1:-1]+P[:,:,:-2])
    rows.append(dict(raw=raw,role='train' if raw in (86,110) else 'diagnostic',
        domain_pixels=int(M.sum()),vertical_stencils=int(V.sum()),horizontal_stencils=int(H.sum()),
        baseline_vertical_E=float(v[:,V].mean()),baseline_horizontal_E=float(h[:,H].mean()),
        baseline_delta_energy=0.,source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
np.savez_compressed(dest,**arrays)
receipt=dict(schema='preregistered_native_frequency_masks.v1',status='frozen_before_formal',
    masks=str(dest),definition='M = fixed D_full & archived oldS2 alpha>0.9; each second-difference stencil requires all3 M pixels',
    pixel_metric='mean absolute second difference of road-only premult native sRGB,3-channel mean; no RGB/alpha',
    primary='vertical second difference measures horizontal stripe energy; horizontal and row-RGB delta energy also reported',
    excess_rule='I0_excess=E_I0-E_c0. If positive, W4 excess<=0.5*I0 excess is stripe gate; nonpositive means not-applicable',
    other_gates='same teacher fit/detail/K/persistence still required; not a standalone image-quality certificate',
    rows=rows,masks_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
    selected_from_future_optimization_results=False,native_GPU_calls=0,human_verdict=None)
(G/'check3/frozen_frequency_masks.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(frozen=True,domain=sum(row['domain_pixels'] for row in rows),rows=rows)))
