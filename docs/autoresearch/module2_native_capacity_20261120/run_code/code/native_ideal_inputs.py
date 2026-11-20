"""Qualified analytic source I for the registered native-GS capacity control.

Pure NumPy; no torch/native import, context, optimizer or automatic execution.
"""
from pathlib import Path
import hashlib,json
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ideal_frame_from_arrays(z,legal,*,case_index=1):
    alpha=np.asarray(z['road_fixed_alpha']);weight=np.asarray(z['road_eligible_weight'])
    score=np.asarray(legal,bool)
    if alpha.shape!=score.shape or weight.shape!=score.shape:raise ValueError('ideal mask/alpha shape mismatch')
    context=np.asarray(z['ideal_available'],bool)&score&(alpha>=.95)&(weight>=.95)&~np.asarray(z['road_K'],bool)
    P=np.asarray(z['road_target_P'][case_index])
    if P.shape!=(3,*score.shape)or not np.isfinite(P[:,context]).all():raise ValueError('ideal premult target shape/value mismatch')
    target=np.zeros_like(P)
    np.divide(P,alpha[None],out=target,where=context[None])
    return dict(alpha=alpha.copy(),targets={'I':target},contexts={'I':context},scores={'I':score.copy()})


def load_ideal_frames(protocol):
    cfg=protocol['arms']['IDEAL_ORACLE'];root=Path(cfg['target_root']);bench=Path(protocol['data']['teacher_root'])
    if cfg['case']!='marking_w120':raise ValueError('only registered12cm control is allowed')
    audit_path=root.parent/'preparation_audit_r2.json';audit=json.loads(audit_path.read_text())
    if audit.get('numeric_qualified')is not True or audit.get('full_target_qualified')is not True:raise RuntimeError('idealr2 numeric/source qualification not passed')
    frames={};files={str(audit_path):sha(audit_path)};counts={}
    for raw in protocol['data']['frames']:
        path=root/f'{raw:03d}.npz'
        with np.load(path,allow_pickle=False)as z:
            if list(z['cases'])!=['asphalt','marking_w120','marking_w240']:raise ValueError('ideal case-index identity differs')
            arrays={k:z[k].copy()for k in ('road_fixed_alpha','road_eligible_weight','ideal_available','road_K','road_target_P')}
        target_path=bench/f'{raw:03d}.npz'
        with np.load(target_path,allow_pickle=False)as z:legal=z['A_pre_alpha']|z['C_pre_alpha']|z['S_pre_alpha']
        frames[int(raw)]=ideal_frame_from_arrays(arrays,legal,case_index=1)
        files.update({str(path):sha(path),str(target_path):sha(target_path)})
        context=frames[int(raw)]['contexts']['I']
        counts[str(raw)]=dict(full_score=int(legal.sum()),qualified_context=int(context.sum()),excluded_context=int((legal&~context).sum()))
    return frames,dict(files=files,counts=counts,role='open_analytic_world_synthetic_I_not_realA_not_held',case_index=1,
        target='T=Pstar/alpha on qualifiedcontext; RegionQuadratic then formsalpha*T=Pstar once',
        source_weights=[['I',1.]],QA=str(audit_path),ordinary_GS_primary=True)
