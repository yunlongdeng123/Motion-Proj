"""Spatial material holdouts from geometry/visibility only; no hidden RGB prior."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,hashlib,sys
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz,save_npz
from surface_material import build_surface_graph,split_holdout,training_median
CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign');OLD=CAM/'scene008_a6_asset_controls_20261105'
RUN=Path(__file__).resolve().parent.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def weighted_quantile(value,weight,q):
    order=np.argsort(value,kind='stable');c=np.cumsum(weight[order]);return float(value[order[np.searchsorted(c,q*c[-1])]])

def main():
    dest=RUN/'cases';dest.mkdir(exist_ok=False)
    A=load_npz(OLD/'m2/design.npz').tocsr();A.eliminate_zeros()
    # Only raw RGB target is consumed. The saved full-fit prior/RGB is forbidden.
    with np.load(OLD/'m2/inputs.npz') as x:target=x['target'].copy()
    geo=dict(np.load(OLD/'geometry_dense.npz'));xyz=geo['xyz'];weight=np.asarray(A.sum(0)).ravel()
    left=weighted_quantile(xyz[:,1],weight,.75);right=weighted_quantile(xyz[:,1],weight,.25);top=weighted_quantile(xyz[:,2],weight,.75)
    hidden={'left':xyz[:,1]>=left,'right':xyz[:,1]<=right,'top':xyz[:,2]>=top}
    contract=dict(status='PRE_REGISTERED_BEFORE_MATERIAL_FIT',geometry_sha256=sha(OLD/'geometry_dense.npz'),
       original_design_sha256=sha(OLD/'m2/design.npz'),raw_target_source_sha256=sha(OLD/'m2/inputs.npz'),
       planes=dict(left_y_ge=left,right_y_le=right,top_z_ge=top),
       plane_rule='Tail25% of BUILD projected column mass on canonical y+,y-,z+; geometry and alpha*T only, no RGB values.',
       training='Discard EVERY BUILD row with any nonzero coefficient in the spatial hidden set; no tolerance exemption.',
       evaluation='Original observed rows with >=.9 normalized contribution from hidden set; mixed boundary rows reported separately.',
       prior='Per-case median of retained raw training target rows only. Never read fitted M2 colours, original prior, or native predicted colours.',
       common_fit='Exactly2000 fullbatch projected FISTA SH0 RGB steps, lambda .001, RGB[0,1]; identical known fit shared by3 extension arms.',
       methods=['median','nearest','graph'],updates='Only mathematically zero columns of retained A; every nonzero column is frozen.',
       hypothesis='Geometry-only colour continuation may improve smooth missing appearance, but cannot guarantee hidden semantic texture.',
       adoption_gate='All3 spatial cases and temporal HELD evaluated; same fixed rule; known RGB preservation and full49 scene checks. Failure reported without parameter sweep.',
       scope='Same-scene material-only pseudo-occlusion. Geometry and PVD source are frozen and already use same-scene context.',
       formal_colour_steps=6000,neural_model_training_steps=0)
    reports=[]
    for name,H in hidden.items():
        folder=dest/name;folder.mkdir();split=split_holdout(A,H,eval_fraction=.9)
        train=np.asarray(split.train_rows);evaluation=np.asarray(split.eval_rows);mixed=np.asarray(split.mixed_rows)
        At=A[train].tocsr();At.eliminate_zeros();unknown=At.getnnz(axis=0)==0
        assert np.all(unknown[H]) and At[:,H].nnz==0
        median=training_median(target,train).astype(np.float32)
        target_train=target[train].astype(np.float32);prior=np.tile(median,(A.shape[1],1))
        # Counterfactual: change all removed RGB answers and rebuild train prior.
        perturbed=target.copy();perturbed[~train]=[.973,.127,.781]
        assert np.array_equal(perturbed[train],target[train]) and np.array_equal(training_median(perturbed,train),training_median(target,train))
        save_npz(folder/'train_design.npz',At)
        np.savez_compressed(folder/'train_inputs.npz',target=target_train,prior=prior,global_rgb=median,unknown=unknown,
             geometry_path=np.asarray(str(OLD/'geometry_dense.npz')))
        np.savez_compressed(folder/'split.npz',train_rows=train,eval_rows=evaluation,mixed_rows=mixed,hidden=H,unknown=unknown)
        row=dict(case=name,train_rows=int(train.sum()),spatial_eval_rows=int(evaluation.sum()),mixed_rows=int(mixed.sum()),
            hidden_primitives=int(H.sum()),unknown_columns=int(unknown.sum()),global_RGB=median.tolist(),
            hidden_training_columns_exact_zero=True,changed_hidden_target_training_bytes_unchanged=True,
            design_sha256=sha(folder/'train_design.npz'),inputs_sha256=sha(folder/'train_inputs.npz'),split_sha256=sha(folder/'split.npz'))
        reports.append(row);print(json.dumps(row),flush=True)
    write(dest/'contract.json',contract);write(dest/'preparation.json',dict(status='PASS',cases=reports,contract_sha256=sha(dest/'contract.json')))

if __name__=='__main__':main()
