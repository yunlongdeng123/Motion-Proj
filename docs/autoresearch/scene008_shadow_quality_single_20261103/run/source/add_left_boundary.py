import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,shutil
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation,binary_erosion
RUN=Path(__file__).resolve().parent.parent;OLD=RUN.with_name('shadow_a5_a37_199_20261103_r4')
def rgb(p):return np.asarray(Image.open(p),np.float32)/255.
def lin(x):return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
def main():
    p=RUN/'quality/metrics.json';shutil.copy2(p,p.with_name('metrics_before_left_boundary.json'));q=json.loads(p.read_text())
    for row in q['rows']:
        raw=row['raw'];d=np.load(RUN/'inputs/views'/('%03d.npz'%raw));base=rgb(RUN/'evaluation/all49'/('%03d_baseline_delete5.png'%raw));b=lin(base);ref=d['reference_linear']
        y=b@np.array([.2126,.7152,.0722]);ry=ref@np.array([.2126,.7152,.0722]);domain=d['repair_mask']
        dark=domain&(y<.78*np.maximum(ry,.01))&((ry-y)>.025)
        ring=binary_dilation(domain,iterations=3)&~binary_erosion(domain,iterations=3)&~binary_dilation(dark,iterations=5)&~d['other_vehicle_guard']
        mask=ring&(np.indices(y.shape)[1]<220);row['left_original_boundary_samples']=int(mask.sum())
        for version,folder in [('R4',OLD),('R8',RUN)]:
            im=rgb(folder/'evaluation/all49'/('%03d_repaired_delete5.png'%raw));res=im-base;g=np.zeros(y.shape)
            g[:,1:]+=np.abs(np.diff(res,axis=1)).mean(2);g[1:]+=np.abs(np.diff(res,axis=0)).mean(2)
            row[version]['left_original_boundary_residual_gradient']=float(g[mask].mean()) if mask.any() else None
    den=sum(x['left_original_boundary_samples'] for x in q['rows'])
    key='left_original_boundary_residual_gradient';q['aggregates'][key]=dict(samples=den,**{v:sum(x[v][key]*x['left_original_boundary_samples'] for x in q['rows'] if x[v][key] is not None)/den for v in ['R4','R8']})
    p.write_text(json.dumps(q,indent=2));print(json.dumps(q['aggregates'][key]))
if __name__=='__main__':main()
