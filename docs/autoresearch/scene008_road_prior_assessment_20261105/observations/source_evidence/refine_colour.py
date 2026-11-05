"""One finer world grid, initialized by the accepted coarse calibration."""
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import lsq_linear
import torch

RUN=Path(__file__).resolve().parent.parent
COARSE=RUN.parent/'scene008_ray_road_colour_20261105'
sys.path.insert(0,str(COARSE/'source'))
import calibrate

calibrate.RUN=RUN
calibrate.GRID=(11,9)
from calibrate import Experiment as Base, MAX_DELTA, grid_reg, write, read


class Experiment(Base):
    def solve(self):
        with np.load(COARSE/'coefficients.npz',allow_pickle=False) as source:
            coarse=source['world'].reshape(6,5,3)
        fine=np.zeros((11,9,3))
        for x in range(11):
            for y in range(9):
                i,j=min(x//2,4),min(y//2,3)
                a,b=x*.5-i,y*.5-j
                fine[x,y]=(1-a)*(1-b)*coarse[i,j]+a*(1-b)*coarse[i+1,j]+(1-a)*b*coarse[i,j+1]+a*b*coarse[i+1,j+1]
        anchor=fine.reshape(-1,3)
        A=[];target=[];flags=[];weights=[]
        for state in ('keep','delete'):
            state_data=[]
            for path in sorted((RUN/'design').glob(f'*_{state}.npz')):
                with np.load(path,allow_pickle=False) as data:
                    state_data.append((data['basis'],data['residual'],data['fit']))
            m=sum(int(item[2].sum()) for item in state_data)
            n=sum(int((~item[2]).sum()) for item in state_data)
            for basis,residual,fit in state_data:
                A.append(basis)
                target.append(residual)
                flags.append(fit)
                weights.append(np.where(fit,1/max(1,m),.40/max(1,n)))
        A=np.concatenate(A).astype(np.float64)
        target=np.concatenate(target).astype(np.float64)
        flags=np.concatenate(flags)
        weights=np.concatenate(weights)
        # Penalize fine-scale deviation from the accepted coarse world field.
        reg=grid_reg()+np.eye(len(anchor))*.035
        fitted=[]
        for k in range(3):
            x=anchor[:,k]
            for _ in range(6):
                e=A@x-target[:,k]
                robust=np.where(flags,1/np.sqrt(e*e+.012**2),1/.025)
                w=weights*robust
                H=A.T@(w[:,None]*A)+reg
                y=A.T@(w*target[:,k])+reg@anchor[:,k]
                chol=np.linalg.cholesky(H)
                x=lsq_linear(chol.T,np.linalg.solve(chol,y),bounds=(-MAX_DELTA,MAX_DELTA),method='bvls',tol=1e-10).x
            fitted.append(x)
        fine=np.stack(fitted,1)
        np.savez_compressed(RUN/'coefficients.npz',world=fine,coarse=anchor,
                            colour_baseline=self.colour.cpu().numpy())
        return dict(coarse=anchor,world=fine)


if __name__=='__main__':
    torch.set_num_threads(1)
    with torch.no_grad():
        experiment=Experiment()
        experiment.cache_design()
        experiment.evaluate(experiment.solve())
