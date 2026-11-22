"""Render stored agreement matrices with unknown/self cells explicitly grey."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    src=a.directory/'agreement_matrices.npz'
    with np.load(src) as z:
        frames=z['frames'];radii=z['radii_m'];signed=z['median_signed_Z_m'][0];coverage=z['coverage_fraction'][0];count=z['matched_counts'][0]
    cmap=plt.get_cmap('coolwarm').copy();cmap.set_bad('#657080');covmap=plt.get_cmap('viridis').copy();covmap.set_bad('#657080')
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained');n=len(frames)
    for j,radius in enumerate(radii):
        q=signed[j].copy();c=coverage[j].copy();np.fill_diagonal(q,np.nan);np.fill_diagonal(c,np.nan)
        # No observed spatial neighbour cannot become a zero-height error.
        q[count[j]==0]=np.nan;c[count[j]==0]=np.nan
        im=axes[0,j].imshow(q*100,vmin=-20,vmax=20,cmap=cmap);axes[0,j].set_title(f'XY <= {radius*100:g} cm: signed Z median (cm)');fig.colorbar(im,ax=axes[0,j],shrink=.7)
        im=axes[1,j].imshow(c,vmin=0,vmax=1,cmap=covmap);axes[1,j].set_title('Matched / all original target points');fig.colorbar(im,ax=axes[1,j],shrink=.7)
        for ax in axes[:,j]:ax.set(xticks=np.arange(n),yticks=np.arange(n),xticklabels=frames,yticklabels=frames,xlabel='reference frame',ylabel='target frame');ax.tick_params(axis='x',rotation=90)
    fig.suptitle('All 17 frames, actual requested domain. Grey = self or no match. No Z filtering / alignment.');out=a.directory/'matrix_unknown_grey.png';fig.savefig(out,dpi=160);plt.close(fig)
    meta=dict(complete=True,CPU_only=True,nativeF=0,nativeB=0,updates=0,source_matrix_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),image_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),display_only=True,unknown_not_zero_error=True)
    (a.directory/'matrix_display_receipt.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))


if __name__=='__main__':main()
