"""输出可用于技术报告的训练支持退化曲线；按更新计数，不冒充epoch末Actor覆盖率。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    groups=defaultdict(list)
    for line in (args.run/'train.jsonl').read_text().splitlines():
        row=json.loads(line); groups[row['epoch']].append(row)
    rows=[]
    for epoch,part in sorted(groups.items()):
        rows.append({'epoch':epoch,'updates':len(part),
            'fallback_fraction':statistics.mean(float(r['lidar_fallback']) for r in part),
            'depth_gradient_fraction':statistics.mean(float(r['native_output_gradient_after_clip']>0) for r in part),
            'mean_free_m':statistics.mean(r['free_intrusion_m'] for r in part),
            'mean_coverage_m':statistics.mean(r['coverage_m'] for r in part)})
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'epoch_support.json').write_text(json.dumps(rows,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(9.3,3.1),layout='constrained')
    x=[r['epoch'] for r in rows]
    axes[0].plot(x,[r['fallback_fraction'] for r in rows],color='#b64034',label='LiDAR fallback')
    axes[0].plot(x,[r['depth_gradient_fraction'] for r in rows],color='#176a9c',label='Nonzero native depth gradient')
    axes[0].set(xlabel='Epoch',ylabel='Fraction of fit updates',ylim=(-.03,1.03),title='Predicted support and gradient path')
    axes[0].legend(fontsize=8,loc='best')
    axes[1].semilogy(x,[max(r['mean_free_m'],1e-8) for r in rows],color='#b64034',label='Hard first-return intrusion')
    axes[1].semilogy(x,[max(r['mean_coverage_m'],1e-8) for r in rows],color='#176a9c',label='Observed-point surface distance')
    axes[1].set(xlabel='Epoch',ylabel='Mean training value (m)',title='Loss reduction can accompany fallback')
    axes[1].legend(fontsize=8,loc='best')
    for axis in axes:
        axis.spines[['top','right']].set_visible(False)
        axis.grid(alpha=.18)
    fig.savefig(args.output/'native_support_training.png',dpi=220)
    fig.savefig(args.output/'native_support_training.pdf')
    plt.close(fig)
    print(json.dumps({'epochs':len(rows),'updates':sum(r['updates'] for r in rows),'output':str(args.output)}))


if __name__=='__main__': main()
