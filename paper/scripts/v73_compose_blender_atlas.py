"""组合真实Blender图；初次排版只使用已完成图，最终要求七方法齐全。"""
from pathlib import Path
import json,os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(os.environ.get('WORLDSIM_PAPER_STAGE',Path(__file__).resolve().parent))
OUT=ROOT/'paper/figures/v73'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
def panel(methods,name,require=False):
 available=[]
 for method in methods:
  root=ROOT/('forensics_r7' if method=='Joint-r7' else 'forensics');files=[root/'renders'/f'{method}_{m}.png' for m in ['whole','detail']]
  if all(p.exists() and p.stat().st_size>500000 for p in files):available.append((method,root,files))
  elif require:raise RuntimeError('缺少通过渲染的图：'+method)
 if not available:return
 fig,axes=plt.subplots(len(available),2,figsize=(10.5,3.65*len(available)),squeeze=False)
 for row,(method,root,files) in enumerate(available):
  info=json.loads((root/'summary.json').read_text())['selected'][method]
  for col,p in enumerate(files):
   ax=axes[row,col];ax.imshow(plt.imread(p));ax.axis('off');ax.set_title(f'{method} | '+('canonical surface' if col==0 else f"first-hit triangle: {info['early_error_m']:.3f} m early"),loc='left',fontsize=11,pad=5)
  axes[row,0].text(.08,.02,info['actor']['scene']+' / '+info['actor']['owner'][:8],transform=axes[row,0].transAxes,fontsize=8,color='#203040',bbox={'facecolor':'white','alpha':.7,'edgecolor':'none','pad':2})
 fig.subplots_adjust(left=.01,right=.995,top=.975,bottom=.01,wspace=.025,hspace=.11)
 for ext in ['pdf','png']:fig.savefig(OUT/(name+'.'+ext),dpi=190,bbox_inches='tight',pad_inches=.04)
 plt.close(fig)
panel(['AdaPoinTr','VGGT-native','Joint-r7'],'blender_atlas',True)
panel(['LiDAR-R8','Open-r3'],'blender_controls_a',True)
panel(['Attraction-r4','First-r6'],'blender_controls_b',True)
