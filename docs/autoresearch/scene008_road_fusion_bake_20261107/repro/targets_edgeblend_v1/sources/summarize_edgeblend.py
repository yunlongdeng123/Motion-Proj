"""只生成C目标解释图/轻量报告，不改target，不执行GPU。"""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
B=Path(__file__).resolve().parents[1];OLD=B.parent/'targets'
def read(base,name):return json.loads((base/name).read_text(encoding='utf8'))
def main():
 manifest=read(B,'target_manifest.json');preflight=read(B,'target_preflight_0F.json');audit=read(OLD,'hardreplace_edge_audit_v1.json')
 assert hashlib.sha256((B/'target_manifest.json').read_bytes()).hexdigest()=='d7cca4167e2bbb68347f63babae356320a1b0ddd60e580bf746e966f2ac8acbe'
 rows=[]
 for raw in [128,137]:
  z=np.load(B/f'raw{raw}.npz')
  bbox=(165,188,208,234) if raw==128 else (118,201,205,233);x0,y0,x1,y1=bbox
  domain='PRIMARY user blue bbox' if raw==128 else 'SECONDARY fixed worldpatch bbox'
  fig,axes=plt.subplots(2,3,figsize=(12,8));fig.subplots_adjust(top=.82,hspace=.35,wspace=.28,bottom=.08)
  names=['Actual original O','Locked hard-Poisson B','C fixed10px edge blend','Common validity M (unchanged)','Fixed w=distance_inside(M)/10','Mean abs(C-B)']
  images=[z['observation_rgb'],z['fusion_rgb'],z['edgeblend_rgb'],z['fusion_replace_mask'].astype(float),z['edgeblend_weight'],abs(z['edgeblend_rgb']-z['fusion_rgb']).mean(-1)]
  for ax,img,name in zip(axes.flat,images,names):
   ax.imshow(img[y0:y1,x0:x1],interpolation='nearest',vmin=0,vmax=.05 if name=='Mean abs(C-B)' else 1,cmap='viridis' if img.ndim==2 else None,extent=[x0,x1,y1,y0]);ax.set_title(name,fontsize=11);ax.set_xlabel('original x pixel');ax.set_ylabel('original y pixel')
  fig.suptitle(f'raw{raw} {domain} | targets shared KEEP/DELETE\nC is strong realRGB anchoring / prior dilution, not a pure atlas target\nHELD target uses query O for diagnostic only; low target MAE is not held prediction',fontsize=12)
  fig.savefig(B/f'edgeblend_raw{raw}_target.png',dpi=170);plt.close(fig)
 # Explicit shared-texture vs view-conditioned blend vs persistent3D inference.
 fig,ax=plt.subplots(figsize=(14,3.8));ax.axis('off')
 blocks=[(.01,.44,.17,.37,'Shared world atlas B\n190 nonHELD witnesses\nsource / Poisson frozen'),(.22,.44,.17,.37,'Per-view legal M + O\nfixed10px distance w\nC=O+w(B-O)'),(.43,.44,.17,.37,'BUILD40 joint bake\n10098 shared parent RGB\ngeometry / alpha fixed'),(.64,.44,.16,.37,'Persistent3D material\nshared4 child colours\nsame KEEP / DELETE'),(.84,.44,.15,.37,'Native49 query\ncam / pose / env\nno O colour condition')]
 for x,y,w,h,t in blocks:
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.01',fc='#edf3f8',ec='#34516b'));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=10)
 for x in [.185,.395,.605,.805]:ax.annotate('',xy=(x+.03,.625),xytext=(x,.625),arrowprops={'arrowstyle':'->','lw':1.5})
 ax.text(.305,.18,'HELD Ctarget reads query O\ndiagnostic only, never fit',ha='center',va='center',color='#a14d32',fontsize=11)
 ax.annotate('',xy=(.305,.40),xytext=(.305,.28),arrowprops={'arrowstyle':'->','color':'#a14d32','lw':1.2})
 ax.text(.73,.16,'O may be loaded for external evaluation/contracts;\nit does not condition native scene/raster colour.\nPreviously exposed same scene; no independent-generalization claim.',ha='center',va='center',fontsize=10)
 fig.savefig(B/'edgeblend_components.png',dpi=170,bbox_inches='tight');plt.close(fig)
 selected=[]
 for r in audit['results']:
  selected.append(dict(raw=r['raw'],native_state='DELETE',domain=r['domain'],pixel_boundary_edge_count=r['cross_mask_edge_metrics']['ALL']['edge_count'],
      mask_edge_reasons={k:v['edge_count'] for k,v in r['cross_mask_edge_metrics'].items() if k!='ALL'},
      residual_gradient_means={k:v['residual_jump']['mean'] for k,v in r['cross_mask_edge_metrics']['ALL']['methods'].items()},
      current_B_world_jumps=r['locked_target_registered_world_jumps'],CPU_hypothetical_C_world_jumps=r['hypothetical_tenpx_registered_world_jumps'],
      prediction_scope='The hypothetical C numbers precede actual C fit and are target diagnostics, never native prediction scores.'))
 summary=dict(schema='FIXED10PX_EDGE_BLEND_TARGET_SUMMARY_V1',status='C_TARGET_LOCKED_NATIVE_ACCEPTANCE_PENDING',
   manifest_sha256=hashlib.sha256((B/'target_manifest.json').read_bytes()).hexdigest(),selected_target_key='edgeblend_rgb',
   formula='C=O+clip(distance_inside(M)/10,0,1)*(B-O)',width=10,parameter_searches=0,original_target_manifest_unchanged=True,
   interpretation='Strong actualRGB anchoring / atlas-prior dilution. Shared atlas remains fixed; full target is view-conditioned via validity M and observedRGB O.',
   query_RGB_scope=manifest['query_RGB_scope'],weight_summary=manifest['group_weight_summary'],
   raw128_primary_weight=preflight['primary_raw128'],hardcopy_diagnostic=selected,
   classification_note='chart_four_corner_geometry_edge is geometric eligibility of the texture chart, not proof of a physical road/car surface edge.',
   new_GPU_forwards=0,world_geometry_depth_updates=0,whole_scene_adopted=False,human_verdict=None,
   figures=['edgeblend_components.png','edgeblend_raw128_target.png','edgeblend_raw137_target.png'])
 (B/'edgeblend_target_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
 md='''# 唯一10px边缘融合控制：强真实RGB锚定

![组件](edgeblend_components.png)

旧目标的world Poisson只约束103个外环节点，没有约束各相机validity-mask与原RGB硬拼接的边界。原target与manifest均保留；这一轮只加一个固定10px控制，替代未执行legacy，不搜索width或来源。

raw128原蓝框、原road1385像素中，216条四邻接跨replace边上的额外残差梯度均值为：旧B .041196、AA .016872、已烘native-B400 .020683。216边分别为chart四角几何资格117、source资格71、scene-depth拒11、A37保守guard17；这里chart edge不是物理首交/真实车辆边。边外AA newroadW均值.472、assetW=0，说明mask资格洞处渲染仍有道路贡献。该结果支持目标拼接缺口，不能用来宣称固定3D表示必然失败。

新目标只用公式 `C = O + clip(distance_inside(M)/10,0,1) * (B-O)`，对所有M边界含内部洞同样处理；同190真实见证、同Poisson、同原fit/protect/M，两状态共享。outside仍C=O逐值相同；不改几何/alpha/深度/来源/旧目标。C字段为edgeblend_rgb，原fusion_rgb仍旧B，不能混用。

BUILD10492个替换像素的平均w=.186557897、max=.728011，0个w=1；raw128主蓝框545替换点平均w=.21037847、max=.6，也无纯atlas内域。HELD2623替换点均w=.1985438、max=.806226，仍0个w1。因此这是很强的真实RGB锚定/先验稀释，不叫纯atlas目标。跨M边C残差约按构造降到B的10%，不能作为融合或native成功证据；验收仍用原冻结世界分桶、trueRGB、native49与保护域。

![raw128同裁剪](edgeblend_raw128_target.png)

![raw137副世界patch同裁剪](edgeblend_raw137_target.png)

HELD9的Ctarget明确读取query真实O，只用于目标构造/诊断，低targetMAE不是held预测。只有BUILD40用于拟合持久3D材质。native查询的scene/raster颜色不以queryO为条件；现有rt.raw可以读取/加载obs用于评价/合同，因此不声称文件不读O。旧场景历史曝光过all199，始终是preexposed same-scene控制，不是独立泛化。原atlas继续世界共享；C的完整投影目标因观测RGB/合法域而view-conditioned，没有新增自由逐帧颜色。

[C合同](edgeblend_contract.json)、[不可变49目标manifest](target_manifest.json)、[0F预检](target_preflight_0F.json)、[queryRGB角色](QUERY_RGB_SCOPE.json)、[详细summary](edgeblend_target_summary.json)。新增目标阶段GPU0F，native训练/恢复/评价由独立执行与评价收据记账。
'''
 (B/'EDGEBLEND_TARGET.md').write_text(md,encoding='utf8');print(str(B/'edgeblend_target_summary.json'))
if __name__=='__main__':main()
