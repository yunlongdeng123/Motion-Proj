"""目标构造轻量交付：真实梯度/保留边界约束与真实RGB偏差分开。"""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
B=Path(__file__).resolve().parents[1]
def read(p):return json.loads((B/p).read_text(encoding='utf8'))
def write(p,v):(B/p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def stat(x):
 x=np.asarray(x,dtype=float);return dict(count=int(x.size),mean=float(x.mean()) if x.size else None,p95=float(np.quantile(x,.95)) if x.size else None,max=float(x.max()) if x.size else None)
def main():
 manifest=read('target_manifest.json');boundary=read('boundary_receipt.json');poisson=read('poisson_receipt.json');source=read('source_chart_receipt.json');consistency=read('target_consistency_receipt.json');a=np.load(B/'world_atlas_targets.npz')
 assert hashlib.sha256((B/'world_atlas_targets.npz').read_bytes()).hexdigest()==manifest['world_atlas_sha256']
 fidelity=[]
 for role in ['BUILD_FIT','HELD_PROJECT_EVAL_ONLY']:
  rows=[r for r in manifest['rows'] if r['role']==role];row=dict(role=role,replace_pixels=sum(r['replace_pixels'] for r in rows),original_legal_pixels=sum(r['fit_pixels'] for r in rows))
  for key in ['direct_vs_true_on_replace','fusion_vs_true_on_replace','fusion_vs_direct_on_replace']:
   n=sum(r[key]['count']//3 for r in rows);s=sum(r[key]['mean']*(r[key]['count']//3) for r in rows if r[key]['count'])
   row[key]=dict(count_pixels=n,sum_pixel_mean_abs_RGB=s,mean=s/n if n else None)
  fidelity.append(row)
 e=a['edges'];g=a['same_frame_gradient_rgb'];measured=a['same_frame_gradient_count']>=2;gradient={}
 for key in ['source_chart_rgb','fusion_chart_rgb']:
  x=a[key];gradient[key]=stat(abs(x[e[:,1]]-x[e[:,0]]-g)[measured])
 preflight=read('final_target_key5_preflight_v1.json');examples=[]
 for raw in [128,137]:
  n=np.load(B/f'raw{raw}.npz')
  if raw==128:
   roi=np.zeros((320,480),bool);roi[188:234,165:208]=True;bbox=(165,188,208,234)
   domain='PRIMARY raw128 fixed user blue bbox';metricdomain='primary_bbox_original_road';edgefield='primary_bbox_boundaries';count=1385;pairtext='near-x6bins matched35/33px'
  else:
   m=np.load(B/'137.npz');roi=m['world_patch_clean_road'];yy,xx=np.where(roi);bbox=(int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1))
   domain='SECONDARY raw137 fixed world patch';metricdomain='world_patch_clean_road';edgefield='world_boundaries';count=int(roi.sum());pairtext='near-x12bins90/76px; lateral15bins101/113px'
  selected=[]
  for method in ['direct','fusion']:
   r=next(r for r in preflight['rows'] if r['raw']==raw and r['method']==method and r['state']=='delete')['layer1_target']
   selected.append(dict(method=method,world_boundary_metrics=r[edgefield],fidelity=r['fidelity_to_original_visible'][metricdomain]))
  meta=dict(raw=raw,target_state='SHARED_KEEP_DELETE',domain=domain,fixed_domain_pixels=count,bbox_xyxy_halfopen=bbox,
    crop_provenance='original registered approximate user bbox' if raw==128 else 'bbox of complete original fixed world_patch_clean_road mask, not selected from improvement',
    crop_fit_pixels=int((roi&n['fit_mask']).sum()),crop_replace_pixels=int((roi&n['fusion_replace_mask']).sum()),
    original_RGB_outside_replacement_is_identity=True,paired_evidence=pairtext,metrics=selected)
  examples.append(meta)
  x0,y0,x1,y1=bbox
  fig,axes=plt.subplots(2,3,figsize=(12,8));fig.subplots_adjust(top=.82,hspace=.35,wspace=.28,bottom=.08)
  imgs=[n['observation_rgb'],n['direct_rgb'],n['fusion_rgb'],n['fusion_replace_mask'].astype(float),n['source_confidence_map'],abs(n['fusion_rgb']-n['direct_rgb']).mean(-1)]
  names=['Original visible RGB','A same190-source chart','B fixed Poisson chart','Common replacement mask','Real-source confidence','Mean abs(B-A)']
  for ax,img,name in zip(axes.flat,imgs,names):
   if img.ndim==3:ax.imshow(img[y0:y1,x0:x1],interpolation='nearest',vmin=0,vmax=1,extent=[x0,x1,y1,y0])
   else:ax.imshow(img[y0:y1,x0:x1],interpolation='nearest',cmap='viridis',vmin=0,vmax=.05 if name=='Mean abs(B-A)' else 1,extent=[x0,x1,y1,y0])
   ax.set_title(name,fontsize=11);ax.set_xlabel('original x pixel');ax.set_ylabel('original y pixel')
  verdict='negative near-x target jump' if raw==128 else 'positive secondary target jump'
  fig.suptitle(f"{domain}: {verdict}\nTargets shared KEEP/DELETE; fixed domain{count}px, legal fit{meta['crop_fit_pixels']}px, replaced{meta['crop_replace_pixels']}px\n{pairtext}; outside replacement A=B=original RGB (identity, not fusion success)",fontsize=12)
  fig.savefig(B/f'target_raw{raw}_source_fusion.png',dpi=170);plt.close(fig)
 # Entire world chart, same colour range, no result-picked ROI.
 keys=a['keys'];lo=keys.min(0);shape=keys.max(0)-lo+1;ij=keys-lo;support=a['source_supported'];core=np.zeros((shape[1],shape[0]));core[ij[:,1],ij[:,0]]=a['core_mask'];pin=np.zeros_like(core);pin[ij[:,1],ij[:,0]]=a['boundary_mask']
 extent=[a['local_xyz'][:,0].min()-.04,a['local_xyz'][:,0].max()+.04,a['local_xyz'][:,1].min()-.04,a['local_xyz'][:,1].max()+.04]
 fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
 for ax,key,title in zip(axes[:2],['source_chart_rgb','fusion_chart_rgb'],['A unblended same-source chart','B one fixed screened-Poisson chart']):
  image=np.full((shape[1],shape[0],3),np.nan);image[ij[support,1],ij[support,0]]=a[key][support]
  ax.imshow(image,origin='lower',extent=extent,vmin=0,vmax=1);ax.contour(core,levels=[.5],colors='black',linewidths=.5,extent=extent,origin='lower');ax.contour(pin,levels=[.5],colors='cyan',linewidths=.5,extent=extent,origin='lower');ax.set_title(title)
 diff=np.full((shape[1],shape[0]),np.nan);diff[ij[support,1],ij[support,0]]=abs(a['fusion_chart_rgb'][support]-a['source_chart_rgb'][support]).mean(1)
 im=axes[2].imshow(diff,origin='lower',extent=extent,vmin=0,vmax=.04,cmap='magma');axes[2].set_title('Mean abs(B-A), supported nodes');fig.colorbar(im,ax=axes[2],shrink=.6)
 for ax in axes:ax.set_xlabel('anchor-local forward x (m)');ax.set_ylabel('anchor-local lateral y (m)')
 fig.suptitle('One8cm world chart; black=canonical core, cyan=reliable retained boundary; unsupported colour nodes blank')
 fig.savefig(B/'world_atlas_source_fusion.png',dpi=170);plt.close(fig)
 # Required direct component diagram; fitting/evaluation owned by sibling agents.
 fig,ax=plt.subplots(figsize=(14,3));ax.axis('off')
 blocks=[(.01,.4,.17,.4,'190 FRONT real witnesses\nexcludeHELD9\nmeasured geometry fixed'),(.22,.4,.18,.4,'Shared world source chart\ntrue same-frame gradients\nreliable retained boundary'),(.44,.4,.15,.4,'A source / B Poisson\nfixed lambda .04\nno per-view colour'),(.63,.4,.15,.4,'Project49 same atlas\ncommon legal mask\nKEEP=DELETE target'),(.82,.4,.17,.4,'Joint3D colour bake\nfixed geometry / alpha\n3-layer independent eval')]
 for x,y,w,h,t in blocks:
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.01',fc='#edf3f8',ec='#34516b'));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=10)
 for x in [.185,.405,.595,.785]:ax.annotate('',xy=(x+.03,.6),xytext=(x,.6),arrowprops={'arrowstyle':'->','lw':1.5})
 ax.text(.5,.13,'Targets are constructed apparent RGB, not hidden ground truth. Source/fusion share information and scope.\nTarget fidelity, writeback-to-target error, and native-render-to-real-RGB error remain separate.',ha='center',va='center',fontsize=11)
 fig.savefig(B/'target_components.png',dpi=170,bbox_inches='tight');plt.close(fig)
 summary=dict(schema='WORLD_SHARED_FUSION_TARGET_SUMMARY_V1',status='TARGET_LOCKED_NOT_SCENE_ACCEPTANCE',source_cameras=['FRONT'],source_view_count=190,
   target_manifest_sha256=hashlib.sha256((B/'target_manifest.json').read_bytes()).hexdigest(),source_chart=source,boundary=boundary,Poisson=poisson,
   measured_same_frame_gradient_fidelity=gradient,projected_target_fidelity_to_actual_RGB=fidelity,examples=examples,world_sampler_consistency=consistency,
   primary_negative_preserved=True,parameters_retuned=False,extra_GPU_forwards=0,baseline_reused_forwards=80,
   strong_control='A/B use exactly the same190-source chart, visibility/projection/mask/domain. Only fixed Poisson fusion differs.',
   limits=['Apparent encoded sRGB, not intrinsic albedo or a shadow decomposition','Only FRONT trajectory viewpoints; no all5camera claim',
     'Boundary zero residual is enforced, not independent accuracy','Target-domain zeros outside replacement are exact originalRGB copies, not fusion success',
     'Raw128 is BUILD same-scene development; HELD9 excluded from construction but scene historically exposed',
     'Poisson clipping of60 nodes/131 channels and increased equation residual explicitly retained','Target accuracy is separate from baked native scene quality'],
   figures=['target_components.png','world_atlas_source_fusion.png','target_raw128_source_fusion.png','target_raw137_source_fusion.png'],failure_ledger_refs=['V82-F13'],human_verdict=None)
 write('target_construction_summary.json',summary)
 md=f'''# 共享世界融合目标：已锁定，非场景采用

![组件](target_components.png)

两试控制只差融合算子：A投影未融合source chart，B投影一次固定screened-Poisson chart。两者使用相同190个非HELD的FRONT轨迹视点实拍见证、同8cm世界几何、同合法投影域及固定BUILD40的714546道路像素。HELD9只投影/评价；原场景曾曝光all199，此处不宣称独立盲测或全五相机验证。目标为encoded sRGB期望完整合成图RGB，非预乘roadRGBT、非albedo；KEEP/DELETE共用同一文件。

chart共13358节点（10098core+3260外halo），11806节点获稳定颜色支持，其中core10094；142节点无有效真实见证。26451邻边中24801有至少2个同帧真实梯度见证，1650弱fallback边另列。沿用固定q75光照cohort，不进行反射/阴影分解；标线没有语义GT，只评价真实梯度保留。

外环只在source资格/guard合法、newroadW≤.05、KEEP保留成像与真实source最大通道残差≤.03、与共享chart一致、至少3 BUILD帧且span≥6时保留。103个节点成为硬边界；1485条RGB前资格记录中接受1411条，残差拒14条，其中staticW>.5的污染拒5条/5独立节点；cohort拒64条可重叠。没有把暗蓝static整体当道路真色。边界pin误差从.0112343变0是硬约束满足，不是独立准确率。

![完整世界chart](world_atlas_source_fusion.png)

一次固定λ=.04，未调权重/阈值/来源。原解范围[-.0103743,.6523792]；60节点/131通道裁剪到[0,1]。方程残差max从1.998e-15增至.00800715，裁剪事实与残差保留。相对于同帧真实梯度，source chart平均绝对残差.00450150→Poisson .00149455；这是构造与源梯度的一致性，不能代替当前真实RGB或最终native质量。

BUILD40实际替换10492/714546像素；atlas外/不合格处A=B=observation RGB，不能把这些恒等区误差0算作融合能力。投影要求四角均有可靠实测source支持、固定测量高度有效，并拒绝可用scene-depth冲突；不为颜色移动道路、不新增单目深度。49帧全部替换像素从同一世界atlas与保存localXY重放，最大RGB误差{consistency['maximum_over_all49']:.3g}，只有float32坐标舍入；这证明共享表示，不是独立跨视角光度准确率。

raw128 **原用户近似蓝框**：固定原road1385、合法fit1071、实际replace545，near-x6bins实际配对35/33。目标A→B相对真实RGB的near jump **.009696462→.010322402回退**，即使road MAE改善，也不写接缝解决。以下同裁剪不重选负例：

![raw128负例](target_raw128_source_fusion.png)

raw137为 **副世界patch**，不是用户蓝框：原road1447，near-x12bins实际90/76、lateral15bins实际101/113。A→B near jump .010576652→.006191509、lateral .017606065→.013906171改善，但不能覆盖raw128负例。裁剪只取原冻结world-patch mask整体bbox[118,201,205,233)，不是按改善点挑选。

![raw137副视图正例](target_raw137_source_fusion.png)

目标trueRGB偏差、native写回目标误差、native对真实RGB误差由独立评价分别报告。无隐藏车/道路纹理GT、无shadow decomposition、无完整PBR/新道路生成/全场景采用声明。目标阶段新增GPU0F，复用engineer80F/B基底；一次未执行括号语法错误与顶层ready schema修复均保留源/首manifest，不改变数组或科研方法。

[目标合同](atlas_contract.json)、[不可变49目标manifest](target_manifest.json)、[源支持](source_chart_receipt.json)、[可靠边界审计](boundary_receipt.json)、[解算与裁剪](poisson_receipt.json)、[共享采样重放](target_consistency_receipt.json)、[详细构造summary](target_construction_summary.json)。
'''
 (B/'TARGET_CONSTRUCTION.md').write_text(md,encoding='utf8');print(str(B/'target_construction_summary.json'))
if __name__=='__main__':main()
