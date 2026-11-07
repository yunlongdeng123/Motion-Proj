"""Finalize blue evidence after all frozen controls, without fitting or GPU work."""
from pathlib import Path
import json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B=Path(__file__).resolve().parent
def read(p):return json.loads((B/p).read_text(encoding='utf8'))
def write(p,v):(B/p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def main():
    labels={'n2':('R0','n2'),'n2_material_vs_n2':('n2','n2 + 18coef'),
            'n2_material_vs_R0':('R0','n2 + 18coef'),'AA_vs_R0':('R0','n2 + RelativeMip (AA)'),
            'AA_frozen_material_vs_AA':('n2 + RelativeMip (AA)','AA + frozen 18coef'),
            'AA_frozen_material_vs_R0':('R0','AA + frozen 18coef')}
    comps={}
    for name,(ref,cand) in labels.items():
        path=f'{name}_fixed_blue_comparison.json';d=read(path);d.update(reference_label=ref,candidate_label=cand)
        write(path,d);comps[name]=d
    spec=[('R0','AA_vs_R0','parent'),('n2','n2_material_vs_n2','parent'),
          ('AA','AA_vs_R0','candidate'),('n2_material','n2_material_vs_R0','candidate'),
          ('AA_frozen_material','AA_frozen_material_vs_R0','candidate')]
    primary=[]
    for method,source,variant in spec:
        d=comps[source];r=d['regions']['original_road'][variant];p=d['regions']['global_protected'][variant]
        primary.append(dict(raw=128,state='delete',domain='PRIMARY_FIXED_APPROXIMATE_USER_BBOX',method=method,count_pixels=r['pixels'],abs_RGB_error_sum=r['RGB_abs_error_sum'],
            abs_RGB_error_mean=r['RGB_abs_error_mean'],near_x_paired_bins=6,
            near_x_band_minus_count=42,near_x_band_plus_count=34,near_x_matched_minus_count=35,near_x_matched_plus_count=33,
            near_x_equal_bin_abs_residual_jump=d['matched_tangent_residual_jumps']['near_x'][variant]['equal_bin_abs_residual_jump_mean'],
            clean_road_count=d['regions']['clean_original_road'][variant]['pixels'],
            clean_road_MAE=d['regions']['clean_original_road'][variant]['RGB_abs_error_mean'],
            direct_depth_clean_road_count=d['regions']['depth_qualified_clean_original_road'][variant]['pixels'],
            direct_depth_clean_road_MAE=d['regions']['depth_qualified_clean_original_road'][variant]['RGB_abs_error_mean'],
            raw128_global_protect_count=p['pixels'],raw128_global_protect_abs_RGB_error_sum=p['RGB_abs_error_sum'],
            raw128_global_protect_MAE=p['RGB_abs_error_mean'],
            newroad_alphaT_mean=r['newroad']['alphaT_mean'],oldroad_alphaT_mean=r['oldroad']['alphaT_mean'],static_alphaT_mean=r['static']['alphaT_mean'],
            global_clip_scope='5key only' if method=='AA_frozen_material' else '49frame dual-state evaluated by parent',
            comparison_source=f'{source}_fixed_blue_comparison.json'))
    world=read('fivekey_fixed_world_patch_comparison.json');both=read('raw128_fixed_roi_both_states.json')
    secondary=[]
    for r in world['per_view']:
        for edge,d in r['edges'].items():
            secondary.append(dict(raw=r['raw'],state=r['state'],domain='SECONDARY_PREDECLARED_WORLD_PATCH',method=r['method'],boundary=edge,
                road_count=r['regions']['world_patch_clean_road']['count_pixels'],band_minus_count=r['regions'][f'{edge}_minus']['count_pixels'],
                band_plus_count=r['regions'][f'{edge}_plus']['count_pixels'],paired_bins=d['paired_bins'],
                matched_minus_count=sum(x['minus_count'] for x in d['bins']),matched_plus_count=sum(x['plus_count'] for x in d['bins']),
                equal_bin_abs_residual_jump_mean=d['equal_bin_abs_residual_jump_mean']))
    fit=read('fit_receipt.json');pre=read('receipt.json');valid=read('linear_validation.json');verification=read('retrieval_verification.json')
    external=[];pooled={}
    for name in ['n2_full49','n2_mip_full49','n2_material_full49','n2_material_pilot','n2_mip_material_pilot']:
        d=read(f'retrieved_source_fitstates_v1/results/{name}/receipt.json')
        external.append(dict(name=name,forwards=d['forwards'],seconds=d['seconds'],raw_count=len(d['raws']),
                             attribution='Parent joint blue/red/protection evaluation, not additional blue fitting. Do not double-count in root ledger.'))
        pooled[name]=[v for v in d['pooled'] if v['metric'] in ['observed_road_RGB','protected_RGB']]
    ledger=dict(schema='BLUE_COMPUTE_LEDGER_V1',own_GPU=dict(smoke_basis_measurement=4,smoke_material_real_render_validation=4,
        formal_basis_measurement=80,formal_material_real_render_validation=4,total_forwards=92,total_backwards=0,
        GPU_training_steps=0,CPU_constrained_QP_channels=3,formal_measurement_seconds_approx=18.15,
        peak_allocated_bytes=pre['peak_allocated_bytes'],peak_reserved_bytes=pre['peak_reserved_bytes'],
        formal_linear_prediction_real_render_maxabs=max(r['linear_prediction_render_maxabs'] for r in valid['records'])),
        own_CPU=dict(fivekey_world_patch_audit_seconds=world['seconds'],colour_hyperparameter_searches=0,
                     note='Other local audit, plotting, QP, hashing and transfer wall time was not instrumented; not invented.'),
        external_joint_evaluation_receipts=external,retrieval_verification=verification,
        no_additional_GPU_after_parent_stop=True,power_operations=0)
    write('blue_compute_ledger.json',ledger)
    summary=read('blue_public_summary.json')
    summary.update(schema='BLUE_FINAL_METRICS_V2',status='BLUE_SEAM_UNRESOLVED_RESEARCH_CONTROL',
        main_engineering_candidate='n2 + RelativeMip, frozen parent cellfit1000 material; root-owned, not whole-scene adoption',
        material_control_decision='18coef retained as negative/limited appearance control; excluded from main engineering candidate',
        primary_state='delete',primary_metric_rows=primary,primary_KEEP_separate_source='raw128_fixed_roi_both_states.json',
        primary_both_states=both['rows'],secondary_world_boundary_rows=secondary,
        metric_numerator='Sum over pixels of mean absolute error over3 RGB channels; denominator1385 original-road pixels. No nonroad/foreground pixels added.',
        final_near_x_counts={'raw':128,'states':['keep','delete'],'domain':'PRIMARY_FIXED_APPROXIMATE_USER_BBOX','minus':42,'plus':34,'matched_minus':35,'matched_plus':33,'matched_tangent_bins':6},
        near_x_residual_jump_definition='For each fixed .2m tangent bin with>=3 original-eligible pixels per side, difference of mean(render-originalRGB) across sides; mean absolute RGB channel jump, then equal mean over6 bins. Not raw colour difference or a physical geometry metric.',
        unsupported_primary=['far-x no supported pixels','lateral146/75 side pixels but no>=3/side matched tangent bin; no primary lateral jump score'],
        fivekey_world_support=dict(source='fivekey_fixed_world_patch_comparison.json',raws=world['raws'],
            states=world['states'],primary_circle_not_redefined=True,
            note='Same predeclared world patch, full original road+guard+raw-ground mapping. Secondary masks built before loading method arrays; this audit occurred after pilot results were open, not independent confirmation.',
            lateral_AA_vs_R0_worsens_in_other_four_DELETE_views=True,
            KEEP_can_show_opposite_direction=True),
        parent_global_metrics_receipt_excerpts=pooled,
        source_and_fitstate_retrieval=dict(status=verification['status'],members=len(verification['verified_members']),
            separately_verified_arrays=len(verification['verified_individual_arrays']),archive_bytes=verification['archive_bytes'],archive_sha256=verification['archive_sha256']),
        shadows='Original raw image supplies visible radiance only; no shadow/material decomposition or hidden road shadow GT. A remaining dark static contribution cannot be certified as either a true shadow or a colour bug from these data.',
        compute_ledger='blue_compute_ledger.json',components_figure='blue_material_components.png',
        final_figure='blue_final_comparison.png',world_figure='blue_fivekey_world_residual_jump.png',
        pending='No additional fit/GPU. Root completes report/whole-scene checks; blue remains unresolved.')
    summary['evidence_files']=list(dict.fromkeys(summary['evidence_files']+['blue_final_metrics.json','blue_compute_ledger.json','fivekey_fixed_world_patch_comparison.json','retrieval_verification.json']))
    write('blue_final_metrics.json',summary);write('blue_public_summary.json',summary)
    # Mirror only small reproducibility artifacts. Full actual-observation probes stay in private retrieval archive.
    src=B/'retrieved_source_fitstates_v1/blue/boundary_material_v1';dst=B/'boundary_material_v1'
    for p in src.rglob('*'):
        if p.is_file() and not p.name.startswith('probe_'):
            rel=p.relative_to(src);target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    # Original pixels and all fixed controls, same crop / interpolation / range.
    files=['layers_128/observations.npz','layers_128/r0_delete.npz','n2_candidate_delete_128.npz','n2_mip_candidate_delete_128.npz','n2_material_candidate_delete_128.npz','n2_mip_material_candidate_delete_128.npz']
    titles=['Original visible RGB','R0 frozen parent','n2 support control','n2 + RelativeMip (main)','n2 +18coef (control)','AA + frozen18coef (5key)']
    fig,ax=plt.subplots(2,3,figsize=(11,10))
    fig.subplots_adjust(hspace=.58,wspace=.30,top=.87,bottom=.10)
    for i,(file,title) in enumerate(zip(files,titles)):
        a=np.load(B/file)['rgb'];a=a.transpose(1,2,0) if a.shape[0]==3 else a
        ax.flat[i].imshow(a[188:234,165:208],interpolation='nearest',vmin=0,vmax=1,extent=[165,208,234,188]);ax.flat[i].set_title(title,fontsize=11)
        if i:ax.flat[i].set_xlabel(f"road MAE {primary[i-1]['abs_RGB_error_mean']:.6f}\nnear residual jump {primary[i-1]['near_x_equal_bin_abs_residual_jump']:.6f}")
    fig.suptitle('Raw128 DELETE | primary fixed approximate user blue bbox\nRoad MAE:1385 pixels | near-x residual jump:6 bins, matched35/33 pixels\nSmall score gains; visible seam remains',fontsize=13)
    fig.savefig(B/'blue_final_comparison.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    colors={'R0':'#777777','n2':'#4488bb','AA':'#008855','n2_material':'#bb8844','AA_frozen_material':'#884488'}
    for method in colors:
        rows=[r for r in world['per_view'] if r['method']==method and r['state']=='delete']
        for j,e in enumerate(['near_x','lateral_y']):
            vals=[r['edges'][e]['equal_bin_abs_residual_jump_mean'] for r in rows]
            ax[j].plot([r['raw'] for r in rows],[np.nan if v is None else v for v in vals],marker='o',label=method,color=colors[method])
    for j,title in enumerate(['near-x boundary','lateral-y boundary']):
        ax[j].set_title(title);ax[j].set_xlabel('raw FRONT view');ax[j].set_ylabel('absolute RGB residual jump');ax[j].grid(alpha=.2)
    ax[1].legend(fontsize=8);fig.suptitle('DELETE secondary fixed world patch: view dependence remains; unsupported bins are gaps')
    fig.savefig(B/'blue_fivekey_world_residual_jump.png',dpi=170);plt.close(fig)
    table='\n'.join(f"|{r['method']}|{r['abs_RGB_error_sum']:.9f} / {r['count_pixels']}|{r['abs_RGB_error_mean']:.9f}|{r['near_x_equal_bin_abs_residual_jump']:.9f}|" for r in primary)
    keeptable='\n'.join(f"|{r['method']}|{r['regions']['original_road']['abs_RGB_error_sum']:.9f} / {r['regions']['original_road']['count_pixels']}|{r['regions']['original_road']['abs_RGB_error_mean']:.9f}|{r['edges']['near_x']['equal_bin_abs_residual_jump_mean']:.9f}|" for r in both['rows'] if r['state']=='keep')
    worldtable=[]
    for state in ['delete','keep']:
        for raw in world['raws']:
            for edge in ['near_x','lateral_y']:
                ref=next(v for v in secondary if v['raw']==raw and v['state']==state and v['method']=='R0' and v['boundary']==edge)
                aa=next(v for v in secondary if v['raw']==raw and v['state']==state and v['method']=='AA' and v['boundary']==edge)
                a='unsupported' if ref['equal_bin_abs_residual_jump_mean'] is None else f"{ref['equal_bin_abs_residual_jump_mean']:.9f}"
                b='unsupported' if aa['equal_bin_abs_residual_jump_mean'] is None else f"{aa['equal_bin_abs_residual_jump_mean']:.9f}"
                worldtable.append(f"|{raw}|{state.upper()}|{edge}|{ref['paired_bins']} / {ref['matched_minus_count']} / {ref['matched_plus_count']}|{a}|{b}|")
    worldtable='\n'.join(worldtable)
    text=f'''# 蓝圈：仅小幅改善，接缝未解决

主工程候选为 **n2 + RelativeMip，沿用父cellfit1000材质**。18系数材质控制保留为对照，不并入主资产。蓝圈与红圈独立验收；此文不宣称整场景采用。

![组件与数据流](blue_material_components.png)

主评价为 **raw128 / DELETE / 原蓝框**。用户蓝圈只登记近似框 `[165,188,208,234)`，不是精确圈选。框内1978px，原观测road1385px，排车辆guard后1340px；其中214px同时有合格直接scene-depth与实测ground支持。593个非road像素不作为道路GT。世界patch在结果比较前固定为anchor-local x[19,32.6]、y[.5,3.5]m，边界x=19.8、x=31.8、y=2.71；两侧带宽.08–.4m。以下第一表始终为raw128 DELETE，near-x原两侧42/34px，实际配对35/33px、6bins。

|方法|绝对RGB误差sum / road像素数|road MAE|near-x两侧残差jump，固定6bin|
|---|---:|---:|---:|
{table}

误差sum先对每像素3通道取平均绝对误差，再对像素求和；float32不同归约顺序最后数位可有舍入差。jump先减原观测RGB，再在同.2m切向bin中比较两侧均值；每侧至少3px，6个bin等权。因此它不直接惩罚真实标线/阴影造成的原图色差。主框far没有样本；lateral虽有146/75px，但无满足配对规则的bin，其jump **unsupported**。

以下单列 **raw128 / KEEP / 同原蓝框**，仍为1385road，near-x同42/34px、配对35/33px、6bins，不能混用DELETE数值：

|方法|绝对RGB误差sum / road像素数|road MAE|near-x两侧残差jump，固定6bin|
|---|---:|---:|---:|
{keeptable}

![全部固定对照，真实像素裁切](blue_final_comparison.png)

在 **raw128 DELETE主框**，AA相对R0的原road MAE下降约4.89%，near-x jump（6bins，配对35/33px）约下降21.83%，仍保留明显接缝。18系数在n2上让ROI MAE再降约1.07%，同near-x jump却从.010822454升至.010833922；冻结迁移到AA后ROI再降约1.07%，jump几乎不变。因此“ROI误差下降”不能写成“接缝解决”。所有材质对照alpha、分组alpha、new/oldroadW与depth相对各自固定表示逐元素完全相同。

**raw128 DELETE父R0主框**的边界机制也不是单一的“新旧road颜色不合”：near-x外42px的new/old/static W均值=.5065/.3729/.1200，内34px=.9312/.0023/.0661；newroad自身mass-normalized RGB两侧接近。lateral-y内146px为.6995/.0052/.2948，外75px为.1822/0/.7313，主要是newroad与native static混合。主框原road中newW>.5的598px，newroad solo alpha均值.99385，joint/solo均值.82081，剩余static颜色偏暗蓝；已有solo支撑，但joint贡献受到其它层/排序影响。此条件分组仅诊断，不替代固定road分母。

“这些cell没有参与fit”的预设已被反证：蓝框中心投影2163行中2026行eligible；这不是每像素真实归属。source_valid128为139行；前方深度冲突71中心均未获source_valid128。source_valid是源资格缓存，不是密集可见性/未观测真值。Gaussian中心深度moment也不是物理首交，不能把中心残差直接记为路面高度误差。

同一预登记世界patch在5个FRONT视图作副审计，使用原road、车辆guard和各raw真实ground，先保存所有mask才读各方法数组。它在pilot已开封后进行，是同场景开发支持证据。它与原蓝框是 **另一套mask**：例如raw128副世界patch有772road，near-x两侧43/35px；即使也有6bins，不能与主框1385road、42/34px、6bins拼接。

副世界patch的raw128 lateral-y仍unsupported；其余4个 **DELETE** view虽然可配对，AA lateral-y jump相对R0反而增加：raw105 DELETE lateral-y 8bins约.011334→.019934；raw137 DELETE lateral-y 15bins约.010415→.015789。KEEP可能反向，以下按frame/state/boundary、配对bin/两侧实际配对像素完整分表，不概括为整条边界一致改善。

|raw|state|副世界patch边界|配对bins / minus像素 / plus像素|R0 residual jump|AA residual jump|
|---:|---|---|---:|---:|---:|
{worldtable}

![五视图相同世界边界残差jump](blue_fivekey_world_residual_jump.png)

唯一材质控制为3条固定世界边界，各一项compact distance taper×[常数,归一化切向]，共6basis/18RGB系数。10098父cell中1270行非零；4个n2子cell继承父basis。BUILD40原观测road与固定车辆guard监督714546个独立像素，keep/delete同掩码共1429092像素状态；域外目标保持同n2父RGB。一次固定约束LS，ridge=.01、域外权重1；每个cell限于原source proxy±.15且0..1，δ0可行。无阈值/lambda/预算搜索、无逐cell加步数、无held9调参；几何/alpha/native/static/env/相机不变。

保存raw90两状态normal equations后，新进程恢复再加raw91两状态，完成拟合入口及真实渲染线性smoke。正式80F约18.15s；smoke测量4F+smoke验证4F+正式验证4F，独占账本共92F、0反向。正式 `baseRGB+Mcoef` 对真实render最大误差3.454e-7；父色最大改.022875，source proxy最终最大差.15，basis域外精确冻结。继承source proxy已看全199，raw128属于BUILD，场景/test曾开封：均属于same-scene开发控制，不能作为盲测或泛化证据。

原图仅提供可见radiance。当前没有阴影/材质分解GT、没有隐藏路面阴影GT，不能将剩余暗static直接认证为真实阴影，也不能从截图认定仅是颜色错误。实测稀疏深度与局部ground插值不是稠密隐藏road GT；无标线准确率、无roadprior成功或360/全相机泛化结论。组合只验5key，不声称full49。

复现与核验：[最终JSON](blue_final_metrics.json)、[原蓝框双状态](raw128_fixed_roi_both_states.json)、[精确计算账本](blue_compute_ledger.json)、[固定原始mask登记](roi_registration.json)、[五视图世界patch证据](fivekey_fixed_world_patch_comparison.json)、[CPU/GPU测量与QP代码](boundary_material_control.py)、[冻结合同](boundary_material_v1/contract.json)、[拟合状态收据](boundary_material_v1/formal/fit_receipt.json)、[真实线性验证](boundary_material_v1/formal/linear_validation.json)、[取回SHA核验](retrieval_verification.json)。源码/fitstates私有取回包19,865,523bytes，SHA256 `b54c61d2291b22cebe8e5f9e93cb3803a9c4d2ee289335a22d60478ea8cd53f0`；29个成员和6个独立读出均核验匹配。该包含实际观测probe供重放，公开HTML仅带轻量代码/JSON/图，不打包probe。
'''
    (B/'BLUE_FINDINGS.md').write_text(text,encoding='utf8')
    print(json.dumps({'findings':str(B/'BLUE_FINDINGS.md'),'metrics':str(B/'blue_final_metrics.json'),'primary':primary},ensure_ascii=False))

if __name__=='__main__':main()
