"""CPU publication of a completed, frozen-asset three-view scale probe."""
import base64,csv,hashlib,html,json,shutil
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RESULT=HERE/'results'
REVIEW=HERE.parent/'review'
OUT=ROOT/'outputs/scene008_vehicle_appearance/diagnosis/scale'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def main():
    r=json.loads((RESULT/'receipt.json').read_text())
    assert r['status']=='PASS_THREE_VIEW_FROZEN_ASSET_SCALE_PROBE'
    assert r['render_calls']==40 and r['raws']==[106,113,130]
    assert digest(RESULT/'receipt.json')==digest(HERE/'receipt.remote.json')
    cov=json.loads((HERE/'covariance_cpu_review.json').read_text())
    assert cov['status']=='PASS_CPU_CALIBRATION_AND_GENERIC_WRAPPER_CONTRACT' and cov['GPU_calls']==0
    OUT.mkdir(parents=True,exist_ok=True);REVIEW.mkdir(exist_ok=True)
    rows=[]
    for v in r['views']:
        for state in ('solo','joint','delete'):
            b=v['arms']['LR_baseline']['vs_area_downsample_HR'][state]['object']
            c=v['arms']['LR_area_compensated']['vs_area_downsample_HR'][state]['object']
            item=dict(raw=v['raw'],state=state,fixed_object_pixels=b['fixed_mask_pixels'],
                chart_factor_mean=v['compensation']['chart_factor']['mean'])
            for k in ('rgb_MAE','alpha_MAE','asset_rgb_MAE','asset_alpha_MAE','asset_straight_RGB_MAE_fixed_ref_alpha_gt05'):
                item['baseline_'+k]=b[k];item['compensated_'+k]=c[k]
                item['relative_drop_'+k]=1-c[k]/b[k]
            rows.append(item)
    with (OUT/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    floors=[max(a['group_conservation_maxabs'] for a in v['arms'].values()) for v in r['views']]
    report=dict(status='PASS_DIAGNOSTIC_NOT_ADOPTED',receipt_sha256=digest(RESULT/'receipt.json'),
        source_sha256=digest(HERE/'scale_probe.py'),asset_sha256=r['asset_sha256'],
        question='Does the native 480px operator reproduce the same-view 1920px actual 3D asset after 4x4 area downsampling?',
        conclusion='Resolution adds a substantial appearance discrepancy. One fixed chart-only area normalization reduces three-view RGB disagreement with the HR actual-render reference; it does not resolve source colours, depth, occlusion, or independently establish geometry accuracy.',
        adoption='No asset or production renderer was changed. Keep this as a diagnostic control pending broader preservation and multiview checks.',
        reference=dict(kind='same-asset actual HR render, not a 2D generated image and not hidden RGB truth',
            raw_ids=r['raws'],mapping=r['pixel_center_mapping'],
            mask='D4(HR solo alpha)>0.01; fixed per view across both LR arms',
            normalized_mask='additional D4(HR state-specific asset alpha)>0.5'),
        fairness=dict(same_physical_asset_rows=50322,chart_rows=47544,retained_rows=2778,
            camera_pose_and_projection_same=True,resolutions=[[480,320],[1920,1280]],
            same_A_rows_and_order_solo_joint=True,zero_native_a6_rows=True,
            a37_deletion='mask false in the same packed rows; no physical packed-row deletion claim',
            black_RGB='native premultiplied RGB with zero raster background',
            street_RGB='native joint premultiplied RGB + (1-joint alpha)*same frozen environment',
            joint_A_RGB='native diagnostic feature accumulation sum_i(alpha_i*T_joint*c_i) over A rows; not jointRGB minus backgroundSoloRGB',
            auxiliary_RGB_vs_solo_native_RGB_maxabs=max(a['aux_asset_RGB_vs_solo_native_RGB_maxabs'] for v in r['views'] for a in v['arms'].values()),
            group_alpha_sum_vs_total_maxabs=max(floors)),
        control=r['opacity_control'],binding=r['binding'],rows=rows,
        covariance_calibration_review=cov,
        caveats=[
            'This is not full Mip-Splatting: no 3D smoothing filter, representation refit, or kernel replacement was implemented.',
            'The bound legacy operator uses neg_fov=True and unconditional +0.3I. Covariance factors reproduce that current operator rather than changing camera conventions.',
            'Joint alpha disagreement does not improve in every view/state; raw106 keep-joint alpha MAE increases from 0.00664839 to 0.01094422.',
            'Near-black opaque pixel counts include legitimate windows/tyres and cannot, by themselves, identify a semantic black-body defect.',
            'Source dark colours, projected chart depth, retained source rows, and a37/static ordering remain independent causes. No claim that alpha alone explains all black edges.',
            'raw113 and raw130 are internal development diagnostics; they are not an untouched final confirmation set.',
            'The actual chart teacher is the persistent 3D asset: hosted strongv2 proposal + true BUILD106 known RGB/source fallback, then 40-BUILD observed fit. SD prompt2_seed3 is a separate baseline and is not the chart teacher.',
        ],
        compute=dict(native_raster_calls=r['render_calls'],timed_main_wall_seconds=r['seconds'],
            wall_time_includes_loading_CPU_and_saving=True,CUDA_only_time_not_measured=True,
            peak_cuda_allocated_bytes=r['peak_cuda_allocated'],peak_cuda_reserved_bytes=r['peak_cuda_reserved'],
            model_weight_updates=0,geometry_updates=0,colour_updates=0,source_opacity_updates=0,
            shared_kernel_edits=0,render_copy_opacity_control_only=True),
        engineering_retries=[
            dict(stage='metadata command quoting',GPU_calls=0,optimization_steps=0),
            dict(stage='local torch import missing typing_extensions; used existing remote CPU environment',GPU_calls=0,optimization_steps=0),
            dict(stage='CPU generic test closure correction',GPU_calls=0,optimization_steps=0)],
        immutable_asset_and_kernel_hashes_checked_before_after=True,automatic_adoption=False)
    write(REVIEW/'scale_resolution_review.json',report);write(OUT/'review.json',report)
    shutil.copy2(RESULT/'receipt.json',OUT/'receipt.json')
    shutil.copy2(HERE/'scale_probe.py',OUT/'scale_probe.py')
    for name in ('relative_area_asset.py','covariance_cpu_review.json','review_covariance_cpu.py','camera_geometry_cpu.json'):
        shutil.copy2(HERE/name,OUT/name)
    if (HERE/'download_verification.json').is_file():
        shutil.copy2(HERE/'download_verification.json',OUT/'download_verification.json')
    architecture='''<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="240" viewBox="0 0 1040 240"><defs><marker id="a" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#304c60"/></marker></defs><rect width="1040" height="240" fill="#f4f8fb"/><g font-family="Arial,sans-serif" font-size="16" text-anchor="middle" fill="#18394e"><g stroke="#718fa2" stroke-width="1.5"><rect x="15" y="72" width="180" height="90" rx="9" fill="white"/><rect x="245" y="25" width="220" height="75" rx="9" fill="white"/><rect x="245" y="140" width="220" height="75" rx="9" fill="white"/><rect x="515" y="72" width="235" height="90" rx="9" fill="white"/><rect x="800" y="72" width="225" height="90" rx="9" fill="white"/></g><text x="105" y="105">Frozen 3D asset</text><text x="105" y="130">same camera / rows</text><text x="355" y="53">1920 actual render</text><text x="355" y="79">premul RGB + alpha</text><text x="355" y="169">480 actual render</text><text x="355" y="195">base / fixed area control</text><text x="632" y="102">HR 4×4 area downsample</text><text x="632" y="128">fixed masks / same crop</text><text x="912" y="103">Solo + joint comparison</text><text x="912" y="129">sampling diagnostic</text></g><g fill="none" stroke="#304c60" stroke-width="2" marker-end="url(#a)"><path d="M195 99L245 62"/><path d="M195 135L245 177"/><path d="M465 62L515 96"/><path d="M465 177L515 140"/><path d="M750 117L800 117"/></g><text x="520" y="233" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="#4c6474">No training; no geometry/colour/source-opacity/kernel changes. HR reference is not hidden ground truth.</text></svg>'''
    (OUT/'architecture.svg').write_text(architecture,encoding='utf-8')
    textlines=['三视角同分辨率分解与采样诊断','',report['conclusion'],'',
        '比较对象为同一3D资产的1920实际渲染经4x4面积下采样；并非隐藏RGB真值。',
        'raw | solo RGB MAE baseline→control | joint RGB MAE baseline→control | chart factor mean']
    for raw in r['raws']:
        a=next(x for x in rows if x['raw']==raw and x['state']=='solo')
        b=next(x for x in rows if x['raw']==raw and x['state']=='joint')
        textlines.append(f"{raw} | {a['baseline_rgb_MAE']:.8f}→{a['compensated_rgb_MAE']:.8f} | {b['baseline_rgb_MAE']:.8f}→{b['compensated_rgb_MAE']:.8f} | {a['chart_factor_mean']:.8f}")
    textlines.extend(['',*report['caveats'],'',f"40 forward calls, timed main {r['seconds']:.6f}s (includes CPU/load/save). No model/geometry/colour updates."])
    (REVIEW/'scale_resolution_review.txt').write_text('\n'.join(textlines)+'\n',encoding='utf-8')
    table=''.join('<tr><td>'+str(x['raw'])+'</td><td>'+x['state']+'</td>'+''.join('<td>'+f"{x[k]:.6f}"+'</td>' for k in ('baseline_rgb_MAE','compensated_rgb_MAE','baseline_alpha_MAE','compensated_alpha_MAE'))+'</tr>' for x in rows)
    sections=[]
    for raw in r['raws']:
        paths=[('HR_baseline/same_resolution_fair.png','1920：同分辨率单体 / 含a37 / 移除a37'),
               ('LR_baseline/same_resolution_fair.png','480：同分辨率单体 / 含a37 / 移除a37'),
               ('base_vs_comp_fair480.png','480基线 / 固定面积补偿 / 1920面积下采样（相同框）'),
               ('LR_area_compensated/sampling_comparison.png','固定补偿与HR参考的单体及joint内A颜色贡献')]
        figures=[]
        for rel,label in paths:
            source=RESULT/f'raw{raw}'/rel
            name=f'raw{raw}_'+rel.replace('/','_')
            assert source.is_file();shutil.copy2(source,OUT/name)
            uri='data:image/png;base64,'+base64.b64encode(source.read_bytes()).decode()
            figures.append(f'<figure><figcaption>{html.escape(label)}</figcaption><a href="{name}"><img src="{uri}" alt="{label}"/></a></figure>')
        sections.append(f'<h2>raw{raw}</h2>'+''.join(figures))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>同分辨率的车体插回诊断</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1300px;margin:auto;padding:30px;background:#f5f8fb;color:#172c3c}h1{font-size:28px}h2{margin-top:36px}p{max-width:1000px}.note{border-left:4px solid #bf7d22;background:#fff4e1;padding:14px}table{border-collapse:collapse;background:white}td,th{padding:8px 14px;border:1px solid #c9d7e0}figure{margin:20px 0}img{max-width:100%;height:auto;border:1px solid #ccd7dd}figcaption{font-weight:600}code{overflow-wrap:anywhere}</style><h1>同分辨率的车体插回与采样诊断</h1><p>冻结现有 a6 资产，对 raw106、113、130 的 480/1920 实际单体与全场景渲染作公平对照。480 原算子相对 1920 面积下采样存在额外采样差异。一次固定的 chart 面积补偿降低三视角 RGB 差异；现有资产和正式渲染器均未改变。</p><p class="note">当前仅为内部诊断。HR实际渲染不是隐藏真值；面积补偿不证明几何正确。raw106 的 joint alpha 差异反而略增。源颜色、PVD深度、源点保留和遮挡仍须分别审查；不自动采用补偿。</p><p>HR与LR使用相同相机姿态、投影和行来源；相同分辨率的 solo / joint 共用精确相同A行。表中误差参考为 D4(HR)，固定对象域为 D4(HR solo alpha)&gt;0.01，RGB为预乘值，不是2D候选质量分数。</p>'''
    page+='<table><thead><tr><th>raw</th><th>状态</th><th>RGB MAE 原480</th><th>RGB MAE 固定补偿</th><th>α MAE 原480</th><th>α MAE 固定补偿</th></tr></thead><tbody>'+table+'</tbody></table>'
    covariance_table=''.join('<tr><td>'+str(x['raw'])+'</td>'+''.join('<td>'+f"{x[k]['median']:.4f}"+'</td>' for k in ('physical_pinhole_HR_unfiltered_sigma_minor','physical_pinhole_HR_unfiltered_sigma_major','active_legacy_HR_unfiltered_sigma_minor','active_legacy_HR_unfiltered_sigma_major','active_legacy_LR_filtered_sigma_minor','active_legacy_LR_filtered_sigma_major'))+'</tr>' for x in cov['views'])
    page+='<h2>新采样与legacy footprint的标定差异</h2><p>原chart在raw106以物理相机fx标定σ=0.6HR像素；CPU按原资产及3pose计算同一协方差在真实针孔相机与当前legacy核中的投影。当前neg_fov=True会使用tan(-0.5)及原负阈值clamp次序，所得footprint与新采样标定不一致。以下仅为解析诊断，未改任何camera、全场景renderer或共享.so。数值为每行主轴σ的中位数；480列已包含原算子+0.3I。</p><table><tr><th>raw</th><th>理想HR σ小</th><th>理想HR σ大</th><th>legacyHR σ小</th><th>legacyHR σ大</th><th>native480 σ小</th><th>native480 σ大</th></tr>'+covariance_table+'</table>'
    page+=''.join(sections)+'<h2>数据流与边界</h2>'+architecture+'<p>Chart的实际颜色来源为 hosted strongv2 外观proposal、真实 BUILD106 known RGB及源色fallback，再经过40个BUILD的观测颜色拟合；SD prompt2_seed3 是独立离线基线，并非这个chart的teacher。图像proposal不能当作几何真值。</p><p>固定控制仅在渲染行副本对chart opacity施加 k=f_R/f_HR，f_R=√[det(G_R)/det(G_R+0.3I)]；HR系数严格1，raw106 HR重放逐元素精确。采用当前算子的 neg_fov 与投影约定，没有修改相机或 .so。这是受 <a href="https://niujinshuchong.github.io/mip-splatting/">Mip-Splatting</a> 面积归一化启发的局部对照，并非完整Mip方法。</p>'
    page+=f"<p>计算：40次native forward，计时主体 {r['seconds']:.3f}秒（含加载、CPU和保存）；权重/几何/颜色更新均为0。峰值CUDA allocated {r['peak_cuda_allocated']/2**30:.2f}GiB。固定原资产 SHA256 <code>{r['asset_sha256']}</code>。</p>"
    page+='<p>复用接口：先将base资产放至目标device，然后 <code>RelativeAreaOpacityAsset(base_asset)</code> 传给原 <code>Runtime.scene</code>。wrapper仅按当前资产 <code>representation_kind==1</code> 选择chart，无N或前缀cut硬编码；HR直接返回base原rows。CPU已验证混排来源、两种N、各向异性四元数与预计算协方差一致、source和nonchart不变。</p><p>下载：<a href="receipt.json">完整实际receipt</a> · <a href="review.json">独立诊断与限制</a> · <a href="metrics.csv">指标CSV</a> · <a href="scale_probe.py">实际推理源代码</a> · <a href="relative_area_asset.py">可复用Asset wrapper</a> · <a href="covariance_cpu_review.json">CPU标定与接口验收</a></p></html>'
    (OUT/'index.html').write_text(page,encoding='utf-8')
    manifest={str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='sha256.json'}
    write(OUT/'sha256.json',manifest)
    print(json.dumps(dict(ready=True,output=str(OUT),review=str(REVIEW/'scale_resolution_review.json'),receipt_sha256=report['receipt_sha256'],files=len(manifest)),ensure_ascii=False))

if __name__=='__main__':main()
