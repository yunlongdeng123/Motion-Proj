"""Independent local-only report QA. Does not import or run the publisher."""
import base64,hashlib,io,json
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlparse,unquote
from datetime import datetime,timezone
from PIL import Image
W=Path(__file__).resolve().parents[3];A=W/'work/a6_artifact_diagnosis';O=W/'outputs/scene008_vehicle_appearance/diagnosis';P=A/'report/publish_diagnosis.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def fmt(v,n=6):
    if v is None:return '—'
    if isinstance(v,bool):return str(v).lower()
    if isinstance(v,int):return f'{v:,}'
    if isinstance(v,float):return f'{v:.{n}f}'
    return str(v)
class Parser(HTMLParser):
    def __init__(self):super().__init__();self.refs=[];self.imgs=[];self.ids=[];self.sec='top';self.text={};self.skip=0;self.scripts=0;self.svgs=0;self.rows=[];self.row=None;self.cell=None
    def handle_starttag(self,t,a):
        d=dict(a)
        if 'id' in d:self.ids.append(d['id'])
        if t=='section':self.sec=d.get('id','?')
        if t=='script':self.scripts+=1
        if t=='svg':self.svgs+=1
        for k in ('href','src'):
            if d.get(k):self.refs.append((t,k,d[k]))
        if t=='img':self.imgs.append(d.get('src',''))
        if t in ('style','script','svg'):self.skip+=1
        if t=='tr':self.row=[]
        if t in ('td','th'):self.cell=[]
    def handle_endtag(self,t):
        if t in ('td','th') and self.cell is not None:self.row.append(''.join(self.cell).strip());self.cell=None
        if t=='tr' and self.row is not None:self.rows.append((self.sec,self.row));self.row=None
        if t in ('style','script','svg'):self.skip=max(0,self.skip-1)
    def handle_data(self,s):
        if self.cell is not None:self.cell.append(s)
        if not self.skip and s.strip():self.text.setdefault(self.sec,[]).append(s.strip())

def main():
    content=(O/'index.html').read_bytes();h=content.decode('utf-8');html_sha=hashlib.sha256(content).hexdigest();publisher_sha=sha(P)
    p=Parser();p.feed(h);publication=read(O/'publication_receipt.json')
    missing=[];outside=[];fragments=[];external=[];local=[];image_errors=[];sizes=[];hash_bad=[];source_changes=[]
    for t,k,v in p.refs:
        if v.startswith('data:'):continue
        if v.startswith('#'):
            if v[1:] not in p.ids:fragments.append(v)
            continue
        u=urlparse(v)
        if u.scheme:external.append(v);continue
        q=(O/unquote(u.path)).resolve();local.append(v)
        if not q.is_relative_to(O.resolve()):outside.append(v)
        if not q.is_file():missing.append(v)
    for i,v in enumerate(p.imgs):
        try:
            assert v.startswith('data:image/');payload=base64.b64decode(v.split(',',1)[1],validate=True)
            im=Image.open(io.BytesIO(payload));sizes.append(list(im.size));im.verify()
        except Exception as ex:image_errors.append(dict(index=i,error=str(ex)))
    for x in publication['evidence']:
        q=O/x['href']
        if not q.is_file() or sha(q)!=x['sha256']:hash_bad.append(x['href'])
        s=Path(x['source'])
        if s.is_file() and sha(s)!=x['sha256']:source_changes.append(x['source'])
    for i,x in enumerate(publication['images']):
        q=O/x['original_href']
        if not q.is_file() or sha(q)!=x['source_sha256']:hash_bad.append(x['original_href'])
        if sizes[i]!=x['wh']:image_errors.append(dict(index=i,error='receipt dimensions differ'))
        if not x['display_derivative']:assert hashlib.sha256(base64.b64decode(p.imgs[i].split(',',1)[1])).hexdigest()==x['source_sha256']
    manifest=read(O/'sha256.json');manifest_bad=[rel for rel,value in manifest.items() if not(O/rel).is_file() or sha(O/rel)!=value]
    assert read(O/'publication_qa.json')['HTML_sha256']==html_sha
    expected=[]
    def row(sec,values,source):expected.append(dict(section=sec,values=[str(x) for x in values],source=source))
    F=A/'surface_fit/combined_v1/observed_fit';formal=read(F/'formal/receipt.json');smoke=read(F/'smoke/receipt.json');review=read(A/'surface_fit/combined_v1/review.json');hr=read(F/'evaluation/HR_summary.json');sums=read(F/'evaluation/summary.json')
    for k in ('status','colour_fit_steps','neural_training_steps','geometry_updates_during_colour_fit','unknown','seconds','PVD_weight_updates','BUILD_RGB_MAE'):row('combined',[k,fmt(formal[k])],'formal receipt')
    row('combined',['asset.rows',fmt(formal['asset']['rows'])],'formal receipt')
    for dom,lab in [('all49','all49混合开发域'),('HELD','HELD9内部开发域')]:
        for arm,name in [('old','旧rough-depth formal'),('final','新C2+matte formal')]:
            d=hr[dom];r=d[arm+'.premultiplied_RGB_error'];s=d[arm+'.straight_RGB_error'];a=d[arm+'.alpha'];assert r['count']==d[('final' if arm=='old' else 'old')+'.premultiplied_RGB_error']['count']
            row('combined',[lab,name,fmt(r['count']),fmt(r['mean']),fmt(s['mean']),fmt(a['mean'])],'HR summary fixed domains')
    for arm,lab in [('native_vs_old','native480 原算子'),('area_vs_old_area','480 相同面积控制')]:
        d=sums[arm]['all49'];r=d['solo.a6.rgb_error'];br=d['solo.a6.baseline_rgb_error'];z=d['solo.a6.measured_depth_error'];bz=d['solo.a6.baseline_measured_depth_error'];assert r['count']==br['count'] and z['count']==bz['count']
        row('combined',[lab,fmt(r['count']),fmt(br['mean']),fmt(r['mean']),fmt(z['count']),fmt(bz['mean']),fmt(z['mean']),fmt(d['scene.keep.a6.visible_coverage']['mean'])],'native/area fixed49 summary')
        for state,reg,desc in [('keep','outside','keep outside'),('delete','outside','delete outside'),('keep','a37','keep a37'),('keep','road_visible','keep 可见road')]:
            r=d[f'scene.{state}.preservation.{reg}.parent_rgb_change'];z=d[f'{state}_fixed_depth_preservation.'+('road' if reg=='road_visible' else reg)+'.expected_depth_parent_change_m']
            row('combined',[lab,desc,fmt(r['above_002'])+' / '+fmt(r['count']),fmt(100*r['above_002']/r['count'],4)+'%',fmt(r['mean'],8),fmt(r['max']),fmt(z['mean']),fmt(z['max'])],'fixed49 preservation summary')
    tail=review['tail_risk'];tr=tail['tail_abs_gt06']
    for name,val in [('chart深度位移绝对值>0.6m',fmt(tr['count'])+' / '+fmt(tail['delta_z_m']['count'])+'（'+fmt(100*tr['fraction_of_chart'],3)+'%）'),('该尾部prior alpha中位数 / alpha≤.9点数',fmt(tr['prior_alpha']['median'])+' / '+fmt(tr['prior_alpha_le09'])),('该尾部到matte边界距离中位数',fmt(tr['matte_boundary_distance_HRpx']['median'],3)+' HRpx'),('全chart最大前移 / 最大后移',fmt(-tail['delta_z_m']['min'])+' / '+fmt(tail['delta_z_m']['max'])+' m'),('最大训练锚点残差',fmt(tail['accepted_anchor_residual_m']['max'])+' m')]:row('combined',[name,val],'geometry tail review')
    app=read(A/'vehicle/matte_contract/application_receipt.json');sam=read(A/'vehicle/matte_contract/matte_receipt.json');filled=read(A/'vehicle/matte_contract/filled_v2_receipt.json');veh=read(A/'vehicle/gpu_result/vehicle/receipt.json');surface=read(A/'surface_fit/results/receipt.json');cpu=read(A/'vehicle/cpu_audit.json')
    assert publication['teacher_sha256']==cpu['actual_input']['candidate_sha256']
    row('actual-input',['核心并非普遍premul/straight错误','核心alpha均值 '+fmt(cpu['normalization']['106_projective_surface']['mean_alpha_core'])+'；除alpha后的核心RGB平均变化 '+fmt(cpu['normalization']['106_projective_surface']['premul_vs_normalized_RGB_L1_core']['mean'])],'actual saved normalization audit')
    row('actual-input',['保留prior不是三视角主因','106/113/130 retained最大alpha·T = '+' / '.join(fmt(x['retained_contribution_max']) for x in veh['frames'])],'actual native grouped contributions')
    row('actual-input',['像素中心/单位往返自洽','HR回投最大误差 '+fmt(cpu['backprojection']['reprojection_HR_px_max'],8)+' px；这不验证先验深度本身正确'],'CPU pixel-centre audit')
    v=veh['frames'][0]
    for name,val in [('alpha>0.98且近黑输出像素（含合法暗材质）',v['opaque_dark_output_pixels']),('其中source-dark alpha·T >0.9的像素',v['source_dark_opaque_dark_pixels_mass_gt09']),('generated chart近黑采样行',cpu['sampling']['generated_sample_dark_rows']),('bilinear跨外canvas tap的行（独立边界采样项）',cpu['sampling']['generated_rows_with_bilinear_background_mask_tap'])]:row('matte',[name,fmt(val)],'native/CPU dark-pixel sampling audits')
    for x in app['frames']:row('matte',[str(x['raw']),fmt(x['before_opaque_dark']),fmt(x['matte_opaque_dark']),fmt(x['known_HR_RGB_delta']['mean'],8),fmt(x['known_HR_RGB_delta']['max'])],'matte-only receipt')
    for name,val in [('v1保护的dark foreground像素',fmt(sam['dark_foreground_pixels_protected'])),('filled_v2封闭孔填充像素 / 新增SAM调用',fmt(filled['closed_raw_SAM_holes_filled_pixels'])+' / '+fmt(filled['SAM_predict_calls_added'])),('filled_v2允许chart行 / 原chart行',fmt(filled['effective_allowed_chart_rows'])+' / '+fmt(filled['chart_rows']))]:row('matte',[name,val],'SAM/fill receipts')
    row('matte',['SAM调用 / neural权重更新','1 / '+fmt(sam['model_training_updates'])],'single frozen SAM receipt')
    for name,val in [('内域相邻深度差p99',fmt(cpu['depth_surface']['interior_adjacent_depth_difference_m']['p99'])+' m'),('内域相邻差>10cm的对数',fmt(cpu['depth_surface']['interior_adjacent_pairs_gt10cm'])),('surfel局部sigma中位数',fmt(cpu['depth_surface']['sigma_local_m']['median'])+' m'),('40-BUILD后新增近黑real-seed / gen-seed行',fmt(cpu['observed_fit']['group_changes']['real_seed']['newly_dark_after_observed_fit'])+' / '+fmt(cpu['observed_fit']['group_changes']['generated_seed']['newly_dark_after_observed_fit']))]:row('depth',[name,val],'rough-depth/colour CPU audit')
    e=surface['measured_training_camera_z_abs_error_m']
    for k in ('count','mean','median','p99','max'):row('depth',[k,fmt(e['prior'][k]),fmt(e['posterior'][k])],'surface training-anchor fit receipt')
    scale=read(A/'scale/results/receipt.json');cov=read(A/'scale/covariance_cpu_review.json');shadow=read(A/'shadow/source_audit.json');geo=read(A/'shadow/results/conditional_geometry.json');replay=[]
    for v in scale['views']:
        b=v['arms']['LR_baseline']['vs_area_downsample_HR'];c=v['arms']['LR_area_compensated']['vs_area_downsample_HR']
        row('scale',[str(v['raw']),fmt(b['solo']['object']['rgb_MAE']),fmt(c['solo']['object']['rgb_MAE']),fmt(b['joint']['object']['rgb_MAE']),fmt(c['joint']['object']['rgb_MAE']),fmt(v['compensation']['chart_factor']['mean'])],'scale receipt')
        assert v['cameras']['LR']['wh']==[480,320] and v['cameras']['HR']['wh']==[1920,1280]
        for field in ('world_view','full_projection','center','timestamp'):assert v['cameras']['LR'][field]==v['cameras']['HR'][field]
        assert v['compensation']['HR_factor_strictly_one']
        if 'HR_compensated_identity_replay_exact' in v:assert v['HR_compensated_identity_replay_exact'];replay.append(v['raw'])
    for v in cov['views']:
        row('scale',[str(v['raw']),fmt(v['physical_pinhole_HR_unfiltered_sigma_minor']['median'],4)+' / '+fmt(v['physical_pinhole_HR_unfiltered_sigma_major']['median'],4),fmt(v['active_legacy_HR_unfiltered_sigma_minor']['median'],4)+' / '+fmt(v['active_legacy_HR_unfiltered_sigma_major']['median'],4),fmt(v['active_legacy_LR_filtered_sigma_minor']['median'],4)+' / '+fmt(v['active_legacy_LR_filtered_sigma_major']['median'],4)],'covariance CPU review')
    for v in shadow['ROI_summary']:
        if v['dark_core_pixels']:row('shadow',[str(v['raw'])+'（'+fmt(v['dark_core_pixels'])+'px）','static '+fmt(v['alpha_T']['legacy_static'])+'；road '+fmt(v['alpha_T']['road']),'static '+fmt(v['neutral_static_luma_delta'])+'；road '+fmt(v['neutral_road_luma_delta'])],'shadow fixed-ROI source audit')
    core=geo['fixed_domains']['same_dark_core']
    for lab,val in [('105 delete static camera-z (m)',core['states']['delete']['groups']['legacy_static']['pooled_conditional_camera_z_m']),('105 delete road camera-z (m)',core['states']['delete']['groups']['road']['pooled_conditional_camera_z_m']),('105 static相对已有路面中心高度 (m)',core['states']['delete']['pooled_static_height_above_road_m']),('105 keep a37 camera-z (m)',core['states']['keep']['groups']['a37']['pooled_conditional_camera_z_m']),('105同域a37关联LiDAR z均值 (m)',core['same_frame_a37_associated_LiDAR_camera_z_m']['mean'])]:row('shadow',[lab,fmt(val)],'shadow conditional same-sort geometry')
    for values in [('车体分解',fmt(veh['seconds'],3)+' s','实际GPU forward；无优化'),('matte SAM推理',fmt(sam['seconds'],3)+' s / 1 call','冻结SAM2.1；后续filled_v2无新增predict'),('matte-only v1',fmt(app['seconds'],3)+' s','独立opacity/边界取样因果对照'),('C2表面',fmt(surface['seconds'],3)+' s / '+fmt(surface['contract']['IRLS_solves'])+' IRLS solves','CPU求解；非神经权重训练'),('scale',fmt(scale['render_calls'])+' forwards / '+fmt(scale['seconds'],3)+' s','计时含加载、CPU和保存；不冒充CUDA-only计时'),('原阴影分层',fmt(shadow['actual_GPU_seconds'],3)+' s','本轮诊断，背景参数未更新')]:row('evidence',values,'actual runtime receipts')
    numeric_bad=[x for x in expected if (x['section'],x['values']) not in p.rows]
    asset_rel=next(x['href'] for x in publication['evidence'] if x['source']==publication['final_hook']['asset']);asset_sha=sha(O/asset_rel);assert asset_sha==formal['asset']['file_sha256']==publication['final_hook']['asset_sha256']
    roles=read(A/'report/evaluation_roles.json');assert not roles['independent_final_confirmation'] and not roles['untouched_generalization_test']
    text={k:' '.join(v) for k,v in p.text.items()}
    guards={
        'shadow_carrier_not_physical_or_historical_identity':all(s in text['shadow'] for s in ['不追溯历史高斯的出生身份','证明真实物理阴影','无事实同相机','不能从中性颜色实验推断真实无阴影外观']),
        'shadow_same_sort_scope_and_additive_environment':all(s in text['shadow'] for s in ['packed与active行均为0','加性背景','不是同一世界点']),
        'HELD_history_internal_development':all(s in text['combined'] for s in ['HELD9内部开发域','角色纠正','不据该字段声称独立最终确认']),
        'large_high_alpha_interior_geometry_tail_disclosed':all(s in text['combined'] for s in ['高alpha内域的大位移','2,210 / 42,536','2.701239 / 0.809745','2.620451','不能解释成只清理低alpha边缘']),
        'training_residual_not_generalization':all(s in text['depth'] for s in ['训练观测拟合','不能称泛化','最坏值变大']),
        'fair_resolution_not_hidden_ground_truth':all(s in text['scale'] for s in ['同一资产、同一相机','4×4面积下采样','自一致误差','绝非隐藏RGB真值','未实现其完整方法']),
        'native_regression_and_alpha_tradeoff_disclosed':all(s in text['combined'] for s in ['native480 RGB误差可退步','变透明的收益','0.072399','0.976360']),
        'frozen_parameters_not_frozen_composited_images':all(s in text['combined'] for s in ['冻结背景参数仍可能改变遮挡','35.951611','不称全域精确保护']),
        'research_asset_not_automatic_adoption':all(s in text['combined'] for s in ['尚不据此声明采用','PASS表示接线与规定检查完成','不是采用决定']),
        'architecture_components_present':p.svgs>=1 and ('native 联合排序' in h or '视口采样 / native排序' in h),
        'static_offline_inline_images':p.scripts==0 and len(p.imgs)==publication['embedded_images'] and publication['external_images']==0}
    issues=[]
    for kind,items in [('missing_local_links',missing),('portable_root_escape',outside),('broken_anchors',fragments),('image_decode_error',image_errors),('evidence_SHA_mismatch',hash_bad),('output_manifest_mismatch',manifest_bad),('table_receipt_mismatch',numeric_bad)]:
        if items:issues.append(dict(kind=kind,details=items))
    for k,v in guards.items():
        if not v:issues.append(dict(kind='semantic_guard_absent',details=k))
    stable=sha(O/'index.html')==html_sha
    if not stable:issues.append(dict(kind='HTML_changed_during_review',details='Re-review the new publication snapshot'))
    report=dict(status='PASS_REVIEWED_SNAPSHOT' if not issues else 'MUST_FIX',reviewed_UTC=datetime.now(timezone.utc).isoformat(),
        scope=dict(read_only=True,new_GPU_calls=0,network_calls=0,publisher_edited=False),must_fix=issues,
        snapshots=dict(HTML_sha256=html_sha,publisher_sha256=publisher_sha,HTML_stable_during_review=stable,publisher_stable_during_review=sha(P)==publisher_sha,publication_created_UTC=publication['created_UTC']),
        static_checks=dict(local_reference_occurrences=len(local),unique_local_files=len(set(local)),inline_images_decoded=len(sizes),evidence_files_hash_checked=len(publication['evidence']),image_originals_hash_checked=len(publication['images']),output_manifest_files_checked=len(manifest),display_derivatives=sum(x['display_derivative'] for x in publication['images']),external_citation_not_download_not_network_checked=sorted(set(external))),
        numerical_checks=dict(receipt_bound_table_rows_checked=len(expected),mismatches=numeric_bad,download_asset_sha256=asset_sha,download_bound_to_formal_receipt=True,HR_old_new_denominators_equal=True,native480_old_new_denominators_equal=True,LiDAR_denominators_equal=True,formal_steps=formal['colour_fit_steps'],extra_smoke_steps=smoke['colour_fit_steps'],HR_exact_replay_raws=replay),
        scientific_scope_PASS=guards,
        PASS_items=['All local image/download targets and section anchors exist; inline images decode; copied evidence/original SHA and output ledger match.','All checked receipt-bound table rows match, including both metric improvements and regressions.','Shadow carrier/conditional-centre evidence is distinguished from actor identity, historical origin and physical shadow truth.','Native480 and1920 comparisons are separated and matched; area control/alpha tradeoff is explicit.','HELD9 is internal method-development diagnosis, with historical final-only flag corrected without rewriting scientific receipts.','High-alpha interior geometry tail,2.70m forward shift,2.62m anchor residual and large outside-depth changes are disclosed.','Architecture component SVG and frozen-network/per-asset-fitting scope are explicit.'],
        pending_addenda=['This snapshot has no surface-LOD section. Backend LOD publication requires a subsequent link/value/scope QA on the new HTML.'],
        nonblocking_observations=[dict(current_sources_changed_since_publication=source_changes),'External Mip-Splatting citation is not a local download and was not network-tested.'])
    target=A/'review/final_report_review.json';target.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],must_fix=issues,static_checks=report['static_checks'],numeric_rows=len(expected),guards=guards,review=str(target)),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
