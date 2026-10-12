"""Focused static QA of actual long-window report evidence; no browser/GPU."""
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'outputs/drivingeditor_vgpu_debug'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Structure(HTMLParser):
    def __init__(self):
        super().__init__();self.section_ids=[];self.videos=[];self.marker_count=0
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='section':self.section_ids.append(a.get('id'))
        if tag=='video':self.videos.append(a)
        if tag=='marker' and a.get('id')=='arr':self.marker_count+=1


page=(OUT/'index.html').read_text(encoding='utf8')
parser=Structure();parser.feed(page)
has_actual_V='long-v-three-arm' in parser.section_ids
summary=json.loads((HERE/'long_original_summary_cpu.json').read_bytes())
videos=json.loads((OUT/'long_B1_video_validation.json').read_bytes())
short=json.loads((OUT/'paired_video_validation.json').read_bytes())
refs=json.loads((OUT/'local_reference_validation.json').read_bytes())
checks={
 'exactly_one_separate_B1_32000_section':parser.section_ids.count('long-b1-32000')==1,
 'B1_exact_9decimal_composite_and_static_means_displayed':'36.430756861' in page and '11.960381004' in page,
 'support_frame_mean_and_pixel_weighted_both_labeled':'99.861365%' in page and '99.860770%' in page and '49帧等权均值' in page and '按区域像素汇总' in page,
 'separate30300_pair_title_and_original_media_retained':'短实验30300：同一时间轴的真实49帧配对' in page and sha(OUT/'assets'/short['file'])==short['mp4_sha256'],
 'long_B1_media_sources_and_all245_decodes_pinned':videos['decoded_frame_count']==245 and all(v['all_frames_decoded'] and sha(OUT/'assets'/v['file'])==v['sha256'] for v in videos['videos']),
 'all_video_players_have_controls_and_no_autoplay':len(parser.videos)==(3 if has_actual_V else 2) and all('controls' in v and 'autoplay' not in v for v in parser.videos),
 'architecture_arrows_and_old3090_reference_retained':parser.marker_count==1 and '参考' not in '' and '../drivingeditor_30k_edit_comparison/index.html?reference=3090-original' in page,
 'B1_completed_B2_complete_actual_evidence':('B1／B2／V均完成，效果仍未验收' if has_actual_V else 'B1／B2完成，V待终点验证') in page and '实际worker／launcher均退出0' in page,
 'B2_restore_and_first10_not_whole_loss_trajectory':'全2,000个camera／time_shift／LR与B1精确一致' in page and '首十标量相等，后续CUDA损失轨迹仍有变化' in page,
 'no_normal_or_paper_reproduction_acceptance_claim':'没有正常效果或完整论文复现结论' in page and '尚未通过论文效果验收' in page,
 'no_dense_edited_GT_and_no_long_short_paired_subtraction':'车体内遮挡区没有dense edited GT' in page and '不能将B1 32k与短候选30300相减' in page,
 'all_current_local_references_exist':refs['status']=='pass' and all(r['exists'] and r['bytes']>0 for r in refs['references']),
}
repeat=json.loads((OUT/'long_B2_report_inputs_validation.json').read_bytes())
noise=json.loads((OUT/'evidence/B2_original30k_to32000_noise_reference/receipts/B1_B2_observed_noise.json').read_bytes())
checks.update({
 'exactly_one_separate_B2_noise_section':parser.section_ids.count('long-b2-noise')==1,
 'actual_B2_raw10_four_PNG_pins_decoded':len(repeat['copied_actual_raw10_PNGs'])==4 and all(sha(OUT/'assets'/r['file'])==r['sha256'] and r['decoded_size']==[480,320] for r in repeat['copied_actual_raw10_PNGs']),
 'B2_9decimal_means_and_weighted_support_distinct':'36.430473769' in page and '11.962831689' in page and '99.859669%' in page and '99.859056%' in page,
 'mean_difference_and_paired_absolute_distribution_distinct':'N=|Δ均值|' in page and '逐帧|Δ|均值为0.008105216' in page and 'P95为0.028672171' in page,
 'all11_preregistered_primary_rows_with_context_and_all49':repeat['primary_metric_count']==len(noise['primary_metrics'])==11 and repeat['context_metric_count']==126 and repeat['frame_count']==49 and '126项六域上下文指标' in page,
 'one_repeat_is_not_CI_or_general_noise_bound':'一次观察到的原版重复差异，不是置信区间、统计显著性或所有CUDA运行的通用噪声界' in page,
 'inverse_depth_explicit_cross_page_and_harmonic900_choice':'下一页1703明确' in page and 'inverse-depth loss 本身是论文明确要求' in page and 'harmonic blend 与 sky_depth=900' in page,
 'no_V_result_placeholder_or_long_candidate_outcome':('长V真实同窗结果已纳入下方' in page and '长V尚无终点效果结论' not in page) if has_actual_V else ('长V尚无终点效果结论' in page and any(phrase in page for phrase in ('长V的实际新binding和CPU独立审阅已就绪', '长V已实际启动，终点结果尚未产生'))),
})
if has_actual_V:
    vroot=HERE/'continuation_diagnostic/opacity_outside30k_to32000'
    own=HERE/'sgcm_data_audit/long_pair_review'
    real=json.loads((vroot/'receipts/three_arm_locked_metrics.json').read_bytes())
    interpretation=json.loads((own/'actual_three_arm_interpretation.json').read_bytes())
    vvideo=json.loads((OUT/'long_V_video_validation.json').read_bytes())
    triple_video=json.loads((OUT/'long_three_arm_video_validation.json').read_bytes())
    copied=OUT/'evidence/V_opacity_outside30k_to32000'
    checks.update({
        'one_actual_V_section_with_finite_scientific_scope':parser.section_ids.count('long-v-three-arm')==1 and '分配方向改善且有重建／目标保真代价' in page and '恢复实验，不是从零训练' in page,
        'actual_V_canonical137_rows_all49_copied_exact':len(real['primary_metrics'])==11 and len(real['context_metrics_all_six_domains'])==126 and all(len(r['V_vs_each_original'][arm]['frames'])==49 for r in real['primary_metrics']+real['context_metrics_all_six_domains'] for arm in ('B1','B2')) and sha(copied/'receipts/three_arm_locked_metrics.json')=='922a3dea16a427539ec944765680b315e9b69b8e7a217af4c5d990b7f7679a9d',
        'independent_numeric_and_locality_receipts_copied_exact':sha(copied/'independent_three_arm_numeric_review.json')=='6b40d9b8e048a23ce3186f16db6db33233b51c9aa7afc8e395db7a849658ef3e' and sha(copied/'removal_locality_actual_cpu.json')=='463fa751241988eb49375d4de447a46000d7c506fbaa755f854d7d3ac8817b8d',
        'all49_adverse_effects_and_actor_tradeoff_not_averaged_away':'全部49帧的背景static MSE均改善' in page and '全部49帧的全图和目标车辆composite MSE均升高' in page and '33.19001' in page,
        'residual_dynamic_and_locality_still_explicit':'79.368807%' in page and '59.281480%' in page and '59.268281%' in page,
        'fixed8_actual16_PNGs_byte_exact':len(interpretation['predeclared_fixed8_actual_images_inspected'])==16 and all(sha(OUT/'assets'/f'V32000_{r["channel"]}_raw{int(Path(r["frame_id"]).name[-3:]):02}.png')==r['sha256'] for r in interpretation['predeclared_fixed8_actual_images_inspected']),
        'adverse_frame_selection_disclosed_not_fixed8':'终点后按不利指标选择' in page and 'raw43–47' in page,
        'V_new_video_actual49_and_old_media_not_reencoded':vvideo['frame_count']==49 and vvideo['all_frames_decoded'] and not vvideo['original_B1_or_short30300_videos_reencoded'] and len(vvideo['input196_PNG_sha256'])==196 and sha(OUT/'assets'/vvideo['file'])==vvideo['sha256'],
        'actual_three_arm_sync_video_all49_source588_pinned':triple_video['frame_count']==49 and triple_video['all_frames_decoded'] and len(triple_video['actual588_source_PNG_sha256'])==588 and triple_video['row_order']==['B1','B2','V'] and not triple_video['old_B1_or_short_videos_reencoded'] and sha(OUT/'assets'/triple_video['file'])==triple_video['sha256'],
        'actor_mask_area_and_original_pooled_error_parallel':'131,747' in page and '226–6,125' in page and 'raw45为307像素' in page and all(value in page for value in ('0.000142594178','0.000138788998','0.000403615477','38.458982051','38.576449590','33.940321891')),
        'no_stale_current_V_pending_claim':all(s not in page for s in ('V待终点验证','长V终点結果尚未产生','长V终点结果尚未产生','长V已获单独批准并运行中','长V须单独批准并验证','本页尚未纳入其完整终态')),
    })
if not all(checks.values()):
    raise AssertionError([key for key,value in checks.items() if not value])
receipt={
 'status':'PASS_FOCUSED_CPU_STATIC_REPORT_QA', 'checks':checks,
 'report_html_sha256':sha(OUT/'index.html'), 'long_summary_cpu_sha256':sha(HERE/'long_original_summary_cpu.json'),
 'long_video_receipt_sha256':sha(OUT/'long_B1_video_validation.json'),
 'short_paired_video_receipt_sha256':sha(OUT/'paired_video_validation.json'),
 'short_paired_MP4_matches_existing_source_receipt':True,
 'B1_294_actual_PNG_hash_and_dimensions_verified':True,
 'actual_images_inspected':['B1 raw10 original GT','B1 raw10 composite','B1 raw10 static','B1 raw10 dynamic','B2 raw10 original GT','B2 raw10 composite','B2 raw10 static','B2 raw10 dynamic']+(interpretation['predeclared_fixed8_actual_images_inspected']+interpretation['adverse_image_supplement_disclosed'] if has_actual_V else []),
 'visual_finding':'B1/B2 static confetti and dynamic road persist; V static road improves but dynamic/background fragments and target fidelity costs remain. No normal-algorithm claim.' if has_actual_V else 'B1/B2 static road has extensive RGB confetti; dynamic includes road/background. No normal-algorithm claim.',
 'browser_UI_or_playback_checked':False, 'CUA_used':False, 'GPU_used':False,
 'task_manifest_or_automation_written_by_this_work':False,
 'new_scripts_sha256':{name:sha(HERE/name) for name in ['build_debug_report.py','summarize_long_original.py','encode_long_original_videos.py','render_long_original_report.py','render_long_repeat_report.py','verify_long_report_cpu.py']},
 'metric_aggregation_scope':summary['aggregation'],
 'B2_report_inputs_validation_sha256':sha(OUT/'long_B2_report_inputs_validation.json'),
 'canonical_original_repeat_noise_sha256':sha(OUT/'evidence/B2_original30k_to32000_noise_reference/receipts/B1_B2_observed_noise.json'),
 'actual_local_reference_count':len(refs['references']),
}
path=OUT/('long_three_arm_report_validation.json' if has_actual_V else 'long_B1_B2_report_validation.json')
path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
print(json.dumps({'status':receipt['status'],'checks':len(checks),'validation_sha256':sha(path),'browser_playback_checked':False}))
