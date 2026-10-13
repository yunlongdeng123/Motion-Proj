"""Record the completed peer pixel review and root's actual selected checks.

Full-sequence claims come from the delegated reviewer's explicit progress report,
not from merely decoding or counting files. Extra alpha_s/body is not assumed.
"""
from pathlib import Path
import json,datetime
W=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-12/w');O=W/'outputs/drivingeditor_red_GN11_30k'
m=json.loads((O/'manifest.json').read_text(encoding='utf-8-sig'));ids=[f['id'] for f in m['frames']]
assert len(ids)==len(set(ids))==49
coverage={'status':'REQUIRED_ACTUAL_PIXEL_REVIEW_COMPLETE','reviewer':'/root/scene8_terminal_audit',
 'basis':'Reviewer explicitly reported actually viewing ALL49x5 observed and ALL49x2 fixed-time contact sheets; root preserved this reported evidence.',
 'observed_reviewed_frame_ids':ids,'observed_channels':['GT','composite','static','dynamic','raw_alpha_d'],
 'fixed_pose':'0029079','fixed_time_reviewed_source_frame_ids':ids,'fixed_channels':['static_RGB','raw_alpha_s'],
 'required_reviewed_PNG_count':343,'required_review_complete':True,
 'extra_alpha_s_and_body_all98_review_claimed':False,'extra_coverage':'Only use a separate actual extra-review receipt if available.',
 'root_direct_pixel_checks':['raw79 static/dynamic real browser','fixed_time_11_44_47.png: raw113-116 static RGB and raw alpha_s'],
 'pixel_transform':'Native480x320 copies in contact sheets; no resize/filter/normalization/pre-mask; GUI screenshot has normal browser display scale.',
 'human_verdict':None}
r={'status':'REAL_REQUIRED49_LAYERED_AND_FIXED49_ENGINEERING_REVIEW_COMPLETE','review_complete':True,'human_verdict':None,
 'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scene':'scene0029_069_117','checkpoint':30000,
 'full_sequence_reviewer':'/root/scene8_terminal_audit','full_sequence_review_evidence_source':'Explicit actual-viewing reviewer progress messages during this task; coverage.json records the required channels and frame IDs.',
 'root_direct_pixel_confirmation':coverage['root_direct_pixel_checks'],'coverage_file':'engineering-visual-coverage.json',
 'observations':[
   {'scope':'observed','frames':'raw69-100','channels':['GT','composite','dynamic'],'finding':'Red sedan shape/color/crossing direction broadly follows GT; raw98-100 right-edge truncation remains; no established whole-body persistent collapse.'},
   {'scope':'observed','frames':'raw71-98','channels':['static'],'finding':'Black underbody/wheel/shadow bars remain on the static road; dynamic branch does not capture a complete grounding shadow.'},
   {'scope':'fixed_pose0029079','frames':'raw71-100','channels':['static_RGB','raw_alpha_s'],'finding':'Black bars move left-to-right with model time; tree shadows and road texture also vary; alpha remains opaque around road residuals.'},
   {'scope':'fixed_pose0029079','frames':'raw101-117','channels':['static_RGB'],'finding':'Black residual blob remains near right curb.'},
   {'scope':'fixed_pose0029079','frames':'raw105-117','channels':['static_RGB'],'finding':'Narrow green vertical line appears near right sky/STOP; not comparably prominent in the 49 observed composites.'},
   {'scope':'fixed_pose0029079','frames':'raw113-117','channels':['static_RGB'],'finding':'Localized pale pink foreground-road patches; root directly confirmed raw113-116 contact sheet.'},
   {'scope':'observed','frames':'raw101-117','channels':['dynamic','raw_alpha_d'],'finding':'Empty body tail shows no obvious car-sized residue in required pixel review; strict zero is not certified.'}
 ],
 'tail_low_threshold_pixel_counts':'unavailable in this required receipt; no zero substitution; see separate extra-review receipt if produced',
 'engineering_quality_boundary':'Static time invariance, complete shadow separation, zero leakage and edited-background GT quality are not established.',
 'relative_GN11_regression_test':'Not applicable: no same-scene C00; do not reinterpret absolute defects as module causal regression or gains.',
 'counterfactual_GT_available':False,'independent_test':False,'restoration_verified':False}
for name,d in [('engineering-visual-review.json',r),('engineering-visual-coverage.json',coverage)]:
    p=O/name;assert not p.exists();p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print('ACTUAL_REQUIRED343_REVIEW_RECORDED_FROM_REVIEWER_AND_ROOT_CHECKS')
