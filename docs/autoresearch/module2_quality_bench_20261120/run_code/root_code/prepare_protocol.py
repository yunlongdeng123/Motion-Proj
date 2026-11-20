from pathlib import Path
import json,hashlib

LOCAL=Path(__file__).resolve().parent
Q='/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1'
G=Q+'/gpu_20261119'
B=G+'/quality_bench_20261120_r1'
spec={
 'schema':'road_asset_quality_benchmark.v2',
 'objective':'Preserve teacher road texture, verifiable marking shape and sharpness in persistent 3D rendering; expose blur, false texture, coverage gaps and static ghosts separately.',
 'scope':{'optimization':False,'new_shared_network':False,'TRELLIS':False,'frames':[86,110,134,179,198],
          'teacher_views':[86,110],'diagnostic_only_views':[134,179,198],'inherited199BUILD_not_blind':True,'GPU_keep_on':True},
 'status':'prospective diagnostic specification after previous results; not a blind or preregistered original experiment',
 'freeze_policy':'Freeze evaluation implementation/config and source hashes before reading new benchmark outputs; corrections retain failed versions and rerun narrowly. No post-result thresholds, ROI picking or aggregate score.',
 'source':{'campaign':G,'prior_frozen_manifest':G+'/evaluation/frozen_r2/manifest.json',
           'historical_ROI':G+'/evaluation/frozen_r2/rois.json','originals_unchanged':True},
 'primary_cases':{
   'M00':G+'/formal_r1/M00/C800','M10':G+'/formal_r1/M10/C800',
   'M01':G+'/formal_r1/M01/C800','M11':G+'/formal_r1/M11/C800'},
 'stage_diagnostics_not_matched_budget':{'initial':G+'/formal_r1/pixel_initial',
       'low_A200':G+'/formal_r1/pixel_common_A200'},
 'filter_diagnostic':{'M12':G+'/source_filtered_training/formal_r1','not_primary_factorial':True},
 'teacher_identity':{'A':'real visible RGB','C':'frozen pseudo hole teacher, not hidden-road GT',
    'S':'frozen pseudo target-shadow teacher','markings':'only verified/manual-supported segments can be called marking structure reference; generic pseudo teacher edge != lane GT'},
 'evaluation_layers':{
   'material':{'legacy_480_masks':'fixed existing A/C/S denominators and context; no mask selected by method output',
       'native_masks':'retain original legal target masks before Gaussian alpha; require trustworthy context for photometric detail',
       'comparison':'same-resolution float teacher and actual rendered packets; no upsampling480 as native result',
       'alpha':'primary detail uses road straight RGB R/alpha versus original T only on FIXED COMMON baseline alpha>=.95 context/score (division<=1/.95); preserve excluded counts. Legacy premult R versus alpha*T remains separate. No de-premult optimization or low-alpha division; new-space metrics are not method improvements over prior reported metrics.'},
   'coverage':{'domain':'full pre-alpha legally expected road/hole/shadow masks',
       'measures':['alpha distribution and coverage thresholds .05/.5/.9/.99','connected gaps and largest gap','thin coverage','fixed K protection'],
       'warning':'conditional texture scores cannot certify missing surface coverage'},
   'joint':{'domain':'target vehicle image footprint, target shadow, kept vehicles, road-only vs joint visibility',
       'measures':['road/joint actual alphaT contribution','fixed target-footprint residual against pseudo teacher in legal visible road only','readonly static vs road contribution comparison'],
       'warning':'other vehicles lack teacher GT after deletion; classify available attribution evidence and diagnostic observations separately'}},
 'detail':{'scales_pixels':[1,2,4],'resolution_units':'input pixel; no pooled native/coarse metric',
   'context':'teacher-available legal context distinct from score mask; teacher and method share context/core',
   'boundary':'strict finite-kernel core primary, partial support/boundary counts explicit; outside-context RGB cannot influence primary scores',
   'measures':['HF correspondence cosine','HF residual MAE/RMSE','predicted and teacher energy plus excess/missing','teacher-flat spurious HF',
      'teacher-edge PR','edge distance','edge orientation disagreement','tile macro mean','tail quantiles','worst tile coordinates'],
   'MAE_role':'auxiliary color fit only, no visual-quality acceptance',
   'tiles':{'origin_xy':[0,0],'coarse480_tile_side':32,'native1920_tile_side':128,'selection':'fixed grid intersected with fixed target domains; retain invalid/N-A counts; do not select by prediction'},
   'no_quality_total_or_unjustified_pass_threshold':True},
 'teacher_controls':{'identity':'exact teacher in same metric space',
    'roundtrip4':'native1920x1280 float encoded-sRGB 4x4 mean then defined reconstruction; diagnostic of lost supervision, not GS output',
    'blur':'fixed declared blur negative control, not a tuned reconstruction',
    'synthetic_validation':['thin edge flattened but global color MAE small','phase-wrong equal-energy texture fails correspondence','outside-context black/color never contaminates scores']},
 'highres_probe':{'target_resolution':[1920,1280],'frames':[110,86],'read_only':True,
    'preconditions':['actual camera/intrinsics','new actual native basis','new fixed-world UV / pixel-footprint read','saved world fields unchanged','constant/decomposition qualification'],
    'forbidden':'bilinear or nearest enlargement of coarse480 rendering labeled native1920; changing geometry or fitting per-frame image',
    'budget':'bounded renderer-only probe, no optimizer; exact forward/memory/time ledger required; final scope recorded before launch'},
 'analysis_order':['1 detail matching and teacher-downsample loss','2 source/multiview/reader and fixed parameter limitations','3 full road coverage and static handoff/ghosts'],
 'report':{'mode':'dark','architecture_components_required':True,'teacher_display':'original teacher and fixed-alpha*teacher separately labelled',
           'views':'native1:1 and same nearest-neighbor zoom; deterministic historical ROI plus metric-ranked worst tiles labelled posthoc diagnostic'},
 'paths':{'remote_output':B,'local_output':'outputs/road_quality_bench_20261120_r1'},
}
text=json.dumps(spec,ensure_ascii=False,indent=2)+'\n'
(LOCAL/'protocol.json').write_text(text,encoding='utf-8')
print(json.dumps({'protocol':str(LOCAL/'protocol.json'),'sha256':hashlib.sha256((LOCAL/'protocol.json').read_bytes()).hexdigest()},ensure_ascii=False))
