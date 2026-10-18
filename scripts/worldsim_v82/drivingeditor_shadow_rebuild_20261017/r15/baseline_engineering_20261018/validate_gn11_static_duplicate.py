from pathlib import Path
import json,re,hashlib,zipfile,subprocess,xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
W=Path(__file__).parent;O=W.parents[1]/'outputs/baseline_engineering_20261018/gn11_static_duplicate_20261019'
r=json.loads((O/'receipt.json').read_text());m=json.loads((O/'motion_feature_receipt.json').read_text());s=json.loads((O/'source_comparison.json').read_text())
assert r['frames']==49 and r['mask_union_exact_all49'] and r['canonical_mask_matches_saved_real_eval_all49'] and not r['GPU_used'] and r['training_updates']==0
assert r['actual_CPU_gradient']['static_RGB_gradient_inside_all_actor_pixels_absmax']==0 and not r['actual_CPU_gradient']['dynamic_opacity_gradient_to_static_alpha_connected']
assert all(s['getter_AST_equal'].values()) and s['renderer_bytes_identical']
assert m['rows_center_span_over_0p1m_and_feature_velocity_below_1e_6']==1195277
with zipfile.ZipFile(O/'actual_source_bytes.zip') as z:
 for k,v in r['actual_source_sha256'].items():assert hashlib.sha256(z.read(k)).hexdigest()==v
links=0
for v in re.findall(r'(?:href|src)="([^"]+)"',(O/'index.html').read_text(encoding='utf-8')):
 if not v.startswith(('http','#')):assert (O/v).exists(),v;links+=1
ET.parse(O/'architecture.svg')
for p in (O/'frames').glob('*.png'):
 with Image.open(p) as im:assert im.size==(480,320);im.verify()
from build_direct_player import video_contract
video_contract(O/'fixed_camera_time_only.mp4')
# Actual decoder, explicit frame40 (raw129) from the already delivered static MP4.
sample=W/'gn11_static_duplicate_20261019/player_static_frame040.png'
subprocess.run(['ffmpeg','-v','error','-y','-i',str(O.parent/'branch_videos/gn11five_static.mp4'),'-vf',r'select=eq(n\,40)','-frames:v','1',str(sample)],check=True)
video=np.asarray(Image.open(sample)).astype(float);png=np.asarray(Image.open(O/'frames/raw129_static.png')).astype(float)
mae=float(np.abs(video-png).mean());dynamic=np.asarray(Image.open(O/'frames/raw129_dynamic.png')).astype(float);dyn_mae=float(np.abs(video-dynamic).mean())
assert mae<3 and dyn_mae>mae*10,(mae,dyn_mae)
out=dict(status='PASSED_ACTUAL_RAW129_VIDEO_FRAME_SOURCE_IMAGE_NUMERIC_AND_FILE_CHECKS',links=links,complete_frame_images=5,
 source_byte_hashes_verified=len(r['actual_source_sha256']),fixed_camera_video_frames=49,raw129_static_player_decoded_RGB_MAE_255=mae,
 decoded_static_to_dynamic_RGB_MAE_255=dyn_mae,actual_media_decoder_checked=True,browser_interaction_verified=False,
 training_updates=0,GPU_used=False,human_verdict=None)
(O/'validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
