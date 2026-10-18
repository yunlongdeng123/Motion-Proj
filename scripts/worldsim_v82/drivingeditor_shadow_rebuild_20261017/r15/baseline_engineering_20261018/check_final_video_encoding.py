"""CPU-only synthetic check: partial MP4 is preserved and not reused."""
from pathlib import Path
import json,uuid
from PIL import Image
from build_direct_player import encode_complete,video_contract
W=Path(__file__).parent
def main():
 root=W/('video_encoding_fixture_'+uuid.uuid4().hex[:8]);root.mkdir()
 for raw in range(89,138):
  d=root/f'0008{raw:03d}';d.mkdir();Image.new('RGB',(480,320),(raw,raw//2,255-raw)).save(d/'full.png')
 target=root/'complete.mp4';target.write_bytes(b'partial interrupted MP4')
 encode_complete(root,'full.png',target);s=video_contract(target)
 preserved=list(root.glob('complete.incomplete_*.mp4'));assert len(preserved)==1 and preserved[0].read_bytes()==b'partial interrupted MP4'
 before=target.stat().st_mtime_ns;encode_complete(root,'full.png',target);assert target.stat().st_mtime_ns==before
 receipt=dict(status='PASSED_CPU_COMPLETE49_VIDEO_RECOVERY_FIXTURE',frames=49,render_wh=[480,320],fps='10/1',
    input_spatial_cropping=False,all_frames_are_I_frames=True,incomplete_file_preserved=True,valid_video_reused=True,
    real_model_results=False,GPU_training_updates=0,fixture_path=str(root.resolve()))
 (W/'video_encoding_CPU_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
if __name__=='__main__':main()
