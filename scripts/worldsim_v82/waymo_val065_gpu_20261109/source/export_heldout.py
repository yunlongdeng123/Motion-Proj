"""Decode held-out RGB only after all fits are fixed; no fitting module reads here."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
from pathlib import Path
import sys,json,io
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent.parent
PREP=Path('/root/autodl-tmp/codex_tmp/waymo_val065_cpu_20261108_r1')
def main():
    sys.path.insert(0,'/root/autodl-tmp/codex_tmp/v43_actor_diagnostics_20261101');import waymo_numpy_cpu as decoder
    pb=decoder.protobuf();media=json.loads((PREP/'media.json').read_text());frames={r['raw']:r for r in media['frames']};out=ROOT/'heldout';out.mkdir(exist_ok=False)
    for n in ['rgb','calib','observations']:(out/n).mkdir()
    rawfile=json.loads((PREP/'inputs/receipt.json').read_text())['raw'];records=[]
    for raw,payload,rec in decoder.records(Path(rawfile),198):
        if raw not in range(79,119):continue
        f=pb.Frame.FromString(payload);front=next(x for x in f.images if x.name==1);cal=next(x for x in f.context.camera_calibrations if x.name==1)
        Image.open(io.BytesIO(front.image)).convert('RGB').resize((480,320),Image.Resampling.BILINEAR).save(out/'rgb'/f'{raw:03d}.png');(out/'calib'/f'{raw:03d}.txt').write_text(decoder.calibration_text(f));d=dict(raw=raw,timestamp_micros=f.timestamp_micros,ego_world=list(f.pose.transform),camera_pose_world=list(front.pose.transform),camera_extrinsic=list(cal.extrinsic.transform),camera_intrinsic_Brown=list(cal.intrinsic),boxes=frames[raw]['boxes']);(out/'observations'/f'{raw:03d}.json').write_text(json.dumps(d));records.append(dict(raw=raw,**rec))
    (out/'receipt.json').write_text(json.dumps(dict(status='PASS_HELDOUT_RGB_DECODED_FOR_EVALUATION_ONLY',frames=records,lidar_decoded=False,fitting_updates=0,source='original CRC-checked TFRecord'),indent=2));print('HELDOUT_RGB',len(records))
if __name__=='__main__':main()
