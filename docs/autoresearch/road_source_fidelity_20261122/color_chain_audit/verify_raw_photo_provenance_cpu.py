"""把选定 bank PNG 与原 TFRecord FRONT JPEG 解码逐像素比较；只读 CPU。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import argparse,io,json,sys,time
import numpy as np
from PIL import Image
sys.dont_write_bytecode=True


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bank',type=Path,default=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs'))
    ap.add_argument('--decoder-root',type=Path,default=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82/drivingeditor_debug/followup_20261013/red_preparation'))
    ap.add_argument('--frames',nargs='+',type=int,default=[86,110,115,117,134,137])
    args=ap.parse_args();started=time.monotonic()
    receipt=json.loads((args.bank/'receipt.json').read_text())
    raw_path=Path(receipt['raw']);selected=set(args.frames)
    sys.path.insert(0,str(args.decoder_root))
    import waymo_numpy_cpu as decoder
    pb=decoder.protobuf();rows=[]
    for index,payload,crc in decoder.records(raw_path,max(selected)):
        if index not in selected:continue
        frame=pb.Frame.FromString(payload)
        front=next(i for i in frame.images if i.name==1)
        calibration=next(c for c in frame.context.camera_calibrations if c.name==1)
        obs=json.loads((args.bank/f'observations/{index:03d}.json').read_text())
        with Image.open(io.BytesIO(front.image)) as im:
            raw_format=im.format;raw_rgb=np.asarray(im.convert('RGB'))
        png_path=args.bank/obs['image_path']
        with Image.open(png_path) as im:bank_rgb=np.asarray(im.convert('RGB'))
        difference=raw_rgb.astype(np.int16)-bank_rgb.astype(np.int16)
        rows.append(dict(raw=index,timestamp_micros=int(frame.timestamp_micros),
            original_image_format=raw_format,original_JPEG_bytes=len(front.image),bank_png=str(png_path),
            image_shape=list(raw_rgb.shape),PIL_decoded_raw_JPEG_equals_bank_PNG=bool(np.array_equal(raw_rgb,bank_rgb)),
            differing_channels=int(np.count_nonzero(difference)),RGB8_max_abs=int(abs(difference).max()),
            raw_Brown_intrinsics=list(calibration.intrinsic),
            raw_Brown_equals_observation=bool(np.array_equal(np.asarray(calibration.intrinsic),obs['camera_intrinsic_Brown'])),
            raw_image_pose_equals_observation=bool(np.array_equal(np.asarray(front.pose.transform),obs['camera_pose_world'])),
            TFRecord_selected_record_CRC=crc))
    assert len(rows)==len(selected)
    print(json.dumps(dict(schema='raw_waymo_JPEG_bank_pixel_provenance_cpu.v1',CPU_only=True,
        original_TFRecord=str(raw_path),frames=args.frames,rows=rows,
        all_selected_bank_PNGs_are_unwarped_JPEG_decodes=all(r['PIL_decoded_raw_JPEG_equals_bank_PNG'] for r in rows),
        bank_receipt_limitations=receipt.get('limitations'),
        export_code=str(args.bank.parent/'source/prepare_cpu.py'),
        distortion_field_pattern=str(args.bank/'observations/{raw:03d}.json')+':camera_intrinsic_Brown',
        raw_images_or_assets_modified=False,GPU_calls=0,
        imported_torch='torch' in sys.modules,imported_tensorflow=any(n.startswith('tensorflow') for n in sys.modules),
        seconds=time.monotonic()-started),ensure_ascii=True,indent=2,allow_nan=False))


if __name__=='__main__':main()
