"""原生样例clean的固定时间对比、完整解码与资源摘要；不虚构三维误差。"""
import argparse
import json
import time
from pathlib import Path
import av
import numpy as np
from PIL import Image,ImageDraw
from prepare_samples import BASE
from catalog_samples import RUN

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene-index',type=int,required=True);a=p.parse_args()
    plan=json.loads((RUN/'samples_cohort_manifest.json').read_text())
    row=[s for s in plan['selected'] if s['role']=='development'][a.scene_index];out=BASE/row['scene_uuid']
    generated=json.loads((out/'generate_result.json').read_text());assert generated['status']=='complete'
    result_path=out/'review_result.json';assert not result_path.exists()
    inputs=json.loads((out/'input_manifest.json').read_text())
    arrays=[np.load(out/f'{name}.npy',mmap_mode='r') for name in ['reference','conditions','clean']]
    sheet=Image.new('RGB',(1920,5*378),'#151c29')
    for row,f in enumerate([0,60,120,180,234]):
        y=row*378
        ImageDraw.Draw(sheet).text((8,y+5),f'Recorded RGB | Official HDmap | Generated clean | t={f/30:.2f}s',fill='white')
        for col,data in enumerate(arrays):sheet.paste(Image.fromarray(data[f]).resize((640,352)),(col*640,y+26))
    sheet.save(out/'comparison.jpg',quality=94)
    with av.open(str(out/'comparison.mp4'),'w') as writer:
        stream=writer.add_stream('libx264',rate=30);stream.width=1920;stream.height=378;stream.pix_fmt='yuv420p';stream.options={'crf':'20'}
        for f in range(237):
            canvas=Image.new('RGB',(1920,378),'#151c29')
            ImageDraw.Draw(canvas).text((8,5),f'Recorded RGB | Official HDmap | Generated clean | t={f/30:.3f}s',fill='white')
            for col,data in enumerate(arrays):canvas.paste(Image.fromarray(data[f]).resize((640,352)),(col*640,26))
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(canvas),format='rgb24')):writer.mux(packet)
        for packet in stream.encode():writer.mux(packet)
    decode={}
    for name in ['clean.mp4','comparison.mp4']:
        with av.open(str(out/name)) as c:decode[name]=sum(1 for f in c.decode(video=0))
        assert decode[name]==237
    result={'status':'complete','scene_uuid':out.name,'role':'frozen_native_development','seed':42,
            'frames':237,'decode_counts':decode,'generation_wall_s':generated['wall_s'],
            'peak_allocated_gib':generated['peak_allocated_gib'],
            'source_initial_frame_mae':inputs['initial_image_alignment']['frame0_mae'],
            'first_input_output_mae':float(np.abs(arrays[0][0].astype(float)-arrays[2][0]).mean()),
            'metric_3d_state_error_available':False,'natural_reconstruction_input':False,
            'claim_boundary':'原生条件输入下的clean工程基线；未作干预、归因或驾驶后果测量',
            'failure_ledger_delta':'none','human_verdict':None}
    result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
