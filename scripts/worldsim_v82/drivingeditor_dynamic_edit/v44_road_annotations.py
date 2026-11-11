"""从现有原始TFRecord导出全部物体框，供固定世界道路来源的遮挡认证。"""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--parser-dir',required=True,type=Path);a=p.parse_args()
    sys.path.insert(0,str(a.parser_dir));from waymo_numpy_cpu import records,protobuf
    pb=protobuf();receipt=json.loads((a.bank/'receipt.json').read_text());wanted=set(receipt['build_raws'])
    out=a.output;out.mkdir(exist_ok=False);rows=[]
    for raw,payload,record in records(receipt['raw'],max(wanted)):
        if raw not in wanted:continue
        frame=pb.Frame.FromString(payload)
        assert frame.context.name==receipt['context_id']
        observed=json.loads((a.bank/'observations'/f'{raw:03d}.json').read_text())
        assert int(frame.timestamp_micros)==observed['timestamp_micros']
        ego=np.array(frame.pose.transform).reshape(4,4);boxes=[];rectangles=[]
        for lab in frame.laser_labels:
            b=lab.box
            if min(b.length,b.width,b.height)<=0:continue
            cs,sn=np.cos(b.heading),np.sin(b.heading);rot=np.array([[cs,-sn,0],[sn,cs,0],[0,0,1.]])
            center=np.array([b.center_x,b.center_y,b.center_z])@ego[:3,:3].T+ego[:3,3]
            boxes.append(dict(track=lab.id,type=int(lab.type),center=center.tolist(),half=[b.length/2,b.width/2,b.height/2],axes=(ego[:3,:3]@rot).tolist()))
        for group in frame.camera_labels:
            if group.name!=1:continue
            for lab in group.labels:
                b=lab.box;rectangles.append(dict(track=lab.id,type=int(lab.type),xyxy=[b.center_x-b.length/2,b.center_y-b.width/2,b.center_x+b.length/2,b.center_y+b.width/2]))
        data=dict(raw=raw,timestamp_micros=int(frame.timestamp_micros),boxes=boxes,front_camera_rectangles=rectangles)
        (out/f'{raw:03d}.json').write_text(json.dumps(data,indent=2)+'\n')
        rows.append(dict(raw=raw,boxes=len(boxes),rectangles=len(rectangles)))
        if raw%25==0:print('ANNOTATIONS',rows[-1],flush=True)
    assert [r['raw'] for r in rows]==sorted(wanted)
    (out/'receipt.json').write_text(json.dumps(dict(status='EXPORTED_ALL_RAW_ANNOTATION_BOXES',frames=rows,source=receipt['raw'],
        all_laser_label_types=True,all_front_camera_label_types=True,no_model_inputs_changed=True),indent=2)+'\n')
if __name__=='__main__':main()
