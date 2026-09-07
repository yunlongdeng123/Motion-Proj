"""构造全部build窗口的CAPA稀疏条件并记录覆盖；不复制已有RGB数据。"""
import argparse,json
from pathlib import Path
import subprocess,sys,time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v73.capa_inputs import build_capa_condition


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--native-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); torch.set_num_threads(4)
    started=time.monotonic()
    scenes=torch.load(args.native_run/'build_observations.pt',map_location='cpu',weights_only=False)
    rows=[]
    for scene in scenes:
        rgb,depth,mask,counts=build_capa_condition(scene)
        rows.append({'scene':scene['scene_id'],'role':scene['role'],'log_id':scene['log_id'],
            'image_shape':list(rgb.shape),'sparse_condition_pixels_per_view':counts,
            'input_projected_measurements':sum(len(view['uv']) for view in scene['views']),
            'views':[{'camera_id':view['camera_id'],'sample_id':view['sample_id'],
                'camera_time_us':view['camera_time_us'],'lidar_time_us':view['lidar_time_us']} for view in scene['views']]})
        print(json.dumps({'scene':scene['scene_id'],'condition_pixels':sum(counts)}),flush=True)
        del rgb,depth,mask
    result={'status':'done','native_run':str(args.native_run),
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'windows':rows,'wall_s':time.monotonic()-started,
        'boundary':'build images + actual calibrated axial LiDAR depths; nearest measurement after pixel quantization; no extra-time label read; counts describe sparse conditions, not occlusion ground truth',
        'storage':'conditions reconstructed on demand; RGB/depth tensors not duplicated to disk'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__': main()
