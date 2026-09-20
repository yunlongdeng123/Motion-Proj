"""正式生成前的输入、编码与权重检查；没有 DiT generate 调用。"""
import argparse
import gc
import json
import os
import time
import zipfile

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('LOCAL_FILES_ONLY', '1')
from common import RUN, SCENE, CAMERA, config

def save(name, value):
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

def scene_inputs():
    import numpy as np
    from PIL import Image
    from scipy.spatial.transform import Rotation, Slerp
    from interactive_drive.config import RasterConfig
    from interactive_drive.scene_loader import load_scene_bundle
    scene = load_scene_bundle(SCENE, CAMERA, 'default', None, RasterConfig())
    RUN.mkdir(parents=True, exist_ok=True)
    Image.fromarray(scene.initial_rgb).save(RUN/'initial_rgb.png')
    (RUN/'prompt.txt').write_text(scene.prompt+'\n')
    with zipfile.ZipFile(SCENE) as z:
        names = z.namelist()
        frames = sorted(n for n in names if n.startswith('frames/'+CAMERA+'/') and n.endswith('.jpeg'))
        first_us = int(frames[0].rsplit('/', 1)[-1].split('.')[0])
        trajectory = json.loads(z.read('rig_trajectories.json'))['rig_trajectories'][0]
    src_us = np.asarray(trajectory['T_rig_world_timestamps_us'], dtype=np.int64)
    poses = np.asarray(trajectory['T_rig_worlds'], dtype=np.float64)
    # 首段5帧、以后每段8帧，共30个完整段；与实际RGB时间对齐，禁止外推。
    timestamps = first_us + np.rint(np.arange(5+29*8)*1e6/30).astype(np.int64)
    assert np.all(np.diff(src_us)>0) and timestamps[0]>=src_us[0] and timestamps[-1]<=src_us[-1]
    source_t = (src_us-src_us[0])/1e6
    target_t = (timestamps-src_us[0])/1e6
    out = np.repeat(np.eye(4)[None], len(timestamps), axis=0)
    out[:, :3, :3] = Slerp(source_t, Rotation.from_matrix(poses[:, :3, :3]))(target_t).as_matrix()
    for axis in range(3):
        out[:,axis,3] = np.interp(target_t, source_t, poses[:,axis,3])
    np.savez(RUN/'trajectory.npz', timestamps_us=timestamps, rig_poses_world=out.astype(np.float32))
    manifest = {'task_id':'WS-V75-PREFLIGHT-01','run_id':'20260920-r1',
        'scene':str(SCENE),'camera':CAMERA,'initial_frame_member':frames[0],
        'rgb_shape':list(scene.initial_rgb.shape),'prompt_characters':len(scene.prompt),
        'input_role':'engineering_development_only','scene_count':1,
        'real_reference_frames':len(frames),'future_rgb_reference_available':len(frames)>1,
        'official_bundle_initial_us':scene.initial_timestamp_us,'initial_rgb_us':first_us,
        'alignment_offset_us':first_us-scene.initial_timestamp_us,
        'trajectory_interpolation':'linear translation + SO(3) Slerp; no extrapolation',
        'input_frames':len(timestamps),'blocks':30,'fps':30,
        'duration_between_frame_centers_s':float((timestamps[-1]-timestamps[0])/1e6),
        'actors':len(scene.vehicle_bbox_tracks),'seed':42,
        'map_layers':len(scene.line_layers)+len(scene.triangle_layers)+len(scene.polygon_layers),
        'world_model_generation_forwards':0,'human_verdict':None}
    assert scene.prompt and scene.initial_rgb.shape==(704,1280,3)
    save('input_manifest.json',manifest)
    return scene

def render_inputs():
    import numpy as np
    import torch
    from PIL import Image
    from interactive_drive.config import RasterConfig
    from interactive_drive.rasterizer import LudusConditionRasterizer
    scene = scene_inputs()
    traj = np.load(RUN/'trajectory.npz')
    count = len(traj['timestamps_us'])
    images = np.lib.format.open_memmap(RUN/'conditions.npy', mode='w+', dtype=np.uint8, shape=(count,704,1280,3))
    renderer = LudusConditionRasterizer(RasterConfig(), max_chunk_frames=8)
    began = time.monotonic()
    try:
        renderer.load_scene(scene)
        for start in range(0,count,8):
            result = renderer.render_chunk(traj['rig_poses_world'][start:start+8],traj['timestamps_us'][start:start+8])
            for i, frame in enumerate(result.frames):
                rgb = np.asarray(frame.rgb_host_uint8)
                assert rgb.shape==(704,1280,3) and rgb.dtype==np.uint8
                images[start+i] = rgb
            if start%32==0:
                print(json.dumps({'rendered':min(start+8,count),'total':count}),flush=True)
        images.flush()
        for i in (0,4,36,count-1):
            Image.fromarray(images[i]).save(RUN/f'condition_{i:03d}.png')
        assert int(images[0].max())>0
    finally:
        renderer.cleanup()
    save('render_result.json', {'status':'passed','frames':count,'elapsed_s':time.monotonic()-began,
        'renderer':'official LudusConditionRasterizer, unchanged','world_model_generation_forwards':0})

def embeddings():
    import numpy as np
    import torch
    from PIL import Image
    cfg = config()
    torch.cuda.reset_peak_memory_stats()
    text_encoder = cfg.text_encoder.setup().to('cuda').eval()
    with torch.inference_mode():
        text = text_encoder([(RUN/'prompt.txt').read_text().strip()]).unsqueeze(0).cpu()
    assert text.shape==(1,1,512,100352) and torch.isfinite(text).all()
    del text_encoder
    gc.collect(); torch.cuda.empty_cache()
    image_encoder = cfg.image_encoder.setup().to('cuda').eval()
    rgb = torch.from_numpy(np.asarray(Image.open(RUN/'initial_rgb.png')).copy())
    rgb = rgb.permute(2,0,1)[None,None,None].to(device='cuda',dtype=torch.bfloat16)/127.5-1
    with torch.inference_mode():
        image = image_encoder(rgb).cpu()
    assert image.shape==(1,1,1,16,88,160) and torch.isfinite(image).all()
    torch.save({'text_embeddings':text,'image_embeddings':image,'negative_text_embeddings':None},RUN/'embeddings.pt')
    save('embedding_result.json',{'status':'passed','text_shape':list(text.shape),'image_shape':list(image.shape),
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'generation_forwards':0,
        'encoder_calls':{'CosmosReason1':1,'LightVAE':1},'text_weight_source':'ModelScope nv-community mirror'})

def weights():
    import torch
    cfg = config()
    cfg.text_encoder = None
    cfg.image_encoder = None
    torch.cuda.reset_peak_memory_stats()
    pipeline = cfg.setup().to('cuda').eval()
    assert all(torch.isfinite(p).all() for p in pipeline.parameters())
    values = torch.load(RUN/'embeddings.pt',map_location='cpu',weights_only=True)
    cache = pipeline.initialize_cache_from_embeddings(**values,view_names=[CAMERA])
    torch.cuda.synchronize()
    save('weight_result.json',{'status':'passed','strict_official_checkpoint_loading':True,
        'pipeline_cache_initialized':True,'generation_forwards':0,'pipeline_training':pipeline.training,
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'note':'只验证加载与cache分配；不是完整推理峰值或输出正确性证明'})

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=['scene','render','embeddings','weights'])
    args = parser.parse_args()
    {'scene':scene_inputs,'render':render_inputs,'embeddings':embeddings,'weights':weights}[args.phase]()
