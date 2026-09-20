"""固定官方配置和本地资产路径；不自动下载或生成视频。"""
import copy
from pathlib import Path

ASSETS = Path('/root/autodl-tmp/models/worldsim_v75')
RUN = Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-PREFLIGHT-01/20260920-r1')
SCENE = ASSETS/'omni-dreams-scenes/scenes/clipgt-0d404ff7-2b66-498c-b047-1ed8cded60d4.usdz'
CAMERA = 'camera_front_wide_120fov'

def config():
    from omnidreams.config import OMNIDREAMS_PIPELINE_CONFIG
    cfg = copy.deepcopy(OMNIDREAMS_PIPELINE_CONFIG)
    cfg.text_encoder.model_name = str(ASSETS/'Cosmos-Reason1-7B-modelscope')
    for name in ('image_encoder', 'encoder'):
        getattr(cfg, name).checkpoint_path = str(ASSETS/'lightvaew2_1.pth')
    cfg.decoder.checkpoint_path = str(ASSETS/'lighttaew2_1.pth')
    cfg.diffusion_model.transformer.checkpoint_path = str(ASSETS/'omni-dreams-models/single_view/2b_res720p_30fps_i2v_hdmap_distilled.pt')
    return cfg
