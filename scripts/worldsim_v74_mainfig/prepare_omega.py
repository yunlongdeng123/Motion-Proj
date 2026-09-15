import pathlib,sys,json,subprocess
P=pathlib.Path('/root/autodl-tmp');repo=P/'external/worldsim_v74_mainfig/vggt-omega';sys.path.insert(0,str(repo))
import torch
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
R=P/'runs/worldsim_v74_h2/WS-V74-MAINFIG-01/20260915-first-return-r1'
view=json.loads((R/'inputs/scene-0520.json').read_text())['views'][0]
images=load_and_preprocess_images([view['image']],image_resolution=416)
status={'status':'WAIT_AUTHORIZED_CHECKPOINT','http_status':401,'configured_hf_token':False,'checkpoint_url':'https://huggingface.co/facebook/VGGT-Omega/resolve/main/vggt_omega_1b_416_reproduce.pt','repo':str(repo),'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'preprocess':'official balanced, image_resolution416','image_shape':list(images.shape),'torch_existing':torch.__version__,'official_torch_requirement':'>=2.6','import_and_preprocess':'PASS; full forward not tested, runtime upgrade may be needed','new_forwards':0,'user_prompted':True}
(R/'omega_status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status))
