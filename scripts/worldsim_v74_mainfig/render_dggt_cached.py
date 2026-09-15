import pathlib,sys,os,json,torch,numpy as np,traceback
P=pathlib.Path('/root/autodl-tmp');R=P/'runs/worldsim_v74_h2/WS-V74-MAINFIG-01/20260915-first-return-r1';repo=P/'external/worldsim_v81/dggt';sys.path.insert(0,str(repo));os.chdir(repo)
from dggt.models.vggt import VGGT
from dggt.utils.load_fn import load_and_preprocess_images
from render_dggt_native import render
torch.set_num_threads(4);model=VGGT();model.load_state_dict(torch.load(P/'models/worldsim_v81/model_latest_nuscenes.pt',map_location='cpu',weights_only=True,mmap=True),strict=True);model=model.eval().float().cuda()
for f in sorted((R/'secondary/predictions/dggt').glob('*/*/result.json')):
 dest=f.parent
 if (dest/'native_render/result.json').exists():continue
 rec=json.loads(f.read_text());m=json.loads(pathlib.Path(rec['input']).read_text());images=load_and_preprocess_images([v['image'] for v in m['views'][:rec['n_images']]],mode='crop').cuda();a=np.load(dest/'native_outputs.npz');pred={k:torch.from_numpy(a[k]).cuda() for k in a.files if k not in ['world_points','world_points_conf']}
 try:render(model,pred,images,dest)
 except Exception:(dest/'render_error.txt').write_text(traceback.format_exc());print(traceback.format_exc(),flush=True)
 del pred,images;torch.cuda.empty_cache()
