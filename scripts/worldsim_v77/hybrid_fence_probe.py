"""栏杆空隙诊断：固定图像提示，检查SAM是否能区分可见金属和孔洞。"""
import torch
from hybrid_common import *
sys.path.insert(0,'/root/autodl-tmp/third_party/worldsim_v32/sam2')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
torch.set_num_threads(4)
out=ROOT/'scene_0255';im=rgb(out/'rgb/00000.png')
p=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml','/root/autodl-tmp/third_party/worldsim_v32/sam2/checkpoints/sam2.1_hiera_large.pt',device='cuda'))
pts=np.array([[447,337],[388,313],[528,324],[400,361],[356,347],[545,346],[376,328],[397,330],[501,334],[521,336],[390,349],[500,357],[397,380],[520,380]],np.float32)
labels=np.array([1]*6+[0]*8)
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 p.set_image(im);ms,scores,_=p.predict(box=np.array([350,301,562,388]),point_coords=pts,point_labels=labels,multimask_output=True)
probe=out/'fence_probe';probe.mkdir(exist_ok=True)
sheet=Image.new('RGB',(640*2,360*2))
for j,m in enumerate(ms):
 write_mask(probe/f'{j}.png',m>0);img=im.copy();img[m>0]=(.4*img[m>0]+.6*np.array([0,230,240])).astype('uint8');sheet.paste(Image.fromarray(img).resize((640,360)),((j%2)*640,(j//2)*360))
sheet.save(probe/'comparison.jpg',quality=95)
dump(probe/'prompts.json',dict(points=pts.tolist(),labels=labels.tolist(),scores=scores.tolist(),role='assistant image-informed diagnostic prompts, not automatic segmentation score'))
print(scores)
