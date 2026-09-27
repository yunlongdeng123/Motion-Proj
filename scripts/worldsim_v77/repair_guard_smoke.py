"""CPU单图检查模型加载/API，避免占用正在推理的GPU。"""
import torch,time
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
from PIL import Image
from repair_common import *
torch.set_num_threads(4);m='/root/autodl-tmp/models/worldsim_v77/grounding-dino-tiny'
p=AutoProcessor.from_pretrained(m,local_files_only=True);model=AutoModelForZeroShotObjectDetection.from_pretrained(m,local_files_only=True).eval()
im=Image.open(ROOT/'official_000/rgb/00000.png').convert('RGB');inputs=p(images=im,text='car. truck. bus. van.',return_tensors='pt');t=time.monotonic()
with torch.inference_mode():pred=model(**inputs)
r=p.post_process_grounded_object_detection(pred,inputs.input_ids,box_threshold=.25,text_threshold=.25,target_sizes=[im.size[::-1]])[0]
assert len(r['boxes'])>0
dump(ROOT/'guard_smoke.json',{'cpu_single_forward':True,'boxes':r['boxes'].tolist(),'scores':r['scores'].tolist(),'labels':r['labels'],'seconds':time.monotonic()-t,'note':'API smoke only，非正式guard结果'})
print(len(r['boxes']),'vehicles',time.monotonic()-t)
