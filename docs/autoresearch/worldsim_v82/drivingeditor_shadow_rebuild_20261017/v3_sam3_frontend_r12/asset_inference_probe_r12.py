"""Real offline inference on a BUILD RGB; no scene training or synthetic masks."""
from pathlib import Path
import os,sys,json,time,gc
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch,numpy as np
from PIL import Image,ImageDraw
torch.set_num_threads(2);torch.manual_seed(0)
sys.path.insert(0,'/root/autodl-tmp/idsplat/asset_sources/sam3')
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');root=R/'ownership_controls_r12_sam3';out=root/'asset_probe';out.mkdir(exist_ok=False)
manifest=json.loads((R/'data_r2/manifest.json').read_text());row=next(x for x in manifest['frames'] if x['frame_id']=='0008107');assert row['split']=='train'
image=Image.open(R/'data_r2'/row['rgb']).convert('RGB');start=time.monotonic()
report=dict(status='RUNNING',frame_id=row['frame_id'],BUILD_RGB=True,scene_training_updates=0,torch=torch.__version__,source_commit='7f473cfd983d5eb43040ff608f66bff0d7bdcb53',shared_environment_modified=False)
(out/'status.json').write_text(json.dumps(report,indent=2))
try:
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  model=build_sam3_image_model(checkpoint_path='/root/autodl-tmp/models/idsplat/sam3/sam3.pt',bpe_path='/root/autodl-tmp/idsplat/asset_sources/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz',load_from_HF=False,device='cuda',compile=False)
  processor=Sam3Processor(model,device='cuda',confidence_threshold=.5);state=processor.set_image(image);state=processor.set_text_prompt('vehicle',state)
  masks=state['masks'].squeeze(1).cpu().numpy();scores=state['scores'].float().cpu().numpy();boxes=state['boxes'].float().cpu().numpy()
 if not len(masks) or not np.isfinite(scores).all():raise RuntimeError('No finite vehicle predictions')
 np.savez_compressed(out/'SAM3_vehicle_predictions.npz',masks=masks,scores=scores,boxes_xyxy=boxes)
 canvas=image.copy();draw=ImageDraw.Draw(canvas)
 for j,(mask,score,box) in enumerate(zip(masks,scores,boxes)):
  edge=mask.copy();edge[1:]=mask[1:]&~mask[:-1];yy,xx=np.where(edge)
  for x,y in zip(xx,yy):draw.point((int(x),int(y)),fill=(255,80,35))
  draw.rectangle(box.tolist(),outline=(255,220,40),width=1);draw.text((float(box[0]),float(box[1])),f'{j}:{score:.2f}',fill=(255,255,0))
 canvas.save(out/'SAM3_vehicle_overlay.png');report['SAM3']={'actual_predictions':len(masks),'seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'native_text_prompt':'vehicle','checkpoint':'/root/autodl-tmp/models/idsplat/sam3/sam3.pt'};print(json.dumps(report['SAM3']),flush=True)
 del model,processor,state;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
 from transformers import AutoImageProcessor,AutoModel
 start=time.monotonic();model_path='/root/autodl-tmp/models/idsplat/dinov3-vitb16'
 processor=AutoImageProcessor.from_pretrained(model_path,local_files_only=True,do_resize=False,do_rescale=True)
 model=AutoModel.from_pretrained(model_path,local_files_only=True).eval().cuda();inputs=processor(images=image,return_tensors='pt');inputs={k:v.cuda() for k,v in inputs.items()}
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):features=model(**inputs,output_hidden_states=True)
 selected=features.hidden_states[7][:,5:,:].float();h,w=inputs['pixel_values'].shape[-2:];assert selected.shape[1]==(h//16)*(w//16)
 assert torch.isfinite(selected).all() and float(selected.std())>0
 np.save(out/'DINOv3_layer7_patch_features.npy',selected.squeeze(0).cpu().numpy())
 report['DINOv3']={'model':'ViT-B/16 LVD1689M','actual_shape':list(selected.shape),'RGB_tensor_shape':list(inputs['pixel_values'].shape),'selected_layer':7,'prefix_tokens_excluded':5,'all_values_finite':True,'standard_deviation':float(selected.std()),'seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'model_directory':model_path,'correspondence_accuracy_not_yet_validated':True}
 report['status']='SAM3_AND_DINOV3_REAL_GPU_INFERENCE_PASSED';(out/'status.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
except BaseException as exc:
 report['status']='FAILED_ENGINEERING';report['error']=repr(exc);(out/'status.json').write_text(json.dumps(report,indent=2));raise
