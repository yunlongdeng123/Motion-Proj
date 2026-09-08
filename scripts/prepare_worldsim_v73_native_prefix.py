"""Prepare immutable VGGT prefixes for exported observations without shared fitting.

Per-image patch encoding may be chunked; all views still jointly enter every
alternating-attention block. Only the four official DPT input layers are cached.
Metric alignment is estimated from build measurements using the original head;
the chosen adapted head is installed afterwards, as in the main V7.3 protocol.
"""
import argparse
import gc
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v73.native_data import sample_depth


class ChunkedImageEncoder(torch.nn.Module):
    """Independent images only; never split the cross-view aggregator context."""
    def __init__(self,encoder,chunk):
        super().__init__(); self.encoder=encoder; self.chunk=chunk

    def forward(self,images):
        outputs=[]
        for start in range(0,len(images),self.chunk):
            result=self.encoder(images[start:start+self.chunk])
            outputs.append(result['x_norm_patchtokens'] if isinstance(result,dict) else result)
        return torch.cat(outputs)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--actor-data',type=Path,required=True)
    parser.add_argument('--head-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--patch-view-chunk',type=int,default=7)
    args=parser.parse_args(); torch.set_num_threads(4); started=time.monotonic()
    args.output.mkdir(parents=True,exist_ok=False)
    def save(name,data): (args.output/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    config=json.loads((args.head_source/'config.json').read_text())
    manifest={'status':'running','phase':'load','actor_data':str(args.actor_data),'head_source':str(args.head_source),
              'patch_view_chunk':args.patch_view_chunk,'optimizer_updates':0,
              'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'cache_boundary':'fully frozen aggregator only; adapted native head is not cached',
              'alignment':'fixed-window IRLS log depth scale from original pretrained head and build sensor measurements',
              'query_selection':'none; no heldout surface or model quality selection'}
    save('manifest.json',manifest); save('status.json',{'status':'running','phase':'load'})
    try:
        scenes=torch.load(args.actor_data/'build_observations.pt',map_location='cpu',weights_only=False)
        # Preserve original observation files without allocating duplicate raw image storage.
        (args.output/'build_observations.pt').symlink_to((args.actor_data/'build_observations.pt').resolve())
        save('cohort.json',[{k:v for k,v in scene.items() if k!='views'} for scene in scenes])
        config['data']['dataset_root']=str(args.actor_data)
        config['data']['camera_channels']=list(dict.fromkeys(v['camera_id'] for s in scenes for v in s['views']))
        config['data']['image_hw']=list(scenes[0]['views'][0]['image'].shape[-2:])
        config['data']['input_kind']='exported_metric_observations'
        save('config.json',config)
        sys.path.insert(0,config['backbone']['repository'])
        from vggt.models.vggt import VGGT
        from safetensors import safe_open
        model=VGGT(enable_camera=False,enable_point=False,enable_track=False)
        with safe_open(config['backbone']['checkpoint'],framework='pt',device='cpu') as checkpoint:
            state={key:checkpoint.get_tensor(key) for key in checkpoint.keys() if key.startswith(('aggregator.','depth_head.'))}
        model.load_state_dict(state,strict=True); del state
        model.requires_grad_(False).eval().cuda()
        layer_ids=model.depth_head.intermediate_layer_idx
        # The vendored official version supports selective output retention.
        model.aggregator.cached_layer_indices=set(layer_ids)
        if args.patch_view_chunk>0:
            model.aggregator.patch_embed=ChunkedImageEncoder(model.aggregator.patch_embed,args.patch_view_chunk).eval()
        cache=args.output/'frozen_prefix'; cache.mkdir(); scales={}; records=[]
        for scene in scenes:
            images=torch.stack([view['image'] for view in scene['views']])[None].cuda()
            save('status.json',{'status':'running','phase':'joint_frozen_aggregator','scene':scene['scene_id'],'views':len(scene['views']),
                                'image_hw':list(images.shape[-2:]),'patch_view_chunk':args.patch_view_chunk})
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                tokens,patch_start=model.aggregator(images)
            ratios=[]; observations=0
            for i,view in enumerate(scene['views']):
                selected=[tokens[j][:,i:i+1].detach() if j in layer_ids else None for j in range(len(tokens))]
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    depth,_=model.depth_head(selected,images[:,i:i+1],patch_start)
                    predicted=sample_depth(depth[0,0,:,:,0],view['uv'].cuda()).float().cpu()
                build=~view['diagnostic_mask']
                ratios.append((view['z_m'][build]/predicted[build].clamp_min(1e-5)).log())
                observations+=int(build.sum())
                torch.save({'tokens':{j:tokens[j][:,i:i+1].detach().cpu().clone() for j in layer_ids},
                            'patch_start':patch_start},cache/f'{scene["scene_id"]}_{i:02}.pt')
                # clone prevents torch serialization of the full 28-view backing storage.
            values=torch.cat(ratios); center=values.median()
            for _ in range(5):
                weights=.25/(values-center).abs().clamp_min(.25)
                center=(weights*values).sum()/weights.sum()
            scales[scene['scene_id']]=float(center.exp())
            row={'scene':scene['scene_id'],'views':len(scene['views']),'image_hw':list(images.shape[-2:]),
                 'scale':scales[scene['scene_id']],'build_scale_measurements':observations,
                 'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30}
            records.append(row); save('status.json',{'status':'running','phase':'prefix_saved',**row})
            print(json.dumps(row),flush=True)
            del images,tokens,selected,depth,predicted; gc.collect(); torch.cuda.empty_cache()
        save('metric_scales.json',scales)
        head=torch.load(args.head_source/'latest.pt',map_location='cpu',weights_only=True)['depth_head']
        torch.save({'depth_head':head,'scales':scales,'config':config,'head_source':str(args.head_source)},args.output/'latest.pt')
        result={'status':'done','scenes':records,'optimizer_updates':0,'wall_s':time.monotonic()-started,
                'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                'cache_bytes':sum(p.stat().st_size for p in cache.glob('*.pt')),
                'method_boundary':'frozen-prefix preparation, not new-source shared fine-tuning or geometry evaluation'}
        save('summary.json',result); save('status.json',{'status':'done'})
        print(json.dumps(result),flush=True)
    except Exception as exc:
        save('status.json',{'status':'failed','exception':type(exc).__name__,'message':str(exc)})
        (args.output/'traceback.txt').write_text(traceback.format_exc())
        raise


if __name__=='__main__': main()
