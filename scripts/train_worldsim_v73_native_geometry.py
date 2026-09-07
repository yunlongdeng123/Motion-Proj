"""训练预训练原生DPT；冻结多层前缀缓存与逐视图反向降低显存，不缩减窗口。"""
import argparse
import gc
import json
from pathlib import Path
import random
import resource
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from motion_proj.worldsim_v73.native_data import load_scene_inputs, sample_depth


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def summarize(errors):
    if not errors:
        return {'count':0, 'mae_m':None, 'hit_02':None}
    x = torch.cat(errors)
    return {'count':len(x), 'mae_m':x.abs().mean().item(), 'median_ae_m':x.abs().median().item(),
            'early_02':(x<-.2).float().mean().item(), 'hit_02':(x.abs()<=.2).float().mean().item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    out = Path(config['runs_root'])/'worldsim_v73'/config['task_id']/args.run_id
    out.mkdir(parents=True, exist_ok=False)
    seed = config['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    started = time.monotonic()
    def progress(value):
        write_json(out/'status.json', {'status':'running', 'elapsed_s':time.monotonic()-started, **value})
        print(json.dumps(value), flush=True)
    write_json(out/'config.json', config)
    write_json(out/'manifest.json', {'task_id':config['task_id'], 'run_id':args.run_id, 'seed':seed,
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip(),
        'failure_ledger_refs':config['failure_ledger_refs'], 'source_test_read':False, 'external_test_read':False,
        'evaluation_boundary':'build-point interpolation diagnostics on existing fit/development logs; not final surface/scene confirmation',
        'native_trainable':'pretrained depth_head all layers', 'frozen':'aggregator',
        'per_point_lidar_time':'unavailable; scan timestamp used, residual motion uncertainty remains'})
    try:
        progress({'phase':'data_index'})
        scenes = load_scene_inputs(config['data'], progress)
        write_json(out/'cohort.json', [{k:v for k,v in s.items() if k!='views'} for s in scenes])
        torch.save(scenes, out/'build_observations.pt')
        sys.path.insert(0, config['backbone']['repository'])
        from vggt.models.vggt import VGGT
        from safetensors import safe_open
        model = VGGT(enable_camera=False, enable_point=False, enable_track=False)
        with safe_open(config['backbone']['checkpoint'], framework='pt', device='cpu') as checkpoint:
            state = {k:checkpoint.get_tensor(k) for k in checkpoint.keys() if k.startswith(('aggregator.','depth_head.'))}
        model.load_state_dict(state, strict=True)
        del state
        model.requires_grad_(False).eval().cuda()
        layer_ids = model.depth_head.intermediate_layer_idx
        cache = out/'frozen_prefix'
        cache.mkdir()
        scales = {}
        for scene in scenes:
            if not scene['views']:
                continue
            progress({'phase':'native_prefix', 'scene':scene['scene_id'], 'views':len(scene['views'])})
            images = torch.stack([v['image'] for v in scene['views']])[None].cuda()
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
                tokens, patch_start = model.aggregator(images)
            ratios = []
            for i, view in enumerate(scene['views']):
                selected = [t[:,i:i+1].detach() if t is not None and j in layer_ids else None for j,t in enumerate(tokens)]
                with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
                    depth,_ = model.depth_head(selected, images[:,i:i+1], patch_start)
                    pred = sample_depth(depth[0,0,:,:,0], view['uv'].cuda())
                build = ~view['diagnostic_mask']
                ratios.append((view['z_m'][build]/pred.cpu()[build].clamp_min(1e-5)).log())
                torch.save({'tokens':{j:t.cpu() for j,t in enumerate(selected) if t is not None},
                            'patch_start':patch_start}, cache/f'{scene["scene_id"]}_{i:02}.pt')
            # 一个场景/窗口共享一个固定米制尺度；开发尺度仅来自自身build输入。
            log_ratios = torch.cat(ratios)
            center = log_ratios.median()
            for _ in range(5):
                weight = (.25/(log_ratios-center).abs().clamp_min(.25))
                center = (weight*log_ratios).sum()/weight.sum()
            scales[scene['scene_id']] = center.exp().item()
            progress({'phase':'prefix_done', 'scene':scene['scene_id'], 'scale':scales[scene['scene_id']],
                      'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30})
            del tokens, selected, images, depth, pred
            gc.collect(); torch.cuda.empty_cache()
        write_json(out/'metric_scales.json',scales)
        head = model.depth_head
        model.depth_head = None
        del model
        gc.collect(); torch.cuda.empty_cache()
        head.requires_grad_(True).train()
        parameters = sum(p.numel() for p in head.parameters())
        optimizer = torch.optim.AdamW(head.parameters(), lr=config['training']['lr'], weight_decay=.01)
        fit = [s for s in scenes if s['role']=='fit' and s['views']]
        before_weight = head.projects[0].weight.detach().clone()

        def predict(scene, i, view):
            saved = torch.load(cache/f'{scene["scene_id"]}_{i:02}.pt', weights_only=True, map_location='cpu')
            tokens = [None]*24
            for j,t in saved['tokens'].items(): tokens[j] = t.cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                depth,_ = head(tokens,view['image'][None,None].cuda(),saved['patch_start'])
                pred = sample_depth(depth[0,0,:,:,0],view['uv'].cuda())*scales[scene['scene_id']]
            return pred

        def evaluate(tag):
            rows = []
            head.eval()
            with torch.no_grad():
                for scene in scenes:
                    errors, actor_errors = [], []
                    for i,view in enumerate(scene['views']):
                        pred = predict(scene,i,view).cpu()
                        select = view['diagnostic_mask']
                        residual = pred - view['z_m']
                        errors.append(residual[select])
                        actor_errors.append(residual[select & view['actor_mask']])
                    rows.append({'scene':scene['scene_id'],'log_id':scene['log_id'],'role':scene['role'],
                                 'all':summarize(errors),'actor':summarize([e for e in actor_errors if len(e)])})
            write_json(out/f'{tag}.json',rows)
            head.train()
            return rows

        progress({'phase':'baseline_evaluation','trainable_parameters':parameters})
        baseline = evaluate('baseline')
        first_grad = None
        for epoch in range(config['training']['epochs']):
            random.shuffle(fit)
            epoch_loss=[]
            for scene in fit:
                torch.cuda.reset_peak_memory_stats()
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.
                step_start = time.monotonic()
                for i,view in enumerate(scene['views']):
                    pred = predict(scene,i,view)
                    mask = ~view['diagnostic_mask'].cuda()
                    actor = view['actor_mask'].cuda() & mask
                    target = view['z_m'].cuda()
                    loss = F.smooth_l1_loss(pred[mask],target[mask],beta=.20)
                    if actor.any():
                        loss = loss + config['training']['actor_weight']*F.smooth_l1_loss(pred[actor],target[actor],beta=.20)
                    loss = loss / len(scene['views'])
                    loss.backward()
                    loss_sum += loss.detach().item()
                    del loss,pred,target
                grad = torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
                if not torch.isfinite(grad):
                    raise FloatingPointError('原生DPT出现非有限梯度')
                if first_grad is None: first_grad=float(grad)
                optimizer.step()
                row={'epoch':epoch+1,'scene':scene['scene_id'],'loss':loss_sum,'grad_norm':float(grad),
                     'step_s':time.monotonic()-step_start,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30}
                with (out/'train.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                epoch_loss.append(loss_sum)
                progress({'phase':'training',**row})
            torch.save({'depth_head':head.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,
                        'scales':scales,'config':config},out/'latest.pt')
        final = evaluate('final')
        summary={'status':'done','trainable_parameters':parameters,'first_gradient_norm':first_grad,
            'first_project_weight_max_change':(head.projects[0].weight.detach()-before_weight).abs().max().item(),
            'epochs':config['training']['epochs'],'scene_count':len(scenes),'baseline':baseline,'final':final,
            'wall_s':time.monotonic()-started,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            'failure_ledger_delta':'none; V73-F01:F04 remain active beyond this native-DPT diagnostic',
            'claim':'native geometry gradient and build-point interpolation only; canonical surface/coverage/hard first-return still pending'}
        write_json(out/'summary.json',summary)
        write_json(out/'status.json',{'status':'done','phase':'native_geometry_diagnostic_complete'})
        print(json.dumps(summary),flush=True)
    except Exception as exc:
        write_json(out/'status.json',{'status':'blocked','exception':type(exc).__name__,'message':str(exc),
                    'elapsed_s':time.monotonic()-started,'gpu_peak_gib':torch.cuda.max_memory_allocated()/2**30})
        (out/'traceback.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
