"""r24：真实DriveEditor的零初始化等价和反向入口，优化0步。"""
from pathlib import Path
import sys, os, json, time, traceback, argparse
os.environ['DRIVEEDITOR_SEQUENTIAL_CFG'] = '1'
os.environ['OMP_NUM_THREADS'] = '4'
S = Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected')
sys.path[:0] = [str(S), str(S/'iteration7'), str(S/'iteration11'), str(S/'iteration12')]
T = Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929'); O = T/'r24'
import torch
import numpy as np
from PIL import Image
from state_adapter import StateResidualAdapter, pack_geometry, contract_checks


def main(resampling_fixed=False):
    global O
    if resampling_fixed: O=O/'resampling_fixed'
    O.mkdir(exist_ok=True)
    if (O/'result.json').exists(): raise RuntimeError('已保存结果，不覆盖探针')
    plan = {'task_id': 'WS-V77-TARGET-PROTECTED-20260929', 'run_id': 'r24', 'phase': 'adapter_engineering_only',
        'case': 'existing_DEV_L009', 'condition_source': str(T/'r22/state/L009'),
        'geometry_only': ['D', 'V', 'O', 'N', 'U', 'Q'], 'RGB_identity_condition': False,
        'backbone_input_channels': 9, 'extra_input_channels': 0, 'architecture_extension': True,
        'adapter_hooks': [2, 5, 8, 11], 'width': 32, 'seed': 6201, 'frames': 30,
        'resolution': [320, 576], 'optimizer_steps': 0, 'failure_ledger_refs': ['V77-F02'],
        'quality_claim': False, 'human_verdict': None,
        'subtrial': 'resampling_fixed' if resampling_fixed else 'initial_probe',
        'condition_resize': 'area_fraction_and_known_normalized_depth_confidence'}
    (O/'run.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n')
    start = time.time(); result = dict(plan, stage='loading', pid=os.getpid())
    def save(name='state.json'): (O/name).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    save(); torch.set_num_threads(4)
    try:
        import train_control as train
        from temporal_scope import selected
        train.selected = selected; train.MODULES = 'temporal_self'
        train.ENCODER = T/'r7/encoder_recovery/official_svd_encoder.safetensors'
        folder = T/'r16/synthetic/L009'
        def load(role, mode): return np.stack([np.asarray(Image.open(p).convert(mode)) for p in sorted((folder/role).glob('*.png'))])
        prepared = train.resized(load('Y','RGB'), load('X','RGB'), load('model_hole','L') > 0, (320,576))
        states = [dict(np.load(T/'r22/state/L009'/f'{i:05}.npz')) for i in range(30)]
        packed = pack_geometry({k: torch.from_numpy(np.stack([s[k] for s in states])) for k in ['D','V','O','N','U','Q']})
        h = torch.from_numpy(load('model_hole','L') > 0)[:, None]
        result['CPU_contracts'] = contract_checks()
        model, params, names = train.model_init(); unet = model.model.diffusion_model
        captured = []
        def capture(module, args, output): captured.append(output[0].detach().float().cpu())
        handle = unet.register_forward_hook(capture)
        torch.cuda.reset_peak_memory_stats(); result['stage']='baseline_forward';save()
        with torch.no_grad(): base_loss = train.loss(model, prepared, 6201)
        base = captured.pop()
        adapter = StateResidualAdapter().cuda(); unet.projected_state_adapter = adapter
        adapter.attach(unet); adapter.set_condition(packed.cuda(), h.cuda())
        result.update(stage='zero_adapter_forward', adapter_parameters=sum(p.numel() for p in adapter.parameters())); save()
        with torch.no_grad(): zero_loss = train.loss(model, prepared, 6201)
        zero = captured.pop(); assert torch.equal(base, zero), '零初始化改变了原网络输出'
        result.update(zero_init_network_exact=True, zero_init_loss_exact=bool(torch.equal(base_loss,zero_loss)), max_output_delta=float((base-zero).abs().max()))
        result['stage']='backward';save(); handle.remove()
        value = train.loss(model, prepared, 6201); value.backward()
        heads = {n: bool(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0)
                 for n,p in adapter.named_parameters() if n.startswith('zero_heads.')}
        assert heads and all(heads.values()), heads
        result.update(stage='passed', zero_head_gradients=heads,
                      initial_feature_branch_zero_grad_expected=True,
                      peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                      peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
                      seconds=time.time()-start, failure_ledger_delta='none; engineering interface passed, no quality inference')
        adapter.detach_hooks(); save('result.json'); save(); print(json.dumps(result, ensure_ascii=False),flush=True)
    except Exception as e:
        result.update(stage='engineering_error',error=repr(e),traceback=traceback.format_exc(),seconds=time.time()-start)
        save('result.json');save();raise


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--resampling-fixed',action='store_true');a=p.parse_args()
    main(a.resampling_fixed)
