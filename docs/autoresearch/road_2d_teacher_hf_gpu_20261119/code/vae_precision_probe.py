"""冻结VAE的运算精度诊断；不运行UNet、不读取隐藏真值、不更改主48对照。"""
from pathlib import Path
import contextlib
import json
import os
import time

for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from teacher_pilot import check_pin, inverse_view, known_metrics, pin, rgb, binary, read, write

C = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_2d_teacher_hf_cpu_20261119_r1')
OUT = C / 'VAE_precision_probe_r1'
OUT.mkdir(exist_ok=False)
m = read(C / 'prepared_r2/pilot_manifest.json')
parent = read(C / 'GPU_pilot_r1/predictions.json')
assert parent['complete'] and parent['completed_diffusion_calls'] == 48
assert len(m['vae_controls']) == 4
controls = m['vae_controls']
cases = {c['id']: c for c in m['cases']}
old = {r['id']: r for r in parent['VAE_controls']}
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
assert torch.cuda.is_available()
processor = VaeImageProcessor(vae_scale_factor=8)
receipt = dict(schema='road_2d_VAE_precision_diagnostic.v1', complete=False,
    scientific_role='post-pilot matched numerical precision diagnostic, not a new teacher arm',
    data_role='only legal original observation and visible road; no hidden GT',
    parent=pin(C / 'GPU_pilot_r1/predictions.json'),
    fixed_pretrained_weight_source='same fp16 variant promoted to the requested compute dtype; not full-precision original weights',
    UNet_calls=0, diffusion_calls=0, training_updates=0, attempted_VAE_roundtrips=0,
    completed_VAE_roundtrips=0, baseline_replay=None, rows=[])
write(OUT / 'receipt.json', receipt)
start = time.perf_counter()
try:
    for dtype_name, dtype, selected in [('bfloat16', torch.bfloat16, controls[:1]),
                                      ('float32', torch.float32, controls),
                                      ('float16', torch.float16, controls)]:
        vae = AutoencoderKL.from_pretrained(m['model'], subfolder='vae', variant='fp16',
            torch_dtype=dtype, local_files_only=True).to('cuda').requires_grad_(False).eval()
        assert vae.dtype == dtype
        for control in selected:
            case = cases[control['case_id']]
            view = case['views'][control['view']]
            for record in [view['VAE_reference_model'], control['visible_native'], case['observation']]:
                check_pin(record)
            image = Image.open(view['VAE_reference_model']['path']).convert('RGB')
            value = processor.preprocess(image).to('cuda', dtype)
            receipt['attempted_VAE_roundtrips'] += 1
            write(OUT / 'receipt.json', receipt)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            cast = contextlib.nullcontext() if dtype == torch.float32 else torch.autocast('cuda', dtype=dtype)
            with torch.inference_mode(), cast:
                latent = vae.encode(value).latent_dist.mode() * vae.config.scaling_factor
                decoded = vae.decode(latent / vae.config.scaling_factor, return_dict=False)[0]
            reconstructed = processor.postprocess(decoded, output_type='pil')[0]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            folder = OUT / dtype_name / control['id']
            folder.mkdir(parents=True)
            reconstructed.save(folder / 'roundtrip_model.png')
            model_rgb = np.asarray(reconstructed).copy()
            native, _ = inverse_view(model_rgb, case['native_hw'], view['plan']['window_xyxy'])
            Image.fromarray(native).save(folder / 'roundtrip_native.png')
            reference, _ = inverse_view(np.asarray(image), case['native_hw'], view['plan']['window_xyxy'])
            visible = binary(control['visible_native']['path'])
            if dtype_name == 'bfloat16':
                check_pin(old[control['id']]['roundtrip_model'])
                exact = bool(np.array_equal(model_rgb, rgb(old[control['id']]['roundtrip_model']['path'])))
                receipt['baseline_replay'] = dict(id=control['id'], exact=exact)
                if not exact:
                    raise RuntimeError('isolated BF16 VAE does not reproduce parent; stop precision attribution')
            receipt['rows'].append(dict(id=control['id'], dtype=dtype_name, seconds=elapsed,
                known_road_vs_native_observation=known_metrics(native, rgb(case['observation']['path']), visible),
                increment_over_resampling=known_metrics(native, reference, visible),
                vs_parent_BF16=known_metrics(native, rgb(old[control['id']]['roundtrip_native']['path']), visible),
                roundtrip_model=pin(folder / 'roundtrip_model.png'), roundtrip_native=pin(folder / 'roundtrip_native.png')))
            receipt['completed_VAE_roundtrips'] += 1
            write(OUT / 'receipt.json', receipt)
            print(json.dumps(dict(id=control['id'], dtype=dtype_name, seconds=elapsed)), flush=True)
        del vae
        torch.cuda.empty_cache()
    receipt.update(complete=True, wall_seconds=time.perf_counter()-start,
                   peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated())
    write(OUT / 'receipt.json', receipt)
except BaseException as error:
    receipt.update(complete=False, failed=True, error=str(error), wall_seconds=time.perf_counter()-start)
    write(OUT / 'receipt.json', receipt)
    raise
