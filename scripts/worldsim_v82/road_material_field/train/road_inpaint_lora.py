"""道路世界纹理小样本后训练；原生/LoRA 使用相同的参考拼接输入。

仅拟合曝光条件下的 sRGB 表观纹理，不声称恢复物理 PBR 材质。
训练真值只用于扩散目标及评价，不进入 masked-image 条件。
"""
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import argparse
import copy
import importlib.metadata
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler, DDPMScheduler
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

MODEL = os.environ.get('ROAD_INPAINT_MODEL')
PROMPT = 'a close-up top-down photograph of rough gray asphalt road surface, continuous texture, natural lighting'
NEGATIVE = ''


def emit(event, **kwargs):
    print(json.dumps(dict(event=event, **kwargs), ensure_ascii=False), flush=True)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def configure(seed):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def set_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


class Sampler:
    """显式排列、游标和独立随机数，保存后下一个样本可以逐值重现。"""
    def __init__(self, n, seed):
        self.n, self.epoch, self.pos = n, 0, 0
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(n, generator=self.generator).tolist()

    def next(self):
        if self.pos == self.n:
            self.epoch += 1
            self.pos = 0
            self.order = torch.randperm(self.n, generator=self.generator).tolist()
        value = self.order[self.pos]
        self.pos += 1
        return value

    def state_dict(self):
        return dict(n=self.n, epoch=self.epoch, pos=self.pos, order=self.order.copy(), generator=self.generator.get_state())

    def load_state_dict(self, state):
        assert self.n == state['n']
        self.epoch, self.pos, self.order = state['epoch'], state['pos'], state['order'].copy()
        self.generator.set_state(state['generator'])


def rgb_tensor(image):
    return torch.from_numpy(np.array(image, dtype=np.float32).copy()).permute(2, 0, 1)[None] / 127.5 - 1


def load_dataset(path, tile_size):
    path = Path(path)
    spec = json.loads(path.read_text())
    assert spec['schema'] == 'road_material_dataset.v1'
    samples = []
    for row in spec['samples']:
        sample = dict(row)
        for key in ['input', 'reference']:
            sample[key + '_image'] = Image.open(path.parent / row[key]).convert('RGB')
        sample['target_image'] = Image.open(path.parent / row['target']).convert('RGB') if row.get('target') else sample['input_image'].copy()
        sample['has_target'] = bool(row.get('target')) and row['split'] != 'query'
        for key in ['hole_mask', 'valid_mask']:
            sample[key + '_image'] = Image.open(path.parent / row[key]).convert('L')
        sizes = {sample[key + '_image'].size for key in ['target', 'input', 'reference', 'hole_mask', 'valid_mask']}
        assert len(sizes) == 1, (row['id'], sizes)
        w, h = sample['input_image'].size
        assert w == h, '方块统一等比放大，不扭曲 world_from_uv'
        hole = np.array(sample['hole_mask_image']) > 127
        valid = np.array(sample['valid_mask_image']) > 127
        if row['split'] != 'query':
            assert (hole & valid).any(), (row['id'], 'empty scoring domain')
        context = np.array(sample['input_image'])
        target = np.array(sample['target_image'])
        assert np.array_equal(context[~hole], target[~hole]), (row['id'], 'context changed outside hole')
        ref = sample['reference_image'].resize((tile_size, tile_size), Image.Resampling.BICUBIC)
        inp = sample['input_image'].resize((tile_size, tile_size), Image.Resampling.BICUBIC)
        tgt = sample['target_image'].resize((tile_size, tile_size), Image.Resampling.BICUBIC)
        canvas = Image.new('RGB', (2 * tile_size, tile_size))
        canvas.paste(ref, (0, 0))
        canvas.paste(inp, (tile_size, 0))
        target_canvas = canvas.copy()
        target_canvas.paste(tgt, (tile_size, 0))
        canvas_mask = Image.new('L', canvas.size, 0)
        canvas_mask.paste(sample['hole_mask_image'].resize((tile_size, tile_size), Image.Resampling.NEAREST), (tile_size, 0))
        scoring = Image.fromarray((hole & valid).astype(np.uint8) * 255).resize((tile_size, tile_size), Image.Resampling.NEAREST)
        loss_canvas = Image.new('L', canvas.size, 0)
        loss_canvas.paste(scoring, (tile_size, 0))
        sample.update(canvas=canvas, target_canvas=target_canvas, canvas_mask=canvas_mask, loss_canvas=loss_canvas, original_size=w)
        samples.append(sample)
    assert len({x['id'] for x in samples}) == len(samples)
    assert samples
    return spec, samples


def make_synthetic(root):
    """工程 smoke 数据；与实拍任务训练账严格分开。"""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    gen = np.random.default_rng(23)
    samples = []
    for i in range(8):
        yy, xx = np.mgrid[:128, :128]
        gray = .30 + .02 * np.sin(xx / 11 + i) + .015 * np.sin(yy / 17) + gen.normal(0, .025, (128, 128))
        rgb = np.clip(np.stack([gray * 1.02, gray, gray * .98], -1) * 255, 0, 255).astype(np.uint8)
        ref = np.roll(rgb, 27, axis=0)
        hole = np.zeros((128, 128), np.uint8)
        hole[28:96, 30:100] = 255
        inp = rgb.copy()
        inp[hole > 0] = 127
        names = {}
        for key, value in dict(target=rgb, input=inp, reference=ref, hole_mask=hole, valid_mask=np.full((128, 128), 255, np.uint8)).items():
            names[key] = f'{i:02d}_{key}.png'
            Image.fromarray(value).save(root / names[key])
        samples.append(dict(id=f'synthetic_{i:02d}', split='train' if i < 6 else 'eval', **names, world_from_uv=np.eye(4).tolist(), provenance=dict(source='synthetic engineering smoke only')))
    spec = dict(schema='road_material_dataset.v1', scene_id='synthetic_wiring_only', seed=23, samples=samples, coordinate=dict(units='m', pixel_centers='world_from_uv @ [u+0.5,v+0.5,0,1]'))
    write_json(root / 'dataset.json', spec)
    return root / 'dataset.json'


class Runner:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.spec, self.samples = load_dataset(args.dataset, args.tile_size)
        self.train_samples = [s for s in self.samples if s['split'] == 'train']
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(args.model, variant='fp16', torch_dtype=torch.bfloat16, safety_checker=None, requires_safety_checker=False, local_files_only=True).to('cuda')
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.model, subfolder='scheduler', local_files_only=True)
        assert self.pipe.unet.config.in_channels == 9
        self.pipe.vae.requires_grad_(False).eval()
        self.pipe.text_encoder.requires_grad_(False).eval()
        self.pipe.unet.requires_grad_(False)
        config = LoraConfig(r=args.rank, lora_alpha=args.rank, target_modules=['to_q', 'to_k', 'to_v', 'to_out.0'], init_lora_weights=True, lora_dropout=0.0)
        self.pipe.unet.add_adapter(config)
        assert all('lora_' in name for name,p in self.pipe.unet.named_parameters() if p.requires_grad)
        self.params = [p for p in self.pipe.unet.parameters() if p.requires_grad]
        for p in self.params:
            p.data = p.data.float()
        self.optimizer = torch.optim.AdamW(self.params, lr=args.lr, betas=(.9, .999), eps=1e-8, weight_decay=.01, foreach=False)
        self.scaler = torch.amp.GradScaler('cuda', enabled=False)
        self.sampler = Sampler(len(self.train_samples), args.seed + 1)
        self.train_generator = torch.Generator('cuda').manual_seed(args.seed + 2)
        self.step = 0
        self.update_seconds = 0.
        self.encode_cache()
        self.pipe.unet.train()
        self.contract = dict(schema='road_inpaint_contract.v1', dataset=str(Path(args.dataset).resolve()), scene_id=self.spec.get('scene_id'), model=args.model, native_unet_channels=9, model_source='inherited SD1.5 inpainting community mirror of deprecated Runway weights', source_url='https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting', lora_rank=args.rank, lora_alpha=args.rank, lora_targets=['to_q', 'to_k', 'to_v', 'to_out.0'], trainable_parameters=sum(p.numel() for p in self.params), base_trainable_parameters=0, vae_frozen=True, text_frozen=True, dtype='bfloat16 base + float32 LoRA/Adam state', prompt=PROMPT, negative_prompt=NEGATIVE, training_seed=args.seed, inference_seeds=args.eval_seeds, inference_steps=args.inference_steps, guidance=args.guidance, scheduler='DDIM eta=0', target_tile_size=self.samples[0]['original_size'], model_tile_size=args.tile_size, canvas=dict(width=2*args.tile_size, height=args.tile_size, left='reference', right='masked input/context', resampling='uniform bicubic RGB, nearest masks; no aspect distortion'), training_objective='epsilon MSE; pixel hole*valid downsampled by area to latent; sum/(4*valid_hole_area); reference/context loss weight=0', learning_rate=args.lr, optimizer='AdamW (.9,.999), eps1e-8, weight_decay=.01; constant LR', train_ids=[s['id'] for s in self.train_samples], eval_ids=[s['id'] for s in self.samples if s['split'] == 'eval'], train_eval_note='fixed spatial split within one scene; no scene generalization claim', query_inputs=['input','reference','hole_mask','valid_mask','world_from_uv'], target_usage='loss and evaluation only; excluded from inference condition', representation='exposure-conditioned sRGB world appearance texture; not physical PBR', torch=torch.__version__, cuda=torch.version.cuda, cuda_device=torch.cuda.get_device_name(0), deterministic_algorithms=True, failure_ledger_refs=['V82-F06'], failure_ledger_delta='none', human_verdict=None)
        self.contract['packages'] = {name: importlib.metadata.version(name) for name in ['torch','diffusers','transformers','peft','safetensors','numpy','Pillow']}
        write_json(self.out / 'contract.json', self.contract)
        emit('runner_ready', trainable=self.contract['trainable_parameters'], train_count=len(self.train_samples), eval_count=len(self.contract['eval_ids']), peak_GB=torch.cuda.max_memory_allocated()/1e9)

    @torch.no_grad()
    def encode_cache(self):
        ids = self.pipe.tokenizer(PROMPT, padding='max_length', max_length=self.pipe.tokenizer.model_max_length, truncation=True, return_tensors='pt').input_ids.to('cuda')
        self.text = self.pipe.text_encoder(ids)[0].detach()
        self.cache = []
        for sample in self.train_samples:
            target = rgb_tensor(sample['target_canvas']).to('cuda', torch.bfloat16)
            inp = rgb_tensor(sample['canvas']).to('cuda', torch.bfloat16)
            mask = torch.from_numpy(np.asarray(sample['canvas_mask'], dtype=np.float32).copy())[None, None].to('cuda') / 255
            masked = inp * (mask < .5)
            z = self.pipe.vae.encode(target).latent_dist.mode() * self.pipe.vae.config.scaling_factor
            zcond = self.pipe.vae.encode(masked).latent_dist.mode() * self.pipe.vae.config.scaling_factor
            m = F.interpolate(mask, z.shape[-2:], mode='nearest')
            weight = torch.from_numpy(np.asarray(sample['loss_canvas'], dtype=np.float32).copy())[None, None].to('cuda') / 255
            weight = F.interpolate(weight, z.shape[-2:], mode='area')
            self.cache.append(dict(z=z.detach(), condition=zcond.detach(), mask=m.to(torch.bfloat16), weight=weight.detach(), id=sample['id']))

    def adapter_state(self):
        return cpu_copy(get_peft_model_state_dict(self.pipe.unet))

    def checkpoint(self, path):
        state = dict(schema='road_lora_full_state.v1', step=self.step, adapter=self.adapter_state(), optimizer=cpu_copy(self.optimizer.state_dict()), scaler=self.scaler.state_dict(), rng=rng_state(), sampler=self.sampler.state_dict(), train_generator=self.train_generator.get_state(), config=vars(self.args).copy(), update_seconds=self.update_seconds, contract=self.contract, gradients='saved only at optimizer step boundary after zero_grad', base_model_frozen=True)
        tmp = Path(str(path) + '.tmp')
        torch.save(state, tmp)
        tmp.replace(path)
        return state

    def restore(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert state['schema'] == 'road_lora_full_state.v1'
        for key in ['model','rank','lr','tile_size','seed','dataset','inference_steps','eval_seeds','guidance']:
            assert state['config'][key] == getattr(self.args, key), (key, state['config'][key], getattr(self.args, key))
        result = set_peft_model_state_dict(self.pipe.unet, state['adapter'])
        assert not result.unexpected_keys, result.unexpected_keys
        self.optimizer.load_state_dict(state['optimizer'])
        self.scaler.load_state_dict(state['scaler'])
        self.sampler.load_state_dict(state['sampler'])
        self.train_generator.set_state(state['train_generator'])
        self.step = state['step']
        self.update_seconds = state['update_seconds']
        set_rng(state['rng'])
        self.pipe.unet.train()
        self.optimizer.zero_grad(set_to_none=True)
        emit('restored', step=self.step, checkpoint=str(path))
        return state

    def update(self, log=True):
        torch.cuda.synchronize()
        start = time.time()
        idx = self.sampler.next()
        item = self.cache[idx]
        z = item['z']
        noise = torch.randn(z.shape, generator=self.train_generator, device='cuda', dtype=z.dtype)
        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (1,), generator=self.train_generator, device='cuda').long()
        noisy = self.noise_scheduler.add_noise(z, noise, t)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = self.pipe.unet(torch.cat([noisy, item['mask'], item['condition']], 1), t, encoder_hidden_states=self.text).sample
            err = (prediction.float() - noise.float()).square()
            loss = (err * item['weight']).sum() / (4 * item['weight'].sum().clamp_min(1e-8))
        assert torch.isfinite(loss), 'non-finite loss'
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(self.params, 1.0)
        assert torch.isfinite(norm), 'non-finite gradient'
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        torch.cuda.synchronize()
        seconds = time.time() - start
        self.update_seconds += seconds
        row = dict(step=self.step, sample_id=item['id'], sampler_epoch=self.sampler.epoch, timestep=int(t.item()), epsilon_mse=float(loss.item()), grad_norm=float(norm.item()), seconds=seconds)
        if log:
            with (self.out / 'training.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(row) + '\n')
            if self.step == 1 or self.step % 20 == 0:
                emit('train', **row)
        return row

    @torch.no_grad()
    def evaluate(self, label, native=False, limit=0):
        state = rng_state()
        was_training = self.pipe.unet.training
        self.pipe.unet.eval()
        if native:
            self.pipe.unet.disable_adapters()
        else:
            self.pipe.unet.enable_adapters()
        dest = self.out / 'eval' / label
        dest.mkdir(parents=True, exist_ok=True)
        rows = []
        selected = self.train_samples[:self.args.train_eval_count] + [s for s in self.samples if s['split'] in ['eval','query']]
        if limit:
            selected = [self.train_samples[0]] + [s for s in self.samples if s['split'] == 'eval'][:max(1, limit-1)]
        start = time.time()
        atlas = []
        for sample in selected:
            name = sample['id']
            sample['canvas'].save(dest / f'{name}_canvas_input.png')
            sample['canvas_mask'].save(dest / f'{name}_canvas_mask.png')
            for seed in self.args.eval_seeds:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    generated = self.pipe(prompt=PROMPT, negative_prompt=NEGATIVE, image=sample['canvas'], mask_image=sample['canvas_mask'], height=self.args.tile_size, width=2*self.args.tile_size, num_inference_steps=self.args.inference_steps, guidance_scale=self.args.guidance, eta=0., generator=torch.Generator('cuda').manual_seed(seed)).images[0]
                generated.save(dest / f'{name}_seed{seed}_raw_canvas.png')
                tile = generated.crop((self.args.tile_size, 0, 2*self.args.tile_size, self.args.tile_size)).resize((sample['original_size'], sample['original_size']), Image.Resampling.LANCZOS)
                raw = np.array(tile)
                inp = np.array(sample['input_image'])
                target = np.array(sample['target_image'])
                hole = np.array(sample['hole_mask_image']) > 127
                valid = np.array(sample['valid_mask_image']) > 127
                score = hole & valid if sample['has_target'] else np.zeros_like(hole)
                composite = inp.copy()
                composite[hole] = raw[hole]
                Image.fromarray(composite).save(dest / f'{name}_seed{seed}_output.png')
                err = np.abs(composite.astype(float) - target.astype(float)) / 255
                mse = ((composite.astype(float) - target.astype(float)) / 255) ** 2
                boundary = hole & ~np.asarray(Image.fromarray(hole.astype(np.uint8)*255).filter(ImageFilter.MinFilter(5))).astype(bool) & valid
                row = dict(id=name, split=sample['split'], seed=seed, native=native, trained_steps=0 if native else self.step, valid_hole_pixels=int(score.sum()), total_hole_pixels=int(hole.sum()), rejected_hole_pixels=int(hole.sum()-score.sum()), mae=float(err[score].mean()) if score.any() else None, rmse=float(np.sqrt(mse[score].mean())) if score.any() else None, psnr=float(-10*np.log10(max(mse[score].mean(), 1e-12))) if score.any() else None, boundary_mae=float(err[boundary].mean()) if boundary.any() and sample['has_target'] else None, outside_hole_max_change=float(np.abs(composite.astype(float)-inp.astype(float))[~hole].max()) if (~hole).any() else 0., output=f'{name}_seed{seed}_output.png')
                rows.append(row)
                if seed == self.args.eval_seeds[0]:
                    atlas.append(dict(id=name, split=sample['split'], image=str((dest / row['output']).resolve()), world_from_uv=sample['world_from_uv'], valid_mask=str((Path(self.args.dataset).parent / sample['valid_mask']).resolve()), hole_mask=str((Path(self.args.dataset).parent / sample['hole_mask']).resolve()), support='generated unknown road is an unverified prior' if sample['split']=='query' else 'real-supported pseudo-mask restoration only; other generated region is prior', source_domain='world appearance sRGB', update_step=0 if native else self.step))
        summary = {}
        for split in ['train', 'eval']:
            group = [r for r in rows if r['split'] == split]
            if group:
                denom = sum(r['valid_hole_pixels'] for r in group)
                summary[split] = dict(samples=len({r['id'] for r in group}), sample_seed_pairs=len(group), valid_hole_pixel_pairs=denom, mae=sum(r['mae']*r['valid_hole_pixels'] for r in group)/denom, rmse=math.sqrt(sum(r['rmse']**2*r['valid_hole_pixels'] for r in group)/denom), median_sample_mae=float(np.median([r['mae'] for r in group])))
        report = dict(label=label, native=native, trained_steps=0 if native else self.step, seconds=time.time()-start, rows=rows, summary=summary, contract=self.contract, domain='independent world tile pseudo-hole supported pixels; not unseen physical GT or scene generalization', human_verdict=None)
        write_json(dest / 'metrics.json', report)
        write_json(dest / 'world_atlas.json', dict(schema='road_material_world_atlas.v1', samples=atlas))
        self.contact_sheet(selected, dest)
        self.pipe.unet.enable_adapters()
        self.pipe.unet.train(was_training)
        set_rng(state)
        emit('eval', label=label, native=native, step=self.step, summary=summary, seconds=report['seconds'])
        return report

    def contact_sheet(self, samples, dest):
        size = 128
        chosen = samples[:4] + [s for s in samples if s['split'] in ['eval','query']][:8]
        chosen = list({s['id']:s for s in chosen}.values())
        sheet = Image.new('RGB', (size*4, (size+22)*len(chosen)+25), '#12161f')
        draw = ImageDraw.Draw(sheet)
        for j, label in enumerate(['reference', 'input/context', 'generated', 'target (eval only)']):
            draw.text((j*size+4, 6), label, fill='white')
        for i, sample in enumerate(chosen):
            y = 25+i*(size+22)
            output = Image.open(dest / f'{sample["id"]}_seed{self.args.eval_seeds[0]}_output.png')
            for j, image in enumerate([sample['reference_image'], sample['input_image'], output, sample['target_image']]):
                sheet.paste(image.resize((size,size)), (j*size, y))
            draw.text((3, y+size+2), f'{sample["split"]}: {sample["id"]}', fill='#b8c4d8')
        sheet.save(dest / 'contact_sheet.png')

    def restore_audit(self):
        """一次真实更新后从磁盘恢复并重放，检查权重、Adam和下个样本。"""
        checkpoint = self.out / f'audit_anchor_step{self.step:06d}.pt'
        self.checkpoint(checkpoint)
        first = self.update(log=False)
        expected_adapter = self.adapter_state()
        expected_optimizer = cpu_copy(self.optimizer.state_dict())
        expected_rng = self.train_generator.get_state().clone()
        expected_sampler = self.sampler.state_dict()
        expected_path = self.out / f'audit_expected_step{self.step:06d}.pt'
        expected_full_state = self.checkpoint(expected_path)
        expected_full_state['replay_row'] = first
        torch.save(expected_full_state, expected_path)
        self.restore(checkpoint)
        second = self.update(log=False)
        actual = self.adapter_state()
        adapter_exact = all(torch.equal(actual[k], value) for k, value in expected_adapter.items())
        opt_actual = cpu_copy(self.optimizer.state_dict())
        optimizer_exact = all(torch.equal(opt_actual['state'][k][name], value) if torch.is_tensor(value) else opt_actual['state'][k][name] == value for k, fields in expected_optimizer['state'].items() for name, value in fields.items())
        sampler_exact = self.sampler.epoch == expected_sampler['epoch'] and self.sampler.pos == expected_sampler['pos'] and self.sampler.order == expected_sampler['order'] and torch.equal(self.sampler.generator.get_state(), expected_sampler['generator'])
        receipt = dict(anchor_step=self.step-1, next_step=self.step, sample_exact=first['sample_id']==second['sample_id'], timestep_exact=first['timestep']==second['timestep'], loss_exact=first['epsilon_mse']==second['epsilon_mse'], adapter_exact=adapter_exact, optimizer_exact=optimizer_exact, sampler_exact=sampler_exact, train_generator_exact=torch.equal(self.train_generator.get_state(), expected_rng), checkpoint_fields=['LoRA','optimizer','scaler disabled bf16 state','Python RNG','NumPy RNG','Torch CPU RNG','all CUDA RNG','independent train CUDA generator','sampler order, position, epoch, generator','step','constant LR contract'], engineering_replayed_updates=1, next_stage_entry='train --resume <checkpoint> --steps <new-total>; eval --resume <checkpoint>')
        receipt['passed'] = all(receipt[k] for k in ['sample_exact','timestep_exact','loss_exact','adapter_exact','optimizer_exact','sampler_exact','train_generator_exact'])
        write_json(self.out / f'restore_audit_step{self.step:06d}.json', receipt)
        emit('restore_audit', **receipt)
        assert receipt['passed'], 'exact restore replay failed'
        return receipt

    def fresh_replay(self, expected_path):
        """由独立进程重载模型、编码缓存和状态后，再验证下一次训练更新。"""
        expected = torch.load(expected_path, map_location='cpu', weights_only=False)
        row = self.update(log=False)
        actual = self.adapter_state()
        opt = cpu_copy(self.optimizer.state_dict())
        adapter_exact = all(torch.equal(actual[k], v) for k, v in expected['adapter'].items())
        optimizer_exact = all(torch.equal(opt['state'][k][name], v) if torch.is_tensor(v) else opt['state'][k][name] == v for k, fields in expected['optimizer']['state'].items() for name, v in fields.items())
        current_rng = rng_state()
        global_rng_exact = torch.equal(current_rng['torch'], expected['rng']['torch']) and all(torch.equal(a,b) for a,b in zip(current_rng['cuda'], expected['rng']['cuda'])) and current_rng['python'] == expected['rng']['python'] and np.array_equal(current_rng['numpy'][1], expected['rng']['numpy'][1]) and current_rng['numpy'][2:] == expected['rng']['numpy'][2:]
        receipt = dict(fresh_process=True, step=self.step, expected=str(expected_path), adapter_exact=adapter_exact, optimizer_exact=optimizer_exact, sample_exact=row['sample_id']==expected['replay_row']['sample_id'], timestep_exact=row['timestep']==expected['replay_row']['timestep'], loss_exact=row['epsilon_mse']==expected['replay_row']['epsilon_mse'], sampler_exact=self.sampler.epoch==expected['sampler']['epoch'] and self.sampler.pos==expected['sampler']['pos'] and self.sampler.order==expected['sampler']['order'] and torch.equal(self.sampler.generator.get_state(),expected['sampler']['generator']), train_generator_exact=torch.equal(self.train_generator.get_state(),expected['train_generator']), global_rng_exact=global_rng_exact, engineering_replayed_updates=1)
        receipt['passed'] = all(receipt[k] for k in ['adapter_exact','optimizer_exact','sample_exact','timestep_exact','loss_exact','sampler_exact','train_generator_exact','global_rng_exact'])
        write_json(self.out / f'fresh_process_replay_step{self.step:06d}.json', receipt)
        emit('fresh_process_replay', **receipt)
        assert receipt['passed'], 'fresh-process exact recovery failed'
        return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['synthetic','train','eval','audit','replay','predict'])
    parser.add_argument('--dataset')
    parser.add_argument('--out', required=True)
    parser.add_argument('--model', default=MODEL, required=MODEL is None,
                        help='existing SD1.5 nine-channel inpainting weights; or set ROAD_INPAINT_MODEL')
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=20261112)
    parser.add_argument('--tile-size', type=int, default=256)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--seconds', type=float, default=0.)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--eval-every', type=int, default=200)
    parser.add_argument('--inference-steps', type=int, default=20)
    parser.add_argument('--eval-seeds', nargs='+', type=int, default=[101,202])
    parser.add_argument('--guidance', type=float, default=3.0)
    parser.add_argument('--resume')
    parser.add_argument('--eval-limit', type=int, default=0)
    parser.add_argument('--train-eval-count', type=int, default=4)
    parser.add_argument('--skip-baseline', action='store_true')
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--label', default='evaluation')
    parser.add_argument('--expected')
    parser.add_argument('--adapter-from')
    parser.add_argument('--native-label',default='native_query_step000000')
    parser.add_argument('--trained-label')
    predict_group=parser.add_mutually_exclusive_group()
    predict_group.add_argument('--native-only',action='store_true')
    predict_group.add_argument('--trained-only',action='store_true')
    args = parser.parse_args()
    if args.mode == 'synthetic':
        args.dataset = str(make_synthetic(Path(args.out) / 'synthetic_data'))
    assert args.dataset
    configure(args.seed)
    start = time.time()
    runner = Runner(args)
    if args.resume:
        runner.restore(args.resume)
    if args.mode == 'predict':
        if args.adapter_from:
            state = torch.load(args.adapter_from, map_location='cpu', weights_only=False)
            assert state['config']['model'] == args.model and state['config']['rank'] == args.rank
            for key in ['inference_steps','eval_seeds','guidance','tile_size']:
                assert state['config'][key] == getattr(args,key), (key, 'query must preserve the paired inference protocol')
            result = set_peft_model_state_dict(runner.pipe.unet, state['adapter'])
            assert not result.unexpected_keys
            runner.step = state['step']
            runner.contract['training_source_contract']=state['contract']
            runner.contract['weights_from_checkpoint']=args.adapter_from
        else:
            assert args.native_only,'trained prediction requires --adapter-from'
        runner.contract['inference_only']=True
        runner.contract['inference_dataset_role']=runner.spec.get('role','query')
        runner.contract['retrieval']=runner.spec.get('retrieval')
        write_json(runner.out/'contract.json',runner.contract)
        if not args.trained_only:
            runner.evaluate(args.native_label, native=True)
        if not args.native_only:
            runner.evaluate(args.trained_label or f'lora_query_step{runner.step:06d}', native=False)
        return
    if args.mode == 'eval':
        runner.evaluate(args.label, native=False, limit=args.eval_limit)
        return
    if args.mode == 'audit':
        runner.restore_audit()
        runner.checkpoint(runner.out / f'checkpoint_step{runner.step:06d}.pt')
        return
    if args.mode == 'replay':
        runner.fresh_replay(args.expected)
        return
    if not args.skip_baseline and not args.resume:
        baseline = runner.evaluate('native_step000000', native=True, limit=args.eval_limit)
        zero = runner.evaluate('lora_step000000', native=False, limit=args.eval_limit)
        exact = all(np.array_equal(np.array(Image.open(runner.out/'eval/native_step000000'/a['output'])), np.array(Image.open(runner.out/'eval/lora_step000000'/b['output']))) for a,b in zip(baseline['rows'], zero['rows']))
        write_json(runner.out / 'zero_step_control.json', dict(passed=exact, same_inputs=True, same_noise_seeds=True, same_steps=True, same_prompt=True, same_reference_collage=True, adapter_B_initially_zero=True))
        assert exact, 'zero-initialized LoRA changed native inference'
        runner.checkpoint(runner.out / 'checkpoint_step000000.pt')
        emit('zero_step_control', exact=exact)
    train_start = time.time()
    last_eval_report = None
    if args.audit:
        runner.restore_audit()
    while runner.step < args.steps or (args.seconds and time.time()-train_start < args.seconds):
        runner.update()
        if runner.step % args.save_every == 0:
            runner.checkpoint(runner.out / f'checkpoint_step{runner.step:06d}.pt')
        if runner.step % args.eval_every == 0:
            last_eval_report = runner.evaluate(f'lora_step{runner.step:06d}', limit=args.eval_limit)
        if args.seconds and time.time()-train_start >= args.seconds:
            break
    final = runner.out / f'checkpoint_step{runner.step:06d}.pt'
    runner.checkpoint(final)
    report = last_eval_report if last_eval_report is not None and last_eval_report['trained_steps'] == runner.step else runner.evaluate(f'lora_step{runner.step:06d}', limit=args.eval_limit)
    write_json(runner.out / 'run_receipt.json', dict(completed=True, total_seconds=time.time()-start, update_seconds=runner.update_seconds, training_updates=runner.step, engineering_replayed_updates=1 if args.audit else 0, checkpoint=str(final), next_stage_command=f'{os.sys.executable} {Path(__file__).resolve()} eval --dataset {args.dataset} --out {args.out} --resume {final} --label restored_final', cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(), cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(), final_summary=report['summary'], formal_method_unchanged=True, purpose='synthetic engineering smoke only' if args.mode=='synthetic' else 'real supported road pseudo-hole native vs LoRA pilot', human_verdict=None))
    emit('completed', steps=runner.step, seconds=time.time()-start, checkpoint=str(final))


if __name__ == '__main__':
    main()
