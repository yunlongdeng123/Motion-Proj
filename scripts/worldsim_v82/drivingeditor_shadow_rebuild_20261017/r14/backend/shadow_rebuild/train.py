"""三阶段训练入口。无 --execute 只读 CPU 预检；不连接/开机/提交远端任务。"""
import os
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from . import SCHEMA
from .config import Config
from .data import Dataset
from .model import SceneModel
from .losses import masked_mean, ssim_loss, geometry_and_reference_terms, instance_body_loss,pixel_valid


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def sample_pixels(h, w, config, generator, device):
    side = min(config.shading_patch, h, w)
    y = int(torch.randint(h-side+1, (), generator=generator))
    x = int(torch.randint(w-side+1, (), generator=generator))
    yy, xx = torch.meshgrid(torch.arange(y, y+side, device=device), torch.arange(x, x+side, device=device), indexing="ij")
    patch = (yy*w+xx).flatten()
    uniform = torch.randint(h*w, (config.uniform_rays,), generator=generator).to(device)
    return patch, uniform, side


def objective(model, frame, actor_count, backend, stage, generator, teacher=None,global_step=0):
    config = model.config
    geometry, package = model.buffers(frame, actor_count, backend)
    terms = geometry_and_reference_terms(model, package, frame, config, stage)
    from .competition import competition_terms
    terms.update(competition_terms(package,frame,config,global_step))
    # Stage1 不把车辆投到道路的暗化用于修改道路几何/本色；跨时刻干净观测监督道路。
    canonical_domain = (~frame.road) & (package["road"].detach() < .05)
    terms["reconstruction"] = masked_mean((package["canonical"]-frame.rgb).abs(), canonical_domain)
    body = (frame.masks & pixel_valid(frame)).any(-1)
    if stage == "geometry":
        terms["rgb"] = terms["reconstruction"]
        terms["ssim"] = ssim_loss(package["canonical"], frame.rgb, canonical_domain)
        terms["body_rgb"] = instance_body_loss((package['canonical']-frame.rgb).abs(),frame) if config.instance_balanced_supervision else masked_mean((package["canonical"]-frame.rgb).abs(), body)
        terms["body_teacher"] = terms["rgb"]*0
    else:
        h, w = frame.rgb.shape[:2]
        patch, uniform, side = sample_pixels(h, w, config, generator, frame.rgb.device)
        indices = torch.cat((patch, uniform))
        image, _ = model.shade_pixels(package, geometry, indices)
        target = frame.rgb.reshape(-1, 3)[indices]
        # uniform 部分对整幅图采样，patch 用于 SSIM；不靠 road ROI 掩掉未解决像素。
        terms["rgb"] = (image[len(patch):]-target[len(patch):]).abs().mean()
        terms["ssim"] = ssim_loss(image[:len(patch)].reshape(side, side, 3), target[:len(patch)].reshape(side, side, 3))
        # 小车体不会因全图像素占比低而失去监督；额外对所有可见车身像素分层采样。
        body_lengths=[]
        if config.instance_balanced_supervision:
            groups=[]
            for actor in range(actor_count):
                if not bool(frame.mask_known[actor]):continue
                ids=(frame.masks[...,actor]&pixel_valid(frame)[...,actor]).flatten().nonzero().flatten()
                if not len(ids):continue
                if len(ids)>128:ids=ids[torch.randperm(len(ids),generator=generator)[:128].to(ids.device)]
                groups.append(ids);body_lengths.append(len(ids))
            body_ids=torch.cat(groups) if groups else body.flatten().nonzero().flatten()
        else:
            body_ids = body.flatten().nonzero().flatten()
            if len(body_ids) > 512:
                choice = torch.randperm(len(body_ids), generator=generator)[:512].to(body_ids.device)
                body_ids = body_ids[choice]
        body_rgb, _ = model.shade_pixels(package, geometry, body_ids)
        body_error=(body_rgb-frame.rgb.reshape(-1,3)[body_ids]).abs()
        if config.instance_balanced_supervision and body_lengths:
            terms['body_rgb']=torch.stack([group.mean() for group in body_error.split(body_lengths)]).mean()
        else:terms["body_rgb"] = body_error.mean() if len(body_ids) else image.sum()*0
        if teacher is not None and config.use_body_teacher:
            with torch.no_grad():
                _, teacher_package = teacher.buffers(frame, actor_count, backend, isolated=False)
            difference=(package['canonical']-teacher_package['canonical']).abs()
            terms["body_teacher"] = instance_body_loss(difference,frame) if config.instance_balanced_supervision else masked_mean(difference, body)
        else:
            terms["body_teacher"] = image.sum()*0
    weights = asdict(config.weights)
    total = sum(terms[name]*weights[name] for name in terms)
    return total, terms, package


def capture(model, optimizer, stage, stage_step, global_step, generator, teacher):
    return dict(schema=SCHEMA, initialization="raw_measurements_no_learned_checkpoint", config=asdict(model.config),
                state=model.state_dict(), optimizer=optimizer.state_dict(), stage=stage, stage_step=stage_step,
                global_step=global_step, generator=generator.get_state(), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if model.field.xyz.is_cuda else [],
                numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                teacher=None if teacher is None else teacher.state_dict(),supervision_lineage=getattr(model,'supervision_lineage',None))


def save(path, value):
    temporary = Path(str(path)+".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def seed_from_state(state):
    """只用于同一 v3 模型恢复（支持 Stage1 增密后点数）；随后严格 load_state_dict。"""
    s = lambda key: state["field."+key].cpu().numpy()
    n = len(s("xyz"))
    seed=dict(xyz=s("xyz"), rgb=np.full((n, 3), .5, np.float32), observed_rgb=np.full((n, 3), .5, np.float32), scales=np.exp(s("log_scales")),
                quaternion=s("quaternion"), owner=s("owner"), road=s("road"), reference_rgb=s("reference_rgb"),
                reference_confidence=s("reference_confidence"), neighbor=s("neighbor"))
    for key in ("source_frame","source_raw_row","source_world","source_candidate_owner","identity_probability","identity_evidence_status"):
        if 'field.'+key in state:seed[key]=s(key)
    return seed


def preflight(dataset, config):
    if len(dataset.seed["xyz"]) > config.max_gaussians:
        raise ValueError("初始点数超过显式容量预算；调整体素/预算后重新审计，不能静默截去场景")
    known = 0;certified_road=0
    for i in range(len(dataset.rows)):
        frame = dataset.frame(i)
        known += int(frame.mask_known.sum())
        certified_road+=int(frame.road_certified.sum()) if frame.road_certified is not None else 0
    refs = int((dataset.seed["reference_confidence"] > 0).sum())
    if config.use_clean_road and refs < 1:
        raise ValueError("没有跨时刻可信干净道路参考；full 臂不能静默变成 no_clean_road 消融")
    if config.weights.road_identity>0 and certified_road==0:raise ValueError('road identity enabled without actual positive road certificates')
    return dict(status="CPU_INPUTS_CHECKED_GPU_NOT_STARTED", frames=len(dataset.rows), actors=len(dataset.actor_ids),
                gaussians=len(dataset.seed["xyz"]), road_gaussians=int(dataset.seed["road"].sum()),
                reference_points=int((dataset.seed["reference_confidence"] > 0).sum()),
                train_frames=len(dataset.train_indices), test_frames=len(dataset.test_indices),
                mask_known_pairs=known,
                certified_road_pixels=certified_road,
                mask_total_pairs=len(dataset.rows)*len(dataset.actor_ids), learned_checkpoint_used=False,
                estimated_GPU_hours=None, real_GPU_ABI_checked=False,
                planned_steps=dict(geometry=config.geometry_steps, material=config.material_steps, joint=config.joint_steps),
                reference_scope="train-frame reprojection pseudo-reference; not independent clean GT")


def checkpoint_config(raw):
    from .config import LossWeights
    value=raw.copy();value['weights']=LossWeights(**value['weights'])
    return asdict(Config(**value).validate())


def validate_supervision_fork(saved,config):
    """只允许同数据/模型、几何阶段的已声明监督修复，不借机混配消融。"""
    previous=checkpoint_config(saved['config']);current=asdict(config)
    expected=copy.deepcopy(previous)
    expected['instance_balanced_supervision']=True
    expected['weights']['mask_outside']=.5
    if previous['instance_balanced_supervision'] or saved['stage']!='geometry' or current!=expected:
        raise ValueError('监督fork仅允许几何阶段false→true和mask_outside=.5；其他配置必须相同')
    return dict(kind='declared_supervision_repair',parent_global_step=saved['global_step'],
                changes={'instance_balanced_supervision':[False,True],'weights.mask_outside':[previous['weights']['mask_outside'],.5]},
                model_optimizer_RNG_preserved=True,not_a_paired_causal_ablation=True)


def run(dataset, config, output, device="cpu", backend="reference", resume=None, max_updates=None,fork_supervision_from=None):
    output = Path(output)
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    else:
        output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    if resume and fork_supervision_from:raise ValueError('resume与监督fork互斥')
    source=resume or fork_supervision_from
    saved = torch.load(source, map_location="cpu", weights_only=False) if source else None
    lineage=saved.get('supervision_lineage') if saved else None
    if saved and (saved.get("schema") != SCHEMA or saved.get("initialization") != "raw_measurements_no_learned_checkpoint"):
        raise ValueError("只允许完全相同配置的 v3 中断恢复；拒绝 GN11 或混配消融 checkpoint")
    if fork_supervision_from:
        lineage=validate_supervision_fork(saved,config);lineage['parent_checkpoint']=str(Path(source).resolve())
        write(output/'supervision_fork.json',lineage)
    elif saved and checkpoint_config(saved['config'])!=asdict(config):
        raise ValueError('中断恢复配置不一致；监督修复必须使用显式fork且写入新目录')
    if saved and saved.get("input_contract") != dataset.manifest:
        raise ValueError("恢复的数据集身份/帧/划分/参考来源与保存时不一致")
    model = SceneModel(seed_from_state(saved["state"]) if saved else dataset.seed, config).to(device)
    if saved:
        model.load_state_dict(saved["state"], strict=True)
    model.supervision_lineage=lineage
    teacher = None
    if saved and saved["teacher"] is not None:
        teacher = SceneModel(seed_from_state(saved["teacher"]), config).to(device)
        teacher.load_state_dict(saved["teacher"], strict=True)
        teacher.requires_grad_(False)
    write(output/"resolved_config.json", asdict(config))
    write(output/"input_contract.json", dataset.manifest)
    write(output/"experiment.json", dict(task_id="drivingeditor_shadow_rebuild_20261017_v3", run_id=output.name,
                                        ablation=config.ablation, seed=config.seed, failure_ledger_refs=["V82-F01"],
                                        failure_ledger_delta="none; new implementation, no real GPU result yet",
                                        human_verdict=None, real_scene=not dataset.manifest.get("synthetic_fixture", False),
                                        backend=backend, allow_partial_masks=dataset.allow_partial_masks,
                                        supervision_lineage=lineage,
                                        initialization="raw_measurements_no_learned_checkpoint"))
    stages = [("geometry", config.geometry_steps), ("material", config.material_steps), ("joint", config.joint_steps)]
    step = saved["global_step"] if saved else 0
    start_step = step
    start_time = time.monotonic()
    resumed_stage_seen = not bool(saved)
    for stage, count in stages:
        if saved and not resumed_stage_seen and stage != saved["stage"]:
            continue
        if stage == "material" and not (saved and not resumed_stage_seen):
            with torch.no_grad():
                model.field.albedo_logit.copy_(model.field.appearance_logit)
            model.field.anchor()
            teacher = copy.deepcopy(model).requires_grad_(False)
        if stage == "joint" and not (saved and not resumed_stage_seen):
            model.field.anchor()
        optimizer = model.optimizer(stage)
        first = 0
        if saved and not resumed_stage_seen:
            first = saved["stage_step"]
            optimizer.load_state_dict(saved["optimizer"])
            generator.set_state(saved["generator"])
            torch.set_rng_state(saved["torch_rng"])
            if device.startswith("cuda") and saved["cuda_rng"]:
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
            np.random.set_state(saved["numpy_rng"])
            random.setstate(saved["python_rng"])
            resumed_stage_seen = True
        for local in range(first+1, count+1):
            frame_index = dataset.train_indices[int(torch.randint(len(dataset.train_indices), (), generator=generator))]
            frame = dataset.frame(frame_index)
            optimizer.zero_grad(set_to_none=True)
            loss, terms, package = objective(model, frame, len(dataset.actor_ids), backend, stage, generator, teacher,step)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"{stage}/{local}: 非有限 loss，保留最近 checkpoint")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
            score = model.field.xyz.grad.detach().norm(dim=-1) if model.field.xyz.grad is not None else None
            optimizer.step()
            model.field.project_constraints(stage)
            if stage == "geometry" and config.densify_interval and local % config.densify_interval == 0 and local <= config.densify_until and score is not None:
                extra = min(int(len(score)*config.densify_fraction), config.max_gaussians-len(score))
                if extra > 0:
                    model.field.split(score.topk(extra).indices, optimizer)
                    if config.use_clean_road:
                        from .references import refresh
                        refreshed = refresh(model.field, dataset)
                        write(output/"reference_refresh.json", refreshed)
                        if refreshed["valid_reference_points"] == 0:
                            raise ValueError("增密后可信道路参考全部丢失；必须诊断输入/几何，不能降格 full 臂")
            if stage == "geometry" and local == count and config.use_clean_road:
                from .references import refresh
                refreshed = refresh(model.field, dataset)
                write(output/"reference_refresh.json", refreshed)
                if refreshed["valid_reference_points"] == 0:
                    raise ValueError("Stage1 后没有可信道路参考，停止进入材质阶段")
            step += 1
            if local == 1 or local % config.log_interval == 0 or local == count:
                progress = dict(stage=stage, stage_step=local, stage_steps=count, global_step=step,
                                total_steps=sum(v for _, v in stages), loss=float(loss.detach()),
                                terms={k: float(v.detach()) for k, v in terms.items()}, frame_id=frame.frame_id,
                                gaussians=len(model.field.xyz), elapsed_seconds=time.monotonic()-start_time)
                write(output/"progress.json", progress)
                with (output/"metrics.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps(progress, ensure_ascii=False, allow_nan=False)+"\n")
                print(json.dumps(progress, ensure_ascii=False), flush=True)
            if local % config.checkpoint_interval == 0 or local == count or (max_updates and step-start_step >= max_updates):
                checkpoint = capture(model, optimizer, stage, local, step, generator, teacher)
                checkpoint["input_contract"] = dataset.manifest
                save(output/"latest.pt", checkpoint)
                if local == count:
                    save(output/(stage+"_final.pt"), checkpoint)
            if max_updates and step-start_step >= max_updates:
                return model, dict(status="BOUNDED_RUN_STOPPED", global_step=step)
    write(output/"completion.json", dict(status="TRAINING_FINISHED_EVALUATION_REQUIRED", global_step=step,
                                         human_verdict=None, problem_solved=None))
    return model, dict(status="TRAINING_FINISHED_EVALUATION_REQUIRED", global_step=step)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ablation", default="full")
    restore=parser.add_mutually_exclusive_group()
    restore.add_argument("--resume")
    restore.add_argument('--fork-supervision-from',help='显式修复实例监督；新输出目录，保留同模型/Adam/RNG和已用步数')
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--allow-partial-masks", action="store_true", help="必须作为显式输入缺口登记；默认拒绝未知 masks")
    args = parser.parse_args()
    if not args.execute:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    config = Config.load(args.config, args.ablation)
    dataset = Dataset(args.data, "cpu", args.allow_partial_masks)
    receipt = preflight(dataset, config)
    if not args.execute:
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return
    # 这里只能在用户另行打开 GPU 后显式运行，预检不会走到此处。
    if not torch.cuda.is_available():
        raise RuntimeError("需要用户已启动的 GPU；本程序不会开机")
    from .gpu_probe import probe
    probe(config)
    dataset.device = "cuda"
    run(dataset, config, args.output, "cuda", "cuda", args.resume, args.max_updates,args.fork_supervision_from)


if __name__ == "__main__":
    main()
