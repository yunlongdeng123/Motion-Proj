"""Create an isolated, hash-bound native overlay; never changes the original run."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'baseline_engineering_20261018/gn11_static_duplicate_20261019/actual_source'
DEST=ROOT/'candidate'


def replace_once(text,old,new):
    if text.count(old)!=1: raise RuntimeError('native source anchor mismatch: '+old[:100])
    return text.replace(old,new)


def build():
    manifest={}
    for relative in ['train.py','scene/gaussian_model.py','gaussian_renderer/__init__.py']:
        raw=(SOURCE/relative).read_bytes()
        text=raw.decode('utf-8').replace('\r\n','\n')
        if relative=='scene/gaussian_model.py':
            text=replace_once(text,'class GaussianModel:',
                'from utils.v2_static_partition import StaticV2Mixin\n\nclass NativeGaussianModel:')
            text=replace_once(text,'class Dynamic_GaussianModel(GaussianModel):',
                'class GaussianModel(StaticV2Mixin, NativeGaussianModel):\n    pass\n\n\nclass Dynamic_GaussianModel(NativeGaussianModel):')
            old=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='Dynamic_GaussianModel')
            new=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='Dynamic_GaussianModel')
            if [ast.dump(n) for n in old.body]!=[ast.dump(n) for n in new.body]:
                raise AssertionError('dynamic class methods changed')
        elif relative=='gaussian_renderer/__init__.py':
            text=replace_once(text,'"static_depth": rendered_depth_s,',
                '"static_depth": rendered_depth_s,\n                "dynamic_depth": rendered_depth_d,\n                "dynamic_feature": rendered_other_d,')
            text=replace_once(text,'rendered_image_d = rendered_image_d + (1 - rendered_opacity_d) * bg_color_from_envmap',
                'rendered_image_d = rendered_image_d + (1 - rendered_opacity_d) * bg_color_from_envmap.detach()')
        else:
            text=replace_once(text,'    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]',
                '    from utils.v2_runtime_context import V2Context\n'
                '    v2_context = V2Context(args, scene, env_map)\n'
                '    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]')
            text=replace_once(text,'        gaussians.update_learning_rate(iteration)',
                '        gaussians.enforce_static()\n        gaussians.update_learning_rate(iteration)')
            anchor='        viewpoint_cam, time_shift = scene.hybrid_sampler.select(scene.getTrainCameras(), args.lambda_self_supervision, scene.time_interval)'
            text=replace_once(text,anchor,anchor+'\n'
                '        viewpoint_cam, time_shift = v2_context.select_gate_camera(scene, viewpoint_cam, time_shift)\n'
                '        v2_context.gate_forward(scene, render, viewpoint_cam, args, background, env_map)')
            text=replace_once(text,'            hybrid_result = compose_hybrid_loss(hybrid_state, args, nps_fn=nps_loss)',
                '            hybrid_state = v2_context.partition_state(hybrid_state, viewpoint_cam, render_pkg, gt_image, dynamic_mask, depth, args, ssim)\n'
                '            v2_loss_config = dict(args)\n'
                '            v2_loss_config.update(lambda_t_reg=0.0, lambda_v_reg=0.0, hybrid_nps_enabled=False)\n'
                '            hybrid_result = compose_hybrid_loss(hybrid_state, v2_loss_config, nps_fn=nps_loss)')
            text=replace_once(text,'        loss.backward()',
                '        ownership_extra, road_loss = v2_context.objectives(viewpoint_cam, render_pkg, gaussians, iteration)\n'
                '        ownership_loss = loss + ownership_extra\n'
                '        v2_context.backward(ownership_loss, road_loss, gaussians, iteration)')
            text=replace_once(text,'                sgcm_statistics = paper_sgcm.refine(gaussians, radii[:end_idx], iteration, visibility_filter=visibility_filter[:end_idx])',
                '                sgcm_statistics = paper_sgcm.refine(gaussians, radii[:end_idx], iteration, visibility_filter=visibility_filter[:end_idx])\n'
                '                gaussians.enforce_static()')
            text=replace_once(text,'            log_dict[\'iter_time\'] = iter_start.elapsed_time(iter_end)',
                '            iter_end.synchronize()\n            log_dict[\'iter_time\'] = iter_start.elapsed_time(iter_end)')
        text='\n'.join(line.rstrip() for line in text.splitlines())+'\n'
        ast.parse(text)
        dest=DEST/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(text,encoding='utf-8',newline='\n')
        manifest[relative]=dict(original_sha256=hashlib.sha256(raw).hexdigest(),
                               candidate_sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    for original,name in [('static_partition.py','v2_static_partition.py'),
                          ('road_objective.py','v2_road_objective.py'),
                          ('runtime_context.py','v2_runtime_context.py')]:
        (DEST/'utils').mkdir(exist_ok=True)
        shutil.copyfile(ROOT/original,DEST/'utils'/name)
        manifest['utils/'+name]=dict(candidate_sha256=hashlib.sha256((DEST/'utils'/name).read_bytes()).hexdigest())
    # Preserve local upstream licence copy; deployment must also preserve base licences.
    license_path=ROOT.parent/'baseline_engineering_20261018/source/LICENSE_gaussian_splatting.md'
    # Text-equivalent working copy; untouched license bytes also ship inside the ZIP.
    license_text='\n'.join(line.rstrip() for line in license_path.read_text(encoding='utf-8').splitlines())+'\n'
    (DEST/'LICENSE_gaussian_splatting.md').write_text(license_text,encoding='utf-8',newline='\n')
    manifest['LICENSE_gaussian_splatting.md']=dict(original_sha256=hashlib.sha256(license_path.read_bytes()).hexdigest(),
        candidate_sha256=hashlib.sha256((DEST/'LICENSE_gaussian_splatting.md').read_bytes()).hexdigest())
    output=dict(status='LOCAL_OVERLAY_BUILT_SOURCE_ANCHORS_VERIFIED',files=manifest,
                dynamic_class_method_AST_unchanged=True,original_files_modified=False,
                trained_model_initialization_used=False,GPU_verified=False,training_started=False,
                required_runtime=['real CPU input bundle','actual fresh initialization binding',
                                  'actual CUDA forward/backward zero-update gate',
                                  'disk admission for all six checkpoints','exclusive GPU slot'])
    (ROOT/'candidate_manifest.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k!='files'},ensure_ascii=False))


if __name__=='__main__':build()
