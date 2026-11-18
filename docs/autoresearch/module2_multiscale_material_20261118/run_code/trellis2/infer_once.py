"""固定网格和参考图的 TRELLIS.2 官方材质推理；大资产仅写远端。"""
import argparse
import functools
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--dino', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--input-manifest', type=Path, required=True)
    parser.add_argument('--lease-file', type=Path, required=True)
    parser.add_argument('--lease-owner', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resolution', type=int, default=512, choices=[512])
    parser.add_argument('--texture-size', type=int, default=1024)
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit('inference requires --execute and exclusive registered lease')
    manifest = json.loads(args.input_manifest.read_text())
    lease = json.loads(args.lease_file.read_text())
    owner = str(lease.get('owner', '')).rstrip('/').split('/')[-1]
    expected_owner = args.lease_owner.rstrip('/').split('/')[-1]
    if lease.get('active') is not True or lease.get('gpu_authorized') is not True or owner != expected_owner:
        raise RuntimeError('exclusive GPU lease not assigned to this caller')
    if (args.seed, args.resolution, args.texture_size) != (101, 512, 1024):
        raise ValueError('single frozen seed101/res512/texture1024 contract')
    if manifest['status'] != 'PASS_CPU_INPUT' or manifest['source_frames'] != [86, 110]:
        raise ValueError('input source manifest not qualified')
    if args.mesh.resolve() != Path(manifest['mesh']).resolve() or args.reference.resolve() != Path(manifest['reference']).resolve():
        raise ValueError('input paths differ from frozen manifest')
    if digest(args.mesh) != manifest['mesh_sha256'] or digest(args.reference) != manifest['reference_sha256']:
        raise ValueError('frozen mesh/reference changed')
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('ATTN_BACKEND', 'sdpa')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

    import numpy as np
    import torch
    import trimesh
    from PIL import Image
    from trellis2 import models
    from trellis2.modules import image_feature_extractor
    from trellis2.pipelines import Trellis2TexturingPipeline
    from trellis2.pipelines import samplers

    record = {
        'task': 'MODULE2-MULTISCALE-MATERIAL-20261118-TRELLIS2',
        'status': 'initializing', 'seed': args.seed,
        'resolution': args.resolution, 'texture_size': args.texture_size,
        'learning_updates': 0, 'preprocess_image': False,
        'input_roles': {'mesh': 'current repaired fixed GS world centers triangulated into one road patch', 'reference': 'legal raw86/110 A/C/S worldwarp; explicit neutral unknowns'},
        'input_manifest': str(args.input_manifest),
        'input_manifest_sha256': digest(args.input_manifest),
        'GPU_lease': {'path': str(args.lease_file), 'owner': owner},
        'new_inference': True, 'historical_output_reused': False,
        'model_root': str(args.model_root), 'dino_root': str(args.dino),
        'mesh': str(args.mesh), 'reference': str(args.reference),
        'mesh_sha256': digest(args.mesh), 'reference_sha256': digest(args.reference),
        'loading_adapter': 'official models and constructor, only used resolution; unused background remover omitted',
        'official_run_modified': False,
        'geometry_generation': False,
        'conditions_differ_from_inpainting': 'TRELLIS accepts fixed mesh plus reference, no hole-mask/context channel',
        'human_verdict': None, 'stages': [],
    }

    def save():
        (args.output / 'run_receipt.json').write_text(json.dumps(record, indent=2, ensure_ascii=False), 'utf-8')

    save()
    start = time.perf_counter()
    try:
        config_file = args.model_root / 'texturing_pipeline.json'
        cfg = json.loads(config_file.read_text())['args']
        if cfg['tex_slat_sampler']['params']['steps'] != 12:
            raise ValueError('official fixed 12-step sampler changed')
        torch.set_num_threads(4)
        record['official_config_sha256'] = digest(config_file)
        record['sampler'] = cfg['tex_slat_sampler']
        selected_names = ['shape_slat_encoder', 'tex_slat_decoder', f'tex_slat_flow_model_{args.resolution}']
        loaded = {}
        for name in selected_names:
            tick = time.perf_counter()
            print(f'LOADING {name}', flush=True)
            loaded[name] = models.from_pretrained(str(args.model_root / cfg['models'][name]))
            record['stages'].append({'name': 'load_' + name, 'seconds': time.perf_counter() - tick,
                                     'parameters': sum(p.numel() for p in loaded[name].parameters())})
            save()
        tick = time.perf_counter()
        dino = image_feature_extractor.DinoV3FeatureExtractor(str(args.dino))
        record['stages'].append({'name': 'load_dinov3_vitl', 'seconds': time.perf_counter() - tick,
                                 'parameters': sum(p.numel() for p in dino.model.parameters())})
        pipeline = Trellis2TexturingPipeline(
            models=loaded,
            tex_slat_sampler=getattr(samplers, cfg['tex_slat_sampler']['name'])(**cfg['tex_slat_sampler']['args']),
            tex_slat_sampler_params=cfg['tex_slat_sampler']['params'],
            shape_slat_normalization=cfg['shape_slat_normalization'],
            tex_slat_normalization=cfg['tex_slat_normalization'],
            image_cond_model=dino,
            rembg_model=None,
            low_vram=True,
        )
        for model in loaded.values():
            model.eval().requires_grad_(False)
        pipeline.cuda()
        mesh = trimesh.load(args.mesh, force='mesh', process=False)
        image = Image.open(args.reference).convert('RGB')
        record['input_vertices'] = len(mesh.vertices)
        record['input_faces'] = len(mesh.faces)
        record['reference_size'] = list(image.size)
        record['torch'] = torch.__version__
        record['device'] = torch.cuda.get_device_name()
        record['load_seconds'] = time.perf_counter() - start
        try:
            module_file = __import__('trellis2').__file__
            record['official_git_sha'] = subprocess.check_output(
                ['git', '-C', str(Path(module_file).resolve().parents[1]), 'rev-parse', 'HEAD'], text=True).strip()
        except Exception:
            record['official_git_sha'] = None

        # 只计时并保存阶段输出；保持官方前向计算和采样原样。
        def wrap(name, function):
            @functools.wraps(function)
            def timed(*a, **kw):
                torch.cuda.synchronize()
                tick = time.perf_counter()
                result = function(*a, **kw)
                torch.cuda.synchronize()
                row = {'name': name, 'seconds': time.perf_counter() - tick}
                if hasattr(result, 'feats'):
                    row['sparse_rows'] = int(result.feats.shape[0])
                    row['channels'] = int(result.feats.shape[1])
                record['stages'].append(row)
                print(json.dumps(row), flush=True)
                save()
                return result
            return timed

        for name in ['get_cond', 'encode_shape_slat', 'sample_tex_slat', 'decode_tex_slat', 'postprocess_mesh']:
            setattr(pipeline, name, wrap(name, getattr(pipeline, name)))
        torch.cuda.reset_peak_memory_stats()
        record['status'] = 'running'
        save()
        tick = time.perf_counter()
        output = pipeline.run(mesh, image, seed=args.seed,
                              preprocess_image=False, resolution=args.resolution,
                              texture_size=args.texture_size)
        torch.cuda.synchronize()
        record['inference_seconds'] = time.perf_counter() - tick
        record['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        record['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        output.export(args.output / 'textured.glb', extension_webp=True)
        # 同时保存无损纹理与 UV，避免 GLB 中的压缩影响定量采样。
        material = output.visual.material
        material.baseColorTexture.save(args.output / 'basecolor_rgba.png')
        material.metallicRoughnessTexture.save(args.output / 'metallic_roughness.png')
        np.savez_compressed(args.output / 'generated_mesh.npz', vertices=np.asarray(output.vertices),
                            faces=np.asarray(output.faces), uv=np.asarray(output.visual.uv))
        # UV seam duplication/reordering is allowed; 3D source vertices/faces must be identical.
        from scipy.spatial import cKDTree
        fixed = np.load(args.input_manifest.parent / 'fixed_geometry.npz', allow_pickle=False)
        transforms = json.loads((args.input_manifest.parent / 'transforms.json').read_text())
        T = np.asarray(transforms['official_output_yup_to_world'])
        observed_world = np.asarray(output.vertices) @ T[:3, :3].T + T[:3, 3]
        distance, nearest = cKDTree(fixed['world_vertices']).query(observed_world)
        mapped = nearest[np.asarray(output.faces)]
        expected_faces = np.sort(fixed['faces'], axis=1)
        observed_faces = np.sort(mapped, axis=1)
        def sorted_faces(a):
            return a[np.lexsort((a[:,2], a[:,1], a[:,0]))]
        geometry_identity = dict(source_vertices=len(fixed['world_vertices']), output_vertices=len(observed_world),
            max_world_vertex_error_m=float(distance.max()), tolerance_m=1e-5,
            faces_same_geometry=np.array_equal(sorted_faces(expected_faces), sorted_faces(observed_faces)),
            source_vertex_identity_covered=len(np.unique(nearest)) == len(fixed['world_vertices']),
            UV_seam_reindexing_allowed=True, GS_geometry_changed=False)
        geometry_identity['passed'] = (geometry_identity['max_world_vertex_error_m'] <= 1e-5
            and geometry_identity['faces_same_geometry'] and geometry_identity['source_vertex_identity_covered'])
        (args.output / 'geometry_identity.json').write_text(json.dumps(geometry_identity, indent=2))
        record['geometry_identity'] = geometry_identity
        if not geometry_identity['passed']:
            raise RuntimeError('official output changed fixed input geometry; preserve rejected output')
        record['output_vertices'] = len(output.vertices)
        record['output_faces'] = len(output.faces)
        record['finite_vertices'] = bool(np.isfinite(output.vertices).all())
        record['output_files'] = ['textured.glb', 'basecolor_rgba.png', 'metallic_roughness.png', 'generated_mesh.npz']
        record['status'] = 'completed'
    except Exception as error:
        record['status'] = 'failed'
        record['error'] = repr(error)
        record['traceback'] = traceback.format_exc()
        raise
    finally:
        record['total_process_seconds'] = time.perf_counter() - start
        save()


if __name__ == '__main__':
    main()
