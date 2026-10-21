"""CPU-only final asset audit and four-arm index; never trains, renders or powers off."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from evaluate_v3_final49 import ARMS, FRAMES, STEPS, VERSION, read, write_new, require_complete, retention_removals


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('non-finite JSON value')
    if isinstance(value, dict):
        for v in value.values(): finite(v)
    if isinstance(value, (tuple, list)):
        for v in value: finite(v)


def check_npz(path):
    with np.load(path, allow_pickle=False) as z:
        for key in z.files:
            a = z[key]
            if a.dtype.kind in 'fc' and not np.isfinite(a).all():
                raise ValueError('nonfinite array: '+str(path)+' '+key)
        return {k: z[k].copy() for k in z.files}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def nested(value, path):
    for part in path.split('.'): value = value[part]
    return value


def weighted(rows, path, weight):
    pairs = [(nested(r, path), nested(r, weight)) for r in rows if nested(r, weight) > 0]
    pairs = [(v, w) for v, w in pairs if v is not None and w > 0]
    return sum(v*w for v, w in pairs)/sum(w for _, w in pairs) if pairs else None


def add_samples(stats, key, values):
    values = np.asarray(values)
    if values.size:
        n, total = stats.get(key, (0, 0.))
        stats[key] = (n+values.size, total+float(values.sum(dtype=np.float64)))


def sample_summary(stats):
    return {k: dict(samples=n, mean=s/n) for k, (n, s) in stats.items()}


def audit(campaign, output):
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    if output.exists() or not output.is_relative_to(campaign/'final49'):
        raise ValueError('fresh output under campaign/final49 required')
    events = require_complete(campaign)
    removed = retention_removals(campaign)
    manifest = {}

    def add(path):
        p = Path(path).resolve()
        if not p.is_relative_to(campaign) or not p.is_file(): raise ValueError('asset escaped or missing')
        name = str(p.relative_to(campaign))
        if name not in manifest:
            manifest[name] = dict(bytes=p.stat().st_size, sha256=digest(p))

    def png(path):
        with Image.open(path) as im:
            if im.size != (480, 320): raise ValueError('wrong PNG size')
            im.verify()
        add(path)

    receipt = read(campaign/'checkpoint_retention.json')
    for path, original in receipt['protected'].items():
        stat = Path(path).stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != (
                original['bytes'], original['mtime_ns'], original['inode']):
            raise ValueError('cleanup-protected asset changed: '+path)
    for item in receipt['kept']:
        if digest(item['path']) != item['sha256']: raise ValueError('retained30K SHA changed')
    add(campaign/'checkpoint_retention.json')
    result = {}
    for arm in ARMS:
        run = campaign/'arms'/arm/'run30k'
        for step in STEPS:
            path = run/'checkpoint_events'/f'iteration_{step:06d}.json'
            event = read(path); add(path); add(event['metrics'])
            for key, item in event['checkpoints'].items():
                if (arm, step, key) not in removed: add(item['path'])
            for frame in event['frames']:
                for p in frame['channels'].values(): png(p)
                z = check_npz(frame['road_float'])
                if set(z) != {'rgb_sum', 'alpha', 'depth_sum'}: raise ValueError('road fields missing')
                if z['alpha'].min() < -1e-5 or z['alpha'].max() > 1.00001: raise ValueError('alpha range')
                add(frame['road_float']); finite(read(frame['hybrid_metrics'])); add(frame['hybrid_metrics'])
        folder = campaign/'final49'/f'{arm}_r2'
        evaluation = read(folder/'evaluation.json'); finite(evaluation)
        if (evaluation['version'] != VERSION or evaluation['optimizer_updates'] != 0
                or evaluation['optimizer_created'] or not evaluation['training_assets_unchanged']
                or evaluation['fixed_road_ids'] != 14034 or evaluation['quality_accepted']
                or evaluation['human_verdict'] is not None
                or evaluation['status'] != 'FORMAL49_RENDERED_REPLAYED_AND_FINITE'
                or tuple(r['frame'] for r in evaluation['frames']) != FRAMES):
            raise ValueError('incomplete final evaluation')
        add(folder/'evaluation.json')
        world = check_npz(folder/'world_units.npz'); add(folder/'world_units.npz')
        fixed = {}
        for actor in (6, 37):
            ids = np.unique(world['query_unit_id'][(world['query_frame'] == 0) & (world['query_actor'] == actor)])
            ids = ids[ids >= 0]
            fixed[actor] = dict(ids=ids, stats={})
        for row in evaluation['frames']:
            frame = folder/row['frame']
            if read(frame/'metrics.json') != row: raise ValueError('per-frame metrics mismatch')
            add(frame/'metrics.json')
            for name in ('joint', 'deleted_five', 'dynamic_raw', 'road_raw', 'road_surface_color'):
                png(frame/(name+'.png'))
            raw = check_npz(frame/'render_float.npz'); units = check_npz(frame/'world_unit_float.npz')
            for key in ('joint', 'deleted_five', 'dynamic_raw', 'road_raw', 'road_surface_color', 'gt'):
                if raw[key].shape != (3, 320, 480): raise ValueError('missing full-frame RGB stream')
            for key in ('dynamic_alpha', 'static_alpha', 'joint_alpha', 'road_alpha',
                        'joint_depth_m', 'static_depth_m', 'dynamic_depth_m', 'road_depth_sum'):
                if raw[key].shape != (1, 320, 480): raise ValueError('missing full-frame scalar stream')
            if len(raw['sample_measured_depth']) > 14: raise ValueError('sample scope changed')
            for a in fixed.values():
                select = np.zeros(len(world['unit_ids']), bool); select[a['ids']] = True
                visible = select & units['in_view']
                verified = select & units['verified_exposure']
                for k in ('road_only_alpha', 'road_alpha_in_background', 'road_alpha_in_joint'):
                    add_samples(a['stats'], k, units[k][visible])
                for k in ('road', 'deleted', 'joint'):
                    add_samples(a['stats'], k+'_verified_RGB_L1', np.abs(units[k]-units['source_RGB'])[verified])
                valid = units['prior_subcell_valid'] & select[:, None]
                add_samples(a['stats'], 'prior_subcell_alpha', units['prior_subcell_alpha'][valid])
            add(frame/'render_float.npz'); add(frame/'world_unit_float.npz')
        rows = evaluation['frames']
        summary = dict(source_reconstruction=evaluation['source_reconstruction_metrics'],
            observed_query={key: weighted(rows, 'verified_exposure.RGB_L1.'+key, 'verified_exposure.units')
                            for key in ('road', 'deleted', 'joint')},
            observed_road_depth_MAE_m=weighted(rows, 'verified_exposure.road_depth_MAE_m', 'verified_exposure.units'),
            observed_context={key: weighted(rows, 'verified_outside_query.RGB_L1.'+key, 'verified_outside_query.units')
                              for key in ('road', 'deleted', 'joint')},
            body_front={key: weighted(rows, 'supervision.'+key, 'supervision.front_valid_pixels')
                        for key in ('front_hard_fraction', 'front_hard_alpha_mass')},
            dynamic_outside={key: weighted(rows, 'supervision.'+key, 'supervision.dynamic_strict_outside_pixels')
                for key in ('dynamic_color_leak_mean', 'dynamic_color_leak_top', 'dynamic_ratio_excess_mean')},
            sampled_front={key: weighted(rows, 'sampled_front.'+key, 'sampled_front.sampled_pixels')
                for key in ('static_front_alpha_mean', 'joint_front_static_alpha_mean',
                            'static_front_center_outside_alpha_mean', 'mean_depth_missed_front_mass_mean')},
            terminal_road_state=evaluation['terminal_road_state'],
            evaluation_wall_seconds=evaluation['zero_update_evaluation_wall_seconds'],
            evaluation_peak_allocated_bytes=evaluation['evaluation_peak_allocated_bytes'],
            evaluation_peak_reserved_bytes=evaluation['evaluation_peak_reserved_bytes'],
            fixed_raw89={str(k): dict(world_units=len(v['ids']),
                provenance_counts={str(p): int((world['provenance'][v['ids']] == p).sum()) for p in (0, 1, 2)},
                repeated_visible_anchor_samples=sample_summary(v['stats'])) for k, v in fixed.items()})
        logs = {}
        for name in ('v3_capacity', 'v3_objectives', 'v3_render_cost', 'v3_densification',
                     'paired_steps', 'gradient_diagnostics'):
            path = run/(name+'.jsonl'); records = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
            finite(records); add(path)
            logs[name] = records
        if not all(x['fixed_road'] == 14034 and x['fixed_road_ids_preserved'] for x in logs['v3_capacity']):
            raise ValueError('road identity/capacity changed')
        cost = logs['v3_render_cost']
        summary['training_cost'] = dict(
            profile_scope=cost[-1]['timing_scope'],
            peak_allocated_bytes=max(x['memory_peak_allocated_bytes'] for x in cost),
            peak_reserved_bytes=max(x['memory_peak_reserved_bytes'] for x in cost),
            raster_calls=cost[-1]['cumulative_raster_calls_all_training_iterations'],
            terminal_capacity=logs['v3_capacity'][-1])
        losses = logs['paired_steps'][-1]['losses']
        summary['last_training_regularizer_record'] = {k: v for k, v in losses.items()
            if any(x in k for x in ('duration', 'static_velocity', 'nps'))}
        gradient = logs['gradient_diagnostics'][-1]
        summary['last_training_gradient_record'] = dict(iteration=gradient['iteration'],
            diagnostic={k: v for k, v in gradient['diagnostic'].items()
                if any(x in k for x in ('duration', 'static_velocity', 'nps', '/t/', '/scaling_t/', '/velocity/'))})
        summary['last_training_source_record'] = logs['v3_objectives'][-1]
        add(run/'status.json'); add(run/'effective_config.json')
        result[arm] = summary
        print(json.dumps(dict(arm=arm, final49_audited=True, manifest_files=len(manifest))), flush=True)
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output/'asset_manifest.json'
    write_new(manifest_path, dict(status='EXISTING_ASSETS_HASHED_AND_FINITE_CHECKED', files=manifest))
    receipt = dict(status='FOUR_ARM_FORMAL49_AND_ASSET_AUDIT_COMPLETE', campaign=str(campaign),
        optimizer_updates=0, fresh_training_updates=120000, failed_training_updates_separate=26152,
        cleanup_exemptions=len(removed), actual_checkpoint_files=56,
        checkpoint_finite_scope='Each formal node previously CPU-loaded when published; terminal models/sidecars CPU-loaded and replayed by final49.',
        asset_manifest_sha256=digest(manifest_path), results=result, quality_accepted=False,
        human_verdict=None, completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        interpretation='Engineering and file completion only. No edited GT, causal source attribution, human acceptance or additional training.')
    finite(receipt); write_new(output/'comparison.json', receipt)
    make_preview(campaign, output)
    return receipt


def make_preview(campaign, output):
    """Uncropped, unretouched scientific contact sheet for visual inspection."""
    frames = ('0008089', '0008113', '0008128', '0008137')
    sheet = Image.new('RGB', (4*480, 4*342), 'white'); draw = ImageDraw.Draw(sheet)
    for i, arm in enumerate(ARMS):
        for j, frame in enumerate(frames):
            with Image.open(campaign/'final49'/f'{arm}_r2'/frame/'deleted_five.png') as image:
                sheet.paste(image, (j*480, i*342+22))
            draw.text((j*480+4, i*342+3), arm+' / '+frame+' / delete five', fill='black')
    sheet.save(output/'deleted_full_frames.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.campaign, args.output)
