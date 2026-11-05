"""Read experiment receipts only; no GPU, optimization, sampling or writes."""
import datetime
import hashlib
import json
from pathlib import Path

RUN = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a6_appearance_formal_20261105')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def evidence(path):
    return {'path': str(path), 'sha256': sha(path)}


receipt_paths=set(RUN.rglob('receipt.json'))|set(RUN.rglob('prepare_receipt.json'))
packaging_roots={'publication_inputs','collected_publication','delivery'}
packaged_receipts=[p for p in receipt_paths if p.relative_to(RUN).parts[0] in packaging_roots]
receipt_paths-=set(packaged_receipts)
receipts = [(p, read(p)) for p in sorted(receipt_paths)]
sd_formal = [(p, r) for p, r in receipts if p.parent.parent == RUN/'candidates_raw106']
sd_smoke = [(p, r) for p, r in receipts if p.parent.parent == RUN/'backend_smoke']
assert len(sd_formal) == 8 and len(sd_smoke) == 1
assert {r['job']['seed'] for _, r in sd_formal} == {0, 1, 2, 3}
assert all(r['training_updates'] == 0 for _, r in sd_formal+sd_smoke)
qa = read(RUN/'candidates_raw106/qa_summary.json')
model = read(RUN/'candidates_raw106/model_manifest.json')
assert qa['independent_process_smoke_first_candidate_replay_sha256_exact']

observed = []
generated = []
evaluation = []
output_adapters = []
for p, r in receipts:
    if 'formal_generated_colour_steps' in r:
        if Path(r['export']['path']).parent.resolve()!=p.parent.resolve():
            packaged_receipts.append(p)
            continue
        assert r['status'].startswith('PASS')
        generated.append({
            'branch': str(p.parent.relative_to(RUN)), 'receipt': evidence(p),
            'status': r['status'], 'execution': 'CPU per-asset bounded colour solver',
            'formal_per_asset_colour_update_steps': r['formal_generated_colour_steps'],
            'neural_weight_training_steps': r['neural_training_steps'],
            'extra_nextstep_replay_evaluations': r['replay_nextstep_checks_extra'],
            'exact_full_state_nextstep_replay': r['exact_full_state_nextstep_replay'],
            'geometry_updates_during_colour_fit': r['geometry_updates'],
            'base_asset_sha256': r['identity']['base_sha256'],
            'fitted_candidate_pixels': sum(v['fitted_pixels'] for v in r['views']),
            'asset': r['export'],
            'initial_objective': r['initial_objective'], 'final_objective': r['final_objective'],
            'seconds_recorded_in_receipt': r.get('seconds'),
            'wall_time_note': None if 'seconds' in r else 'Solver receipt does not record elapsed time; not estimated.',
            'neural_or_PVD_model_weights_changed': False,
        })
    elif 'colour_fit_steps' in r:
        if Path(r['asset']['path']).parent.resolve()!=p.parent.resolve():
            packaged_receipts.append(p)
            continue
        assert r['status'] == 'PASS'
        observed.append({
            'branch': str(p.parent.relative_to(RUN)), 'receipt': evidence(p),
            'stage': 'smoke' if p.parent.name == 'smoke' else 'formal',
            'execution': 'CUDA per-asset SH0 colour solver; pretrained/PVD weights frozen',
            'per_asset_colour_update_steps': r['colour_fit_steps'],
            'neural_weight_training_steps': r['neural_training_steps'],
            'PVD_weight_training_steps': r['PVD_weight_updates'],
            'seconds_recorded_in_receipt': r['seconds'],
            'full_state_exact_resume': r['full_state_exact_resume'],
            'extra_nextstep_replay_evaluations': 2,
            'replay_evidence': 'Executed fit source computes one next step from live state and one from fully restored state, compares exact x/y; receipt full_state_exact_resume=true.',
            'next_stage_native_scene_entry': r['next_stage_scene'],
            'formal_projection_seconds': r['projected_formal_seconds'],
            'peak_vram_bytes': r['peak_vram'],
            'source_sha256': r['source_sha256'],
            'asset': r['asset'],
            'kernel_change_before_fit': r.get('kernel_change_before_fit'),
        })
    elif '/evaluation/' in str(p) or '/preview_' in str(p):
        evaluation.append({'branch': str(p.parent.relative_to(RUN)), 'receipt': evidence(p),
                           'status': r.get('status'), 'seconds_recorded_in_receipt': r.get('seconds'),
                           'not_training': True})
    elif p.parent == RUN/'output_adapter' or 'projective_surface' in str(p):
        output_adapters.append({'branch':str(p.parent.relative_to(RUN)), 'receipt':evidence(p),
                                'status':r.get('status'), 'seconds_recorded_in_receipt':r.get('seconds'),
                                'operation_class':'output resampling inference: view-conditioned single-reference representation' if 'projective_surface' in str(p) else 'deterministic thickness-only kernel output adapter',
                                'neural_weight_training_updates':0,
                                'scope_note':'No claim of 360-degree completion or independent geometry truth' if 'projective_surface' in str(p) else 'Point positions/sigma/opacity/source fixed; thickness=sigma',
                                'actual_receipt':r})

failure_log = RUN/'highres_smoke.log'
failure_text = failure_log.read_text()
assert 'self.step' in failure_text or 'state=s.step(state)' in failure_text
assert 'x.dtype==torch.float32' in failure_text and 'AssertionError' in failure_text
failed = {
    'stage': 'initial observed-HR smoke dtype wiring failure',
    'log': evidence(failure_log), 'accepted_colour_update_steps': 0,
    'attempted_first_update': True, 'neural_weight_training_steps': 0,
    'PVD_weight_training_steps': 0,
    'reason': 'First proposed update reached deterministic CSR dtype assertion before step() returned; outer state assignment never committed step1.',
    'seconds_recorded': None,
    'compute_note': 'Zero accepted updates does not mean zero compute; initial objective/gradient and part of first proposal were executed. Elapsed cost was not recorded.',
    'recovery': 'Float32 correction, then successful 32-step real smoke and unchanged 1000-step formal run in fresh output directories.',
}

preparation_failures = []
for p in sorted((RUN/'inputs').glob('*.log')):
    txt = p.read_text(errors='replace')
    if 'Traceback' in txt:
        preparation_failures.append({'log': evidence(p), 'stage': 'input collection/preparation',
                                     'neural_weight_training_steps': 0,
                                     'seconds_recorded': None,
                                     'last_error': next((x for x in reversed(txt.splitlines()) if x.strip()), '')})

collections = []
for p in sorted(RUN.rglob('collection.json')):
    if p.relative_to(RUN).parts[0] in packaging_roots:
        continue
    r = read(p)
    collections.append({'branch': str(p.parent.relative_to(RUN)), 'source': evidence(p),
                        'status': r.get('status'), 'seconds_recorded': r.get('seconds'),
                        'rows': r.get('rows'), 'frames': len(r.get('frames', [])),
                        'no_weight_training': True})

smoke_fits = [r for r in observed if r['stage'] == 'smoke']
formal_fits = [r for r in observed if r['stage'] == 'formal']
result = {
    'schema': 'a6.appearance.experiment_ledger.v1',
    'snapshot_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'run': str(RUN),
    'scope': 'This run only; accepted optimisation steps, inference, smoke/replay overhead and known failed preparation are distinguished.',
    'packaged_receipt_copies_excluded_from_compute_accounting':[str(p) for p in sorted(set(packaged_receipts))],
    'terminology': {
        'neural_weight_training': 'Updating PVD, SD1.5, IP-Adapter, CLIP, VAE or other pretrained network parameters. Count is zero in this run.',
        'per_asset_colour_update': 'Fitting this fixed Gaussian asset SH0/RGB variables with frozen pretrained models. CPU or CUDA colour iterations are not neural weight training.',
        'denoising_step': 'A frozen model inference step, not a training update.',
        'kernel_output_adapter': 'Deterministic thickness=sigma change before sphere fitting; not learned geometry or neural weight training.',
    },
    'weight_training_totals': {'PVD': 0, 'SD15': 0, 'IP_Adapter': 0, 'CLIP_text_and_vision': 0, 'VAE': 0,
                               'all_pretrained_neural_weight_updates': 0},
    'offline_SD15_inference': {
        'formal_candidates': len(sd_formal), 'smoke_candidates_separate': len(sd_smoke),
        'formal_denoising_steps_per_candidate': [r['parameters']['num_inference_steps'] for _, r in sd_formal],
        'formal_inference_seconds_excluding_model_load': sum(r['seconds'] for _, r in sd_formal),
        'smoke_inference_seconds_separate_excluding_model_load': sum(r['seconds'] for _, r in sd_smoke),
        'prompts': len({r['job']['prompt'] for _, r in sd_formal}), 'seeds_per_prompt': [0,1,2,3],
        'independent_process_first_candidate_exact_PNG_SHA256_replay': True,
        'repeatability_scope': 'Exact replay established for prompt1_seed0 on this recorded software/hardware stack; no cross-device/version guarantee is claimed.',
        'model_and_tokenizer_file_hashes': model['files'],
        'model_manifest': evidence(RUN/'candidates_raw106/model_manifest.json'),
        'batch_receipt': evidence(RUN/'candidates_raw106/batch_receipt.json'),
        'formal_receipts': [evidence(p) for p, _ in sd_formal], 'smoke_receipts': [evidence(p) for p, _ in sd_smoke],
        'parameters': model['parameters'], 'environment': model['environment'],
        'model_loading_time_recorded': False,
        'no_weight_training': True,
    },
    'strong_hosted_generation': {
        'calls': 2, 'call_count_evidence': 'Explicit parent task record: strong hosted2calls; two returned v1/v2 original images exist locally.',
        'checkpoint_hash_available': False, 'checkpoint_hash': None,
        'deterministic_replay_established': False,
        'provider_random_seed_exposed': False,
        'elapsed_compute_seconds_recorded': None,
        'classification': 'Hosted pretrained inference supplies inferred material proposals, not measured RGB, hidden ground truth, or completed validated 3D geometry.',
        'no_locally_performed_neural_weight_training': True,
    },
    'observed_HR_asset_colour_fits': observed,
    'observed_HR_failed_dtype_attempt_separate': failed,
    'generated_asset_colour_fits': generated,
    'automatically_counted_totals_at_snapshot': {
        'completed_generated_asset_fit_branches': len(generated),
        'generated_asset_formal_colour_update_steps': sum(r['formal_per_asset_colour_update_steps'] for r in generated),
        'generated_asset_extra_nextstep_replay_evaluations': sum(r['extra_nextstep_replay_evaluations'] for r in generated),
        'observed_asset_smoke_branches': len(smoke_fits),
        'observed_asset_smoke_colour_steps_separate': sum(r['per_asset_colour_update_steps'] for r in smoke_fits),
        'observed_asset_formal_branches': len(formal_fits),
        'observed_asset_formal_colour_steps': sum(r['per_asset_colour_update_steps'] for r in formal_fits),
        'observed_asset_extra_nextstep_replay_evaluations': sum(r['extra_nextstep_replay_evaluations'] for r in observed),
        'accepted_colour_steps_in_failed_dtype_attempt': 0,
        'all_pretrained_neural_weight_updates': 0,
    },
    'collection_and_preparation_receipts': collections,
    'other_failed_input_preparation_attempts': preparation_failures,
    'native_evaluation_and_preview_receipts': evaluation,
    'deterministic_output_adapter_and_representation_tests': output_adapters,
    'accounting_limits': [
        'Recorded SD inference seconds exclude model loading, hashing and transfer.',
        'Generated CPU colour-solver receipts do not record duration; none is invented.',
        'Hosted calls have no recorded provider duration, checkpoint hash or deterministic replay proof.',
        'Stage durations can overlap and are not added into a claimed end-to-end GPU bill.',
        'Replay evaluations and smoke are outside formal optimisation-step contracts.',
    ],
}
hosted_path=RUN/'hosted_generation/hosted_generation.json'
if hosted_path.is_file():
    hosted_requests=read(hosted_path)
    assert len(hosted_requests['calls'])==2
    assert hosted_requests['checkpoint_hash'] is None and hosted_requests['deterministic_rerun_claim'] is False
    result['strong_hosted_generation']['calls']=len(hosted_requests['calls'])
    result['strong_hosted_generation']['call_count_evidence']='Actual hosted_generation.json request/tool-response receipt, independently read and hashed'
    result['strong_hosted_generation']['request_and_tool_response_receipt']=evidence(hosted_path)
    result['strong_hosted_generation']['actual_requests_and_responses']=hosted_requests
archived_eval=RUN/'output_adapter/projective_surface/evaluation_before_pool_fix'
old_eval_source=RUN/'source/output_adapter/projective_surface/evaluate_projective_surface.py.before_metric_pool_fix'
if archived_eval.is_dir():
    archived_metrics=sorted(archived_eval.glob('*_metrics.json'))
    result['failed_projective_aggregate_evaluation_separate']={
        'stage':'Prototype fixed49 native evaluation completed its frame loop; aggregation/pool wiring failed afterward',
        'archived_output_directory':str(archived_eval),
        'completed_frame_evaluations_before_failure':49,
        'frame_count_evidence':'Parent execution record; archived evaluator source puts pool aggregation after fixed raw089..137 loop',
        'saved_archived_per_frame_metric_JSON_count':len(archived_metrics),
        'old_evaluator_source':evidence(old_eval_source) if old_eval_source.is_file() else None,
        'additional_model_inference_or_native_evaluation_frames':49,
        'accepted_per_asset_colour_updates':0,'neural_weight_training_updates':0,'PVD_weight_updates':0,
        'seconds_recorded':None,
        'compute_note':'This is extra native evaluation overhead, not zero compute and not a training/fitting replay. Runtime was not recorded.',
        'recovery':'Aggregation-only fix; subsequent fixed49 evaluation retained separately under evaluation/.'}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
