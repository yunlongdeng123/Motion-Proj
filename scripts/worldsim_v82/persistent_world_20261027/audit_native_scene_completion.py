"""CPU audit of a completed native run, respecting the explicit retention ledger."""
import argparse, datetime, gc, json, math
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene-run', type=Path, required=True)
    args = parser.parse_args()
    scene = args.scene_run.resolve(); run = scene / 'formal30k'
    returned = json.loads((run / 'returned.json').read_text())
    assert returned['reason'] == 'native_returned' and returned['iteration'] == 30000
    assert returned['first_iteration'] == 1 and not returned['extra_smoke']
    count = 0
    with (run / 'paired_steps.jsonl').open() as stream:
        for count, line in enumerate(stream, 1):
            record = json.loads(line)
            assert record['iteration'] == count
            assert all(math.isfinite(value) for value in record['losses'].values() if isinstance(value, (float, int)))
    assert count == returned['observed_finite_step_calls'] == 30000
    ledger_path = scene / 'checkpoint_retention.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    removed = {item['path']: item for item in ledger.get('files', []) if item.get('deleted')}
    nodes = []; final = []
    for iteration in range(5000, 30001, 5000):
        event = json.loads((run / 'checkpoint_events' / f'iteration_{iteration:06d}.json').read_text())
        assert event['status'] == 'complete' and event['iteration'] == iteration
        assert len(event['frames']) == 49
        for frame in event['frames']:
            assert frame['channels']
            assert all(Path(value).is_file() for value in frame['channels'].values())
        for field in ['metrics', 'paper_metrics']:
            assert Path(event[field]).is_file()
        assert set(event['checkpoints']) == {'static', 'dynamic', 'environment', 'training_state'}
        for item in event['checkpoints'].values():
            path = Path(item['path'])
            if path.exists():
                assert path.stat().st_size == item['bytes']
            else:
                assert iteration < 30000 and str(path) in removed
                assert removed[str(path)]['bytes'] == item['bytes'] and len(removed[str(path)]['sha256']) == 64
            if iteration == 30000:
                value = torch.load(str(path), map_location='cpu', mmap=True)
                assert value is not None
                del value; gc.collect()
                final.append(dict(path=str(path), bytes=path.stat().st_size))
        nodes.append(iteration)
    result = dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), context=scene.name,
                  formal_native_updates=30000, ordered_finite_step_records=count, complete_save_eval_nodes=nodes,
                  frames_per_node=49, final_four=final, final_four_cpu_readable=True,
                  native_returned=returned, cuda_allocator_peak=json.loads((run / 'cuda_memory_peak.json').read_text()),
                  retention_ledger=str(ledger_path) if ledger else None,
                  pruned_checkpoint_files_accounted_for=len(removed), quality_accepted=False, human_verdict=None)
    (scene / 'formal30k_agent_acceptance.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__': main()
