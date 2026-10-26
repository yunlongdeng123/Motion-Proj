"""CPU-only completion packaging for an already running diagnostic controller."""
import argparse, json, os, subprocess, time
from pathlib import Path
from benchmark_geometry import ROOT, read
from resource_control import require_start


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--context', required=True)
    args = p.parse_args()
    dest = ROOT/'benchmark_execution_v1'/args.context
    receipt = dest/'CPU_delivery_watch.json'
    state = dict(pid=os.getpid(), start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21], uses_GPU=False)
    while True:
        require_start()
        actual = read(dest/'status.json')
        state['parent_phase'] = actual['phase']
        receipt.write_text(json.dumps(state, indent=2))
        if (dest/'cpu_delivery/summary.json').exists():
            state['phase'] = 'CPU_delivery_verified_already_present'
            break
        if actual['phase'] == 'diagnostic_execution_complete_formal_admission_pending':
            subprocess.run(['/root/autodl-tmp/envs/drivingeditor/bin/python', str(ROOT/'code/analyze_benchmark_suite.py'), '--execution', str(dest)], check=True)
            state['phase'] = 'CPU_delivery_completed'
            break
        if actual['phase'] in ['engineering_repair_needed_preserved', 'resource_stop']:
            state['phase'] = 'parent_not_complete_outputs_preserved'
            break
        time.sleep(10)
    receipt.write_text(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
