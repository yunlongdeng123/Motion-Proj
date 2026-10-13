"""Preserve failed metadata; derive paper metrics from the actual event metrics directory."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
collector = (HERE / 'collect_red_terminal_cpu.py').read_text(encoding='utf-8')
collector = collector.replace("BASE / 'metadata/red_terminal_20261013_1056'", "BASE / 'metadata/red_terminal_20261013_1056_attempt2'")
collector = collector.replace("copy_new(RUN / ('eval_%d/paper_region_metrics.json' % step)", "copy_new(Path(event['metrics']).parent / 'paper_region_metrics.json'")
collector = collector.replace("'new_hash_or_fingerprint_computed': False, 'human_verdict': None}", "'new_hash_or_fingerprint_computed': False, 'human_verdict': None,\n         'prior_tooling_attempt_preserved': str(BASE / 'metadata/red_terminal_20261013_1056')}")
launch = (HERE / 'launch_red_cpu_collector.py').read_text(encoding='utf-8')
launch = launch.replace("BASE / 'metadata/red_terminal_20261013_1056'", "BASE / 'metadata/red_terminal_20261013_1056_attempt2'")
launch = launch.replace('tools/red_terminal_20261013/collect_red_terminal_cpu.py', 'tools/red_terminal_20261013/collect_red_terminal_cpu_attempt2.py')
for name, body in (('collect_red_terminal_cpu_attempt2.py', collector), ('launch_red_cpu_collector_attempt2.py', launch)):
    compile(body, name, 'exec')
    with (HERE / name).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(body)
print('Attempt2 new tool names/metadata; actual event-derived metrics directory; syntax PASS')
