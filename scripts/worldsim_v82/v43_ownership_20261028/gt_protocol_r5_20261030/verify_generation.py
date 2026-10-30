"""对整套文档生成做前后字节比较；不执行任何重建或训练。"""
from pathlib import Path
import json
import subprocess
import sys

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / 'outputs/V43_Ownership_20261027'
NAMES = ['V43_DESIGN.md', 'index.html', 'architecture.svg', 'architecture.png',
         'data_architecture.svg', 'data_architecture.png', 'architecture_validation.json',
         'ownership_record_contract.json', 'mechanism_experiment_plan.json',
         'direction_state.json', 'design_revision_validation.json']
before = {name: (OUT / name).read_bytes() for name in NAMES}
commands = [[sys.executable, 'work/v43_ownership_20261027/build_v43.py'],
            ['node', 'work/v43_ownership_20261027/render_architecture.cjs'],
            [sys.executable, 'work/v43_joint_direction_20261030/validate_revision.py']]
for command in commands:
    result = subprocess.run(command, cwd=BASE, capture_output=True)
    if result.returncode:
        sys.stdout.buffer.write(result.stdout)
        sys.stderr.buffer.write(result.stderr)
        raise SystemExit(result.returncode)
changed = [name for name in NAMES if before[name] != (OUT / name).read_bytes()]
assert not changed, changed
receipt = {'architecture_revision': 5, 'passed': True,
           'comparison': 'before_after_byte_equality_without_hashes',
           'files_checked': NAMES, 'changed_after_full_regeneration': changed,
           'document_contract_and_both_diagram_generators_repeated': True,
           'new_experiment_runs': 0, 'new_training_updates': 0, 'new_scene_renders': 0}
Path(__file__).with_name('generator_idempotency.json').write_text(
    json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
print(json.dumps(receipt, ensure_ascii=True))
