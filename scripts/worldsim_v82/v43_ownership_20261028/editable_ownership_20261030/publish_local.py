"""Add report navigation only; preserve all existing R2/edit/white data."""
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[2]
site = root / 'outputs/GS_Scene_Diagnostics_20261027'
ctx = '14705303724557273004_3105_000_3125_000'
report = root / 'outputs/V43_Ownership_20261027/gpu_v43/editable_ownership_20261030'
assert (report / 'index.html').is_file()
backup = Path(__file__).parent / 'entry_backup'
backup.mkdir(exist_ok=True)
checks = []
for label, path, prefix in [('root', site / 'index.html', '../'), ('scene1', site / ctx / 'index.html', '../../')]:
    original = path.read_text(encoding='utf8')
    link = prefix + 'V43_Ownership_20261027/gpu_v43/editable_ownership_20261030/index.html'
    marker = '<!-- editable-ownership-20261030 -->'
    if marker not in original:
        assert '<nav>' in original
        shutil.copy2(path, backup / (label + '.html'))
        updated = original.replace('<nav>', '<nav>' + marker + '<a href="' + link + '">V43 身份与表示修复 · 场景1新对照</a>', 1)
        path.write_text(updated, encoding='utf8')
    assert (path.parent / link).resolve() == (report / 'index.html').resolve()
    checks.append(dict(path=str(path), link=link, link_exists=True,
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(report / 'navigation_publication.json').write_text(json.dumps(dict(entries=checks,
    data_mutations=False, R2_or_white_outputs_replaced=False, added_navigation_only=True), ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps(checks, ensure_ascii=False))
