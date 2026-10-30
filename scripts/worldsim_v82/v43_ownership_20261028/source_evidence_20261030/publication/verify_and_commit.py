"""Verify bounded remote publication, then commit it; no experiment or power action."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import hashlib
import json
import re
import subprocess
import tarfile
import xml.etree.ElementTree as ET

REPO = Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
BASE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030/publication_source_evidence')


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], stderr=subprocess.STDOUT).decode().strip()


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(v for k, v in attrs if k in ('href', 'src') and v)


def main():
    with tarfile.open(BASE / 'publication.tar.gz', 'r:gz') as archive:
        manifest = json.loads(archive.extractfile('PUBLICATION_MANIFEST.json').read())
    receipt = json.loads((BASE / 'publication_receipt.json').read_text())
    links, broken, syntax = 0, [], 0
    for name in manifest['files']:
        path = REPO / name
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == manifest['sha256'][name]
        urls = []
        if path.suffix == '.html':
            parser = Links()
            parser.feed(data.decode('utf-8-sig'))
            urls = parser.links
        elif path.suffix == '.md' and name.startswith('docs/'):
            urls = re.findall(r'\]\(([^)]+)\)', data.decode('utf-8-sig'))
        elif path.suffix == '.py':
            compile(data.decode('utf-8-sig'), name, 'exec')
            syntax += 1
        elif path.suffix == '.json':
            json.loads(data)
        elif path.suffix == '.svg':
            ET.fromstring(data)
        for url in urls:
            u = urlsplit(url)
            if u.scheme or not u.path:
                continue
            links += 1
            if not (path.parent / unquote(u.path)).exists():
                broken.append([name, url])
    result = {'passed': not broken, 'files_sha256': len(manifest['files']), 'python_syntax': syntax,
              'relative_links': links, 'broken_links': broken, 'power_actions': 0}
    (BASE / 'validation_receipt.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    assert not broken, broken
    assert git('branch', '--show-current') == 'research/worldsim-v8.2-drivingeditor-v42'
    staged = set(git('diff', '--cached', '--name-only', '-z').strip('\0').split('\0'))
    assert staged and staged <= set(receipt['stage_allowlist'])
    assert not git('diff', '--cached', '--check')
    unrelated = REPO / receipt['known_unrelated_dirty_path_preserved']
    before = hashlib.sha256(unrelated.read_bytes()).hexdigest()
    message = BASE / 'commit_message.txt'
    message.write_text('research(v43): trace exposed residuals and control observation evidence\n\nSeparate post-edit per-row sources from pre-edit contribution coverage. Replay all static rows across original observations, probe the frozen old model under new queries, and preserve environmental contribution diagnostics.\n\nReconstruct the same world from complementary legal observations with shared physical gauge, native schedule and a fresh 10K budget. Record real 600-second smoke, full-state restore, common-query edits, held-out views, visibility strata and fetched hashes.\n\nCorrect earlier M1/M2 interpretation: selected-rule and heuristic-repair failures do not establish accurate-assignment or representation limits. Preserve old artifacts and unrelated changes; keep AutoDL running.\n')
    print(git('commit', '-F', str(message)))
    assert hashlib.sha256(unrelated.read_bytes()).hexdigest() == before
    commit = {'commit': git('rev-parse', 'HEAD'), 'staged_files': len(staged),
              'remaining_status': git('status', '--short'), 'unrelated_sha256': before, 'power_actions': 0}
    (BASE / 'commit_receipt.json').write_text(json.dumps(commit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'validation': result, 'commit': commit}, ensure_ascii=False))


if __name__ == '__main__':
    main()
