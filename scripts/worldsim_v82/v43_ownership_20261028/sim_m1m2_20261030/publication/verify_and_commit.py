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
BASE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030/publication_m1m2')


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
    message.write_text('research(v43): compare truth-assisted assignment and representation repair\n\n'
        'Separate frozen-parameter routing from same-candidate position, scale and split controls on actual reconstructed Gaussians. Keep world truth independent of per-Gaussian assignment, and preserve native environment versus white-background diagnostics.\n\n'
        'Record three keyframes, four edits, independent held-out references, and two matched 200-update appearance refits. Split/refit improves some deletion errors but unedited and shifted views still regress; no successful generalization or mathematical upper bound is claimed. Disclose unequal capacity, corrected depth replay and failed pilots.\n\n'
        'Validate native contribution conservation, selection parity, checkpoint reload, depth replay RGB parity, fetched evidence hashes, 50 report browser combinations, diagram layout and publication links. Keep AutoDL running.\n')
    print(git('commit', '-F', str(message)))
    assert hashlib.sha256(unrelated.read_bytes()).hexdigest() == before
    commit = {'commit': git('rev-parse', 'HEAD'), 'staged_files': len(staged),
              'remaining_status': git('status', '--short'), 'unrelated_sha256': before, 'power_actions': 0}
    (BASE / 'commit_receipt.json').write_text(json.dumps(commit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'validation': result, 'commit': commit}, ensure_ascii=False))


if __name__ == '__main__':
    main()
