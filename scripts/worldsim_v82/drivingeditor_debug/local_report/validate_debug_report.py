"""Check the published local report references; does not claim UI playback."""
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

OUT = Path(__file__).resolve().parents[2] / 'outputs/drivingeditor_vgpu_debug'


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('href', 'src', 'poster') and value:
                self.links.append(value)


page = OUT / 'index.html'
parser = References()
parser.feed(page.read_text(encoding='utf8'))
result = []
for link in parser.links:
    parts = urlsplit(link)
    if parts.scheme not in ('', 'file') or not parts.path:
        continue
    path = unquote(parts.path)
    if parts.scheme == 'file':
        if path.startswith('/') and len(path) > 2 and path[2] == ':':
            path = path[1:]
        target = Path(path)
    else:
        target = (OUT / path).resolve()
    result.append({'reference': link, 'exists': target.is_file(), 'bytes': target.stat().st_size if target.is_file() else None})
evidence = {'status': 'pass' if all(r['exists'] and r['bytes'] > 0 for r in result) else 'fail', 'references': result, 'browser_playback_checked': False}
(OUT / 'local_reference_validation.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf8')
manifest_path = OUT / 'report_manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf8'))
manifest['files'] = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name != 'report_manifest.json'}
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps({'status': evidence['status'], 'count': len(result), 'bad_references': [r for r in result if not r['exists'] or not r['bytes']], 'browser_playback_checked': False}))
if evidence['status'] != 'pass':
    raise SystemExit(1)
