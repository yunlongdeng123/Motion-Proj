"""仅校验和解包交付文件；不加载任何研究模型。"""
from pathlib import Path
import hashlib
import json
import shutil
import tarfile
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2] / 'outputs/scene008_road_r0_v44'
PACKET = ROOT / 'delivery.tar.gz'
EXPECTED = '12f51b14602fd928b02fb3a3f3b449d873f4d8b6840ae6e058e584bb22470d6e'
digest = hashlib.sha256(PACKET.read_bytes()).hexdigest()
assert digest == EXPECTED, (digest, EXPECTED)
DEST = (ROOT / 'delivery').resolve()
assert not DEST.exists(), '保留已解包现场，不覆盖'
DEST.mkdir()
with tarfile.open(PACKET, 'r:gz') as tar:
    for member in tar:
        parts = Path(member.name.replace('\\', '/')).parts
        assert parts[0] == 'delivery_final' and '..' not in parts
        rel = Path(*parts[1:])
        out = (DEST / rel).resolve()
        assert out == DEST or DEST in out.parents
        if member.isdir():
            out.mkdir(parents=True, exist_ok=True)
        else:
            assert member.isfile(), member.name
            out.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, out.open('wb') as target:
                shutil.copyfileobj(source, target)
manifest = json.loads((DEST / 'publication_manifest.json').read_text(encoding='utf-8'))
for rel, item in manifest.items():
    data = (DEST / rel).read_bytes()
    assert len(data) == item['bytes'], rel
    assert hashlib.sha256(data).hexdigest() == item['sha256'], rel
actual = {p.relative_to(DEST).as_posix() for p in DEST.rglob('*') if p.is_file()}
assert actual == set(manifest) | {'publication_manifest.json'}

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs = []
    def handle_starttag(self, tag, attrs):
        self.refs.extend(v for k, v in attrs if k in ('href', 'src') and v)

parser = Links()
html = (DEST / 'index.html').read_text(encoding='utf-8')
assert '@@' not in html
parser.feed(html)
broken = []
for ref in parser.refs:
    url = urlsplit(ref)
    if url.scheme or not url.path:
        continue
    if not (DEST / unquote(url.path)).exists():
        broken.append(ref)
assert not broken, broken
from PIL import Image
pngs = list(DEST.rglob('*.png'))
for p in pngs:
    with Image.open(p) as im:
        im.verify()
for name in ['frame_index.json', 'wide_frame_index.json']:
    obj = json.loads((DEST / name).read_text())
    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, str):
            assert (DEST / value).is_file(), value
    walk(obj)
receipt = dict(packet_sha256=digest, manifest_entries=len(manifest), files=len(actual),
               png_decodes=len(pngs), static_links_checked=len(parser.refs), broken_links=broken,
               extraction='regular_files_only, all resolved paths within delivery',
               scientific_result='12/13; RESEARCH_CANDIDATE_NOT_ADOPTED')
(ROOT / 'retrieval_verification.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(receipt, ensure_ascii=False))
