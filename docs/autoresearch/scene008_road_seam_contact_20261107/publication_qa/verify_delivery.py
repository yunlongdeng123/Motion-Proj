"""校验取回包与离线静态资源，不运行任何研究模型。"""
from pathlib import Path, PurePosixPath
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
import hashlib
import json
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[2]/'outputs/scene008_road_seam_contact'
receipt=json.loads((ROOT/'packet_receipt.json').read_text(encoding='utf-8'))
packet=ROOT/'delivery.tar.gz'
assert packet.stat().st_size==receipt['bytes']
assert hashlib.sha256(packet.read_bytes()).hexdigest()==receipt['sha256']
dest=(ROOT/'delivery').resolve();assert not dest.exists(),'不覆盖已取回现场'
dest.mkdir()
with tarfile.open(packet,'r:gz') as tar:
    for member in tar:
        parts=PurePosixPath(member.name.replace('\\','/')).parts
        assert parts and parts[0]==receipt['archive_prefix'] and '..' not in parts
        out=(dest/Path(*parts[1:])).resolve()
        assert out==dest or dest in out.parents
        if member.isdir():out.mkdir(parents=True,exist_ok=True)
        else:
            assert member.isfile(),member.name
            out.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as stream,out.open('wb') as output:shutil.copyfileobj(stream,output)
manifest=json.loads((dest/'publication_manifest.json').read_text(encoding='utf-8'))
for rel,item in manifest.items():
    data=(dest/rel).read_bytes()
    assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],rel
actual={p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file()}
assert actual==set(manifest)|{'publication_manifest.json'}
assert len(actual)==receipt['files']
class References(HTMLParser):
    def __init__(self):super().__init__();self.refs=[]
    def handle_starttag(self,tag,attrs):self.refs.extend(v for k,v in attrs if k in ['href','src'] and v)
parser=References();html=(dest/'index.html').read_text(encoding='utf-8');parser.feed(html)
assert '@@' not in html
broken=[]
for ref in parser.refs:
    u=urlsplit(ref)
    if not u.scheme and u.path and not (dest/unquote(u.path)).exists():broken.append(ref)
assert not broken,broken
d=json.loads((dest/'report_data.json').read_text(encoding='utf-8'))
image_paths=[]
for variant in d['variants']:image_paths.extend(variant['images'].values())
image_paths.extend(d['observed'].values())
assert all((dest/p).is_file() for p in image_paths)
from PIL import Image
pngs=list(dest.rglob('*.png'))
for p in pngs:
    with Image.open(p) as im:im.verify()
qa=dict(packet_sha256=receipt['sha256'],files=len(actual),manifest_entries=len(manifest),PNG_decodes=len(pngs),
    gallery_images=len(image_paths),static_refs=len(parser.refs),broken_static_refs=broken,
    extraction='regular files only; resolved paths inside delivery',
    primary=d['main_candidate'],boundary_material_selected=d['boundary_material_selected'],AutoDL='KEEP_ON')
(ROOT/'retrieval_verification.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(qa))
