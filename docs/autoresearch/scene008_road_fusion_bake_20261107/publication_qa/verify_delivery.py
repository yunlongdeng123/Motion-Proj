"""安全解包自己的交付目录，核对源文件并检查离线图集。"""
from pathlib import Path
import argparse,hashlib,json,tarfile
from html.parser import HTMLParser
from PIL import Image
BASE=Path(__file__).resolve().parents[3]/'outputs/scene008_road_fusion_bake'
ap=argparse.ArgumentParser();ap.add_argument('--verify-existing',action='store_true');args=ap.parse_args()
receipt=json.loads((BASE/'packet_receipt.json').read_text(encoding='utf-8'))
archive=BASE/'delivery.tar.gz';assert archive.stat().st_size==receipt['bytes']
assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
out=(BASE/'delivery').resolve()
if out.exists() and not args.verify_existing:raise FileExistsError('保留现有交付，不覆盖：'+str(out))
if not out.exists():
    out.mkdir()
    with tarfile.open(archive,'r:gz') as tar:
        for m in tar.getmembers():
            parts=Path(m.name).parts;assert parts[0]==receipt['prefix']
            if len(parts)==1:continue
            target=out.joinpath(*parts[1:]).resolve();assert target.is_relative_to(out)
            assert not m.issym() and not m.islnk()
            if m.isdir():target.mkdir(parents=True,exist_ok=True);continue
            assert m.isfile();target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(m) as src,target.open('wb') as dest:
                while data:=src.read(2**20):dest.write(data)
manifest=json.loads((out/'publication_manifest.json').read_text(encoding='utf-8'));issues=[]
for rel,info in manifest.items():
    p=out/rel
    if not p.is_file() or p.stat().st_size!=info['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=info['sha256']:issues.append(rel)
assert not issues,issues
decoded=0
for p in out.rglob('*.png'):
    with Image.open(p) as im:im.load();assert min(im.size)>0
    decoded+=1
class LinkParser(HTMLParser):
    def __init__(self):super().__init__();self.urls=[]
    def handle_starttag(self,tag,attrs):
        self.urls.extend(v for k,v in attrs if k in ('href','src') and v)
text=(out/'index.html').read_text(encoding='utf-8');parser=LinkParser();parser.feed(text);local=[]
for url in parser.urls:
    if url.startswith(('http:','https:','#','data:')):continue
    local.append(url);assert (out/url.split('#')[0]).is_file(),url
for raw in range(89,138):
    for state in ('keep','delete'):
        for variant in ('baseline','direct','fusion','edgeblend'):
            for folder in ('frames','crops'):assert (out/f'{folder}/{variant}/{raw}_{state}.png').is_file()
    assert (out/f'frames/observed/{raw}.png').is_file()
    for key in ('direct_rgb','fusion_rgb','edgeblend_rgb','domain','weight'):assert (out/f'frames/targets/{key}_{raw}.png').is_file()
result=dict(archive_SHA_exact=True,manifest_files_exact=len(manifest),PNG_decodes=decoded,
    static_references=len(local),all49_gallery_states_present=True,report=str(out/'index.html'),AutoDL='KEEP_ON')
(BASE/'retrieval_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
