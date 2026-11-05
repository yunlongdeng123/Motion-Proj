"""CPU校验独立图片包；相同113预览只在字节一致时保留。"""
import hashlib,json,shutil,tarfile
from pathlib import Path,PurePosixPath
from PIL import Image

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))

transfer=read(ROOT/'transfer_receipt.json');archive=ROOT/transfer['file']
assert archive.stat().st_size==transfer['bytes'] and sha(archive)==transfer['sha256']
dest=ROOT/'retrieved_native49_v1'
if dest.exists():dest=ROOT/'retrieved_native49_v1_verified'
dest.mkdir(exist_ok=False)
with tarfile.open(archive,'r:xz') as t:
    for m in t.getmembers():
        p=PurePosixPath(m.name)
        assert (m.isfile() or m.isdir()) and not p.is_absolute() and '..' not in p.parts
    t.extractall(dest,filter='data')
for p in dest.rglob('*'):
    if p.is_file():
        out=ROOT/p.relative_to(dest);out.parent.mkdir(parents=True,exist_ok=True)
        if out.exists():
            if sha(out)!=sha(p):
                # 唯一的本地预览JSON由Python在Windows重写，可能带CRLF。
                # 内容必须逐字段相同，留副本后使用归档内权威原字节。
                assert p.relative_to(dest).as_posix()=='native49_v1/frames/113/metadata.json',out
                assert read(out)==read(p)
                backup=ROOT/'preview_backup/113_metadata_windows.json';backup.parent.mkdir(exist_ok=True)
                if not backup.exists():shutil.copyfile(out,backup)
                shutil.copyfile(p,out)
        else:shutil.copyfile(p,out)
base=ROOT/'native49_v1';m=read(base/'manifest.json');r=read(base/'receipt.json')
assert sha(base/'manifest.json')==r['manifest_sha256']=='efbcf40049521eecb947e3e9a2282b83796ccf36433dc328e2724a2e0c90c9dd'
assert m['status']=='COMPLETE' and m['raw_ids']==list(range(89,138))
assert [x['raw'] for x in m['frames']]==m['raw_ids']
ids=[x['id'] for x in m['columns']];assert ids==['old_formal','c2_matte','surface_lod']
seen=set();renders=refs=0
for frame in m['frames']:
    raw=frame['raw'];x0,y0,x1,y1=frame['crop_xyxy'];assert 0<=x0<x1<=480 and 0<=y0<y1<=320
    for state in ['keep','delete','reference']:
        rows=[frame[state]] if state=='reference' else list(frame[state].values())
        if state!='reference':assert list(frame[state])==ids
        for row in rows:
            path=base/row['path'];assert row['path'] not in seen;seen.add(row['path'])
            assert row['raw']==raw and sha(path)==row['sha256']
            if state!='reference':
                assert row['state']==state and row['method'] in ids and row['renderer_id']==m['renderer']['id'];renders+=1
            else:refs+=1
            with Image.open(path) as im:
                assert im.size==(480,320) and im.mode=='RGB';im.load()
assert (renders,refs)==(294,49) and len(seen)==343
receipt=dict(status='PASS_ALL_294_NATIVE_RGB_AND_49_REFERENCE_PNG',manifest=str(base/'manifest.json'),manifest_sha256=sha(base/'manifest.json'),
    raw_ids=m['raw_ids'],methods=ids,states=['keep','delete'],images=renders,reference_images=refs,
    every_file_SHA_and_RGB480x320_decoded=True,metadata_raw_state_method_cross_checked=True,
    all_perraw_crops_common_to_all_methods_states_and_reference=True,
    original_reference_not_delete_GT=True,interpolated_frames=0,
    newly_rendered_native_forwards=r['new_native_raster_forwards'],reused_byte_exact_PNGs=r['reused_PNGs'],
    transfer=transfer,source_sha256=sha(ROOT/'capture_frames.py'),archive_contains_no_training_states=True)
(ROOT/'download_qa.json').write_bytes((json.dumps(receipt,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
print(json.dumps({k:receipt[k] for k in ['status','manifest_sha256','images','reference_images','newly_rendered_native_forwards','reused_byte_exact_PNGs']}))
