"""只裁剪 protobuf 字段，不解 JPEG / range image；保持每条原CRC收据。"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import ctypes,json,pathlib,struct,time,argparse
BASE=pathlib.Path('/root/autodl-tmp/codex_tmp/v43_actor_diagnostics_20261101')
lib=ctypes.CDLL('/root/autodl-tmp/drivingeditor_repro/campaign/scene_search_mirror/libpreview_crc32c.so')
lib.preview_crc32c.argtypes=[ctypes.c_void_p,ctypes.c_size_t];lib.preview_crc32c.restype=ctypes.c_uint32
assert lib.preview_crc32c(b'123456789',9)==0xe3069283
def crc(x):
    c=lib.preview_crc32c(x,len(x));return (((c>>15)|(c<<17))+0xa282ead8)&0xffffffff
def varint(b,i):
    n=0;s=0
    while True:
        t=b[i];i+=1;n|=(t&127)<<s
        if t<128:return n,i
        s+=7
        if s>63:raise ValueError('varint')
def vint(n):
    out=bytearray()
    while n>127:out.append((n&127)|128);n>>=7
    out.append(n);return bytes(out)
def field(n,b):return vint((n<<3)|2)+vint(len(b))+b
def fields(b):
    i=0
    while i<len(b):
        start=i;tag,i=varint(b,i);wire=tag&7
        if wire==2:
            size,i=varint(b,i);value=b[i:i+size];i+=size
        elif wire==0:value,i=varint(b,i)
        elif wire==1:value=b[i:i+8];i+=8
        elif wire==5:value=b[i:i+4];i+=4
        else:raise ValueError(('wire',wire))
        if i>len(b):raise ValueError('truncated field')
        yield tag>>3,wire,b[start:i],value
def image_front(b):
    rows=list(fields(b));name=next(v for n,w,r,v in rows if n==1)
    return b''.join(r for n,w,r,v in rows if n in (1,2,3,4,5,6,7,8)) if name==1 else None
def laser(b):
    out=[]
    for n,w,r,v in fields(b):
        if n==1:out.append(r)
        elif n in (2,3):out.append(field(n,b''.join(rr for nn,ww,rr,vv in fields(v) if nn in (2,4))))
    return b''.join(out)
def select(payload):
    front=[];lidar=[];context=None;timestamp=None
    for n,w,r,v in fields(payload):
        if n in (1,2,3):
            front.append(r);lidar.append(r)
            if n==1:context=next(vv.decode() for nn,ww,rr,vv in fields(v) if nn==1)
            if n==2:timestamp=v
        elif n==4:
            im=image_front(v)
            if im:front.append(field(4,im))
        elif n==5:lidar.append(field(5,laser(v)))
        elif n in (6,9):front.append(r)
    return b''.join(front),b''.join(lidar),context,timestamp
def write_record(f,p):
    header=struct.pack('<Q',len(p));f.write(header);f.write(struct.pack('<I',crc(header)));f.write(p);f.write(struct.pack('<I',crc(p)))
def run(scene_key):
    cfg=json.loads((BASE/'scenes.json').read_text());scene=next(x for x in cfg['scenes'] if x['scene_key']==scene_key)
    inv=next(x for x in json.loads((BASE/'raw_inventory.json').read_text()) if x['scene_key']==scene_key)
    out=BASE/'selected'/scene_key;out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():return
    start=time.monotonic();rows=[];source=pathlib.Path(inv['path']);before=source.stat()
    with source.open('rb') as src,(out/'front.tfrecord.tmp').open('xb') as front,(out/'lidar.tfrecord.tmp').open('xb') as lidar:
        while True:
            offset=src.tell();h=src.read(12)
            if not h:break
            if len(h)!=12:raise ValueError('short header')
            size,c=struct.unpack('<QI',h)
            if crc(h[:8])!=c or not 0<size<96*1024**2:raise ValueError('header CRC')
            payload=src.read(size);tail=src.read(4)
            if len(tail)!=4 or len(payload)!=size or crc(payload)!=struct.unpack('<I',tail)[0]:raise ValueError('payload CRC')
            a,b,context,timestamp=select(payload)
            if context!=scene['context_id']:raise ValueError('wrong context')
            write_record(front,a);write_record(lidar,b)
            rows.append(dict(raw=len(rows),timestamp_micros=timestamp,original_offset=offset,original_payload_bytes=size,
                original_length_masked_crc32c=c,original_payload_masked_crc32c=struct.unpack('<I',tail)[0],
                selected_front_payload_bytes=len(a),selected_lidar_payload_bytes=len(b),front_masked_crc32c=crc(a),lidar_masked_crc32c=crc(b)))
    after=source.stat()
    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    assert len(rows)==scene['expected_frames']
    for n in ('front','lidar'):(out/(n+'.tfrecord.tmp')).rename(out/(n+'.tfrecord'))
    (out/'source_receipts.json').write_text(json.dumps(rows,separators=(',',':')))
    summary=dict(scene_key=scene_key,context_id=scene['context_id'],frames=len(rows),source_path=str(source),source_bytes=before.st_size,
        front_bytes=(out/'front.tfrecord').stat().st_size,lidar_bytes=(out/'lidar.tfrecord').stat().st_size,wall_seconds=time.monotonic()-start,
        exact_JPEG_and_compressed_range_payloads=True,source_CRC_every_record='passed',source_unchanged_size_mtime=True,
        role='derived_protobuf_field_subset_not_complete_original_TFRecord',CPU_threads=1,GPU_queries=0)
    (out/'manifest.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene');a=p.parse_args()
    cfg=json.loads((BASE/'scenes.json').read_text());order=['waymo-train-212','waymo-train-537','waymo-train-564','waymo-train-008']
    order+= [s['scene_key'] for s in cfg['scenes'] if s['scene_key'] not in order and s['scene_key']!='waymo-val-035']
    for key in ([a.scene] if a.scene else order):run(key)
