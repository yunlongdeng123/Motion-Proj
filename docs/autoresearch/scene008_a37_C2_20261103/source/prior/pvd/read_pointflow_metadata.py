"""Read only the official PointFlow ZIP directory over HTTP ranges."""
import io
import json
from pathlib import Path
import time
import zipfile
import requests

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a37_C2_20261103_r1/source/prior/pvd')
SIZE = 7650585133
URL = 'https://drive.usercontent.google.com/download?id=1sw9gdk_igiyyt7MqALyxZhRrtPvAn0sX&export=download&confirm=t'


class RangeFile(io.RawIOBase):
    def __init__(self):
        self.position = 0
        self.bytes_fetched = 0
        self.session = requests.Session()
        self.session.trust_env = False
        self.cache = {}

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else SIZE + offset
        return self.position

    def read(self, n=-1):
        n = SIZE-self.position if n < 0 else min(n, SIZE-self.position)
        if n == 0: return b''
        if n > 32*1024**2: raise RuntimeError('Metadata-only reader refuses large range')
        key = (self.position, n)
        if key not in self.cache:
            with self.session.get(URL, headers={'Range':f'bytes={self.position}-{self.position+n-1}'},
                  stream=True, timeout=(8,45), proxies={'https':'http://127.0.0.1:45957'}) as response:
                if response.status_code != 206:
                    raise RuntimeError(f'ZIP metadata range failed HTTP {response.status_code}')
                data = response.content
            if len(data) != n: raise RuntimeError(f'Range length mismatch {len(data)} != {n}')
            self.cache[key] = data
            self.bytes_fetched += n
        result = self.cache[key]
        self.position += n
        return result


if __name__ == '__main__':
    started=time.monotonic()
    remote=RangeFile()
    archive=zipfile.ZipFile(remote)
    car=[info for info in archive.infolist() if '/02958343/train/' in info.filename and info.filename.endswith('.npy')]
    payload=dict(official_source='https://github.com/stevenygd/PointFlow#dataset', drive_id='1sw9gdk_igiyyt7MqALyxZhRrtPvAn0sX',archive_bytes=SIZE,
                 metadata_bytes_fetched=remote.bytes_fetched,metadata_wall_seconds=time.monotonic()-started,
                 car_train_count=len(car),car_train_compressed_bytes=sum(i.compress_size for i in car),
                 car_train_uncompressed_bytes=sum(i.file_size for i in car),
                 car_entries=[dict(name=i.filename,header_offset=i.header_offset,compressed_bytes=i.compress_size,
                                   bytes=i.file_size,compression=i.compress_type,crc32=i.CRC) for i in car])
    (ROOT/'pointflow_car_training_directory.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({key:value for key,value in payload.items() if key!='car_entries'}),flush=True)
