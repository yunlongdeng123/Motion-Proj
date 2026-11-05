"""CPU-only BUILD appearance references, image/mask checks, and SHA manifest."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.sparse import load_npz

RUN = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a6_appearance_formal_20261105')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    root = RUN/'inputs'
    collection = json.loads((root/'collection.json').read_text())
    collection['black_background'] = 'Outside union(object,known) is black and not an inpaint target'
    write(root/'collection.json',collection)
    receipts = []
    for raw in (106,114,129):
        per = root/f'raw{raw:03d}'
        with np.load(per/'raster_original.npz') as saved:
            known = saved['known_mask'].copy()
            support = saved['object_mask'].copy()
            unknown = saved['inpaint_mask'].copy()
            source = saved['real_rgb_source_exact'].copy()
            rgb = saved['real_rgb_uint8'].copy()
        assert np.array_equal(unknown, support & ~known)
        image = np.array(Image.open(per/'input_512.png'))
        mk = np.array(Image.open(per/'known_mask_512.png')) > 0
        mo = np.array(Image.open(per/'object_mask_512.png')) > 0
        mu = np.array(Image.open(per/'inpaint_mask_512.png')) > 0
        assert np.array_equal(mu, mo & ~mk)
        assert np.all(image[mu] == 128) and np.all(image[~(mo|mk)] == 0)
        assert not np.any(mu & mk)
        # Real visible-a6 upper appearance only: every nonblack source pixel
        # comes from its original BUILD RGB under the original a6 visible mask.
        ys,xs = np.where(known)
        bbox = np.array([xs.min(),ys.min(),xs.max()+1,ys.max()+1],np.int64)
        side = int(max(bbox[2:]-bbox[:2])+8)
        left = int(np.floor((bbox[0]+bbox[2]-side)/2))
        top = int(np.floor((bbox[1]+bbox[3]-side)/2))
        masked = np.zeros_like(rgb)
        masked[known] = rgb[known]
        reference = Image.fromarray(masked).crop((left,top,left+side,top+side)).resize((512,512),Image.Resampling.BILINEAR)
        reference.save(per/'reference_build_a6_upper_512.png')
        provenance = dict(split='BUILD',actor_id='a6',raw_frame=raw,
            held_rgb_used=False,user_reference_used=False,native_a6_rgb_used=False,
            source=f'Original observations/frames/{raw:03d}.npz rgb masked by original visible[0] (real visible a6 upper surface), tight square crop+black pad+512bilinear resize',
            source_rgb_dtype=str(source.dtype),source_rgb_shape=list(source.shape),
            known_bbox_xyxy=bbox.tolist(),reference_crop_xyxy=[left,top,left+side,top+side],
            mask_source='Original observation visible[0], without predicted native-a6 content',
            reference_sha256=sha(per/'reference_build_a6_upper_512.png'))
        write(per/'reference_provenance.json',provenance)
        A = load_npz(per/'design_A_normalized.npz')
        with np.load(per/'operator_samples.npz') as samples:
            assert A.shape == (len(samples['pixels']),18432)
            assert np.array_equal(samples['pixels'],np.flatnonzero(support))
        with np.load(per/'transform.npz') as tr:
            F=tr['original_to_generated_pixel_center'];I=tr['generated_to_original_pixel_center']
            roundtrip=float(np.max(np.abs(F@I-np.eye(3))))
            assert roundtrip<1e-10
        receipts.append(dict(raw=raw, status='PASS', mask_disjoint=True,
            gray_unknown_exact=True, black_background_exact=True,
            source_rgb_preserved_in_npz=True, original_transform_roundtrip_maxabs=roundtrip,
            operator_shape=list(A.shape)))
    write(root/'input_qa.json',dict(status='PASS',frames=receipts))
    files = [p for p in root.rglob('*') if p.is_file() and p.suffix in ('.png','.json','.npz') and p.name!='file_manifest.json']
    write(root/'file_manifest.json',dict(files=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]))
    print(json.dumps(dict(status='PASS',files=len(files),frames=receipts)),flush=True)


if __name__=='__main__':
    main()
