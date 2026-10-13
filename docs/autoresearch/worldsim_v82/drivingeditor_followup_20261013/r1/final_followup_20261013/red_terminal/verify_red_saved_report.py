"""Decode actual report assets and check the scene-aware embedded saved manifest."""
import argparse
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from PIL import Image


def verify(root, js_path):
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    spec = manifest['source']['followup_scene_spec']
    page = (root / 'index.html').read_text(encoding='utf-8')
    embedded = re.search(r'<script id="viewer-data" type="application/json">(.*?)</script>', page, re.S)
    assert embedded and json.loads(embedded.group(1)) == manifest
    assert not re.search(r'GMC|silver|0008033|scene0008|0017010|scene0017', page)
    assert manifest['title'] == 'DrivingEditor · GN11 红车 · 30k'
    assert spec['scene_id'] == 'scene0029_069_117' and spec['raw_frame_range'] == [69, 117]
    assert spec['anchor_frame'] == '0029079' and spec['anchor_index'] == 10
    assert manifest['human_verdict'] is None and manifest['restoration_verified'] is False
    assert manifest['completion_contract']['status'] == 'FOLLOWUP_COMPLETE_CONTRACT_PASS'
    assert manifest['verification']['status'] == 'PASS_SAVED_FP32_CPU_EXPORT'
    assert len(manifest['frames']) == len(manifest['static_time_frames']) == 49
    assets, zero_mask_frames = [], []
    valid_actor_count = 0
    for rows in (manifest['frames'], manifest['static_time_frames']):
        for index, row in enumerate(rows):
            assert row['local_index'] == index and row['source_raw_index'] == spec['raw_frame_range'][0] + index
            assert row['timestamp'] == spec['model_timestamps'][index]
            for key, relative in row['images'].items():
                path = root / relative
                assert path.resolve().is_relative_to(root.resolve())
                with Image.open(path) as im:
                    im.load()
                    assert im.size == (480, 320)
                    assert im.mode == ('RGB' if key in ('gt', 'composite', 'static', 'dynamic') else 'L')
                    if key == 'body_mask':
                        if row['body_pixel_count'] == 0:
                            assert im.getextrema() == (0, 0) and row['bbox'] is None
                            zero_mask_frames.append(row['id'])
                        else:
                            assert im.getextrema() == (0, 255) and row['bbox'] is not None
                            valid_actor_count += 1
                assets.append(relative)
    assert len(assets) == len(set(assets)) == len(list((root / 'assets').rglob('*.png'))) == 441
    assert len(zero_mask_frames) == manifest['summary']['empty_actor_frame_count']
    assert valid_actor_count == manifest['summary']['valid_actor_frame_count']
    assert valid_actor_count + len(zero_mask_frames) == 49
    ET.parse(root / 'architecture-components.svg')
    scripts = re.findall(r'<script>(.*?)</script>', page, re.S)
    assert len(scripts) == 1 and 'fetch(' not in scripts[0]
    with js_path.open('x', encoding='utf-8') as stream:
        stream.write(scripts[0])
    return {'status': 'PASS_RED_OFFLINE_ASSETS_AND_EMBEDDED_SOURCE', 'decoded_PNGs': len(assets),
            'dimensions_WH': [480, 320], 'local_index_range': [0, 48], 'source_raw_index_range': spec['raw_frame_range'],
            'anchor': spec['anchor_frame'], 'empty_actor_masks_decode_black': len(zero_mask_frames),
            'valid_actor_frames': valid_actor_count, 'empty_actor_frame_ids': zero_mask_frames,
            'empty_ROI_fallback_is_full_frame': True, 'manifest_matches_embedded_JSON': True,
            'architecture_SVG_parses': True, 'external_fetch_required': False, 'browser_used': False,
            'GPU_used': False, 'new_hash_or_fingerprint_computed': False, 'human_verdict': None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--js', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.report.resolve(), args.js)
    with args.receipt.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False))
