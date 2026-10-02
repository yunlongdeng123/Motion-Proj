"""检查诊断页链接、逐帧图片与全部视频实际解码，不做模型实验。"""
import argparse
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('review_dir', type=Path)
    ap.add_argument('--node', required=True)
    a = ap.parse_args()
    root = a.review_dir.resolve()
    page = (root / 'index.html').read_text(encoding='utf-8')
    manifest = json.loads((root / 'diagnostic_manifest.json').read_text(encoding='utf-8'))
    human = json.loads((root / 'human_review.json').read_text(encoding='utf-8'))
    assert len(manifest['cases']) == 2 and len(human['cases']) == 14
    embedded = re.search(r'<script id="manifest" type="application/json">(.*?)</script>', page, re.S).group(1)
    assert json.loads(embedded) == manifest
    missing = []
    for link in re.findall(r'(?:href|src)="([^"]+)"', page):
        if link.startswith(('#', 'http:', 'https:')) or link == 'delivery_validation.json':
            continue
        if not (root / unquote(link.split('#')[0])).is_file():
            missing.append(link)
    assert not missing, missing
    roles = list(manifest['cases'][0]['videos'])
    frame_images = []
    videos = set()
    for case in manifest['cases']:
        videos.update(case['videos'].values())
        videos.update(case['native'].values())
        for frame in range(10):
            for role in roles:
                path = root / case['frame_pattern'].format(i=f'{frame:03d}', role=role)
                with Image.open(path) as im:
                    im.verify()
                frame_images.append(str(path.relative_to(root)))
    contacts = list((root / 'contacts').glob('*.jpg'))
    for path in contacts:
        with Image.open(path) as im:
            im.verify()
    assert len(videos) == 24 and len(frame_images) == 140 and len(contacts) == 7
    def decode(rel):
        path = root / rel
        result = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,nb_read_frames,r_frame_rate', '-of', 'json', str(path)], capture_output=True, text=True, check=True)
        meta = json.loads(result.stdout)['streams'][0]
        assert meta['width'] == 1024 and meta['height'] == 576 and int(meta['nb_read_frames']) == 10, (rel, meta)
        subprocess.run(['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(path), '-f', 'null', '-'], capture_output=True, check=True)
        return {'path': rel, **meta, 'actual_ffmpeg_decode': True}
    with ThreadPoolExecutor(max_workers=4) as pool:
        video_checks = list(pool.map(decode, sorted(videos)))
    code = re.findall(r'<script>(.*?)</script>', page, re.S)
    for script in code:
        subprocess.run([a.node, '--check', '-'], input=script, text=True, capture_output=True, check=True)
    result = {'stage': 'passed', 'cases': 2, 'human_rows_preserved': 14,
              'videos': len(videos), 'actual_decoded_frames': sum(int(v['nb_read_frames']) for v in video_checks),
              'frame_images': len(frame_images), 'contact_images': len(contacts),
              'missing_links': [], 'JS_syntax_passed': True,
              'browser_playback_verified': False, 'video_checks': video_checks}
    (root / 'delivery_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'video_checks'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
