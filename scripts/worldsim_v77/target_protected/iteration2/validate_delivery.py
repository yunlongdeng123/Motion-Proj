"""静态核验交付：资源可用、判据隔离、人工留空；不替代浏览器播放验证。"""
import argparse
import json
import re
import subprocess
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('src', 'href') and value:
                parsed = urlsplit(value)
                if not parsed.scheme and parsed.path:
                    self.paths.append(unquote(parsed.path))


def main(root, node, out):
    manifest = json.loads((root / 'training_review_manifest.json').read_text(encoding='utf-8'))
    clips = manifest['clips']
    assert len(clips) == len({c['case_id'] for c in clips}) == 37
    assert Counter(c['review']['synthetic_status'] for c in clips) == {'pass': 27, 'uncertain': 9, 'reject': 1}
    assert manifest['run_id'] == 'r2' and manifest['human_scoring_scope'] == 'training_input_five_checks_v2'
    assert manifest['training_ready'] == 0
    paths = []
    videos = []
    images = []
    frame_count = 0
    human_frames = 0
    for c in clips:
        assert c['original_ai_score'] == 1 and c['human_verdict'] is None and not c['training_ready']
        assert c['frame_count'] == len(c['preview_frames'])
        assert c['review']['reviewed_frames'] == [0, c['frame_count'] // 2, c['frame_count'] - 1]
        frame_count += c['frame_count']
        human_frames += c['frame_count'] if c['review']['synthetic_status'] == 'pass' else 0
        videos.extend(c['videos'].values())
        images.extend(c['contacts'])
        for f in c['preview_frames']:
            images.extend(f[k] for k in ('gt', 'input', 'labels', 'condition'))
    assert frame_count == 630 and len(videos) == 148
    assert next(c for c in clips if c['case_id'] == 'D006')['review']['synthetic_status'] == 'reject'
    paths.extend(videos + images)
    out.parent.mkdir(parents=True, exist_ok=True)
    scripts_checked = 0
    for page in ('index.html', 'data_review.html'):
        text = (root / page).read_text(encoding='utf-8')
        assert '<svg' in text
        parser = References()
        parser.feed(text)
        paths.extend(parser.paths)
        for i, code in enumerate(re.findall(r'<script[^>]*>(.*?)</script>', text, flags=re.S)):
            if not code.strip():
                continue
            script = out.parent / f'{Path(page).stem}_{i}.js'
            script.write_text(code, encoding='utf-8')
            subprocess.run([str(node), '--check', str(script)], check=True, capture_output=True)
            scripts_checked += 1
        if page == 'data_review.html':
            assert "KEY='v77-training-input-frames-r2'" in text
            assert "KEY='v77-target-protected-synthetic-frames-v1'" not in text
            assert 'V77_training_input_r2_frame_review.json' in text
            assert '车身颜色、材质、受光、贴片感本身不扣分' in text
    missing = [p for p in paths if not (root / p).is_file() or (root / p).stat().st_size == 0]
    assert not missing, missing[:20]
    data = {'scope': 'static delivery validation; browser interaction not tested', 'cases': 37,
            'sampled_visual_review': 'first/middle/last; not full-frame human review',
            'engineering_frames': frame_count, 'human_candidate_frames': human_frames,
            'video_links': len(videos), 'image_links': len(images), 'missing_resources': missing,
            'javascript_syntax_checks': scripts_checked, 'human_verdicts_all_null': True,
            'new_review_storage_isolated': True, 'old_video_decode_not_repeated': True,
            'local_delivery': str(root.resolve())}
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(data, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--node', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.root, a.node, a.out)
