"""审核包静态完整性检查；不替代浏览器交互与人工视觉审核。"""
import argparse
import json
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path


class Resources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ('href', 'src') and value and '${' not in value:
                if not value.startswith(('http:', 'https:', '#', 'data:')):
                    self.links.append(value)


def main(root, sample, node, output):
    manifest = json.loads((root / 'review_manifest.json').read_text(encoding='utf-8'))
    frames = {c['case_id']: c['frame'] for c in json.loads(sample.read_text(encoding='utf-8'))['clips']}
    assert manifest['run_id'] == 'r3'
    assert manifest['human_scoring_scope'] == 'training_input_five_checks_r3'
    assert manifest['training_ready'] == 0
    clips = manifest['clips']
    assert len(clips) == len({c['case_id'] for c in clips}) == 24
    assert {c['case_id'] for c in clips} == set(frames)
    video_paths, preview_paths, contacts = [], [], []
    for c in clips:
        assert c['human_verdict'] is None and c['training_ready'] is False
        q = c['review']
        assert q['human_verdict'] is None and q['temporal_visual_scope'] == 'not_reviewed'
        assert q['reviewed_frames'] == [frames[c['case_id']]]
        assert q['synthetic_status'] == 'pass' and c['pixel_contract_pass']
        video_paths.extend(c['videos'].values())
        assert len(c['preview_frames']) == 10
        for f in c['preview_frames']:
            preview_paths.extend(f[k] for k in ('gt', 'input', 'labels', 'condition'))
        contacts.extend(c['contacts'])
    assert len(set(video_paths)) == 96 and len(set(preview_paths)) == 960
    pages = ['index.html', 'data_review.html']
    references = video_paths + preview_paths + contacts
    for name in pages:
        content = (root / name).read_text(encoding='utf-8')
        assert '\ufffd' not in content
        parser = Resources()
        parser.feed(content)
        references.extend(parser.links)
        if name == 'index.html':
            assert '<svg' in content
        else:
            assert "KEY='v77-training-input-frames-r3'" in content
            assert 'V77_training_input_r3_frame_review.json' in content
            embedded = content.split('const DATA=', 1)[1].split(';\nconst $=', 1)[0]
            assert json.loads(embedded) == manifest
        for i, script in enumerate(re.findall(r'<script[^>]*>(.*?)</script>', content, re.S)):
            path = output.parent / f'{name}.{i}.check.js'
            path.write_text(script, encoding='utf-8')
            subprocess.run([str(node), '--check', str(path)], check=True)
    missing = sorted({p for p in references if not (root / p.split('#')[0]).is_file()})
    assert not missing, missing
    result = dict(run_id='r3', case_count=len(clips), actual_frame_count=240,
                  video_count=96, preview_count=960, referenced_files_checked=len(set(references)),
                  missing_resources=missing, javascript_syntax='pass', embedded_manifest='pass',
                  independent_review_frame_selection='pass', human_verdict='all null',
                  training_ready=0, unicode_replacement_characters=0,
                  browser_interaction='not tested', video_decode='remote delivery_validation.json')
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--sample', type=Path, required=True)
    p.add_argument('--node', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    main(a.root, a.sample, a.node, a.output)
