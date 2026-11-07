"""Static publication checks only; never opens a browser or starts a server."""
from __future__ import annotations

import argparse
import ast
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    def __init__(self, text: str):
        super().__init__(convert_charrefs=True)
        self.ids: dict[str, tuple[str, dict]] = {}
        self.duplicates: list[str] = []
        self.images: list[dict] = []
        self.links: list[str] = []
        self.scripts: list[tuple[dict, str]] = []
        self.script_attributes: dict | None = None
        self.script_text: list[str] = []
        self.feed(text)

    def handle_starttag(self, tag: str, attributes: list[tuple[str, str | None]]) -> None:
        attrs = dict(attributes)
        if attrs.get('id'):
            if attrs['id'] in self.ids:
                self.duplicates.append(attrs['id'])
            self.ids[attrs['id']] = (tag, attrs)
        if tag == 'img':
            self.images.append(attrs)
        if tag == 'a' and attrs.get('href'):
            self.links.append(attrs['href'])
        if tag == 'script':
            self.script_attributes = attrs
            self.script_text = []

    def handle_data(self, data: str) -> None:
        if self.script_attributes is not None:
            self.script_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == 'script' and self.script_attributes is not None:
            self.scripts.append((self.script_attributes, ''.join(self.script_text)))
            self.script_attributes = None
            self.script_text = []


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def png_size(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    assert header[:8] == b'\x89PNG\r\n\x1a\n' and header[12:16] == b'IHDR', path
    return struct.unpack('>II', header[16:24])


def validate_links(page: Page, folder: Path) -> int:
    checked = 0
    for value in [image['src'] for image in page.images] + page.links:
        parsed = urlsplit(value)
        assert not parsed.scheme and not parsed.netloc, f'Nonlocal dependency: {value}'
        if parsed.path:
            target = folder / unquote(parsed.path)
            assert target.is_file(), f'Missing relative file: {value}'
            checked += 1
    return checked


def validate_viewer(folder: Path, filename: str) -> dict:
    source = (folder / filename).read_text(encoding='utf-8')
    page = Page(source)
    assert not page.duplicates, f'Duplicate HTML IDs: {page.duplicates}'
    viewer_scripts = [content for attrs, content in page.scripts if attrs.get('id') == 'v44-viewer-data']
    assert len(viewer_scripts) == 1
    data = json.loads(viewer_scripts[0])
    raws = [105, 113, 130, 137] if data['mode'] == 'pilot' else list(range(89, 138))
    assert data['raw_ids'] == raws and data['default_raw'] == 113 and data['default_state'] == 'delete'
    assert set(data['frames']) == {str(raw) for raw in raws}
    slider = page.ids['v44-range'][1]
    expected_range = ('0', '3', '1') if data['mode'] == 'pilot' else ('89', '137', '113')
    assert tuple(slider[key] for key in ('min', 'max', 'value')) == expected_range
    assert slider['step'] == '1'
    for raw in raws:
        frame = data['frames'][str(raw)]
        crop = frame['views']['crop']
        assert len(crop) == 4 and 0 <= crop[0] < crop[2] <= 480 and 0 <= crop[1] < crop[3] <= 320
        for variant in ('current', 'v44'):
            for state in ('keep', 'delete'):
                entry = frame[variant][state]
                path = folder / entry['path']
                assert png_size(path) == (480, 320) and sha(path) == entry['sha256']
        observed = frame['observed']
        assert sha(folder / observed['path']) == observed['sha256']
    for column in ('current', 'v44', 'observed'):
        tag, initial = page.ids['v44-image-' + column]
        assert tag == 'img' and initial['data-raw'] == '113' and initial['data-state'] == 'delete'
        expected = data['frames']['113']['observed'] if column == 'observed' else data['frames']['113'][column]['delete']
        assert initial['src'] == expected['path'], f'Initial src is not actual raw113: {column}'
        assert 'loading' not in initial, 'Initial images must be directly available without interaction'
    for required in ('frozen-transfer', '未重标定', 'HELD', 'engine pass ≠ human', '没有独立', 'cpu_stage.html'):
        assert required in source, f'Missing report boundary: {required}'
    javascript = '\n'.join(content for attrs, content in page.scripts if attrs.get('type') != 'application/json')
    assert not re.search(r'\b(?:fetch|XMLHttpRequest|WebSocket|EventSource)\s*\(', javascript)
    assert not any(attrs.get('src') for attrs, _ in page.scripts), 'Script must be self-contained'
    for marker in ('Promise.all(paths.map(preload))', 'ticket !== sequence', 'im.decode()', 'grid.replaceChildren(fragment)'):
        assert marker in javascript, f'Missing group-switch protection: {marker}'
    node = shutil.which('node')
    assert node, 'Node is required for static JavaScript syntax validation'
    with tempfile.TemporaryDirectory(prefix='v44_static_qa_', dir=Path(__file__).resolve().parent) as temporary:
        script = Path(temporary) / 'inline.js'
        script.write_text(javascript, encoding='utf-8')
        subprocess.run([node, '--check', str(script)], check=True, capture_output=True, text=True)
    static_source = (folder / data['static_page']).read_text(encoding='utf-8')
    static = Page(static_source)
    actual_images = [image for image in static.images if image.get('data-column') in ('current', 'v44', 'observed')]
    assert len(actual_images) == len(raws) * 2 * 3
    assert not static.scripts and not static.duplicates
    static_keys = {(int(image['data-raw']), image['data-state'], image['data-column']) for image in actual_images}
    assert static_keys == {(raw, state, column) for raw in raws for state in ('keep', 'delete') for column in ('current', 'v44', 'observed')}
    return dict(page=filename, mode=data['mode'], frames=len(raws), initial_raw=113, initial_state='delete',
                checked_local_links=validate_links(page, folder) + validate_links(static, folder),
                static_images=len(actual_images), javascript_syntax='PASS',
                atomic_switch_source_guards='PRESENT_NOT_BROWSER_EXECUTED')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    folder = args.output.resolve(strict=True)
    manifest = read(folder / 'publication_manifest.json')
    assert not list(folder.rglob('*.npz')), 'Large float NPZ leaked into report package'
    assert sha(folder / manifest['cpu_stage']['path']) == manifest['cpu_stage']['sha256']
    for item in manifest['files']:
        path = folder / item['path']
        assert sha(path) == item['sha256'] and path.stat().st_size == item['bytes'], path
    for path in (folder / 'code/report').glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'))
    pages = [validate_viewer(folder, 'index.html')]
    if manifest.get('pilot_comparison_sha256'):
        pages.append(validate_viewer(folder, 'pilot.html'))
    report = dict(status='PASS_V44_STATIC_PUBLICATION_QA', browser_or_server_used=False,
                  pages=pages, copied_file_hashes_checked=len(manifest['files']),
                  cpu_stage_sha256_preserved=True, no_float_NPZ=True,
                  human_verdict_inferred=False, full49_executed=manifest['full49_executed'])
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
