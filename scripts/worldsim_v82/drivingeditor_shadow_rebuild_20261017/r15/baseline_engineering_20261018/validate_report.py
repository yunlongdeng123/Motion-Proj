"""Check report assets and script syntax, without claiming browser acceptance."""
from pathlib import Path
from html.parser import HTMLParser
import json, re, subprocess, xml.etree.ElementTree as ET
from urllib.parse import unquote, urlparse
from PIL import Image

W = Path(__file__).parent
O = W.parents[1] / 'outputs/baseline_engineering_20261018'


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.items = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.items += [attrs[k] for k in ['src', 'href'] if k in attrs]


def main():
    content = (O / 'index.html').read_text(encoding='utf-8')
    parser = Links()
    parser.feed(content)
    checked = 0
    for link in parser.items:
        p = urlparse(link)
        if p.scheme or not p.path:
            continue
        assert (O / unquote(p.path)).is_file(), link
        checked += 1
    pngs = list((O / 'prior_trace').rglob('*.png'))
    assert len(pngs) == 196
    for p in pngs:
        with Image.open(p) as im:
            assert im.size == (480, 320)
            im.load()
    ET.parse(O / 'architecture.svg')
    script = '\n'.join(re.findall(r'<script>(.*?)</script>', content, re.S))
    assert script and 'fetch(' not in script
    (W / 'report_inline.js').write_text(script, encoding='utf-8')
    result = subprocess.run(['node', '--check', str(W / 'report_inline.js')], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    receipt = dict(status='PASSED_STATIC_MEDIA_AND_JS_SYNTAX', PNG_count=len(pngs), PNG_size=[480, 320],
        relative_links_checked=checked, component_SVG_parsed=True, JS_syntax_passed=True,
        local_JSON_fetch_required=False, browser_file_UI_acceptance=False,
        original_control_isolated_body_restoration_claim_retracted=True)
    (O / 'validation.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
