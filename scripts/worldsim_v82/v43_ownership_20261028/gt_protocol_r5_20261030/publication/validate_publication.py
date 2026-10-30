"""仅检查本地发布包、路径和报告骨架，不运行远端发布入口。"""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
import json
import posixpath
import re
import xml.etree.ElementTree as ET

from publish_remote import read_bundle

HERE = Path(__file__).resolve().parent
manifest, payload = read_bundle(HERE / 'publication.tar.gz')
assert set(payload) == set(manifest['files'])
if manifest['m0_report_included']:
    published_m0 = json.loads(payload[manifest['optional_m0_doc_root'] + '/report_state.json'])
    assert published_m0['artifact_kind'] != 'skeleton_not_final_report'
    assert manifest['m0_source_files'], '实际M0报告须同时归档源码'
    for name in manifest['m0_source_files']:
        assert name.startswith(manifest['optional_m0_code_root'] + '/') and name in payload
        assert Path(name).suffix.lower() in {'.py', '.json', '.cjs', '.md'}
        assert not any(part in {'__pycache__', 'output', 'outputs', 'tmp', 'temp', '.cache'} for part in Path(name).parts)
else:
    assert not manifest['m0_source_files']
external_dependencies = {
    'docs/autoresearch/v43_ownership_20261028/joint_direction_r4_20261030/index.html',
    'docs/autoresearch/v43_ownership_20261028/r2_recovery_20261029/index.html',
    'docs/autoresearch/v43_ownership_20261028/unknown_white_20261030/index.html',
    'docs/autoresearch/v43_ownership_20261028/editable_ownership_20261030/index.html',
}


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attributes):
        self.links.extend(value for key, value in attributes if key in ('src', 'href'))


broken = []
checked = 0
used_dependencies = set()
archived_source_markdown = []
for name, data in payload.items():
    links = []
    if name.endswith('.html'):
        parser = Links()
        parser.feed(data.decode('utf-8-sig'))
        links = parser.links
    elif name.endswith('.md'):
        if name.startswith('scripts/'):
            archived_source_markdown.append(name)
        else:
            links = re.findall(r'\]\(([^)]+)\)', data.decode('utf-8-sig'))
    elif name.endswith('.svg'):
        ET.fromstring(data)
    elif name.endswith('.json'):
        json.loads(data)
    elif name.endswith('.py'):
        compile(data.decode('utf-8-sig'), name, 'exec')
    for link in links:
        url = urlsplit(link)
        if url.scheme or not url.path:
            continue
        checked += 1
        target = posixpath.normpath(posixpath.join(posixpath.dirname(name), unquote(url.path)))
        if target in external_dependencies:
            used_dependencies.add(target)
        elif target not in payload:
            broken.append((name, link))
assert not broken, broken
template = HERE / 'm0_report_template'
state = json.loads((template / 'report_state.json').read_text(encoding='utf8'))
assert state['artifact_kind'] == 'skeleton_not_final_report'
assert state['gpu_smoke']['status'] == 'pending_not_claimed_complete'
assert state['formal_reconstruction']['checkpoint'] is None
assert not state['method_success_or_generalization_claim']
assert state['dataset']['frames'] == 16
assert not state['ground_truth']['unique_gaussian_owner_gt']
assert not state['ground_truth']['finite_sample_shadow_difference_is_pure_shadow_gt']
assert not state['training_permission']['complete_mesh_dense_depth_surface_or_object_ids_and_edit_references_to_training']
parser = Links()
parser.feed((template / 'index.html').read_text(encoding='utf8'))
for link in parser.links:
    url = urlsplit(link)
    if not url.scheme and url.path:
        assert (template / unquote(url.path)).exists(), link
ET.fromstring((template / 'architecture.svg').read_text(encoding='utf8'))
diagram = json.loads((template / 'architecture_validation.json').read_text(encoding='utf8'))
assert diagram['native_canvas_rendered'] and not diagram['text_overflow']
for name in ['prepare_publication.py', 'publish_remote.py', 'validate_publication.py']:
    compile((HERE / name).read_text(encoding='utf8'), name, 'exec')
receipt = {'passed': True, 'revision': 5, 'payload_files_checked': len(payload),
           'publication_links_checked': checked, 'broken_links': broken,
           'remote_existing_dependencies': sorted(used_dependencies),
           'source_markdown_preserves_original_local_references': archived_source_markdown,
           'm0_skeleton_links_checked': len(parser.links), 'm0_template_diagram_rendered': True,
           'm0_template_claims_smoke_complete': False, 'm0_template_published_as_final_report': False,
           'filled_m0_report_included': manifest['m0_report_included'],
           'm0_source_files_archived': len(manifest['m0_source_files']),
           'remote_publish_executed': False, 'ssh_executed': False, 'commits': 0,
           'validation_scope': 'local_artifact_and_script_syntax_not_remote_execution'}
(HERE / 'publication_validation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
print(json.dumps(receipt, ensure_ascii=True))
