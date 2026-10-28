"""Validate saved local entry links and manifests, with no browser or network."""
from __future__ import annotations
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlsplit

import patch_shared_scene_entries as patch


class Links(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.references = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('href', 'src') and value:
                self.references.append((tag, value))


def main():
    pages = [patch.OUT / 'index.html'] + [patch.OUT / context / 'index.html' for context in patch.CONTEXTS[1:]]
    statuses = [patch.audit_scene(context, number)[0] for number, context in enumerate(patch.CONTEXTS, 1)]
    assert all(s['current_execution'] == 'local_delivery_available' for s in statuses)
    groups = {s['context']: {g['id']: g for g in s['groups']} for s in statuses}
    checked, group_links = 0, []
    for page in pages:
        source = page.read_text(encoding='utf8')
        assert '?group=' not in source.split('?group=', 1)[-1].split('"', 1)[0], 'duplicated group query'
        for tag, relative in Links(source).references:
            parts = urlsplit(relative)
            if parts.scheme or parts.netloc or relative.startswith(('data:', 'javascript:')):
                continue
            target = (page.parent / unquote(parts.path)).resolve() if parts.path else page
            assert target.exists(), (page, tag, relative)
            checked += 1
            if not parts.fragment.startswith('workspace?'):
                continue
            assert parts.fragment.count('?') == 1, relative
            pairs = parse_qsl(parts.fragment.split('?', 1)[1])
            assert len({key for key, value in pairs}) == len(pairs), relative
            params = dict(pairs)
            if 'group' not in params:
                continue
            context = target.parent.name
            group = groups[context][params['group']]
            raw = int(params['raw'])
            assert raw in group['complete_raws'], (relative, 'raw not available')
            group_links.append(dict(page=str(page.relative_to(patch.BASE)), context=context, group=group['id'], raw=raw))
    for number, context in enumerate(patch.CONTEXTS[1:], 2):
        status = json.loads((patch.OUT / context / 'v43_entry_status.json').read_text(encoding='utf8'))
        assert status['current_execution'] == 'local_delivery_available'
        if number == 3:
            assert status['execution_authorization'] == 'resumed_by_current_user'
            assert status['old_hold_is_current'] is False
        assert {link['group'] for link in group_links if Path(link['page']).parent.name == context}.issuperset(gid for gid, _, _ in patch.PLANS)
    report = json.loads((patch.OUT / 'scene_entry_audit_20261029.json').read_text(encoding='utf8'))
    assert report['case1_files_written'] is False and report['bounded_report_written'] is False
    for relative, digest in report['protected_sha256'].items():
        assert hashlib.sha256((patch.BASE / relative).read_bytes()).hexdigest() == digest, relative
    result = dict(schema='scene-entry-static-link-qa-v1', local_references_checked=checked, valid_group_links=group_links, original_case1_and_bounded_hashes_preserved=True, all_deliveries_actual=True, browserQA=False, ssh_used=False)
    (Path(__file__).parent / 'static_link_qa.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(dict(local_references_checked=checked, valid_group_links=len(group_links), case1_bounded_preserved=True, browserQA=False)))


if __name__ == '__main__':
    main()
