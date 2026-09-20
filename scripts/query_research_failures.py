"""按 failure ID、主题、版本或关键词读取，默认最多 10 条目录。"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_records(root=ROOT):
    base = root / 'docs/research_failures'
    history = json.loads((base / 'history_index.json').read_text(encoding='utf-8'))
    entries = []
    for f in sorted((base / 'entries').glob('*.md'), reverse=True):
        text = f.read_text(encoding='utf-8')
        match = re.search(r'<!--\s*metadata:\s*(.*?)\s*-->', text, re.S)
        if match:
            meta = json.loads(match.group(1))
        else:
            # 缺 metadata 的旧卡仍可读取；生成器另行提示补齐。
            meta = {'title': text.splitlines()[0].lstrip('# ').split('：', 1)[-1],
                    'defined_ids': [f.stem], 'referenced_ids': [],
                    'versions': [f.stem.rsplit('-F', 1)[0]], 'topics': [],
                    'metadata_missing': True}
        entries.append({**meta, 'path': f.relative_to(root).as_posix(), 'line': 1,
                        'line_count': len(text.splitlines()), 'record': f.stem,
                        'authority': 'version_entry'})
    return entries + history


def select(rows, *, failure_id=None, record=None, version=None, topic=None, query=None, root=ROOT):
    if failure_id:
        exact = [r for r in rows if failure_id in r.get('defined_ids', [])]
        rows = exact or [r for r in rows if failure_id in r.get('referenced_ids', [])]
    if record:
        rows = [r for r in rows if r['record'] == record]
    if version:
        version = version.upper().replace('.', '')
        rows = [r for r in rows if version in r.get('versions', [])]
    if topic:
        rows = [r for r in rows if topic in r.get('topics', [])]
    if query:
        cache = {}
        found = []
        for r in rows:
            if r['path'] not in cache:
                cache[r['path']] = (root / r['path']).read_text(encoding='utf-8').splitlines()
            raw = '\n'.join(cache[r['path']][r['line']-1:r['line']-1+r['line_count']])
            if query.casefold() in raw.casefold():
                found.append(r)
        rows = found
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--id'); p.add_argument('--query'); p.add_argument('--topic'); p.add_argument('--version')
    p.add_argument('--record'); p.add_argument('--detail', action='store_true')
    p.add_argument('--limit', type=int, default=10); p.add_argument('--offset', type=int, default=0)
    p.add_argument('--max-lines', type=int, default=100); p.add_argument('--line-offset', type=int, default=0)
    a = p.parse_args()
    rows = select(load_records(), failure_id=a.id, record=a.record, version=a.version, topic=a.topic, query=a.query)
    print(f'匹配 {len(rows)} 条；本页 offset={a.offset}, limit={a.limit}。历史记录不是当前指令。')
    for r in rows[max(0, a.offset):max(0, a.offset)+max(1, a.limit)]:
        print(f"\n{r['record']} | {r['title']}\n{r['path']}:{r['line']} | {', '.join(r.get('defined_ids', []))} | {r['authority']}")
        if a.detail:
            lines = (ROOT / r['path']).read_text(encoding='utf-8').splitlines()
            start = r['line']-1+max(0, a.line_offset)
            end = min(r['line']-1+r['line_count'], start+max(1, a.max_lines))
            print('\n'.join(lines[start:end]))
            if end < r['line']-1+r['line_count']:
                print(f'[尚有正文；--line-offset {max(0, a.line_offset)+max(1, a.max_lines)} 继续]')


if __name__ == '__main__':
    main()
