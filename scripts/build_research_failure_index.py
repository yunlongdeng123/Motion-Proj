"""从冻结历史索引和独立失败卡生成导航；不改历史正文与研究状态。"""
import collections
import posixpath
from query_research_failures import ROOT, load_records


def write(rel, body):
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(body.rstrip()+'\n')


def link(row, parent):
    rel = posixpath.relpath(row['path'], parent)
    return f"[{row['record']}]({rel})（行{row['line']}）"


def clean(text):
    return text.replace('|', ' / ').replace('\n', ' ')


def main():
    rows = load_records()
    entries = [r for r in rows if r['authority'] == 'version_entry']
    missing = [r['record'] for r in entries if r.get('metadata_missing')]
    if missing:
        raise SystemExit('请补齐 metadata 后生成目录：'+', '.join(missing))
    versions = collections.defaultdict(list)
    defined = collections.defaultdict(list)
    referenced = collections.defaultdict(list)
    for row in rows:
        for version in row.get('versions', []) or ['misc']:
            versions[version].append(row)
        for failure_id in row.get('defined_ids', []):
            defined[failure_id].append(row)
        for failure_id in row.get('referenced_ids', []):
            referenced[failure_id].append(row)
    families = collections.defaultdict(list)
    for failure_id in sorted(set(defined) | set(referenced)):
        families[failure_id.rsplit('-F', 1)[0]].append(failure_id)
    id_intro = '# 全量 failure ID 目录\n\n由 `scripts/build_research_failure_index.py` 生成。独立标题定义与只有引用的 ID 分开标注，不等于全部是科学失败。无 ID 的阶段请按[版本](VERSIONS.md)查询。\n\n'
    id_intro += '| ID 系列 | 有定义 | 仅引用 | 全部 ID |\n|---|---:|---:|---|\n'
    for family, ids in sorted(families.items()):
        n = sum(i in defined for i in ids)
        id_intro += f'| {family} | {n} | {len(ids)-n} | [{len(ids)} 个](ids/{family}.md) |\n'
        body = f'# {family} failure ID\n\n[全部 ID](../IDS.md) · [版本阶段记录](../VERSIONS.md)\n\n每个 ID 优先链接独立卡，否则链接原定义；只有引用时明确标注。多条历史用 `python scripts/query_research_failures.py --id ID --detail` 读取。\n\n| ID | 类型 | 首选证据 | 原标题 / 说明 |\n|---|---|---|---|\n'
        for failure_id in ids:
            r = (defined.get(failure_id) or referenced[failure_id])[0]
            kind = '独立卡' if r['authority'] == 'version_entry' else ('历史标题定义' if failure_id in defined else '正文记录，未标注标题定义')
            body += f'| {failure_id} | {kind} | {link(r, "docs/research_failures/ids")} | {clean(r["title"])} |\n'
        write(f'docs/research_failures/ids/{family}.md', body)
    write('docs/research_failures/IDS.md', id_intro)

    body = '# 全部版本与历史阶段\n\n由 `scripts/build_research_failure_index.py` 生成。旧版本未删除；表中是阶段记录数，一个 ID 可出现多次，不能作为独立失败数。V73 对应 V7.3，V74-H2 单列。\n\n| 版本 | 历史记录 | 独立卡 | ID 目录 | 阶段目录 |\n|---|---:|---:|---|---|\n'
    for version, selected in sorted(versions.items()):
        n = sum(r['authority'] == 'version_entry' for r in selected)
        pages = []
        for offset in range(0, len(selected), 50):
            page = offset//50+1
            pages.append(f'[{page}](versions/{version}-{page}.md)')
            content = f'# {version} / 第 {page} 页\n\n[全部版本](../VERSIONS.md) · [全部 ID](../IDS.md) · [研究边界](../BOUNDARIES.md)\n\n此处是阶段索引，不维护当前执行状态。独立卡优先，历史保持原顺序。\n\n| 记录 | 原标题 | 定义 ID |\n|---|---|---|\n'
            for r in selected[offset:offset+50]:
                content += f'| {link(r, "docs/research_failures/versions")} | {clean(r["title"])} | {", ".join(r.get("defined_ids", [])) or "阶段记录 / 引用"} |\n'
            write(f'docs/research_failures/versions/{version}-{page}.md', content)
        family_link = f'[{version}](ids/{version}.md)' if version in families else '按阶段/关键词查阅'
        body += f'| {version} | {len(selected)-n} | {n} | {family_link} | {" / ".join(pages)} |\n'
    write('docs/research_failures/VERSIONS.md', body)
    body = '# 独立失败卡\n\n由 `scripts/build_research_failure_index.py` 生成；卡片记录各次证据与结论，不是当前研究状态。[旧版本](VERSIONS.md) · [全部 ID](IDS.md) · [维护方式](README.md)。\n\n| ID | 证据主题 |\n|---|---|\n'
    for r in sorted(entries, key=lambda r:r['record']):
        body += f'| [{r["record"]}](entries/{r["record"]}.md) | {clean(r["title"])} |\n'
    write('docs/research_failures/ENTRIES.md', body)
    print(f'Indexed {len(rows)} records, {len(entries)} cards, {len(defined)} defined IDs, {len(set(referenced)-set(defined))} reference-only IDs, {len(versions)} versions.')


if __name__ == '__main__':
    main()
