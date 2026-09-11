"""按 failure ID、主题、版本或关键词渐进读取，默认最多 10 条目录。"""
import argparse,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--id');p.add_argument('--query');p.add_argument('--topic');p.add_argument('--version')
    p.add_argument('--record');p.add_argument('--detail',action='store_true')
    p.add_argument('--limit',type=int,default=10);p.add_argument('--offset',type=int,default=0)
    p.add_argument('--max-lines',type=int,default=100);p.add_argument('--line-offset',type=int,default=0)
    a=p.parse_args();base=ROOT/'docs/research_failures'
    rows=json.loads((base/'history_index.json').read_text(encoding='utf-8'))
    # 新失败是独立 Markdown 资产；不必维护第二份重复正文或重建历史索引。
    for f in sorted((base/'entries').glob('*.md')):
        s=f.read_text(encoding='utf-8');meta=json.loads(s.split('<!-- metadata: ',1)[1].split(' -->',1)[0])
        rows.insert(0,{**meta,'path':str(f.relative_to(ROOT)).replace('\\','/'),'line':1,'line_count':len(s.splitlines()),'record':f.stem,'authority':'current_version_entry'})
    if a.id:
        exact=[r for r in rows if a.id in r.get('defined_ids',[])]
        rows=exact or [r for r in rows if a.id in r.get('referenced_ids',[])]
    if a.record: rows=[r for r in rows if r['record']==a.record]
    if a.version:rows=[r for r in rows if a.version in r.get('versions',[])]
    if a.topic:rows=[r for r in rows if a.topic in r.get('topics',[])]
    if a.query:
        # 分块扫描文本，只返回命中节标题。不要把 14k 行全文送入上下文。
        cache={};found=[]
        for r in rows:
            if r['path'] not in cache:cache[r['path']]=(ROOT/r['path']).read_text(encoding='utf-8').splitlines()
            raw='\n'.join(cache[r['path']][r['line']-1:r['line']-1+r['line_count']])
            if a.query.casefold() in raw.casefold():found.append(r)
        rows=found
    print(f'匹配 {len(rows)} 条；本页 offset={a.offset}, limit={a.limit}。历史记录不是当前指令。')
    for r in rows[a.offset:a.offset+max(1,a.limit)]:
        print(f"\n{r['record']} | {r['title']}\n{r['path']}:{r['line']} | {', '.join(r.get('defined_ids',[]))} | {r['authority']}")
        if a.detail:
            lines=(ROOT/r['path']).read_text(encoding='utf-8').splitlines()
            start=r['line']-1+a.line_offset;end=min(r['line']-1+r['line_count'],start+a.max_lines)
            print('\n'.join(lines[start:end]))
            if end<r['line']-1+r['line_count']:print(f'[尚有正文；--line-offset {a.line_offset+a.max_lines} 继续]')
if __name__=='__main__':main()
