from pathlib import Path
import json
from html.parser import HTMLParser

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'work/vgpu_bugfix_20261012'
OUT=ROOT/'outputs/drivingeditor_vgpu_debug/evidence'

def shrink(x, depth=0):
    if isinstance(x, list):
        if len(x)>6:
            return {'list_length':len(x), 'sample': [shrink(v,depth+1) for v in x[:2]]}
        return [shrink(v,depth+1) for v in x]
    if isinstance(x, dict):
        return {k:shrink(v,depth+1) for k,v in x.items() if k not in ('all49_counts','per_frame','frames','rows','records','all2000_steps','all49')}
    return x

paths=[
    OUT/'S_observed_static_rgb30k_to32000/receipts/S_vs_originals_locked_metrics.json',
    OUT/'V_opacity_outside30k_to32000/receipts/three_arm_locked_metrics.json',
    WORK/'sgcm_data_audit/observed_static_rgb_S_review/actual_S_locality_core_cpu.json',
]
for p in paths:
    if not p.exists():
        print('MISSING',p.relative_to(ROOT));continue
    j=json.loads(p.read_text(encoding='utf-8-sig'))
    print('\nFILE',p.relative_to(ROOT))
    if 'locked_metrics' in p.name:
        print('KEYS',list(j))
        for k,v in j.items():
            if k not in ('primary_metrics','aggregation','noise_definition','limitations'):continue
            if k=='primary_metrics':
                v=[{a:b for a,b in row.items() if not isinstance(b,(dict,list))} for row in v]
            t=json.dumps(v,ensure_ascii=False,indent=2)
            print(k,t[:14000])
    else:
        print('KEYS',list(j))
        for k,v in j.items():
            if 'domain' in k and isinstance(v,dict):
                for domain,data in v.items():
                    if not isinstance(data,dict):continue
                    print(domain)
                    for arm in ['B1','B2','S','V']:
                        if arm in data:
                            print(arm,json.dumps(data[arm].get('metrics',{}).get('composite_clamped_MSE',{})))
            elif not isinstance(v,(list,dict)):
                print(k,v)

class Parse(HTMLParser):
    def __init__(self):super().__init__();self.in_title=False;self.title=[];self.hrefs=[]
    def handle_starttag(self,tag,attrs):
        if tag=='title':self.in_title=True
        if tag=='a':self.hrefs.extend(v for k,v in attrs if k=='href')
    def handle_endtag(self,tag):
        if tag=='title':self.in_title=False
    def handle_data(self,data):
        if self.in_title:self.title.append(data)
print('\nREPORTS')
for p in sorted((ROOT/'outputs').glob('*/index.html')):
    a=Parse();a.feed(p.read_text(encoding='utf-8-sig'));print(str(p.parent.relative_to(ROOT)),''.join(a.title))
print('\n3090 LOCAL REFERENCE')
p=ROOT/'outputs/drivingeditor_30k_edit_comparison/index.html'
if p.exists():
    s=p.read_text(encoding='utf-8-sig')
    for term in ['21','49','3090','baseline']:
        pos=s.find(term)
        print(term, s[max(0,pos-130):pos+300])
