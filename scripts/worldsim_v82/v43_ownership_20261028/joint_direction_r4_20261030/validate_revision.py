from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import sys, json, re, xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parents[2]
OUT = BASE/'outputs/V43_Ownership_20261027'
sys.path.insert(0, str(BASE/'work/v43_ownership_20261027'))
from build_v43 import update_index
page = (OUT/'index.html').read_text(encoding='utf8')
assert update_index(page) == page, '首页生成必须幂等'
assert page.count('<!-- v43-design-revision4-start -->') == 1
assert 'v43-design-revision3-start' not in page
assert page.count('<h1>') == 1
ET.fromstring((OUT/'architecture.svg').read_text(encoding='utf8'))
class Links(HTMLParser):
    def __init__(self): super().__init__(); self.links=[]
    def handle_starttag(self, tag, attrs):
        self.links.extend(v for k,v in attrs if k in ('src','href'))
p = Links(); p.feed(page)
links = p.links + re.findall(r'\]\(([^)]+)\)', (OUT/'V43_DESIGN.md').read_text(encoding='utf8'))
broken=[]
for link in links:
    u=urlsplit(link)
    if u.scheme or not u.path: continue
    if not (OUT/unquote(u.path)).exists(): broken.append(link)
assert not broken, broken
schema=json.loads((OUT/'ownership_record_contract.json').read_text(encoding='utf8'))
plan=json.loads((OUT/'mechanism_experiment_plan.json').read_text(encoding='utf8'))
state=json.loads((OUT/'direction_state.json').read_text(encoding='utf8'))
assert not schema['model_contract']['scene_judgment_receives_edit_instruction']
assert not schema['candidate_selection']['candidate_is_action_set']
assert not plan['session_execution']['autodl_auto_shutdown']
assert not state['automatic_autodl_shutdown_allowed']
assert state['new_training_updates_this_revision'] == 0
assert not plan['scene1']['train_new_transferable_modules']
old=(BASE/'work/v43_joint_direction_20261030/revision3_backup/index.html').read_text(encoding='utf8')
historical_headers=['历史CPU候选：1 / 0 / unknown', '历史CPU代理预览：尚不能通过归属验收', '2026-10-27历史调查：199帧与共享工作台']
for title in historical_headers:
    pattern=r'<section><h2>'+re.escape(title)+r'</h2>.*?</section>'
    assert re.search(pattern,old,re.S).group() == re.search(pattern,page,re.S).group(), title
receipt={'architecture_revision':4,'passed':True,'local_links_checked':len(links),'broken_links':broken,
    'generator_idempotent':True,'historical_sections_preserved':historical_headers,
    'document_and_contract_consistency':True,'browser_interaction_qa':False,
    'new_experiment_runs':0,'new_training_updates':0,'human_verdict':None}
(OUT/'design_revision_validation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(receipt,ensure_ascii=False))
