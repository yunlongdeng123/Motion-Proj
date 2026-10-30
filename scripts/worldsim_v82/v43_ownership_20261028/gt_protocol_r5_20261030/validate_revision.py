from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import sys, json, re, xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parents[2]
OUT = BASE/'outputs/V43_Ownership_20261027'
backup=BASE/'work/v43_gt_protocol_20261030/revision4_backup'
sys.path.insert(0, str(BASE/'work/v43_ownership_20261027'))
from build_v43 import update_index, schema
from build_architecture import architecture, data_architecture
from revision5_contract import DESIGN, upgrade_schema, experiment_plan
page = (OUT/'index.html').read_text(encoding='utf8')
assert update_index(page) == page, '首页生成必须幂等'
assert page.count('<!-- v43-design-revision5-start -->') == 1
assert page.count('<!-- v43-design-revision5-end -->') == 1
assert 'v43-design-revision4-start' not in page
assert 'v43-design-revision3-start' not in page
assert page.count('<h1>') == 1
assert (OUT/'V43_DESIGN.md').read_text(encoding='utf8') == DESIGN
for filename, generate in [('architecture.svg', architecture), ('data_architecture.svg', data_architecture)]:
    source=(OUT/filename).read_text(encoding='utf8')
    ET.fromstring(source)
    assert source == generate(), filename+' generation must be idempotent'
main_svg=ET.fromstring((OUT/'architecture.svg').read_text(encoding='utf8'))
rects=main_svg.findall('.//{http://www.w3.org/2000/svg}rect')
assert sum(1 for rect in rects if rect.get('y')=='345' and rect.get('height')=='1030') == 3, '保留原三大框'
old_main=ET.fromstring((backup/'outputs/V43_Ownership_20261027/architecture.svg').read_text(encoding='utf8'))
old_rects=old_main.findall('.//{http://www.w3.org/2000/svg}rect')
rect_geometry=lambda elements: [tuple(rect.get(key) for key in ('x','y','width','height','rx')) for rect in elements]
assert rect_geometry(rects) == rect_geometry(old_rects), '保留大框及全部内部小框的布局'
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
contract=json.loads((OUT/'ownership_record_contract.json').read_text(encoding='utf8'))
plan=json.loads((OUT/'mechanism_experiment_plan.json').read_text(encoding='utf8'))
state=json.loads((OUT/'direction_state.json').read_text(encoding='utf8'))
assert contract == upgrade_schema(schema()), '合同生成必须幂等'
assert plan == experiment_plan(), '实验矩阵生成必须幂等'
assert contract['schema_version'].endswith('-5')
assert not contract['model_contract']['scene_judgment_receives_edit_instruction']
assert not contract['candidate_selection']['candidate_is_action_set']
assert not contract['ground_truth_contract']['unique_gt_for_every_reconstructed_gaussian']
assert not contract['responsibility']['is_ground_truth']
assert not contract['correspondence']['projected_overlap_implies_surface_identity']
assert not contract['correspondence']['missing_observation_is_no_correspondence']
assert not contract['truth_assisted_reference_contract']['mathematical_upper_bound']
assert not contract['truth_assisted_reference_contract']['complete_clean_surface_reference_is_fair_method']
assert not contract['effect']['independent_shadow_reference']['vehicle_present_minus_absent_is_pure_shadow_gt']
assert not contract['legal_input_contract']['test_gt_for_actual_method_selection_or_adaptation']
assert set(['wrong_depth','mixed_support','no_correspondence','missing_observation']) <= set(contract['correspondence']['vocabulary'])
assert 'actual_training_with_current_reconstruction_code' in contract['data_protocol']['primary_route']
assert not contract['data_protocol']['natural_contamination_guaranteed']
assert not plan['gt_input_isolation']['deployment_receives_world_gt']
assert not plan['real_scenes']['complete_gt']
assert not plan['real_scenes']['human_or_subagent_reference_is_perfect_gt']
assert not plan['comparison_contract']['truth_assisted_M1_M2_are_mathematical_upper_bounds']
assert plan['minimal_first_gate']['scene_count'] == 1
assert plan['minimal_first_gate']['actual_reconstruction_required']
assert not plan['session_execution']['autodl_auto_shutdown']
assert not state['automatic_autodl_shutdown_allowed']
assert state['new_training_updates_this_revision'] == 0
assert state['new_scene_renders_this_revision'] == 0
assert state['architecture_revision'] == 5
run_receipt = OUT / state['minimal_actual_reconstruction_run_receipt']
run_state = json.loads(run_receipt.read_text(encoding='utf8'))
assert run_state['artifact_kind'] == 'actual_run_report' and run_state['formal_updates'] == 10000
assert not run_state['new_V43_joint_model_trained']
assert json.loads((OUT / state['mechanism_dataset_run_receipt']).read_text(encoding='utf8'))['status'] == 'passed'
assert not plan['scene1']['train_new_transferable_modules']
old=(backup/'outputs/V43_Ownership_20261027/index.html').read_text(encoding='utf8')
historical_headers=['历史CPU候选：1 / 0 / unknown', '历史CPU代理预览：尚不能通过归属验收', '2026-10-27历史调查：199帧与共享工作台']
for title in historical_headers:
    pattern=r'<section><h2>'+re.escape(title)+r'</h2>.*?</section>'
    assert re.search(pattern,old,re.S).group() == re.search(pattern,page,re.S).group(), title
diagram_receipt=json.loads((OUT/'architecture_validation.json').read_text(encoding='utf8'))
assert diagram_receipt['architecture_revision'] == 5
assert len(diagram_receipt['diagrams']) == 2
assert all(not item['text_overflow'] and item['native_canvas_rendered'] for item in diagram_receipt['diagrams'])
required_fragments=['GT 定义在独立世界','alpha×T','真值辅助归属参考','真值辅助表示修复参考',
                    '1 个最小普通场景的实际重建','3–5','有车减无车不能直接称纯阴影 GT','场景1仅用于开发诊断和迁移推理']
for fragment in required_fragments:
    assert fragment in DESIGN, fragment
receipt={'architecture_revision':5,'passed':True,'local_links_checked':len(links),'broken_links':broken,
    'generator_idempotent':True,'historical_sections_preserved':historical_headers,
    'document_and_contract_consistency':True,'nested_main_architecture_preserved':True,
    'diagrams_canvas_rendered':2,'diagram_text_overflow':False,'browser_interaction_qa':False,
    'validation_scope':'document_generator_only_not_experiment_acceptance',
    'world_gt_and_legal_input_isolation_contract_checked':True,
    'truth_assisted_references_not_mathematical_bounds_checked':True,
    'backup_path':str(backup.relative_to(BASE)).replace('\\','/'),
    'new_experiment_runs':0,'new_training_updates':0,'new_scene_renders':0,'human_verdict':None}
(OUT/'design_revision_validation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(receipt,ensure_ascii=False))
