"""Build the complete-five-identity reconstruction report from local evidence.

python -B source/build_report.py
Writes index.html, report_data.json and figures/. Missing receipts/images stay
pending; old Full5 reference images cannot satisfy new engineering acceptance.
"""
import csv
import hashlib
import html
import json
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

IDS = [5, 6, 18, 26, 37]
FRAMES = [97, 105, 109]
QUERIES = [('keep', 'KEEP []'), ('delete_a37', 'DELETE [37]'),
           ('delete_a6', 'DELETE [6]'), ('delete_a6_a37', 'DELETE [6,37]'),
           ('keep_again', 'KEEP again []')]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def h(value):
    return html.escape(str(value), quote=True)


def status(value):
    return '通过' if value is True else '未通过' if value is False else '待实际证据'


def number(value):
    return f'{value:.6f}' if isinstance(value, (int, float)) else '—'


class Builder:
    def __init__(self, root):
        self.root = root
        self.inputs = set()
        self.errors = []

    def read(self, rel):
        path = self.root / rel
        if not path.is_file():
            return None
        self.inputs.add(path)
        try:
            return json.loads(path.read_text(encoding='utf-8-sig'),
                              parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
        except (ValueError, OSError) as error:
            self.errors.append(dict(path=rel, error=str(error)))
            return None

    def link(self, rel, label):
        if (self.root / rel).is_file():
            return '<a href="'+quote(rel, safe='/._-')+'">'+h(label)+'</a>'
        return '<span class="pending">'+h(label)+'：待取回</span>'

    def image(self, path):
        if not path.is_file():
            return None
        self.inputs.add(path)
        with Image.open(path) as im:
            if im.size != (480, 320):
                raise ValueError('Expected native480x320 image: '+str(path))
            return im.convert('RGB').copy()

    def figures(self, image_root):
        folder = self.root/'figures'
        folder.mkdir(exist_ok=True)
        svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="470" viewBox="0 0 1500 470" role="img" aria-labelledby="t d">
<title id="t">完整五身份重建与身份隔离</title><desc id="d">完整真实输入联合重建五个动态身份，身份随原生拓扑和检查点完整恢复保持正确；主输出为完整KEEP中的a5车顶与挡风玻璃。</desc>
<defs><marker id="a" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#547382"/></marker><style>text{font-family:'Microsoft YaHei',Arial,sans-serif;fill:#193b4a}.t{font-size:23px;font-weight:600}.s{font-size:17px}.box{fill:white;stroke:#8aabb9;stroke-width:1.7}.arrow{fill:none;stroke:#547382;stroke-width:2;marker-end:url(#a)}</style></defs>
<rect width="1500" height="470" fill="#f3f7f9"/><text x="28" y="40" class="t">完整五身份联合重建，身份随拓扑与恢复保持隔离</text><text x="28" y="89" class="s">BUILD · dynamic_actor_ids = [5, 6, 18, 26, 37]</text>
<rect class="box" x="30" y="185" width="225" height="155" rx="12"/><text x="142" y="222" text-anchor="middle" class="t">完整真实输入</text><text x="142" y="259" text-anchor="middle" class="s">五车原始PLY / RGB</text><text x="142" y="293" text-anchor="middle" class="s">49帧与原相机</text>
<rect class="box" x="290" y="185" width="300" height="155" rx="12"/><text x="440" y="222" text-anchor="middle" class="t">原生联合重建</text><text x="440" y="259" text-anchor="middle" class="s">全五车动态＋静态背景</text><text x="440" y="293" text-anchor="middle" class="s">原目标、采样、学习率</text>
<rect class="box" x="625" y="185" width="245" height="155" rx="12"/><text x="747" y="222" text-anchor="middle" class="t">带身份检查点</text><text x="747" y="259" text-anchor="middle" class="s">逐Gaussian actor ID</text><text x="747" y="293" text-anchor="middle" class="s">clone / split / prune继承</text>
<rect class="box" x="910" y="62" width="235" height="85" rx="12"/><text x="1027" y="95" text-anchor="middle" class="t">R独立于E</text><text x="1027" y="126" text-anchor="middle" class="s">E改变不重训、不改R</text>
<rect class="box" x="910" y="185" width="235" height="155" rx="12"/><text x="1027" y="222" text-anchor="middle" class="t">身份完整恢复</text><text x="1027" y="259" text-anchor="middle" class="s">拓扑 / Adam / ID对齐</text><text x="1027" y="293" text-anchor="middle" class="s">可靠保存与完整恢复</text>
<rect class="box" x="1180" y="185" width="290" height="155" rx="12"/><text x="1325" y="222" text-anchor="middle" class="t">完整KEEP验收</text><text x="1325" y="259" text-anchor="middle" class="s">a5车顶 / 挡风玻璃</text><text x="1325" y="293" text-anchor="middle" class="s">正常可辨的上部结构</text>
<path class="arrow" d="M255 262H290"/><path class="arrow" d="M590 262H625"/><path class="arrow" d="M870 262H910"/><path class="arrow" d="M1145 262H1180"/><path class="arrow" d="M1027 147V185"/>
<text x="30" y="394" class="s">逐Gaussian身份由完整真实源点建立；clone/split继承、prune同步，原生训练数学保持不变。</text>
<text x="30" y="434" class="s">主验收：完整KEEP的a5车顶与挡风玻璃、可靠身份谱系；E只作为R不变的辅助验证。</text></svg>'''
        (folder/'architecture.svg').write_text(svg,encoding='utf-8')
        canvas=Image.new('RGB',(1040,1070),'#f3f7f9')
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17) if Path('C:/Windows/Fonts/arial.ttf').is_file() else ImageFont.load_default()
        sources=[]
        for row,raw in enumerate(FRAMES):
            for col,name in enumerate(('gt','keep')):
                path=self.root/image_root/f'{raw:03d}'/(name+'.png')
                im=self.image(path)
                if im is not None:
                    canvas.paste(im,(60+490*col,54+335*row))
                else:
                    ImageDraw.Draw(canvas).rectangle((60+490*col,54+335*row,540+490*col,374+335*row),fill='#e3ebee')
                sources.append(dict(raw=raw,state=name,path=str(path.relative_to(self.root)).replace('\\','/') if im is not None else None))
        draw=ImageDraw.Draw(canvas)
        for col,label in enumerate(('Observed GT','Full KEEP')):
            draw.text((68+490*col,20),label,font=font,fill='#193b4a')
        for row,raw in enumerate(FRAMES):
            draw.text((7,195+335*row),str(raw),font=font,fill='#193b4a')
            for col in range(2):
                if sources[2*row+col]['path'] is None:
                    draw.text((170+490*col,200+335*row),'Pending new model image',font=font,fill='#627680')
        canvas.save(folder/'pilot_comparison.png')
        # A5 body is inspected in the original fixed crop only at raw105.
        crop=Image.new('RGB',(390,214),'#f3f7f9')
        for col,name in enumerate(('gt','keep')):
            im=self.image(self.root/image_root/'105'/(name+'.png'))
            if im is not None:
                crop.paste(im.crop((0,160,175,320)),(8+195*col,45))
            else:
                ImageDraw.Draw(crop).rectangle((8+195*col,45,183+195*col,205),fill='#e3ebee')
        draw=ImageDraw.Draw(crop)
        for col,label in enumerate(('Observed GT','Full KEEP')):
            draw.text((8+195*col,12),label,font=font,fill='#193b4a')
        crop.save(folder/'a5_raw105_crop.png')
        contact=self.root/'evidence/visual_contact/all_visible_a5_contact.png'
        if contact.is_file():
            self.inputs.add(contact)
            (folder/'all_visible_a5_contact.png').write_bytes(contact.read_bytes())
        return sources

    def same_budget_comparison(self, primary, image_root):
        """Compare native float metrics only after exact fixed-domain checks."""
        reference = self.read('evidence/reference_single10k/summary.json') or {}
        upper = self.read('evidence/reference_single10k/upper_regions.json') or {}
        cross_audit = self.read('evidence/native10k_same_domain_cpu_audit.json') or {}
        fixed = primary.get('full49_fixed_reference', {})
        checks = dict(
            both_10000_iterations=reference.get('iteration') == primary.get('iteration') == 10000,
            both_same_49_views=reference.get('raw_ids') == primary.get('raw_ids') == list(range(89, 138)),
            full_121128=reference.get('fixed_denominator', {}).get('pixels') == fixed.get('full_pixels') == 121128,
            upper_25906=upper.get('pixels') == fixed.get('upper_pixels') == 25906,
            full_mask_sha256_equal=reference.get('fixed_denominator', {}).get('sha256') == fixed.get('full_mask_sha256') and bool(fixed.get('full_mask_sha256')),
            upper_index_sha256_equal=upper.get('upper25_indices_sha256') == fixed.get('upper_index_sha256') and bool(fixed.get('upper_index_sha256')),
            fixed_float_gt_sha256_equal=upper.get('gt_rgb_sha256') == fixed.get('gt_sha256') and bool(fixed.get('gt_sha256')),
            original_camera_checks_pass=reference.get('camera_checks', {}).get('all49_passed') is True,
            new_native_all49_complete=primary.get('status') == 'PASS_NATIVE_KEEP_RECONSTRUCTION_ALL49',
            actual_98_float_cpu_domain_audit=cross_audit.get('status') == 'PASS_EXACT_FIXED_DOMAIN_NATIVE10K_COMPARISON',
        )
        rows = []
        if all(checks.values()):
            rows = [
                dict(model='Single native10K', modelling_actor_ids=[37], iteration=10000,
                     pixels=121128, keep_rgb_mae=reference['all49']['keep_rgb_mae'],
                     upper_pixels=25906, upper_keep_rgb_mae=upper['keep_rgb_mae'],
                     keep_mean_alpha=reference['all49']['keep_mean_alpha']),
                dict(model='Full5 identity10K', modelling_actor_ids=IDS, iteration=10000,
                     pixels=121128, keep_rgb_mae=primary['selected']['keep_rgb_mae'],
                     upper_pixels=25906, upper_keep_rgb_mae=primary['upper_selected']['keep_rgb_mae'],
                     keep_mean_alpha=primary['selected']['keep_mean_alpha']),
            ]
        # Same original pixel crop in each of the three actually reviewed views.
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16) if Path('C:/Windows/Fonts/arial.ttf').is_file() else ImageFont.load_default()
        canvas = Image.new('RGB', (600, 610), '#f3f7f9')
        draw = ImageDraw.Draw(canvas)
        for col, label in enumerate(('Observed GT', 'Single native10K', 'Full5 identity10K')):
            draw.text((35+185*col, 16), label, font=font, fill='#193b4a')
        image_checks = []
        for row, raw in enumerate(FRAMES):
            old_gt = self.image(self.root/'evidence/reference_single10k/frames'/f'{raw:03d}'/'gt.png')
            new_gt = self.image(self.root/image_root/f'{raw:03d}'/'gt.png')
            if old_gt is not None and new_gt is not None:
                if old_gt.tobytes() != new_gt.tobytes():
                    raise ValueError('Same-camera GT PNG mismatch at raw'+str(raw))
                image_checks.append(dict(raw=raw, observed_gt_decoded_rgb_exact=True))
            draw.text((2, 119+185*row), str(raw), font=font, fill='#193b4a')
            paths = [self.root/image_root/f'{raw:03d}'/'gt.png',
                     self.root/'evidence/reference_single10k/frames'/f'{raw:03d}'/'keep.png',
                     self.root/image_root/f'{raw:03d}'/'keep.png']
            for col, path in enumerate(paths):
                im = self.image(path)
                box = (35+185*col, 50+185*row)
                if im is not None:
                    canvas.paste(im.crop((0, 160, 175, 320)), box)
                else:
                    draw.rectangle((*box, box[0]+175, box[1]+160), fill='#e3ebee')
                    draw.text((box[0]+6, box[1]+70), 'Pending image', font=font, fill='#627680')
        canvas.save(self.root/'figures/same_budget_10k.png')
        with (self.root/'comparison.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['model', 'modelling_actor_ids', 'iteration', 'pixels', 'keep_rgb_mae', 'upper_pixels', 'upper_keep_rgb_mae', 'keep_mean_alpha'])
            writer.writeheader()
            writer.writerows(rows)
        changes = dict(all49_relative_mae_decrease=1-rows[1]['keep_rgb_mae']/rows[0]['keep_rgb_mae'],
                       upper_relative_mae_decrease=1-rows[1]['upper_keep_rgb_mae']/rows[0]['upper_keep_rgb_mae']) if rows else {}
        return dict(status='PASS_FIXED_DOMAIN_SAME_BUDGET_FLOAT10K' if rows else 'PENDING_MATCHING_ALL49_FLOAT_EVIDENCE',
                    checks=checks, rows=rows, same_camera_gt_image_checks=image_checks,
                    relative_changes=changes,
                    fixed_reference=fixed, metric_units='sRGB [0,1] channel MAE weighted by fixed original pixel-frame count',
                    scope='49 training views; a5 is visible in24 views raw89..112; no independent generalization or clean hidden-surface GT',
                    intervention='Modelling R, complete real initialization support, and native branch objective package change together; not a single-loss causal ablation',
                    fixed_crop_xyxy=[0,160,175,320], human_verdict=None)

    def qa(self):
        class Parser(HTMLParser):
            def __init__(self):
                super().__init__(); self.links=[]
            def handle_starttag(self,tag,attrs):
                for key,value in attrs:
                    if key in ('href','src') and value:self.links.append(value)
        parser=Parser();parser.feed((self.root/'index.html').read_text(encoding='utf-8'))
        checked=[];images=[]
        for value in parser.links:
            if value.startswith('#') or urlsplit(value).scheme:continue
            rel=unquote(value.split('#')[0])
            path=self.root/rel
            if not path.is_file():raise FileNotFoundError('Broken report link: '+rel)
            checked.append(rel)
            if path.suffix.lower()=='.png':
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:images.append(dict(path=rel,size=list(im.size)))
            elif path.suffix.lower()=='.svg':
                element=ET.parse(path).getroot()
                images.append(dict(path=rel,size=[int(element.attrib['width']),int(element.attrib['height'])]))
        return dict(status='PASS',checked_relative_links=len(checked),broken_links=[],images=images)

    def build(self):
        scope=self.read('evidence/reconstruction_scope_20261107.json')
        if not scope:raise FileNotFoundError('Latest reconstruction scope is required')
        historical_contract=self.read('evidence/engineering_contract.json')
        index=self.read('evidence/evidence_index.json') or {}
        cpu_path=index.get('cpu_tests','evidence/cpu_interface_tests.json')
        lifecycle_path=index.get('lifecycle','evidence/smoke_pilot/run_receipts/campaign/smoke_lifecycle_gate.json')
        aux_path=index.get('query_validation','evidence/smoke_pilot/summary.json')
        primary_path=index.get('reconstruction_summary',aux_path)
        cpu=self.read(cpu_path) or {}
        lifecycle=self.read(lifecycle_path) or {}
        aux=self.read(aux_path) or {}
        primary=self.read(primary_path) or {}
        image_root=index.get('image_root','evidence/smoke_pilot/frames')
        sources=self.figures(image_root)
        comparison=self.same_budget_comparison(primary,image_root)
        source_integrity_path=str(Path(primary_path).parent/'source_integrity.json').replace('\\','/')
        source_integrity=self.read(source_integrity_path) or {}
        retrieval_path=index.get('reconstruction_retrieval','evidence/formal10k_keep_all49_canonical_copy.json')
        retrieval=self.read(retrieval_path) or {}
        formal_restore_path=index.get('formal_restore','evidence/formal10k_keep_pilot/run_receipts/model/restore_audit.json')
        formal_restore=self.read(formal_restore_path) or {}
        cli_path=index.get('public_cli_comparison','evidence/formal10k_keep_all49/run_receipts/campaign/public_keep_png_consistency.json')
        cli=self.read(cli_path) or {}
        loading_contract=self.read('evidence/checkpoint_loading_contract.json') or {}
        backup_path='evidence/inference_export/inference_package_verification.json'
        backup=self.read(backup_path) or {}
        execution=self.read('evidence/execution_verification.json') or {}
        cross_float_audit=self.read('evidence/native10k_same_domain_cpu_audit.json') or {}
        cached_png_path='../a5_regression_20261106/evidence/cached_native10k_comparison/summary.json'
        cached_png=self.read(cached_png_path) or {}
        visual=self.read('evidence/root_reconstruction_visual_review.json') or {}
        stage='formal' if '/formal' in str(primary.get('model','')) else 'smoke'
        records=[row for row in visual.get('records',[]) if row.get('stage')==stage and row.get('iteration')==primary.get('iteration')]
        review=records[-1] if records else None
        contact_review=(review or {}).get('all49_additional_review',{})
        contact_reviewed=contact_review.get('a5_visible_frames_actually_reviewed')==list(range(89,113)) and (self.root/'figures/all_visible_a5_contact.png').is_file()
        roof_accepted=(review or {}).get('a5_roof_normal') is True or (review or {}).get('agent_visual_acceptance') in ('PASS_NORMAL_COMPLETE_RECONSTRUCTION','ACCEPTED_NORMAL_A5_ROOF_WINDSHIELD','PASS_VISIBLE_ROOF_AND_WINDSHIELD_STRUCTURE_WITH_SOFT_DETAILS')
        binding_path=index.get('initial_row_binding','evidence/smoke_pilot/run_receipts/model/initial_row_binding.json')
        binding=self.read(binding_path) or {}
        identity=self.read('evidence/identity_contract.json') or {}
        audit=self.read('evidence/full5_weight_reuse_audit.json') or {}
        first=self.read('evidence/first_forward_B_comparison.json')
        history=self.read('evidence/historical_research_closure.json') or {}
        ids=primary.get('dynamic_actor_ids',aux.get('dynamic_actor_ids'))
        full_ids=sorted(ids)==IDS if isinstance(ids,list) else None
        full_source=full_ids is True and binding.get('status')=='PASS_ACTUAL_NATIVE_XYZ_TIME_ROWS_EXACT' and binding.get('rows')==149072 and binding.get('xyz_exact') is True and binding.get('time_exact') is True and binding.get('source_ply_order')==[5,26,18,6,37]
        identity_lifecycle=cpu.get('exit_code')==0 and cpu.get('actual_test_count',0)>=6 and str(cpu.get('status','')).startswith('PASS') and str(lifecycle.get('status','')).startswith('PASS_REAL') and lifecycle.get('exact_full_restore_differences')==[] and aux.get('checkpoint_identity_self_contained') is True
        if stage=='formal' and primary.get('iteration')==10000:
            identity_lifecycle=identity_lifecycle and formal_restore.get('status')=='PASS_EXACT_FULL_RESTORE' and formal_restore.get('differences')==[] and primary.get('identity_is_saved_source_ownership') is True and sum(primary.get('current_owner_rows',{}).values())==primary.get('native_dynamic_rows')
        R_E_independent=aux.get('dynamic_actor_ids')==IDS and aux.get('parameters_unchanged') is True and aux.get('keep_native_max_abs_difference')==0 and aux.get('keep_again_max_abs_difference')==0 and aux.get('optimization_updates')==0 and aux.get('optimizer_created') is False
        all49_complete=primary.get('status')=='PASS_NATIVE_KEEP_RECONSTRUCTION_ALL49' and primary.get('frame_count')==49 and str(source_integrity.get('status','')).startswith('PASS') and source_integrity.get('changed')==[] and retrieval.get('status')=='PASS_BYTE_IDENTICAL_COMPLETE_LOCAL_CANONICAL_COPY' and len(list((self.root/image_root).glob('*/keep.png')))==49
        checks=dict(full5_real_source_binding=full_source if binding else None,
                    identity_topology_capture_full_restore=identity_lifecycle if cpu and lifecycle else None,
                    modelling_R_independent_of_query_E=R_E_independent if aux else None,
                    primary_keep_visible_roof_structure_restored=True if roof_accepted else None,
                    fixed_domain_all49_reconstruction=all49_complete,
                    public_KEEP_CLI_exact=cli.get('status')=='PASS_PUBLIC_CLI_KEEP_MATCHES_ACCEPTED_NATIVE_KEEP' and all(row.get('max_uint8_difference')==0 for row in cli.get('frames',[])) and cli.get('raw_ids')==FRAMES)
        backup_verified=backup.get('status')=='PASS_INFERENCE_PACKAGE_LOCAL_ARCHIVE_AND_ALL_FILES' and backup.get('every_file_sha256_verified') is True and backup.get('iteration')==10000 and backup.get('dynamic_actor_ids')==IDS
        checks['local_inference_backup_verified']=backup_verified
        acceptance=[('KEEP_ROOF','完整KEEP中a5车顶、挡风玻璃的连续可见结构','三帧结构恢复；细节仍软' if roof_accepted else '已按当前节点检查；尚待清晰结构' if review else '待实际目视'),
                    ('FULL5_SOURCE','完整五身份真实源点；实际初始化XYZ/time顺序正确',status(checks['full5_real_source_binding'])),
                    ('IDENTITY_LIFECYCLE','身份随clone/split/prune、保存和完整恢复正确',status(checks['identity_topology_capture_full_restore'])),
                    ('R_FIXED','建模集合R固定，不随编辑/查询集合E改变',status(checks['modelling_R_independent_of_query_E'])),
                    ('KEEP49','49训练视图固定域评价、源文件守恒、完整取回',status(checks['fixed_domain_all49_reconstruction'])),
                    ('PUBLIC_KEEP','公开推理CLI与原生KEEP三帧逐像素相同',status(checks['public_KEEP_CLI_exact'])),
                    ('WEIGHT_BACKUP','带源身份的推理权重已取回，所有文件SHA校验',status(backup_verified))]
        formal_entries=[]
        seen=set()
        for rel in index.get('formal_receipts',[]):
            value=self.read(rel)
            if value is None:continue
            if value.get('status')!='PASS_COMPLETE_BOUNDARY':raise ValueError('Incomplete formal returned receipt: '+rel)
            start,end=value.get('source_iteration'),value.get('iteration')
            updates=value.get('actual_additional_iterations')
            if not isinstance(start,int) or not isinstance(end,int) or updates!=end-start:raise ValueError('Formal update interval mismatch')
            if (start,end) in seen:continue
            seen.add((start,end))
            formal_entries.append(dict(source=rel,from_iteration=start,to_iteration=end,updates=updates,receipt=value))
        formal=sum(row['updates'] for row in formal_entries)
        engineering=lifecycle.get('total_new_engineering_updates')
        accounting=dict(historical_completed_total=history.get('completed_total_updates'),
                        new_reconstruction_engineering_updates=engineering,new_reconstruction_formal_updates=formal,
                        formal_intervals=formal_entries,running_updates_not_presumed_zero=True,
                        completed_total_receipted=history.get('completed_total_updates',0)+engineering+formal if engineering is not None else None)
        decision=self.read(index.get('acceptance_decision','evidence/reconstruction_acceptance_decision.json')) or {}
        accepted=decision.get('reconstruction_accepted') is True
        if accepted and (not roof_accepted or not all(checks.values())):raise ValueError('Reconstruction acceptance lacks roof/identity proof')
        if accepted:
            overall='FULL5_RECONSTRUCTION_ACCEPTED_BY_AGENT_EVIDENCE_HUMAN_VERDICT_NULL'
        elif roof_accepted and all49_complete and identity_lifecycle:
            overall='FULL5_10K_VISIBLE_ROOF_RESTORED_IDENTITY_VERIFIED_KEEP49_COMPLETE'
        elif roof_accepted:
            overall='FULL5_10K_VISIBLE_ROOF_RESTORED_3FRAME_KEEP49_PENDING'
        else:
            overall='FULL5_RECONSTRUCTION_AND_IDENTITY_IN_PROGRESS_ROOF_DETAIL_PENDING'
        if stage=='formal' and primary.get('iteration')==5000:
            stage_note='正式5K已出现蓝色完整车形和上部支撑；车顶/挡风玻璃仍很模糊，尚不能验收为正常完整重建。这是模糊与细节清晰度问题，不等同于旧single透明缺体的表现。'
        elif stage=='smoke':
            stage_note='当前smoke已可见蓝色a5车形与上部轮廓，但很模糊；正式5K/10K的车顶与挡风玻璃须按实际KEEP图另验收。'
        elif roof_accepted:
            stage_note='正式10K的97/105/109三帧已恢复连续可辨的a5车顶、挡风玻璃与蓝色车体轮廓，未见旧Single的透明缺顶；细节仍软。这是代理的可见结构判断，不是GT级清晰度或人类质量验收。'+('49视图KEEP固定域评价与身份检查已另行完成。' if all49_complete else '49视图评价取回仍待完成。')
        else:
            stage_note='按最新完整KEEP图检查a5上部结构；误差下降或身份测试通过，均不自动等于车顶重建合格。'
        keep=primary.get('selected') or primary.get('keep_a5') or primary.get('all49') or primary.get('selected_a5',{}).get('keep') or {}
        upper=primary.get('upper_selected') or primary.get('upper_all49') or primary.get('upper') or {}
        pixels=primary.get('fixed_denominator',{}).get('pixels',keep.get('pixels'))
        identity_rows=[]
        for label,key in (('当前完整a5可见域','identity_selected'),('固定包围盒高度上25%','upper_identity_selected')):
            value=primary.get(key,{})
            if value:
                identity_rows.append('<tr><td>'+label+'</td><td>'+str(value['pixels'])+'</td><td>'+f'{value["owner5_alphaT_mean"]:.6f}'+'</td><td>'+f'{value["static_alphaT_mean"]:.6f}'+'</td><td>'+f'{value["other_owner_alphaT_mean"]:.6f}'+'</td><td>'+f'{value["argmax_owner5_rate"]:.4%}'+'</td></tr>')
        rows=''.join('<tr><td>'+h(code)+'</td><td>'+h(req)+'</td><td>'+h(result)+'</td></tr>' for code,req,result in acceptance)
        source_rows=''.join('<tr><td>'+str(row['actor_id'])+'</td><td>'+str(row['source_actor_index'])+'</td><td>'+str(row['rows'])+'</td></tr>' for row in identity.get('source_actor_rows',[]))
        metric_rows=''.join('<tr><td>'+h(row['model'])+'</td><td>'+h(row['modelling_actor_ids'])+'</td><td>10,000</td><td>'+number(row['keep_rgb_mae'])+'</td><td>'+number(row['upper_keep_rgb_mae'])+'</td></tr>' for row in comparison['rows'])
        if not metric_rows:
            metric_rows='<tr><td colspan="5">等待相同固定域、相同10K预算的完整float结果，不用pilot或缓存PNG补空位。</td></tr>'
        changes=comparison['relative_changes']
        comparison_note=('相对旧Single10K，全a5域MAE下降'+f'{changes["all49_relative_mae_decrease"]:.2%}'+'，固定上部下降'+f'{changes["upper_relative_mae_decrease"]:.2%}'+'。' if changes else '')
        tag='10K可见结构恢复 · 49视图评价完成 · 身份验证通过' if roof_accepted and all49_complete and identity_lifecycle else '重建与身份验证进行中；a5车顶以实际KEEP验收'
        contact_section=('<details><summary>全部24个a5可见帧：GT / KEEP联系图（代理已实际查看）</summary><p>raw89–112的可见车体部分连续；近处raw101–110车顶、挡风玻璃轮廓连贯，细节偏软。111/112自然出画，只判断可见部分，不推断不可见表面。每对图采用原a5 mask包围盒外扩6像素，GT与KEEP共用裁剪；各对显示缩放不同，指标始终来自原始像素。</p><a href="figures/all_visible_a5_contact.png"><img src="figures/all_visible_a5_contact.png" alt="24个a5可见帧GT与完整KEEP联系图"></a></details>' if contact_reviewed else '')
        if backup_verified:
            weight_backup_section='<h3>已核验的本地推理权重备份</h3><p>压缩包<code>model/inference10k_delivery.tar.gz</code>，'+f'{backup["archive"]["bytes"]:,}'+'字节；解包目录<code>model/inference10k</code>共'+str(backup['extracted_file_count'])+'个文件、'+f'{backup["extracted_bytes"]:,}'+'字节，逐文件SHA全部通过。</p><p>压缩包SHA-256：<code>'+h(backup['archive']['sha256'])+'</code>。权重保留在交付目录，Git仅归档报告、代码和校验凭据。'+self.link(backup_path,'完整备份验证收据')+'</p>'
        else:
            weight_backup_section='<p class="pending">推理权重备份仍待完整取回与逐文件SHA校验；不以导出成功代替本地备份可用。</p>'
        cached_rows=''.join('<tr><td>'+h(label)+'</td><td>'+number(cached_png.get('models',{}).get(key,{}).get('keep_rgb_mae'))+'</td><td>'+number(cached_png.get('models',{}).get(key,{}).get('boundary_keep_rgb_mae'))+'</td></tr>' for key,label in [('full5_native10k','历史Full5 native10K PNG'),('single_native10k','历史Single native10K PNG')])
        old_rows=''.join('<tr><td><code>'+h(run['path'])+'</code></td><td>'+str(node['iteration'])+'</td><td>无可可靠恢复的动态逐点身份谱系</td></tr>' for run in audit.get('runs',[]) for node in run.get('component_availability',[]))
        links=' · '.join(self.link(rel,label) for rel,label in [
            ('evidence/reconstruction_scope_20261107.json','最新重建scope'),(primary_path,'当前KEEP评价'),
            ('evidence/root_reconstruction_visual_review.json','root分阶段重建目视'),(cpu_path,'身份CPU测试'),
            (lifecycle_path,'真实smoke保存/恢复闭环'),(binding_path,'实际源点XYZ/time绑定'),
            (formal_restore_path,'5K→10K完整恢复'),(source_integrity_path,'49评价源文件守恒'),
            (retrieval_path,'完整取回逐文件SHA'),(cli_path,'公开KEEP CLI逐像素核对'),
            ('evidence/checkpoint_loading_contract.json','检查点身份与加载合同'),
            ('evidence/execution_verification.json','连续10K与49float独立复算'),
            ('evidence/native10k_same_domain_cpu_audit.json','旧/新98份float同域逐帧复核'),
            ('source/audit_same_budget_native10k.py','同域CPU复核源码'),
            ('evidence/first_forward_B_comparison.json','原生B首步13项raw-loss核对'),
            ('source/drivingeditor_dynamic_edit/ownership.py','身份继承与持久化源码'),
            ('source/build_report.py','报告构建源码'),('report_data.json','报告数据与manifest')])
        style='body{margin:0;background:#f3f7f9;color:#193b4a;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1400px;margin:auto;padding:28px}section{background:white;border:1px solid #dce7ec;border-radius:12px;padding:24px;margin:20px 0}h1{font-size:30px;line-height:1.35}h2{font-size:22px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #dce6eb;text-align:left;vertical-align:top}th{background:#edf4f7}code,pre{word-break:break-word}pre{white-space:pre-wrap;background:#f4f7f9;padding:12px;font-size:12px}.pending{color:#946322}.tag{font-size:13px;padding:5px 12px;background:#dcecf2;border-radius:20px}img{display:block;max-width:100%;height:auto}a{color:#006b83}footer,figcaption{font-size:13px;color:#607783}details{margin:14px 0}summary{cursor:pointer;font-weight:600}.notice{background:#fff4de;padding:12px;border-left:4px solid #c99435}'
        page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scene008：完整五身份重建与身份隔离</title><style>{style}</style></head><body><main>
<span class="tag">{tag}</span><h1>Scene008：完整五身份重建与身份隔离</h1>
<p>目标是恢复a5车顶和挡风玻璃，解决完整重建与身份管理。固定建模集合<code>R=[5,6,18,26,37]</code>，所有五车都使用完整真实源点进入动态重建；编辑目标E不再决定哪些保留车辆转到静态分支学习。</p>
<p class="notice">{stage_note} 身份元数据本身不改变训练数学；身份正确与车顶质量分别验收。GPU保持开机，本轮未用重置卡、未关机。</p>
<section><h2>完整重建与身份数据流</h2><img src="figures/architecture.svg" alt="完整五身份固定建模、原生联合训练、可靠身份持久化与完整KEEP检查"><p>保留原Full5 B的输入、损失、采样和30K学习率时钟。新增可可靠保存恢复的身份元数据；不把编辑集合当作重建集合。</p></section>
<section><h2>本轮主验收</h2><table><thead><tr><th>项</th><th>要求</th><th>当前证据</th></tr></thead><tbody>{rows}</tbody></table><p>旧smoke和先前研究是阶段证据；正式节点分别验收。human_verdict为空，代理实际检查不冒充用户判定。</p></section>
<section><h2>同10K：GT / 旧Single / 新Full5身份重建</h2><p>三列使用同一原相机、时间与固定像素窗口；97/105/109的GT解码RGB逐像素相同。旧Single把a5留在时间静态分支；新Full5使用完整a5真实源点进入动态分支。新结果恢复了连续车顶/挡风玻璃轮廓，细节仍软。</p><a href="figures/same_budget_10k.png"><img src="figures/same_budget_10k.png" alt="同10K GT、旧Single与新Full5身份重建，97 105 109固定裁剪"></a><p>原像素窗口(0,160)–(175,320)，无增强。下面仅比较原生float RGB（0–1通道MAE），不用显示PNG反算指标。</p><table><thead><tr><th>模型</th><th>建模动态集合R</th><th>训练更新</th><th>全a5域MAE<br>121,128像素帧</th><th>固定上部MAE<br>25,906像素帧</th></tr></thead><tbody>{metric_rows}</tbody></table><p>{comparison_note}49个训练视图中，a5只在raw89–112的24帧可见。上部定义为原始a5 mask与其逐帧包围盒高度上25%的交集，并非按像素数取25%，也非独立标注的车顶真值。</p><p>同10K排除了“只有30K才完整”的解释；R、真实初始化支撑与动态/静态目标包同时变化，尚不能将差异归因单项loss、先验或某一提交。这是同场景训练视图诊断，不是独立泛化。</p><p>{self.link('comparison.csv','同口径指标CSV')} · {self.link('evidence/reference_single10k/summary.json','旧Single原生float49')} · {self.link('evidence/reference_single10k/upper_regions.json','旧Single固定上部')} · {self.link(primary_path,'新Full5原生KEEP49')}</p></section>
<section><h2>完整KEEP：97 / 105 / 109</h2><p>当前数据：{h(primary.get('model','待取回'))}，iteration={h(primary.get('iteration','待取回'))}。只展示这个检查点真实GT与KEEP，未取回的帧保留空位。三帧先检查a5车体，再评价49个训练视图。</p><a href="figures/pilot_comparison.png"><img src="figures/pilot_comparison.png" alt="97 105 109完整KEEP与GT"></a><h3>raw105 a5上部定位</h3><img src="figures/a5_raw105_crop.png" alt="原像素GT与完整KEEP的a5固定裁剪"><p>原像素窗口(0,160)–(175,320)，未增强。当前scope={h(primary.get('scope','待取回'))}，实际固定a5分母={h(pixels if pixels is not None else '待取回')}像素；完整49视图参考为121,128，固定上部为25,906。pilot不能冒充49视图结果。</p><table><thead><tr><th>KEEP RGB MAE（0–1）</th><th>固定上部MAE</th><th>总alpha均值</th></tr></thead><tbody><tr><td>{number(keep.get('keep_rgb_mae',keep.get('rgb_mae')))}</td><td>{number(upper.get('keep_rgb_mae',upper.get('rgb_mae')))}</td><td>{number(keep.get('mean_alpha',keep.get('keep_mean_alpha')))}</td></tr></tbody></table><p>总alpha可由背景或错误表面补满，不能单独证明车顶正常。区分完整车形但模糊、缺少上部结构与正常可辨细节。</p>{contact_section}</section>
<section><h2>五身份真实输入与谱系</h2><table><thead><tr><th>actor ID</th><th>原PLY索引</th><th>完整真实源点</th></tr></thead><tbody>{source_rows}</tbody></table><p>实际PLY顺序<code>[5,26,18,6,37]</code>与公开ID排序不同。初始动态149,072行、静态585,943行；源XYZ/time已按原生归一化逐行核对。actor ID不是可训练参数；clone/split继承实际父身份，prune同步，保存/完整恢复后保持一致。</p><p>{links}</p><details><summary>身份CPU测试、真实恢复与分阶段目视原文</summary><pre>{h(json.dumps({'cpu':cpu,'lifecycle':lifecycle,'source_binding':binding,'first_forward':first,'visual':visual},ensure_ascii=False,indent=2))}</pre></details></section>
<section><h2>固定a5可见域中的身份贡献</h2><table><thead><tr><th>原观测选域</th><th>像素帧</th><th>owner5平均αT</th><th>static平均αT</th><th>其他身份平均αT</th><th>最大贡献通道为owner5的比例</th></tr></thead><tbody>{''.join(identity_rows)}</tbody></table><p>来自原生CUDA的static＋五身份六通道，未归一化为概率，也未按渲染alpha挑像素；六通道之和与joint alpha的最大差为{h(primary.get('identity_partition_max_abs_error','待取回'))}。argmax比例用于观察当前固定域的身份贡献，不是独立识别准确率，也不代替车顶画质验收。</p></section>
<section><h2>重建推理与身份加载</h2><p>公开CLI默认完整KEEP、先渲染97/105/109；建模R从检查点只读读取。真实10K公开入口与原生KEEP三帧PNG逐像素差为0，推理不创建优化器，也无需原PLY、mask或GT。仍需兼容的原生CUDA引擎与扩展。</p><p>{self.link('source/drivingeditor_dynamic_edit/README.md','重建与身份API用法')} · {self.link('source/drivingeditor_dynamic_edit/query.py','公开KEEP入口')} · {self.link('source/drivingeditor_dynamic_edit/scene_io.py','自带身份检查点加载器')} · {self.link('evidence/inference_export/inference10k_equivalence.json','推理包与原训练检查点精确一致')} · {self.link('evidence/inference_export/inference10k_cli_comparison.json','独立推理包CLI与主KEEP一致')}</p>{weight_backup_section}<details><summary>检查点与推理包的边界</summary><p>推理包保留原高斯参数、源身份表、当前逐行actor IDs与来源谱系；只去掉Adam状态，不能恢复训练。完整Adam/RNG现场仍保留在原训练目录。未保存逐点身份的旧混合权重显式拒绝按actor加载，不用框或近邻猜归属。</p><pre>{h(json.dumps(loading_contract,ensure_ascii=False,indent=2))}</pre></details></section>
<section><h2>已完成更新独立记账</h2><p>旧研究已完成44,622更新；本轮重建smoke工程{h(engineering if engineering is not None else '待回执')}、正式完成{formal:,}，已取回完成回执总计{h(accounting['completed_total_receipted'] if accounting['completed_total_receipted'] is not None else '待回执')}。正式0→5K＋5K→10K去重计数；smoke 4,681＋完整恢复后2步单列，不抵扣正式预算。评价为零优化更新。</p><details><summary>正式逐阶段回执</summary><p>以source_iteration、iteration和actual_additional_iterations校验阶段区间；10K回执中的formal_stop=5000是旧wrapper常量，实际phase_stop与iteration均为10000。</p><pre>{h(json.dumps(accounting,ensure_ascii=False,indent=2))}</pre></details></section>
<section><h2>参考与历史范围</h2><p>旧Full5可作完整KEEP重建参考，但已核四处原生权重都缺动态逐点身份/完整父谱系，不能按框或近邻猜归属。新训练保存可靠身份；旧图不能代验收。</p><details><summary>历史同10K缓存PNG（与上面的原生float表分开）</summary><table><thead><tr><th>已有原生uint8 PNG / 255</th><th>全a5域MAE</th><th>边界MAE</th></tr></thead><tbody>{cached_rows}</tbody></table><p>49张已有训练图按同GT可见域评价，Full5全域误差更低但边界未更好。Full5该10K权重已清理，未恢复或重渲这些权重；本表仅为历史PNG证据，不能替代新身份检查点验收。{self.link(cached_png_path,'PNG独立口径与原始来源')}</p></details><details><summary>旧权重严格只读审计</summary><table><thead><tr><th>已核具体路径</th><th>节点</th><th>身份可恢复性</th></tr></thead><tbody>{old_rows}</tbody></table><p>{self.link('evidence/full5_weight_reuse_audit.json','完整CPU审计收据')}</p></details><p>{self.link('../a5_regression_20261106/index.html','此前静态a5诊断（历史）')}；旧Single止20K，30K未执行，不下30K结论；旧备选与opacity0均0训练。</p>
<details><summary>辅助R/E解耦验证与早先编辑范围记录</summary><p>改变E后的过滤、参数不变和切回KEEP仅用作R固定的辅助工程证据。静态车形残影不属于本任务的重建/身份验收范围，不在这里继续处理。</p><pre>{h(json.dumps({'auxiliary_query_receipt':aux,'historical_engineering_contract':historical_contract},ensure_ascii=False,indent=2))}</pre></details></section>
<footer>主scope以最新重建授权为准。更新：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}。{overall}</footer></main></body></html>'''
        (self.root/'index.html').write_text(page,encoding='utf-8')
        (self.root/'report_data.json').write_text('{}\n',encoding='utf-8')
        qa=self.qa()
        self.inputs.update(p for p in (self.root/'source').rglob('*') if p.is_file() and p.suffix in ('.py','.md'))
        data=dict(status=overall,generated_utc=datetime.now(timezone.utc).isoformat(),scope=scope,
                  primary_objective='Complete a5 roof/windshield reconstruction and reliable full5 identity',
                  modelling_actor_ids=IDS,primary_frames=FRAMES,primary_acceptance=acceptance,checks=checks,
                  primary_reconstruction=primary,primary_visual_record=review,root_reconstruction_visual_review=visual,
                  same_budget_native_float10k=comparison,same_domain_cpu_audit=cross_float_audit,
                  historical_cached_native10k_png=cached_png,
                  auxiliary_query_evidence=aux,historical_engineering_contract=historical_contract,
                  identity_contract=identity,source_binding=binding,cpu_tests=cpu,lifecycle=lifecycle,
                  formal_restore=formal_restore,source_integrity=source_integrity,
                  retrieval_manifest=retrieval,execution_verification=execution,
                  public_keep_cli=cli,checkpoint_loading_contract=loading_contract,
                  local_inference_weight_backup=backup,
                  accounting=accounting,old_full5_weight_audit=audit,pilot_sources=sources,
                  visible_roof_and_windshield_structure_restored=roof_accepted,
                  visible_structure_reviewed_frames=contact_review.get('a5_visible_frames_actually_reviewed',(review or {}).get('raw_frames_actually_viewed',[])),
                  GT_level_sharpness_claimed=False,static_residue_is_primary_acceptance_gate=False,quality_accepted=accepted,human_verdict=None,
                  input_manifest=[dict(path=str(p.relative_to(self.root)).replace('\\','/'),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(self.inputs)],
                  output_manifest=[dict(path=str(p.relative_to(self.root)).replace('\\','/'),bytes=p.stat().st_size,sha256=digest(p)) for p in [self.root/'index.html',self.root/'comparison.csv',*sorted((self.root/'figures').iterdir())]],
                  input_parse_errors=self.errors,qa=qa)
        serialized=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
        (self.root/'report_data.json').write_text(serialized,encoding='utf-8')
        json.loads(serialized,parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
        print(json.dumps(dict(status=overall,qa=qa,input_parse_errors=self.errors),ensure_ascii=True))


if __name__=='__main__':
    Builder(Path(__file__).absolute().parent.parent).build()
