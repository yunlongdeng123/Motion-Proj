"""本地报告一致性检查；不执行build_report、CUDA、SSH或训练，不写outputs。

默认核验最终HTML、72张交互图、独立参考配对和已取回JSON收据链。
--source-only只检查源码并列出待处理问题，不声称实际报告已通过。
本工具不重新验证远端checkpoint内容、FP32指标或科学结论。
"""
from pathlib import Path, PurePosixPath
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
import argparse
import ast
import html
import json
import math
import re
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
DEFAULT_OUT = BASE / 'outputs/V43_Ownership_20261027/gpu_v43/sim_mechanism_20261030'
FRAMES = ('000000', '000008', '000015')
EDITS = ('none', 'delete', 'shift', 'rotate')
CHANNELS = ('final', 'white', 'C', 'alpha', 'fill')
EXPECTED = {(frame, edit) for frame in FRAMES for edit in EDITS}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def finite(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), label)
    return value


def read(path):
    def invalid_constant(value):
        raise ValueError(f'{path}: non-finite JSON {value}')
    return json.loads(path.read_text(encoding='utf-8-sig'), parse_constant=invalid_constant)


def normalized(value):
    return f'{int(value):06d}' if str(value).isdigit() else str(value)


def safe_relative(root, relative):
    require(not PurePosixPath(relative).is_absolute() and '..' not in PurePosixPath(relative).parts,
            '引用路径越界：' + relative)
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), '引用路径越界：' + relative)
    return path


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.images, self.selects, self.tables, self.scripts = [], {}, {}, [], []
        self.image_ids = []
        self.select = self.option = self.table = self.row = self.cell = None
        self.heading = ''
        self.capture_heading = self.in_script = False

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        self.links.extend(value for key, value in attributes if key in ('src', 'href') and value)
        if tag == 'img' and attrs.get('id'):
            self.image_ids.append(attrs['id'])
            self.images[attrs['id']] = attrs
        elif tag == 'select':
            self.select = attrs.get('id')
            self.selects[self.select] = []
        elif tag == 'option' and self.select:
            self.option = {'value': attrs.get('value'), 'text': ''}
        elif tag == 'h2':
            self.capture_heading, self.heading = True, ''
        elif tag == 'table':
            self.table = {'heading': self.heading.strip(), 'rows': []}
        elif tag == 'tr' and self.table is not None:
            self.row = []
        elif tag in ('td', 'th') and self.row is not None:
            self.cell = ''
        elif tag == 'script':
            self.in_script = True
            self.scripts.append('')

    def handle_data(self, data):
        if self.option is not None:
            self.option['text'] += data
        if self.capture_heading:
            self.heading += data
        if self.cell is not None:
            self.cell += data
        if self.in_script:
            self.scripts[-1] += data

    def handle_endtag(self, tag):
        if tag == 'option' and self.option is not None:
            self.selects[self.select].append(self.option['value'] or self.option['text'].strip())
            self.option = None
        elif tag == 'select':
            self.select = None
        elif tag == 'h2':
            self.capture_heading = False
        elif tag in ('td', 'th') and self.cell is not None:
            self.row.append(self.cell.strip())
            self.cell = None
        elif tag == 'tr' and self.row is not None:
            self.table['rows'].append(self.row)
            self.row = None
        elif tag == 'table' and self.table is not None:
            self.tables.append(self.table)
            self.table = None
        elif tag == 'script':
            self.in_script = False


def source_review():
    source = (HERE / 'build_report.py').read_text(encoding='utf-8-sig')
    tree = ast.parse(source, filename='build_report.py')
    findings = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'get'
                and len(node.args) >= 2 and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in ('clamped_mae', 'raw_mae')
                and isinstance(node.args[1], ast.Constant) and node.args[1].value == 0):
            findings.append({'line': node.lineno, 'issue': 'missing_metric_defaults_to_zero',
                             'detail': '缺测或零有效像素不能显示为0误差；应拒绝未评分行并显示N/A'})
    if "['architecture.svg','architecture.png']" in source.replace(' ', ''):
        findings.append({'issue': 'template_png_copy_requires_final_svg_rerender',
                         'detail': '最终SVG改状态后须执行render_report_diagram.cjs，不能继续交付旧骨架PNG'})
    return {'source_syntax_passed': True, 'issues_for_author_review': findings,
            'build_report_executed': False, 'actual_report_evidence_validated': False}


def expression_value(expression, values):
    """只解释报告当前的字符串拼接路径，不执行HTML里的JavaScript。"""
    pieces = expression.strip().split('+')
    result = []
    for piece in pieces:
        piece = piece.strip()
        if piece in values:
            result.append(values[piece])
        else:
            value = ast.literal_eval(piece)
            require(isinstance(value, str), '交互图路径只能由字符串与f/e/k拼接')
            result.append(value)
    return ''.join(result)


def check_png(path):
    from PIL import Image
    with Image.open(path) as image:
        image.load()
        require(image.format == 'PNG' and image.size == (480, 320), f'交互图尺寸/格式错误：{path}')


def check_smoke(evidence):
    smoke = read(evidence / 'smoke_r2/smoke_summary.json')
    require(smoke['status'] == 'PASS_REAL_NATIVE_SMOKE', 'smoke未通过')
    require(smoke['approximately_ten_minutes_contract'] is True, '约10分钟smoke门槛未通过')
    require(smoke['each_leg_reached_requested_loop_wall_seconds'] is True, 'smoke未逐段达到登记时长')
    triggers = smoke['stop_trigger_timing']
    require(set(triggers) == {'fresh', 'resumed'}, 'smoke两段计时不全')
    seconds = []
    for name, trigger in triggers.items():
        require(trigger['reason'] == 'smoke', name + '未由真实smoke时长触发停止')
        value = finite(trigger['training_loop_wall_seconds_excluding_measured_eval_checkpoint'], name + '秒数无效')
        require(value >= 300, name + '训练循环未达到300秒')
        seconds.append(value)
    total = finite(smoke['timed_leg_training_loop_wall_seconds_excluding_measured_eval_checkpoint'], 'smoke总时长无效')
    require(total >= 600 and abs(total - sum(seconds)) < 1e-5, 'smoke总时长与两段不一致')
    for folder in ('resumed', 'next_stage_entry'):
        restored = read(evidence / 'smoke_r2' / folder / 'restore_verification.json')
        require(restored['all_passed'] is True and restored['checks'] and all(value is True for value in restored['checks'].values()),
                folder + '完整恢复收据未通过')
    require(smoke['complete_restore']['all_passed'] is True, 'smoke摘要完整恢复未通过')
    entry = read(evidence / 'smoke_r2/render_entry/render_manifest.json')
    require(entry['status'] == 'NATIVE_BRANCH_ROUTING_EDIT_RENDERED', '下一阶段实际渲染收据未通过')
    require(entry['training_updates'] == 0 and entry['original_checkpoint_unchanged'] is True, '下一阶段渲染修改了模型')
    require(finite(entry['native_none_identity_max_abs'], '入口none差异无效') <= 2e-6, '入口native none不一致')
    require({row['edit'] for row in entry['frames']} == {'none', 'delete'}, '下一阶段未实际覆盖none/delete')
    phases = smoke['phases']
    require(smoke['extra_smoke_updates'] == phases['fresh']['observed_finite_step_calls'] + phases['resumed']['observed_finite_step_calls'],
            'smoke额外更新记账不一致')
    require(smoke['extra_entry_updates'] == phases['entry']['observed_finite_step_calls'], '恢复入口额外更新不一致')
    return smoke


def check_heldout(root, page, state, references):
    heldout = state.get('heldout')
    require(isinstance(heldout, dict), 'report_state需明确heldout={status,views}，未执行不能省略状态')
    folder = root / 'evidence/heldout_r3_10k'
    plain = re.sub(r'<[^>]*>', '', page)
    if heldout.get('status') == 'not_executed':
        require(heldout.get('views') == 0, '未执行heldout的views应为0')
        require(not (folder / 'render_manifest.json').exists(), 'heldout已有实际渲染收据，不能仍标未执行')
        require(re.search(r'留出.{0,25}(尚未执行|未执行|待执行|未运行|尚未运行)', plain)
                or re.search(r'(尚未执行|未执行|待执行|未运行|尚未运行).{0,12}留出', plain),
                'HTML未明确留出视角尚未执行')
        return {'status': 'not_executed', 'views': 0, 'execution_claimed': False}
    require(heldout.get('status') == 'executed' and heldout.get('views') == 1, 'heldout状态只接受not_executed/0或executed/1')
    render = read(folder / 'render_manifest.json')
    metrics = read(folder / 'reference_metrics.json')
    execution = read(folder / 'execution.json')
    query = read(folder / 'query_camera.json')
    require(render['status'] == 'NATIVE_SINGLE_HELDOUT_VIEW_RENDERED' and render['iteration'] == 10000, 'heldout不是实际10K原生渲染')
    require(render['heldout_views'] == 1 and render['edit'] == 'none', 'heldout视图/操作错误')
    require(render['training_updates'] == 0 and render['original_checkpoint_unchanged'] is True, 'heldout更改模型')
    require(render['ground_truth_rgb_or_geometry_used_for_forward'] is False, 'heldout前向使用了答案')
    require(finite(render['native_none_identity_max_abs'], 'heldout none误差无效') <= 2e-6, 'heldout原生none不一致')
    require(render['actual_camera_reconstruction_max_abs'] <= 1e-5, 'heldout实际Camera重建错误')
    require(metrics['status'] == 'SCORED_ONE_INDEPENDENT_HELDOUT_VIEW' and metrics['heldout_views'] == 1, 'heldout独立评分未完成')
    require(metrics['ground_truth_read_after_forward_render'] is True and metrics['ground_truth_used_for_training'] is False,
            'heldout GT角色或读取顺序声明错误')
    require(execution['query_only'] is False and execution['status'] == metrics['status'] and execution['training_updates'] == 0,
            '只有CPU query核验不能冒充heldout渲染')
    require(query['train_input'] is False and query['ground_truth_rgb_or_geometry_used_for_query'] is False,
            'heldout query不应读取RGB/几何答案')
    require(query['original_training_camera_restore_max_abs'] <= 1e-5, 'heldout query未复现训练相机映射')
    selected = [item for item in references.get('heldout', [])
                if item['view_id'] == render['view_id'] and normalized(item['frame_id']) == normalized(render['frame'])]
    require(len(selected) == 1 and selected[0]['train_input'] is False, 'heldout独立参考与实际query不匹配')
    require(metrics['reference_rgb'].replace('\\', '/').endswith('/dataset_r2/privileged/' + selected[0]['rgb']),
            'heldout评分参考不对应独立manifest')
    families = [prefix for prefix in ('heldout_none_', '')
                if all((folder / (prefix + name + '.png')).is_file() for name in CHANNELS)]
    displayed = [prefix for prefix in families if f'evidence/heldout_r3_10k/{prefix}final.png' in page]
    require(len(displayed) == 1, 'heldout五分量文件族不完整，或HTML没有唯一对应实际final')
    for name in CHANNELS:
        check_png(folder / (displayed[0] + name + '.png'))
    check_png(folder / 'reference_rgb.png')
    require('evidence/heldout_r3_10k/reference_rgb.png' in page,
            '已执行heldout至少应在HTML展示实际final与独立reference')
    return {'status': 'executed', 'views': 1, 'json_reference_and_images_checked': True,
            'generalization_success_claim_validated': False}


def validate_report(root):
    evidence = root / 'evidence'
    page = (root / 'index.html').read_text(encoding='utf-8-sig')
    parser = ReportParser()
    parser.feed(page)
    state = read(root / 'report_state.json')
    require(state['artifact_kind'] == 'actual_run_report', '仍是报告骨架')
    require(state['smoke_complete'] is True and state['formal_complete'] is True, '报告未声明实际完成，不能走最终验收')
    require(state['new_V43_joint_model_trained'] is False and state['unique_gaussian_GT_assumed'] is False, '研究边界声明错误')
    require(state['human_verdict'] is None, '人工verdict不由本工具代填')
    require(state['observed_frames'] == 16 and state['formal_updates'] == 10000, '报告预算/帧数错误')
    for link in parser.links:
        url = urlsplit(link)
        if not url.scheme and url.path:
            require(safe_relative(root, unquote(url.path)).exists(), '本地链接缺失：' + link)
    smoke = check_smoke(evidence)
    formal = read(evidence / 'formal_r3_10k/returned.json')
    require(formal['iteration'] == formal['formal_config_budget'] == formal['observed_finite_step_calls'] == 10000,
            '正式更新没有完成合同10K或记账不一致')
    require(formal['inherited_engine_and_learning_rate_horizon'] == 30000, '正式LR周期改变')
    require(formal['first_iteration'] == 1 and formal['extra_smoke'] is False and formal['extra_entry'] is False, '正式训练不是fresh独立预算')
    require(read(evidence / 'formal_execution.json')['returncode'] == 0, '正式运行进程未成功返回')
    require(read(evidence / 'edit_execution.json')['returncode'] == 0, '实际编辑进程未成功返回')
    for iteration in (5000, 10000):
        event = read(evidence / f'formal_r3_10k/checkpoint_events/iteration_{iteration:06d}.json')
        require(event['status'] == 'complete' and event['iteration'] == iteration, '正式checkpoint event未完整发布')
        require(set(event['checkpoints']) == {'static', 'dynamic', 'environment', 'training_state'}, '正式event组件缺失')
    dataset = read(evidence / 'dataset_r2/validation.json')
    observations = read(evidence / 'dataset_r2/observations/manifest.json')
    require(dataset['status'] == 'passed' and len(dataset['frame_checks']) == 16, '16帧数据未通过')
    require(observations['status'] == 'complete' and len(observations['frames']) == 16, '观测清单未完整生成')
    render = read(evidence / 'edit_r3_10k/render_manifest.json')
    metrics = read(evidence / 'edit_r3_10k/reference_metrics.json')
    require(render['status'] == 'NATIVE_BRANCH_ROUTING_EDIT_RENDERED' and render['iteration'] == 10000, '实际编辑不是完整10K输出')
    require(render['training_updates'] == 0 and render['original_checkpoint_unchanged'] is True and render['same_camera_for_all_edits'] is True,
            '编辑渲染模型/相机配对声明不符')
    require(render['privileged_reference_used_for_training'] is False and metrics['privileged_reference_used_for_training'] is False,
            '独立参考角色错误')
    require(set(map(normalized, render['native_none_identity_checked_frames'])) == set(FRAMES), 'native none未覆盖三帧')
    require(finite(render['native_none_identity_max_abs'], 'native none误差非有限') <= 2e-6, 'native none不一致')
    render_keys = [(normalized(row['frame']), row['edit']) for row in render['frames']]
    metric_keys = [(normalized(row['frame']), row['edit']) for row in metrics['rows']]
    require(len(render_keys) == len(set(render_keys)) == 12 and set(render_keys) == EXPECTED, '实际渲染3×4键不全或重复')
    require(len(metric_keys) == len(set(metric_keys)) == 12 and set(metric_keys) == EXPECTED, '分数3×4键不全或重复')
    require(metrics['scored'] == 12 and metrics['missing_reference_count'] == 0, '仍有pending参考，不能显示为0误差')
    require(all(row['status'] == 'SCORED_INDEPENDENT_REFERENCE' for row in metrics['rows']), '有未评分的参考行')
    require(state['native_edit_views'] == state['independent_reference_views_scored'] == 12, '报告状态计数与收据不一致')
    camera = read(evidence / 'edit_r3_10k/training_camera_restore_verification.json')
    require(camera['all_passed'] is True and camera['cameras'] == 16, '训练相机重载核对未通过')
    require(camera['maximum_extrinsic_abs_error'] <= 1e-5 and camera['maximum_intrinsic_abs_error'] <= 1e-5, '相机重载误差超限')
    require(tuple(parser.selects.get('frame', ())) == FRAMES, 'HTML帧选择不对应三关键帧')
    require(tuple(parser.selects.get('edit', ())) == EDITS, 'HTML操作选择不对应四操作')
    require({'reference', *CHANNELS}.issubset(parser.images), '交互区缺少独立reference或五种实际分量')
    require(len(parser.image_ids) == len(set(parser.image_ids)), 'HTML图像id重复会导致交互更新错误')
    script = '\n'.join(parser.scripts)
    ref_match = re.search(r"getElementById\(\s*['\"]reference['\"]\s*\)\.src\s*=\s*([^;]+)", script)
    image_match = re.search(r'getElementById\(\s*k\s*\)\.src\s*=\s*([^;]+)', script)
    require(ref_match is not None and image_match is not None, '无法核验HTML动态路径表达式')
    loop = re.search(r'for\s*\(\s*const\s+k\s+of\s+(\[[^\]]+\])\s*\)', script)
    require(loop is not None and tuple(ast.literal_eval(loop.group(1))) == CHANNELS, '交互更新没有覆盖final/white/C/alpha/fill五分量')
    require(re.search(r'getElementById\(\s*id\s*\)\.onchange\s*=\s*update', script)
            and re.search(r'update\(\)\s*;', script), '帧/操作事件或首次图像加载没有绑定')
    references_root = evidence / 'dataset_r2/privileged'
    references = read(references_root / 'reference_manifest.json')
    require(references['schema'] == 'v43_sim_reference_v1', '独立参考清单schema错误')
    require(references['edits']['shift']['translation_world_m'] == [0, 1.5, 0]
            and references['edits']['rotate']['yaw_delta_deg'] == 30.0, '独立编辑参考参数错误')
    records = {}
    for item in references['references']:
        require(type(item['vehicle_shadow']) is bool, 'shadow开关不是布尔值')
        key = (normalized(item['frame_id']), item['edit'], item['vehicle_shadow'])
        require(key not in records, '独立参考key重复')
        records[key] = item
    by_key = dict(zip(metric_keys, metrics['rows']))
    gallery_paths = []
    for frame, edit in sorted(EXPECTED):
        item = records[(frame, edit, True)]
        reference_relative = 'evidence/dataset_r2/privileged/' + item['rgb']
        expected_reference = f'evidence/dataset_r2/privileged/references/{edit}/shadow_on/rgb/{frame}.png'
        require(reference_relative == expected_reference, '参考清单与操作/帧/影状态路径不一致')
        actual_ref = expression_value(ref_match.group(1), {'f': frame, 'e': edit})
        require(actual_ref == reference_relative, 'HTML独立参考指向错误帧、操作或shadow状态')
        metric = by_key[(frame, edit)]
        remote_ref = metric['reference'].replace('\\', '/')
        require(remote_ref.endswith('/dataset_r2/privileged/' + item['rgb']), '指标并非对照当前显示的独立参考')
        require(metric['reference_only_used_for_evaluation'] is True, '指标参考角色错误')
        for relative in [actual_ref, *[expression_value(image_match.group(1), {'f': frame, 'e': edit, 'k': name}) for name in CHANNELS]]:
            path = safe_relative(root, relative)
            require(path.is_file(), '交互图缺失：' + relative)
            check_png(path)
            gallery_paths.append(relative)
        for name in CHANNELS:
            expected_image = f'evidence/edit_r3_10k/{edit}/{frame}_{name}.png'
            require(expression_value(image_match.group(1), {'f': frame, 'e': edit, 'k': name}) == expected_image,
                    '实际编辑分量指向错误帧/操作/通道')
        off = records[(frame, edit, False)]
        require(safe_relative(references_root, off['rgb']).is_file(), '独立shadow-off配对图缺失')
    require(len(gallery_paths) == len(set(gallery_paths)) == 72, '交互区3帧×4操作×6图未覆盖72个文件')
    tables = [table for table in parser.tables if '区域误差' in table['heading']]
    require(len(tables) == 1 and len(tables[0]['rows']) == 13, '区域误差表不含完整12行')
    table_keys = []
    for cells in tables[0]['rows'][1:]:
        require(len(cells) == 5, '指标表列数不符')
        key = (normalized(cells[0]), cells[1])
        require(key in by_key, '指标表出现未知帧/操作')
        table_keys.append(key)
        regions = by_key[key]['regions']
        body, road = regions['original_body_footprint'], regions['road_reference']
        require(int(cells[2]) == body['pixels'], '指标表原车体分母不一致')
        for cell, region in ((cells[3], body), (cells[4], road)):
            if region['pixels'] == 0:
                require(cell.upper() in {'N/A', 'NA', '—', 'NULL', '无有效像素', 'N/A（空区域）'}, '零有效像素不应显示为0误差')
            else:
                expected = finite(region['clamped_mae'], 'MAE缺失/非有限')
                require(abs(float(cell) - expected) <= 0.00005001, 'HTML的4位MAE与JSON不一致')
    require(len(table_keys) == len(set(table_keys)) == 12 and set(table_keys) == EXPECTED, '表格帧/操作重复或缺失')
    assessment = read(root / 'assessment.json')
    require(state['assessment'] == assessment, '状态与assessment内容不一致')
    for value in [assessment['lead'], *assessment['findings'], *assessment['next_steps']]:
        require(html.escape(str(value)) in page, 'assessment内容没有进入HTML或被改写')
    if '317步计时' in page:
        require(read(evidence / 'timing_r1/returned.json')['observed_finite_step_calls'] == 317, 'HTML计时探测317步与收据不符')
    for value in (smoke['extra_smoke_updates'], smoke['extra_entry_updates']):
        require(str(value) in page, 'smoke记账数值未显示')
    svg_path = root / 'architecture.svg'
    svg_text = svg_path.read_text(encoding='utf8')
    svg = ET.fromstring(svg_text)
    labels = ''.join(svg.itertext())
    for marker in ('GPU smoke待跑', 'GPU smoke 待跑', '当前均等待实际凭据', 'skeleton_not_final_report', 'REFERENCE_NOT_YET_AVAILABLE'):
        require(marker not in page + labels, '最终报告含旧pending标签：' + marker)
    require(len(svg.findall('.//{http://www.w3.org/2000/svg}rect')) >= 9, '架构大模块/内部小框疑似被简化')
    for label in ('static', 'dynamic', 'environment', '恢复', '评价'):
        require(label in labels, '组件图缺少关键组件：' + label)
    diagram = read(root / 'architecture_validation.json')
    require(diagram['native_canvas_rendered'] is True and not diagram['text_overflow'], '最终组件图未通过文字边界检查')
    require(str(svg_path.stat().st_mtime_ns) == diagram['svg_mtime_ns'], '最终SVG更新后没有重渲染PNG')
    require(str((root / 'architecture.png').stat().st_mtime_ns) == diagram['png_mtime_ns'], 'PNG并非本次最终SVG重渲染结果')
    heldout = check_heldout(root, page, state, references)
    return {'report_validated': True, 'actual_report_evidence_validated': True,
            'gallery_images_checked': 72, 'independent_reference_pairs_checked': 12,
            'metric_rows_checked': 12, 'static_local_links_checked': len(parser.links),
            'heldout': heldout,
            'svg_png_render_receipt_checked': True, 'completed_receipts_cross_checked': True,
            'gpu_checkpoint_contents_revalidated': False, 'raw_fp32_metrics_recomputed': False,
            'scientific_success_or_human_verdict_assigned': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-root', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--source-only', action='store_true')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    destination = args.receipt or HERE / ('report_source_validation.json' if args.source_only else 'report_validation.json')
    require(not destination.resolve().is_relative_to(args.report_root.resolve()), '校验收据默认留在work，不覆盖报告outputs')
    receipt = {'report_validated': False, 'actual_report_evidence_validated': False}
    failure = None
    try:
        receipt.update(source_review())
        if not args.source_only:
            receipt.update(validate_report(args.report_root.resolve()))
    except Exception as error:
        failure = error
        receipt['validation_error'] = type(error).__name__ + ': ' + str(error)
    receipt.update(validation_scope='local_source_links_gallery_and_json_consistency_only',
                   source_only=args.source_only, writes_outputs=False, remote_execution=False)
    destination.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps(receipt, ensure_ascii=True))
    if failure is not None:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
