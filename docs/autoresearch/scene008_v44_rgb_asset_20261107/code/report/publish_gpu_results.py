"""归档 v44 GPU 结果：大图从原远端结果直拷，不重传或改变方法。"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tarfile

REPO = Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
RUN = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_rgb_asset_20261107_r1')
DEST = REPO/'docs/autoresearch/scene008_v44_rgb_asset_20261107'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if git('diff', '--cached', '--name-only'):
        raise RuntimeError('保留其他暂存工作，停止自动stage')
    protected = REPO/'scripts/run_worldsim_v72_e3_render_bridge.py'
    before = sha(protected)
    original_head = git('rev-parse', 'HEAD')
    if not (DEST/'index.html').is_file():
        raise RuntimeError('缺少已发布CPU阶段，不创建伪历史')
    shutil.copy2(DEST/'index.html', DEST/'cpu_stage_source_repository.html')
    with tarfile.open(RUN/'v44_gpu_metadata.tgz') as archive:
        for member in archive.getmembers():
            if not (DEST/member.name).resolve().is_relative_to(DEST.resolve()) or not (member.isfile() or member.isdir()):
                raise RuntimeError('Unsafe publication member')
        archive.extractall(DEST)
    manifest = json.loads((DEST/'publication_manifest.json').read_text())
    assert manifest['full49_executed'] and manifest['quality_accepted'] is None
    main_prefix = 'data/results_'+manifest['main_comparison_sha256'][:12]+'/'
    pilot_prefix = 'data/pilot_'+manifest['pilot_comparison_sha256'][:12]+'/'
    layer_prefix = manifest['supplemental_layer_replay'].rsplit('/', 1)[0]+'/'
    source_map = {main_prefix: RUN/'results/full49_r1', pilot_prefix: RUN/'results/pilot_r1',
                  layer_prefix: RUN/'results/pilot_layers_r1'}
    copied = 0
    for item in manifest['files']:
        relative = item['path']
        if relative.startswith('data/'):
            matching = [(prefix, folder) for prefix, folder in source_map.items() if relative.startswith(prefix)]
            if len(matching) != 1:
                raise RuntimeError('Unknown result source '+relative)
            prefix, folder = matching[0]
            source = folder/relative[len(prefix):]
            if sha(source) != item['sha256']:
                raise RuntimeError('Original run file differs '+relative)
            target = DEST/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied += 1
        if sha(DEST/relative) != item['sha256']:
            raise RuntimeError('Published result differs '+relative)
    shutil.copy2(DEST/'cpu_stage.html', DEST/'cpu_stage_source_local.html')
    navigation = []
    for page in DEST.glob('*.html'):
        original = page.read_text(encoding='utf-8-sig')
        updated = original.replace('../scene008_vehicle_appearance/diagnosis/',
            '../scene008_a6_insert_artifacts_20261105/delivery/')
        if updated != original:
            old_sha = sha(page)
            page.write_text(updated)
            navigation.append(dict(path=page.name, local_HTML_sha256=old_sha, repository_HTML_sha256=sha(page),
                                   change='Only sibling report navigation; scientific data unchanged'))
    (DEST/'repository_navigation_receipt.json').write_text(json.dumps(navigation, indent=2)+'\n')
    status_file = REPO/'docs/RESEARCH_STATUS.md'
    old = status_file.read_text()
    (DEST/'prior_research_status_gpu.md').write_text(old)
    marker = 'RCR-1支线当前由用户采用'
    tail = old[old.index(marker):]
    status_file.write_text('''# 当前研究状态

更新：2026-11-07（Asia/Singapore）。

Scene008 当前保留 C2 连续表面、固定 RGBA 与同一 S/F SurfaceLOD 车体。v44已完成RCR-1背景接入的4帧pilot和完整49帧双状态：同一车体，背景为已保存RGB×1 native10K及其匹配road1K/gates，11×9色场仅冻结迁移，未重标定。PVD、车体和所有预训练权重本轮无更新。

本轮优先道路蓝紫异常；用户明确将银残片留到下一轮。固定原观测道路1075281像素的绝对蓝误差keep .0165024→.0081845、delete .0143270→.0084500，49帧两状态逐帧均改善。全49帧delete道路proxy RGB .0186062→.0096190；该proxy不是隐藏道路独立真值。详见[v44逐帧报告](autoresearch/scene008_v44_rgb_asset_20261107/index.html)。

a6整体delete RGB .0549473→.0549451、深度 .171958→.170030m，无新增低alpha洞；局部并非零回退：a6最坏raw137，深度最坏raw101，a37 keep最坏raw122，均在报告披露。raw105真实蓝车局部超蓝尾部仍存在，不做整图去色。原预登记蓝ROI为空，另存有真实RGB支持的独立背景域与全49原观测道路统计，未改正式分母。HELD9仍为内部开发，不声称独立泛化。

工程与来源复核完成：pilot16＋full49 196＋必要分组诊断3＝215次native前向；独立CPU全49 1550项、跨阶段42项、分组90项通过。无新训练或求解。PNG/JSON/报告已取回；大float与components保留原campaign。银片主要为static增量和environment×T减量抵消的成像问题，训练根因未证；本轮不处理银片。

执行边界：subagent仅gpt-6.1-sol/max。按用户要求AutoDL保持开机，不自动关机；无待启动训练控制器。人工画质verdict仍为null，未替用户接受整体画质。failure_ledger_refs=[]；failure_ledger_delta=none。旧报告及CPU阶段保存。

'''+tail)
    index = REPO/'docs/EXPERIMENTS.md'
    text = index.read_text()
    lines = text.splitlines()
    matches = [i for i, line in enumerate(lines) if '`v44 / scene008_v44_rgb_asset_20261107_r1`' in line]
    if len(matches) != 1:
        raise RuntimeError('v44 index not unique')
    lines[matches[0]] = '- `v44 / scene008_v44_rgb_asset_20261107_r1`：[4帧pilot＋完整49帧双状态去蓝与冻结车体对照](autoresearch/scene008_v44_rgb_asset_20261107/index.html)；道路蓝偏差98/98帧状态改善，a6整体近稳定、局部回退保留；银残片用户指定延期，0训练，215次native前向。failure_ledger_delta=none。'
    index.write_text('\n'.join(lines)+'\n')
    paths = ['docs/autoresearch/scene008_v44_rgb_asset_20261107', 'docs/RESEARCH_STATUS.md', 'docs/EXPERIMENTS.md']
    git('add', '--', *paths)
    git('diff', '--cached', '--check')
    staged = git('diff', '--cached', '--name-only').splitlines()
    assert all(p.startswith(paths[0]+'/') or p in paths[1:] for p in staged)
    assert sha(protected) == before
    receipt = dict(status='STAGED_V44_GPU_RESULTS_PENDING_DIFF_REVIEW', parent=original_head,
                   directly_copied_remote_files=copied, source_data_files_verified=len(manifest['files']),
                   staged_files=staged, protected_script_sha256=before,
                   full49_complete=True, new_training_updates=0, shutdown_performed=False,
                   report_path=str(DEST/'index.html'))
    (RUN/'git_gpu_stage_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    print(git('diff', '--cached', '--stat'))
    print(json.dumps({k: v for k, v in receipt.items() if k != 'staged_files'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
