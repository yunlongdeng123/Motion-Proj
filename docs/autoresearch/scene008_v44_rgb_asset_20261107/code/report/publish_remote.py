"""将已完成的 CPU 阶段接线归档，不启动 GPU 任务或控制器。"""
from pathlib import Path
import ast
import hashlib
import json
import shutil
import subprocess
import tarfile

REPO = Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_rgb_asset_20261107_r1')
DEST = REPO/'docs/autoresearch/scene008_v44_rgb_asset_20261107'


def git(*args):
    result = subprocess.run(['git', *args], cwd=REPO, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    archive = CAMPAIGN/'v44_cpu_delivery.tar.gz'
    staged_before = git('diff', '--cached', '--name-only')
    if staged_before:
        raise RuntimeError('已有其他暂存工作，停止自动归档：'+staged_before)
    protected = REPO/'scripts/run_worldsim_v72_e3_render_bridge.py'
    protected_before = sha(protected)
    head_before = git('rev-parse', 'HEAD')
    if DEST.exists():
        raise FileExistsError(f'保留已有目录，请检查后另开归档：{DEST}')
    DEST.mkdir()
    with tarfile.open(archive, 'r:gz') as saved:
        for member in saved.getmembers():
            target = (DEST/member.name).resolve()
            if not target.is_relative_to(DEST.resolve()) or not (member.isfile() or member.isdir()):
                raise ValueError('Unsafe archive member')
        saved.extractall(DEST)
    # 本地 outputs 和 Git docs 的相邻报告路径不同，正文与数据不改。
    page = DEST/'index.html'
    page.write_text(page.read_text().replace(
        '../scene008_vehicle_appearance/diagnosis/index.html#frame-compare',
        '../scene008_a6_insert_artifacts_20261105/delivery/index.html#frame-compare').replace(
        '../scene008_vehicle_appearance/diagnosis/data/shadow/qa_receipt_v2.json',
        '../scene008_a6_insert_artifacts_20261105/delivery/data/shadow/qa_receipt_v2.json'))
    (DEST/'.gitattributes').write_text('* -text whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n')
    for name in ('source', 'contract', 'review'):
        shutil.copytree(DEST/'code'/name, CAMPAIGN/name, dirs_exist_ok=True)
    for source in (DEST/'code').rglob('*.py'):
        ast.parse(source.read_text(encoding='utf-8-sig'))
    # 只改变当前状态段；后面的 RCR/a5 独立工作保留。
    status_file = REPO/'docs/RESEARCH_STATUS.md'
    old_status = status_file.read_text()
    (DEST/'prior_research_status.md').write_text(old_status)
    marker = 'RCR-1支线当前由用户采用'
    position = old_status.index(marker)
    status = '''# 当前研究状态

更新：2026-11-07（Asia/Singapore）。

Scene008 当前车体阶段保留 C2 连续表面、固定 RGBA 与同一 S/F SurfaceLOD；PVD 与预训练权重不更新。已有49帧双状态和拖动报告保留，几何前移、少量低alpha及内部开发域边界仍以原报告为准，不扩为360度或隐藏真值准确性结论。

当前工作为 v44：把 RCR-1 已保存的 RGB×1 原生10K及其 freshroad1K/gates 接入当前冻结车体。独立代码和实际远端CPU输入检查已完成（552项）；原11×9世界色场仅为来自旧M2联合场景的冻结迁移，未重标定。新组合未运行GPU、未得到画质verdict；当前CUDA不可用，下一步带GPU后先raw105/113/130/137成对pilot，通过工程检查再完整49帧。详见[v44接线与异常评估](autoresearch/scene008_v44_rgb_asset_20261107/index.html)。

银亮片与暗斑分别保留诊断：历史亮片净增亮由static主导、环境×T仍有车形，不能称为PVD车体产生或去蓝已修复。v44将输出同排序四组RGB/alpha贡献；新组合来源待实际渲染核对。无新native/road/asset训练，无待启动控制器。

执行边界：subagent仅gpt-6.1-sol/max；本会话保持AutoDL开机，不自动关机。failure_ledger_refs=[]，本次failure_ledger_delta=none；人工画质verdict未代填。

'''
    status_file.write_text(status+old_status[position:])
    index = REPO/'docs/EXPERIMENTS.md'
    entry = '\n- `v44 / scene008_v44_rgb_asset_20261107_r1`：[当前冻结车体与RCR-1去蓝背景接线、异常评估及CPU预检](autoresearch/scene008_v44_rgb_asset_20261107/index.html)；552项实际CPU检查通过，GPU pilot/full49未运行；固定色场迁移非新校准，银亮片未修复，failure_ledger_delta=none。\n'
    if 'scene008_v44_rgb_asset_20261107_r1' in index.read_text():
        raise RuntimeError('已有v44索引，请手动检查避免重复')
    with index.open('a') as stream:
        stream.write(entry)
    paths = ['docs/autoresearch/scene008_v44_rgb_asset_20261107', 'docs/RESEARCH_STATUS.md', 'docs/EXPERIMENTS.md']
    git('add', '--', *paths)
    git('diff', '--cached', '--check')
    staged = git('diff', '--cached', '--name-only').splitlines()
    if not all(p.startswith(paths[0]+'/') or p in paths[1:] for p in staged):
        raise RuntimeError('暂存出现范围外文件')
    assert sha(protected) == protected_before
    stats = git('diff', '--cached', '--stat')
    # 在提交之前输出供根代理检查；此脚本不commit或push。
    receipt = dict(status='STAGED_V44_CPU_ONLY_PENDING_ROOT_DIFF_REVIEW', head_before=head_before,
                   staged_paths=staged, diff_stat=stats, protected_script_sha256=protected_before,
                   source_archived=True, remote_campaign_source_ready=True, GPU_renders=0,
                   training_updates=0, controller_started=False, shutdown_performed=False)
    (CAMPAIGN/'git_stage_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
