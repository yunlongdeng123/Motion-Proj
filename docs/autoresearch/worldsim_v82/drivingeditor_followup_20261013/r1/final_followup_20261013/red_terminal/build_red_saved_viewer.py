"""Call the existing saved viewer builder with red scene labels and evidence links."""
import argparse
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
ORIGINAL = Path('/root/autodl-tmp/drivingeditor_repro/campaign/followup_scene_prep_20261013/tools/gmc_saved_cpu_20261013_0905/build_saved_followup_viewer.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    module = runpy.run_path(str(ORIGINAL), run_name='SAVED_VIEWER_SCENE_LABEL_ADAPTER')
    build = module['build']
    page = build.__globals__['PAGE'].replace('GMC', '红车')
    marker = '<p class="about-text" id="resource-overview"></p>'
    links = '<p class="about-text">终态真实记录：<a href="scene0029_terminal_summary.json">终态汇总</a> · <a href="scene0029_terminal_resources_summary.json">完整资源</a> · <a href="scene0029_NPS6_SGCM145_accounting_summary.json">NPS / SGCM / sampler</a> · <a href="scene0029_30000_all49_region_metrics.json">49 帧区域指标</a> · <a href="scene0029_fixed_pose_49times_summary.json">固定相机 49 时间</a></p>'
    if marker not in page:
        raise ValueError('Original saved viewer resource marker changed')
    build.__globals__['PAGE'] = page.replace(marker, marker + links)
    output = build(args.output / 'manifest.json', args.output)
    assert 'torch' not in sys.modules
    print('Original saved viewer generated with red scene labels: ' + str(output))


if __name__ == '__main__':
    main()
