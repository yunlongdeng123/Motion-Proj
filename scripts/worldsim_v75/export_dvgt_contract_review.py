"""将原生接口检查导出为真实输入、空间误差图和完整配对数据。"""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from probe_dvgt_temporal_contract import OUT
from audit_dvgt_native_contract import OUT as AUDIT


def main():
    dst = OUT/'review'; dst.mkdir(exist_ok=True)
    audit = dst/'single_frame_audit'; audit.mkdir(exist_ok=True)
    for name in ['protocol.json', 'result.json']:
        shutil.copy2(AUDIT/name, audit/name)
    for name in ['protocol.json', 'preflight.json', 'result.json']:
        shutil.copy2(OUT/name, dst/name)
    result = json.loads((OUT/'result.json').read_text())
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    cams = ['FC', 'FL', 'FR', 'RL', 'RR', 'SL', 'SR']; table = []
    for case in result['cases']:
        log = case['log_id']; folder = OUT/log; target = dst/log; target.mkdir(exist_ok=True)
        for name in ['input_manifest.json', 'inference_result.json', 'evaluation.json', 'projection_display.npz']:
            shutil.copy2(folder/name, target/name)
        rgb = np.load(folder/'network_rgb.npy', mmap_mode='r')[-1]
        np.save(target/'last_input_rgb.npy', rgb)
        maps = np.load(folder/'projection_display.npz')
        fig = plt.figure(figsize=(14, 8.8)); gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=.32, wspace=.25)
        ax = fig.add_subplot(gs[0, 0]); ax.imshow(rgb[0]); ax.axis('off'); ax.set_title('Actual front input (same last frame)', loc='left')
        for j, key in enumerate(['T1', 'T3'], 1):
            ax = fig.add_subplot(gs[0, j]); ax.imshow(rgb[0]); im = ax.imshow(maps[f'0_{key}_angles'], extent=(-.5,511.5,511.5,-.5),
                                                                         vmin=0, vmax=60, cmap='magma', interpolation='nearest', alpha=.9)
            med = case['projection'][0][key]['median_angular_error_deg']
            ax.set_title(f'{key} front: median {med:.2f}°', loc='left'); ax.axis('off')
        # 色条单独占据两行之间的空白，避免覆盖真实场景。
        cax = fig.add_axes([.57, .495, .35, .016])
        cb = fig.colorbar(im, cax=cax, orientation='horizontal', extend='max')
        cb.set_label('Point / calibrated-ray angle (degrees)', fontsize=10, labelpad=1)
        ax = fig.add_subplot(gs[1, :2]); x = np.arange(7)
        for j, key in enumerate(['T1', 'T3']):
            vals = [r[key]['median_angular_error_deg'] for r in case['projection']]
            bars = ax.bar(x+(j-.5)*.36, vals, .36, color=['#bf583a','#267c91'][j], label=['1 timestamp / 7 RGB','3 timestamps / 21 RGB'][j])
            for b, v in zip(bars, vals):
                ax.text(b.get_x()+b.get_width()/2, v+.9, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x, cams); ax.set_ylim(0,120); ax.set_ylabel('Median angular error (degrees)')
        ax.set_title('All seven views, same paired valid pixels', loc='left'); ax.legend(frameon=False, fontsize=10); ax.grid(axis='y',alpha=.15)
        ax = fig.add_subplot(gs[1, 2]); ax.axis('off')
        front = case['projection'][0]
        msg = ('Front points in front of camera\n'
               f'T1: {front["T1"]["positive_fraction"]*100:.1f}%   T3: {front["T3"]["positive_fraction"]*100:.1f}%\n\n'
               'No coordinate / scale fitting\n'
               'Known calibration: evaluation only\n'
               'History adds real observations\n\n'
               'No new world-model generation\n'
               'No closed-loop harm inferred')
        ax.text(.03,.96,msg,va='top',linespacing=1.65,color='#264350')
        fig.suptitle(f'DVGT-1 native geometry: {log[:8]} | temporal-context control',x=.07,ha='left',fontsize=17)
        fig.text(.07,.025,'2..80 m paired-valid samples; 8 px grid; padding excluded. Heatmap saturates at 60°. FC/FL/FR/RL/RR/SL/SR = ring cameras.',fontsize=10,color='#53666e')
        fig.subplots_adjust(top=.9,bottom=.10,left=.07,right=.95)
        for ext in ['png','svg']: fig.savefig(dst/f'{log[:8]}-native-contract.{ext}',dpi=170)
        plt.close(fig)
        for row in case['projection']:
            table.append({'log_id':log,'camera':row['camera'],'common_samples':row['common_samples'],
                          **{f'{key}_{metric}':value for key in ['T1','T3'] for metric,value in row[key].items()}})
    (dst/'paired_metrics.json').write_text(json.dumps(table,indent=2)+'\n')
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="260" viewBox="0 0 1120 260"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="none" stroke="#376377"/></marker></defs><rect width="1120" height="260" fill="#f5f8fa"/><g fill="white" stroke="#96afbc" stroke-width="1.5"><rect x="24" y="55" width="230" height="108" rx="8"/><rect x="300" y="55" width="190" height="108" rx="8"/><rect x="536" y="55" width="254" height="108" rx="8"/><rect x="836" y="55" width="260" height="108" rx="8"/></g><g stroke="#376377" stroke-width="2" marker-end="url(#a)"><path d="M254 109H290"/><path d="M490 109H526"/><path d="M790 109H826"/></g><g font-family="Arial,Microsoft YaHei,sans-serif" fill="#233f50" text-anchor="middle"><text x="139" y="84" font-size="17">真实截止前 RGB</text><text x="139" y="116" font-size="15">1 时刻 × 7 视角</text><text x="139" y="143" font-size="15">3 时刻 × 7 视角（2Hz）</text><text x="395" y="97" font-size="19">官方 DVGT-1</text><text x="395" y="132" font-size="15">固定权重 / 原生点图</text><text x="663" y="84" font-size="17">截止时刻点图 → 世界坐标</text><text x="663" y="116" font-size="15">已知相机射线仅用于评价</text><text x="663" y="143" font-size="15">共同像素：角误差 / 正深度</text><text x="966" y="89" font-size="18">检查原生几何可用范围</text><text x="966" y="121" font-size="15">两例均有方向残余</text><text x="966" y="148" font-size="15">本轮不接入生成闭环</text><text x="560" y="214" font-size="16">没有 LiDAR、GT 对象形状或拟合变换输入 DVGT；新增历史是额外信息；0 次世界模型生成</text></g></svg>'''
    (dst/'architecture.svg').write_text(svg+'\n',encoding='utf-8')
    for path in dst.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    print(dst)


if __name__ == '__main__': main()
