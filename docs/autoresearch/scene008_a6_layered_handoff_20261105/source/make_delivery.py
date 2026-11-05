"""CPU-only delivery; requires all nine completed evaluations and saved replays."""
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RUN = Path(__file__).resolve().parent.parent
PREV = RUN.parent / 'scene008_a6_local_fusion_20261105'
LAYERS = ('near', 'same', 'far')
STATES = ('black', 'keep', 'delete')
HELD = (89, 95, 101, 105, 113, 121, 125, 130, 137)
PROBES = (105, 113, 130)
ARMS = ('parent', 'retirement', 'proxy_only', 'coupled')
LABELS = ('Parent', 'Retire only', 'New only', 'Coupled')

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def mean(value):
    return value['mean'] if 'mean' in value else value['sum']/value['count'] if value['count'] else None

def fmt(value, percent=False):
    return '—' if value is None else format(value, '.2%' if percent else '.5f')

def grid(paths, box=None, scale=1):
    panels = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert('RGB')
            if box is not None: image = image.crop(box)
            if scale != 1: image = image.resize((image.width*scale, image.height*scale), Image.Resampling.NEAREST)
            panels.append(image)
    width, height = max(panels[0].width, 205), panels[0].height+27
    output = Image.new('RGB', (width*4, height), '#070b10')
    draw = ImageDraw.Draw(output)
    try: font = ImageFont.truetype('DejaVuSans.ttf', 14)
    except OSError: font = ImageFont.load_default()
    for index, (image, label) in enumerate(zip(panels, LABELS)):
        draw.text((index*width+7, 5), label, fill='#e5edf2', font=font)
        output.paste(image, (index*width+(width-image.width)//2, 27))
    return output

ARCH = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 165" role="img" aria-label="原始观测到分层接管与保护回退">
<defs><marker id="a" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7Z" fill="#506c7c"/></marker></defs><g fill="none" stroke="#506c7c" stroke-width="2" marker-end="url(#a)"><path d="M180 72H225"/><path d="M405 72H450"/><path d="M650 72H695"/><path d="M875 72H920"/></g>
<g stroke="#b9cbd5"><rect x="5" y="25" width="175" height="94" rx="8" fill="#eef5f8"/><rect x="225" y="25" width="180" height="94" rx="8" fill="#e4f2ec"/><rect x="450" y="25" width="200" height="94" rx="8" fill="#fff0d8"/><rect x="695" y="25" width="180" height="94" rx="8" fill="#e4f2ec"/><rect x="920" y="25" width="175" height="94" rx="8" fill="#eef5f8"/></g>
<g font-family="sans-serif" font-size="16" text-anchor="middle" fill="#183240"><text x="92" y="60">原始 RGB / LiDAR</text><text x="92" y="88">固定车姿与父场景</text><text x="315" y="60">BUILD 实测表面组</text><text x="315" y="88">固定代理面与登记行</text><text x="550" y="60">近 / 同 / 远物理 crop</text><text x="550" y="88">车体参考法向 ±12 cm</text><text x="785" y="60">联合透明合成</text><text x="785" y="88">固定退役 50% + γ</text><text x="1007" y="60">49 帧保护验收</text><text x="1007" y="88">失败 → zero operation</text><text x="550" y="150">原生中心深度排序仍在；没有不透明表面首交保证</text></g></svg>'''

def main():
    summary, verification = read(RUN/'summary.json'), read(RUN/'verification.json')
    assert summary['status'] == 'EXECUTION_PASS' and len(summary['records']) == 9
    assert verification['status'] == 'PASS' and verification['exact_saved_metric_replay']
    assert verification['saved_selected_state_reloaded'] and len(verification['selected']) == 81
    expected = {(rank, layer, raw, state, arm) for rank in range(3) for layer in LAYERS for raw in PROBES for state in STATES for arm in ARMS}
    assert set(map(tuple, verification['checked'])) == expected and len(verification['checked']) == 324
    cases, csv_rows, table = [], [], []
    for layer in LAYERS:
        for rank in range(3):
            stem = f'{layer}_group{rank}'
            result, fit = read(RUN/f'{stem}_evaluation.json'), read(RUN/f'{stem}_fit.json')
            assert result['status'] == fit['status'] == 'EXECUTION_PASS'
            assert len(result['rows']) == 49 and [r['raw'] for r in result['rows']] == list(range(89, 138))
            accepted = bool(result['method_accepted'])
            with np.load(RUN/f'group_ledger_{rank}.npz', allow_pickle=False) as ledger: proxy_count = len(ledger['xyz'])
            with np.load(RUN/f'{stem}_selected.npz', allow_pickle=False) as selected:
                assert bool(selected['candidate_accepted']) == accepted and str(selected['layer']) == layer
                assert str(selected['group_ledger_sha256']) == sha(RUN/f'group_ledger_{rank}.npz')
                assert float(selected['retirement']) == (result['retirement'] if accepted else 0.)
                assert np.array_equal(selected['gamma'], np.full(proxy_count, result['selected_gamma'] if accepted else 0., np.float32))
            choice = fit['choice']
            assert choice['status'] == result['build_choice']['status'] and choice['gamma'] == result['selected_gamma']
            final_asset = 'candidate' if accepted else 'zero operation'
            assert result['final_asset'] == final_asset
            replay = [r for r in verification['selected'] if r['group'] == rank and r['layer'] == layer]
            assert len(replay) == 9 and all(bool(r['candidate_accepted']) == accepted for r in replay)
            no_mass = choice['reason'] == 'NO_LOST_GROUP_SUPPORT'
            failures = {s: result['pools']['all'][s]['coupled']['failed_frames'] for s in STATES}
            held = result['pools']['held']['black']
            item = dict(layer=layer, rank=rank, accepted=accepted, final_asset=final_asset, build_status=choice['status'], reason=choice['reason'], applicable=not no_mass, removed_alphaT=choice['removed_alphaT'], held={a: held[a]['group'] for a in ARMS}, failed_frames=failures, issues=result['issues'])
            cases.append(item)
            cells = [layer, str(rank), str(held['parent']['group']['rgb']['count'])]
            cells += [fmt(mean(held[a]['group']['rgb']))+' / '+fmt(mean(held[a]['group']['depth'])) for a in ('parent', 'retirement', 'coupled')]
            cells += [fmt(mean(held['coupled']['group']['coverage_error']), True), ' / '.join(str(failures[s]) for s in STATES), choice['status'], choice['reason'], fmt(choice['removed_alphaT']), '不适用：无旧组支撑损失' if no_mass else ('通过' if accepted else '未通过')]
            table.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in cells)+f'<td><a href="data/{stem}_selected.npz">{final_asset}</a> · <a href="data/{stem}_evaluation.json">明细</a></td></tr>')
            for row in result['rows']:
                assert set(row['states']) == set(STATES)
                for state in STATES:
                    assert set(row['states'][state]) == set(ARMS)
                    depth = next(v for v in result['non_target_depth'] if v['raw'] == row['raw'] and v['state'] == state)
                    for arm in ARMS:
                        values = row['states'][state][arm]; g = values['group']
                        csv_rows.append([layer, rank, row['raw'], row['held'], state, arm, g['rgb']['count'], *[mean(g[k]) for k in ('rgb', 'depth', 'coverage_error', 'old_new_coverage', 'total_alpha', 'known_rgb', 'known_change', 'outside_change')], '|'.join(values['failures']), depth['mean'] if arm == 'coupled' else None])
            for raw in HELD:
                for state in STATES:
                    for arm in ARMS:
                        if not (RUN/f'{stem}_render' / f'{raw:03d}_{state}_{arm}.png').is_file(): raise FileNotFoundError(f'{stem}: missing {raw}/{state}/{arm}')
    assert len(csv_rows) == 5292
    crops = {}
    for raw in HELD:
        with Image.open(PREV/'render'/f'{raw:03d}_pvd_pvd.png') as mask:
            y, x = np.nonzero(np.asarray(mask.convert('RGB'))[:, :, 0] > 5)
            assert len(x), f'Empty fixed PVD crop for raw{raw}'
            crops[raw] = (max(0, int(x.min())-12), max(0, int(y.min())-16), min(mask.width, int(x.max())+13), min(mask.height, int(y.max())+14))
    sources = sorted((RUN/'source').glob('*.py')); assert sources
    out = RUN/'delivery'; out.mkdir(exist_ok=False)
    for folder in ('frames', 'data', 'source'): (out/folder).mkdir()
    for pattern in ('*.json', '*.npz'):
        for source in sorted(RUN.glob(pattern)): shutil.copyfile(source, out/'data'/source.name)
    for source in sources: shutil.copyfile(source, out/'source'/source.name)
    for item in cases:
        layer, rank = item['layer'], item['rank']
        for raw in HELD:
            for state in STATES:
                paths = [RUN/f'{layer}_group{rank}_render'/f'{raw:03d}_{state}_{arm}.png' for arm in ARMS]
                stem = f'{layer}_{rank}_{raw:03d}_{state}'
                grid(paths, crops[raw], 3).save(out/'frames'/f'{stem}.jpg', quality=94)
                if raw in PROBES and state != 'black': grid(paths).save(out/'frames'/f'{stem}_full.jpg', quality=92)
    accepted_count = sum(c['accepted'] for c in cases); no_mass_count = sum(not c['applicable'] for c in cases)
    dump(out/'summary.json', dict(status='EXECUTION_VERIFIED', cases=cases, accepted_cases=accepted_count, no_lost_group_support_cases=no_mass_count, applicable_rejected_cases=sum(c['applicable'] and not c['accepted'] for c in cases), final_asset=summary['final_asset'], checked=len(verification['checked']), saved_selected_states=9, csv_rows=len(csv_rows), training_updates=0, selection_scope=verification['selection_scope'], depth_scope=verification['depth_scope']))
    with (out/'fixed_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['layer', 'group_rank', 'raw', 'held', 'state', 'arm', 'group_pixels', 'rgb_mae', 'depth_mae_m', 'group_coverage_abs_error', 'group_contribution', 'total_alpha', 'known_rgb_mae', 'known_rgb_change', 'outside_rgb_change', 'guard_failures', 'coupled_non_target_a6_depth_change_m'])
        writer.writerows(csv_rows)
    (out/'architecture.svg').write_text(ARCH, encoding='utf-8')
    audit_link = '<a href="data/retired_depth_audit.json">退役深度审计原始数据</a> · ' if (out/'data'/'retired_depth_audit.json').is_file() else ''
    depth_section = ''
    if (RUN/'retired_depth_audit.json').is_file():
        audit_rows = []
        for group in read(RUN/'retired_depth_audit.json')['groups']:
            for layer in LAYERS:
                values = group['physical_layers'][layer]['pooled']
                total = values['first_order_pre_retirement_total']
                bins = values['rendered_center_depth_bins']
                cells = [str(group['rank']), layer, fmt(total['mass']),
                         *[fmt(bins[name]['fraction_of_total'], True) for name in LAYERS],
                         fmt(total['weighted_signed_mean_m'])]
                audit_rows.append('<tr>'+''.join('<td>'+html.escape(value)+'</td>' for value in cells)+'</tr>')
        depth_section = '<section><h2>物理裁剪与实际合成深度是否一致</h2><p>固定 BUILD 测量射线上，使用父场景 T × 原生 α × 50% × 条件裁剪质量做介入前一阶归因，再按合成使用的高斯中心深度相对实测值分为近/同/远（±12 cm）。该权重不等于退役后的真实 αT 差；中心深度与物理法向位置是两套分类。它检查本轮干预实际对应了哪些原生合成贡献。</p><table><tr><th>rank</th><th>物理层</th><th>一阶归因 αT 总量</th><th>合成近层占比</th><th>合成同层占比</th><th>合成远层占比</th><th>加权中心深度偏差 / m</th></tr>'+''.join(audit_rows)+'</table></section>'
    raw_options = ''.join(f'<option value="{r}"'+(' selected' if r == 113 else '')+f'>{r}</option>' for r in HELD)
    probe_figures = ''.join(f'<figure><a id="link{r}" href="frames/near_0_{r}_delete_full.jpg"><img id="probe{r}" src="frames/near_0_{r}_delete_full.jpg" alt="raw{r}街景四分支全图"></a><figcaption>raw{r}：点击查看原分辨率街景全图。</figcaption></figure>' for r in PROBES)
    report = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Scene008 · 近同远物理支撑接管</title>
<style>body{{margin:0;background:#f3f6f8;color:#172f3b;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:28px 24px}}section{{background:white;border:1px solid #d9e3e9;border-radius:9px;padding:20px;margin:18px 0;overflow:auto}}h1{{font-size:28px}}h2{{font-size:21px}}svg,img{{display:block;width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}}td,th{{padding:8px;border-bottom:1px solid #dbe5eb;text-align:left;white-space:nowrap}}a{{color:#006875}}select{{font:inherit;padding:5px;margin:0 15px 0 5px}}figure{{margin:16px 0}}figcaption,.small{{font-size:14px;color:#4a6170}}.notice{{border-left:5px solid #b86a2d;padding-left:14px}}</style>
<main><h1>Scene008 · 近 / 同 / 远物理支撑接管</h1><p class="notice">9 个预注册 case 已完成评价与保存状态重放；实际通过 <strong>{accepted_count}/9</strong>，无旧组支撑损失而不适用 <strong>{no_mass_count}/9</strong>。最终汇总状态：<strong>{html.escape(summary['final_asset'])}</strong>。单个 case 通过不能替代多组同时启用检查；表中每项都链接实际保存的 candidate 或 zero operation。</p>
<section><h2>输入、组件与输出</h2>{ARCH}<p>固定实测代理面、父场景、PVD、颜色与尺度。旧行登记仅证明 BUILD 测量射线上的渲染责任，不能把 mixed 行整体称为坏支撑。近/同/远是沿车体参考法向、以实测平面 n=0 和 ±0.12 m 划分的物理 slab；切向沿用原薄壳，完整法向范围来自测量车盒每边扩 0.28 m 的八角投影。BUILD 相机必须位于平面同侧。</p><p>这里裁剪的是每条射线上条件高斯的物理支撑质量；并非按原生中心深度分层，也不依靠逐像素图像门控。最终仍用原生中心深度排序和透明合成，没有不透明表面首交保证。四分支固定为父资产、仅退役、仅新面、联合接管；退役 50%，γ 上限 0.35，仅用 BUILD 求解。</p></section>
<section><h2>九个固定 case</h2><p>下表为相同 HELD 黑底实测组像素：RGB MAE / 深度 MAE（米），coverage 为相对父资产指定组贡献的平均绝对偏差，要求 ≤1%。保护失败帧来自完整49帧黑底 / keep / delete；BUILD 状态和最终保存资产分别列出。NO_LOST_GROUP_SUPPORT 表示没有待补偿的旧组支撑损失，本分支不适用，不当作接管机制失败的反证。</p>
<table><thead><tr><th>层</th><th>rank</th><th>测量像素</th><th>Parent RGB / depth</th><th>Retirement RGB / depth</th><th>Coupled RGB / depth</th><th>coverage偏差</th><th>fail 黑/keep/delete</th><th>BUILD</th><th>拟合原因</th><th>旧组贡献净减少</th><th>判定</th><th>最终保存资产</th></tr></thead><tbody>{''.join(table)}</tbody></table><p>旧组贡献净减少是只退役后与父场景的组贡献差，包含传输变化；下方另列介入前的一阶归因量。</p></section>{depth_section}
<section><h2>四分支同裁剪放大</h2><p>四列依次 Parent、Retire only、New only、Coupled。各 raw 统一使用上一轮 PVD mask 取裁剪，所有 case/分支共用，最近邻放大三倍。下方三个探针提供 keep/delete 原始街景全图。</p>
<label>层<select id="layer"><option>near</option><option>same</option><option>far</option></select></label><label>组<select id="group"><option>0</option><option>1</option><option>2</option></select></label><label>raw<select id="raw">{raw_options}</select></label><label>状态<select id="state"><option value="black">黑底 a6</option><option value="keep">无编辑街景</option><option value="delete">删 a37 街景</option></select></label>
<figure><img id="frame" src="frames/near_0_113_black.jpg" alt="四分支固定裁剪比较"></figure>{probe_figures}</section>
<section><h2>验收范围与可复核材料</h2><p>40帧 BUILD 构建及拟合，9帧 HELD 为 {', '.join(map(str, HELD))}。这些 HELD 已重复用于机制探针，不是未触碰测试集。参数拟合只用 BUILD；HELD veto 参与最终资产选择，因此不能把最终选择称为完全 BUILD-only。</p><p>严格相对父资产的目标 RGB/深度检查仅在 black 状态执行。新增非目标深度保护仅覆盖 a6 固定实测点中 group 外的点，在三种状态、49帧检查，逐帧均值变化阈值为 0.005 m；其他组件继承已有 RGB guard，不声称已对整幅街景深度实施严格保护。</p><p>已保存并重新加载9份 selected.npz，拒绝 case 保存 retirement=0、γ=0；在三个重放探针上，其零操作与原生父资产精确一致。独立重放3探针 × 3状态 × 4分支 × 9case，共 {len(verification['checked'])} 项指标比较，另核对81项最终保存状态。训练更新为0；全部 case 明细均保留。</p><p><a href="summary.json">交付汇总</a> · <a href="fixed_metrics.csv">5292行逐帧指标</a> · <a href="data/summary.json">运行汇总原件</a> · <a href="data/verification.json">重放原件</a> · {audit_link}<a href="architecture.svg">组件图</a> · <a href="MANIFEST.sha256">文件校验</a></p></section>
<script>const ids=['layer','group','raw','state'],c=Object.fromEntries(ids.map(k=>[k,document.getElementById(k)]));function update(){{const l=c.layer.value,g=c.group.value,r=c.raw.value.padStart(3,'0'),s=c.state.value;document.getElementById('frame').src=`frames/${{l}}_${{g}}_${{r}}_${{s}}.jpg`;for(const p of [105,113,130]){{const path=`frames/${{l}}_${{g}}_${{p}}_${{s==='keep'?'keep':'delete'}}_full.jpg`;document.getElementById('probe'+p).src=path;document.getElementById('link'+p).href=path;}}}}ids.forEach(k=>c[k].addEventListener('change',update));</script></main></html>'''
    (out/'index.html').write_text(report, encoding='utf-8')
    manifest = [f'{sha(path)}  {path.relative_to(out).as_posix()}' for path in sorted(out.rglob('*')) if path.is_file()]
    (out/'MANIFEST.sha256').write_text('\n'.join(manifest)+'\n', encoding='utf-8')
    print(json.dumps(dict(delivery=str(out), files=len(manifest)+1, csv_rows=len(csv_rows), accepted_cases=accepted_count)))

if __name__ == '__main__':
    main()
