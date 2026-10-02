"""r23候选固定配比选择、真实磁盘合同及逐帧审核材料；不自动准入。"""
from pathlib import Path
import sys, json, copy, time
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))
import reveal_factory as factory
from reveal_factory import O, T, ROOT, read, dump, prepare_masks
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg


def select():
    p = O/'selected.json'
    if p.exists(): return read(p)['selected']
    pool = [r for f in sorted((O/'lane_candidates').glob('*.json')) for r in read(f)['candidates']]
    quotas = {'train': {'protected_reveal': 24, 'dense_known_background': 10, 'ordinary_background': 6},
              'validation': {'protected_reveal': 6, 'dense_known_background': 3, 'ordinary_background': 1}}
    chosen = []; used = Counter(); source_families = set(); deficits = {}
    for split, families in quotas.items():
        for family, n in families.items():
            available = sorted([r for r in pool if r['source_split'] == split and r['data_family'] == family],
                               key=lambda r: (r['source_id'], r['lane_token'], r['lane_midpoint_distance_m'], r['speed_mps']))
            selected = []
            # 先每scene一条，再允许每scene最多两条；不能靠一个scene凑数量。
            for cap in [1, 2]:
                for r in available:
                    key = (r['source_id'], family)
                    if len(selected) >= n: break
                    if key in source_families or used[r['scene']] >= cap: continue
                    selected.append(copy.deepcopy(r)); used[r['scene']] += 1; source_families.add(key)
            deficits[split+'/'+family] = n-len(selected); chosen += selected
    for i, r in enumerate(chosen, 1):
        r.update(case_id=f'R{i:03}', edge_mode=['hard', 'feather_05', 'feather_10'][(i-1)%3],
                 independent_quality='pending', human_verdict=None, training_admission=False)
    dump(p, {'selected': chosen, 'pool': len(pool), 'requested': 50, 'count': len(chosen), 'quotas': quotas,
             'deficits': deficits, 'scenes': len(used), 'max_cases_per_scene': max(used.values(), default=0),
             'selection_model_outputs_used': False, 'training_admission': 0})
    return chosen


def video(path, frames):
    writer = imageio_ffmpeg.write_frames(str(path), (1024, 576), fps=10, codec='libx264',
                quality=7, pix_fmt_out='yuv420p', macro_block_size=1, output_params=['-movflags', '+faststart'])
    writer.send(None)
    for f in frames: writer.send(np.ascontiguousarray(f))
    writer.close()


def main():
    cv2.setNumThreads(1); selected = select()
    factory.legacy.O = O; factory.legacy.ROOT = ROOT; g = factory.legacy.geometry()
    review = O/'data_review'; (review/'contacts').mkdir(parents=True, exist_ok=True)
    rows = []; checks = []; font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)
    for row in selected:
        cid = row['case_id']; sid = row['source_id']; dest = O/'synthetic'/cid
        if (dest/'pair_manifest.json').exists():
            done = read(dest/'pair_manifest.json'); rows.append(done); checks.append(read(dest/'technical.json')); continue
        start = time.time(); src = g.sources[sid]; g.prepare(sid); pm = factory.legacy.protections(g, sid)
        mesh = dict(np.load(T/'r8/assets'/f'{row["asset"]}.npz'))
        aa = [factory.legacy.old.silhouette(mesh['vertices'], mesh['faces'], f['actor'], sf) for f, sf in zip(row['frames'], src['frames'])]
        q, why = factory.exact(g, src, row, aa, pm); assert q is not None, (cid, why)
        for role in ['Y', 'X', 'influence', 'model_hole', 'write_alpha', 'protected', 'condition_preview']:
            (dest/role).mkdir(parents=True, exist_ok=True)
        panels = {k: [] for k in ['gt', 'labels', 'mask', 'condition']}; metrics = []
        for i, (f, silhouette) in enumerate(zip(src['frames'], aa)):
            with Image.open(ROOT/'rgb'/f['filename']) as im:
                y = np.asarray(im.convert('RGB').resize((1024, 576), Image.Resampling.LANCZOS)).copy()
            a = silhouette.astype('float32'); sigma = {'hard': 0., 'feather_05': .5, 'feather_10': 1.}[row['edge_mode']]
            if sigma: a = cv2.GaussianBlur(a, (3, 3), sigma)
            influence = a > 1/65535
            # 中性轮廓只表达遮挡，不生成车身GT；完整隐藏RGB不进入训练条件。
            x = np.rint(y*(1-a[..., None])+np.array([100, 112, 124])*a[..., None]).clip(0, 255).astype('uint8')
            contract, _ = prepare_masks(silhouette); h = contract['model_mask']
            assert h[influence].all() and not h[512:].any()
            cp = x.copy(); cp[h] = 127
            expected = y.copy(); expected[h] = 127
            assert np.array_equal(cp, expected), '擦除后的X仍残留合成影响'
            label = y.copy(); label[silhouette] = np.rint(y[silhouette]*.35+np.array([250, 190, 25])*.65).astype('uint8')
            mask_img = y.copy(); mask_img[h] = np.rint(y[h]*.3+np.array([70, 130, 250])*.7).astype('uint8')
            for t, ms in pm.items():
                m = ms[i]; cv2.drawContours(label, cv2.findContours(m.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (70, 240, 110), 2)
                Image.fromarray(m.astype('uint8')*255).save(dest/'protected'/f'{i:03}_{t}.png')
            arrays = {'Y': y, 'X': x, 'influence': influence.astype('uint8')*255,
                      'model_hole': h.astype('uint8')*255, 'write_alpha': np.rint(contract['alpha']*255).astype('uint8'),
                      'condition_preview': cp}
            for role, arr in arrays.items(): Image.fromarray(arr).save(dest/role/f'{i:03}.png')
            # 检查实际落盘数据，而非只相信渲染内存。
            yy = np.asarray(Image.open(dest/'Y'/f'{i:03}.png')); xx = np.asarray(Image.open(dest/'X'/f'{i:03}.png'))
            hh = np.asarray(Image.open(dest/'model_hole'/f'{i:03}.png')) > 0
            assert np.array_equal(yy, y) and not np.any((xx != yy).any(-1)&~hh)
            panels['gt'].append(y); panels['labels'].append(label); panels['mask'].append(mask_img); panels['condition'].append(cp)
            metrics.append({'frame': i, 'hole_fraction': float(h.mean()), 'RGB_leak': 0, 'ego_pixels': 0,
                            'masked_X_equals_masked_Y': True, 'GT_exact': True})
        assetdir = review/'assets'/cid; assetdir.mkdir(parents=True, exist_ok=True)
        sheet = Image.new('RGB', (1600, 830), (18, 24, 32)); draw = ImageDraw.Draw(sheet)
        for j, i in enumerate([0, 15, 29]):
            for k, (role, frames) in enumerate(panels.items()):
                draw.text((400*k+6, j*275+8), f'{cid} f{i} | {role}', font=font, fill='white')
                sheet.paste(Image.fromarray(frames[i]).resize((400, 225)), (k*400, j*275+38))
        sheet.save(review/'contacts'/f'{cid}.jpg', quality=95)
        for role, frames in panels.items(): video(assetdir/(role+'.mp4'), frames)
        completed = row | q | {'folder': str(dest), 'receiver_scene': src['scene'], 'split': row['source_split'],
            'frame_count': 30, 'camera': src['camera'], 'pixel_metrics': metrics,
            'videos': {r: f'assets/{cid}/{r}.mp4' for r in panels}, 'review_contact': f'contacts/{cid}.jpg',
            'model_H_policy': 'sam_full_v2', 'RGB_training_input': 'X masked before resize; synthetic RGB fully removed'}
        check = {'case_id': cid, 'technical_pass': True, 'frames': 30, 'training_admission': 0,
                 'independent_quality': 'pending', 'RGB_leak': 0, 'GT_exact': True, 'human_verdict': None}
        dump(dest/'pair_manifest.json', completed); dump(dest/'technical.json', check)
        rows.append(completed); checks.append(check)
        dump(review/'synthetic_manifest.json', {'clips': rows}); dump(O/'technical_checks.json', {'stage': 'running', 'cases': checks})
        print('REVEAL_RENDER', cid, src['scene'], row['data_family'], round(time.time()-start, 1), flush=True)
    dump(review/'synthetic_manifest.json', {'clips': rows})
    dump(O/'technical_checks.json', {'stage': 'complete_pending_independent_QA', 'cases': checks, 'training_admission': 0})


if __name__ == '__main__': main()
