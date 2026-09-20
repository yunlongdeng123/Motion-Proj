"""实际生成画面、策略动作和独立参考轨迹的对照；不补画事故或重建表面。"""
import argparse
import json
from pathlib import Path
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_following_closed_loop import OUT, ARMS

NAMES = {'gt_clean':'GT state', 'dvgt_metric':'DVGT state', 'dvgt_lidar_scaled':'Global LiDAR scale', 'reference_lidar':'Target LiDAR repair'}
COLORS = {'gt_clean':'#526e84', 'dvgt_metric':'#d05549', 'dvgt_lidar_scaled':'#9470b6', 'reference_lidar':'#168879'}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--figures-only', action='store_true'); parser.add_argument('--refresh-media', action='store_true'); args = parser.parse_args()
    assert json.loads((OUT/'queue_result.json').read_text())['status'] == 'complete'
    dest = OUT/'review'
    if args.figures_only or args.refresh_media: assert (dest/'comparison.json').exists()
    else: assert not dest.exists(); dest.mkdir()
    protocol = json.loads((OUT/'protocol.json').read_text()); base = Path(protocol['base'])
    original = np.load(base/'trajectory.npz'); samples = {}; summary = []
    read = json.loads((base.parent/'reconstruction/readout_ray_control_result.json').read_text())
    for name in ARMS:
        folder = OUT/name
        samples[name] = {'run': json.loads((folder/'result.json').read_text()),
                         'dense': json.loads((folder/'dense_reference_result.json').read_text()),
                         'decisions': json.loads((folder/'decisions.json').read_text()),
                         'video': np.load(folder/'generated.npy', mmap_mode='r')}
    reference = samples['gt_clean']; ref_pos = np.array([r['ego_xy_yaw'][:2] for r in reference['dense']['rows']])
    ref_acc = np.array([r['policy']['acceleration_mps2'] for r in reference['decisions']])
    for name, values in samples.items():
        positions = np.array([r['ego_xy_yaw'][:2] for r in values['dense']['rows']])
        acceleration = np.array([r['policy']['acceleration_mps2'] for r in values['decisions']])
        assert np.array_equal(values['video'][0], reference['video'][0]), '生成初帧必须配对一致'
        summary.append({'arm': name, 'initial_target_center_residual_m': 0. if name=='gt_clean' else read['readouts'][name]['center_error_m'],
                        'progress_m': values['dense']['progress_m'],
                        'progress_change_vs_gt_m': values['dense']['progress_m']-reference['dense']['progress_m'],
                        'end_position_change_vs_gt_m': float(np.linalg.norm(positions[-1]-ref_pos[-1])),
                        'mean_abs_acceleration_change_vs_gt_mps2': float(np.mean(abs(acceleration-ref_acc))),
                        'max_abs_acceleration_change_vs_gt_mps2': float(np.max(abs(acceleration-ref_acc))),
                        'braking_decisions': int((acceleration < 0).sum()), 'all_decisions': len(acceleration),
                        'minimum_reference_clearance_m': values['dense']['minimum_reference_clearance_m'],
                        'reference_overlap_frames': len(values['dense']['overlap_frames']), 'frames': len(positions)})
    (dest/'comparison.json').write_text(json.dumps({'status':'complete', 'cases':summary, 'seed':42, 'source_logs':1,
        'human_verdict':None, 'failure_ledger_delta':'none', 'boundary':'first exposed task with fixed ordinary policy; full causal feedback but not a driving SOTA/generalization result'}, indent=2)+'\n')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    origin = original['ego_world'][0, :2, 3]; rotation = original['ego_world'][0, :2, :2]
    for name, values in samples.items():
        dense = values['dense']['rows']; ts = np.array([r['time_s'] for r in dense]); positions = np.array([r['ego_xy_yaw'][:2] for r in dense])
        relative = (positions-origin)@rotation
        axes[0, 0].plot(-relative[:, 1], relative[:, 0], color=COLORS[name], label=NAMES[name])
        target = next(t for t in json.loads((OUT/name/'condition_scene.json').read_text())['tracks'] if t['id'] == protocol['target'])
        selected = np.array(target['frames']) < 117
        condition_xy = (np.array(target['centers'])[selected, :2]-origin)@rotation
        axes[0, 0].plot(-condition_xy[:, 1], condition_xy[:, 0], '--', color=COLORS[name], alpha=.7, linewidth=1.1)
        axes[0, 0].scatter(-condition_xy[0, 1], condition_xy[0, 0], s=22, color=COLORS[name])
        ds = values['decisions']; times = [r['observation_frame']/30 for r in ds]
        acceleration = [r['policy']['acceleration_mps2'] for r in ds]
        axes[0, 1].step(times, acceleration, where='post', color=COLORS[name], label=NAMES[name])
        axes[1, 0].plot(ts, [r['clearance']['distance_m'] if r['clearance'] else np.nan for r in dense], color=COLORS[name], label=NAMES[name])
        axes[1, 1].plot(ts, np.linalg.norm(positions-ref_pos, axis=1), color=COLORS[name], label=NAMES[name])
    axes[0, 0].set(xlabel='Initial ego right (m)', ylabel='Initial ego forward (m)', title='Solid: executed ego / dashed: target condition')
    axes[0, 0].set_aspect('equal', adjustable='datalim')
    axes[0, 1].set(xlabel='Policy observation time (s)', ylabel='Acceleration (m/s^2)', title='Actions from actual generated RGB')
    axes[0, 1].axhline(0, color='#777777', lw=.8)
    axes[1, 0].set(xlabel='Time (s)', ylabel='Footprint clearance (m)', title='Independent recorded actors; every frame')
    axes[1, 1].set(xlabel='Time (s)', ylabel='Position difference vs GT-state loop (m)', title='Closed-loop execution difference')
    for ax in axes.flat:
        ax.grid(alpha=.2); ax.spines[['top','right']].set_visible(False); ax.legend(fontsize=7)
    fig.savefig(dest/'closed-loop-results.png', dpi=180); fig.savefig(dest/'closed-loop-results.svg'); plt.close(fig)
    for svg in dest.glob('*.svg'): svg.write_bytes(('\n'.join(x.rstrip() for x in svg.read_text().splitlines())+'\n').encode())
    if args.figures_only:
        print('Updated figure with actual target-condition trajectories'); return
    chosen = [4, 44, 84]
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    sheet = Image.new('RGB', (3*640, 4*390), '#122335'); draw = ImageDraw.Draw(sheet)
    for row_index, name in enumerate(ARMS):
        values = samples[name]
        for column, f in enumerate(chosen):
            row = next(x for x in values['decisions'] if x['observation_frame'] == f)
            img = Image.fromarray(values['video'][f]); d = ImageDraw.Draw(img)
            lead = row['policy']['lead']
            if lead: d.rectangle(lead['box'], outline='#20ebb0', width=4)
            x, y = column*640, row_index*390
            draw.text((x+8,y+3), f'{NAMES[name]} | policy input t={f/30:.2f}s', fill='white', font=font)
            draw.text((x+8,y+21), f'next action a={row["policy"]["acceleration_mps2"]:.3f} m/s^2 | green=selected lead', fill='white', font=small)
            sheet.paste(img.resize((640,352)), (x,y+38))
    sheet.save(dest/'actual-policy-inputs.jpg', quality=95)
    with av.open(str(dest/'four-arm-feedback.mp4'), 'w') as writer:
        stream=writer.add_stream('libx264',rate=30);stream.width=1280;stream.height=768;stream.pix_fmt='yuv420p';stream.options={'crf':'18'}
        for f in range(117):
            grid=Image.new('RGB',(1280,768),'#122335'); draw=ImageDraw.Draw(grid)
            for i,name in enumerate(ARMS):
                x,y=(i%2)*640,(i//2)*384
                decisions=samples[name]['decisions']; active=next(r for r in decisions if r['first_condition_frame']<=f<=r['last_condition_frame'])
                draw.text((x+8,y+5),f'{NAMES[name]} | t={f/30:.2f}s | applied a={active["policy"]["acceleration_mps2"]:.3f} m/s^2',fill='white',font=font)
                grid.paste(Image.fromarray(samples[name]['video'][f]).resize((640,352)),(x,y+32))
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.array(grid),format='rgb24')):writer.mux(packet)
        for packet in stream.encode():writer.mux(packet)
    with av.open(str(dest/'four-arm-feedback.mp4')) as video: decoded=sum(1 for _ in video.decode(video=0))
    assert decoded==117
    print(json.dumps({'status':'complete','comparison_decoded_frames':decoded,'summary':summary}),flush=True)


if __name__=='__main__':main()
