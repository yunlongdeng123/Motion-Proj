"""并排展示真实输入条件与生成视频；标记来源，避免将条件当作生成或真值。"""
from pathlib import Path
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

run = Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-BASELINE-01/20260920-single3090-r1')
conditions = np.load('/root/autodl-tmp/runs/worldsim_v75/WS-V75-PREFLIGHT-01/20260920-r1/conditions.npy', mmap_mode='r')
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
font_small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
sample_ids = [0, 58, 118, 177, 236]
tiles = []
with av.open(str(run / 'clean-seed42.mp4')) as src, av.open(str(run / 'condition-and-clean.mp4'), mode='w') as dst:
    stream = dst.add_stream('libx264', rate=30)
    stream.width, stream.height, stream.pix_fmt = 1280, 394, 'yuv420p'
    stream.options = {'crf': '18'}
    for i, frame in enumerate(src.decode(video=0)):
        canvas = Image.new('RGB', (1280, 394), '#e9edf2')
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 10), 'Official scene condition (input)', fill='#123047', font=font)
        draw.text((652, 10), 'OmniDreams clean (generated)', fill='#123047', font=font)
        draw.text((1112, 12), f't = {i / 30:.2f} s', fill='#123047', font=font_small)
        canvas.paste(Image.fromarray(conditions[i]).resize((640, 352)), (0, 42))
        canvas.paste(frame.to_image().resize((640, 352)), (640, 42))
        for packet in stream.encode(av.VideoFrame.from_image(canvas)):
            dst.mux(packet)
        if i in sample_ids:
            canvas.save(run / f'comparison-{i:03d}.jpg', quality=94)
            tiles.append(canvas)
    assert i + 1 == len(conditions) == 237
    for packet in stream.encode():
        dst.mux(packet)
sheet = Image.new('RGB', (1280, 394 * len(tiles)), 'white')
for row, tile in enumerate(tiles):
    sheet.paste(tile, (0, row * 394))
sheet.save(run / 'condition-and-clean-contact.jpg', quality=94)
print('Saved paired video and 5-time-point contact sheet.')
