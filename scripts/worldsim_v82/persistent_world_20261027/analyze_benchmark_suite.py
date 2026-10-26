"""CPU packaging of executed eight-family diagnostics, never draft counts."""
import argparse, json, shutil, subprocess
from pathlib import Path
from PIL import Image, ImageDraw
from benchmark_geometry import read


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--execution', type=Path, required=True)
    args = p.parse_args()
    state = read(args.execution/'status.json')
    output = args.execution/'cpu_delivery'
    output.mkdir(exist_ok=True)
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    items, panels = [], []
    for stage in ['camera_objects', 'foreign_asset']:
        if stage not in state['stages']:
            continue
        batch_path = Path(state['stages'][stage]['summary']).parent
        batch = read(batch_path/'summary.json')
        assert batch['adapter_actual_native_max_difference'] <= 2e-6
        assert batch['master_files_and_tensor_versions_unchanged']
        for record in batch['completed']:
            family = record['family']
            source = batch_path/family
            target = output/family
            target.mkdir(exist_ok=True)
            frames = sorted((source/'native_rgb').glob('*.png'))
            assert len(frames) == 31
            summary = read(source/'summary.json')
            shutil.copy2(source/'summary.json', target/'native_summary.json')
            contact_index = 3*len(frames)//4
            shutil.copy2(frames[contact_index], target/'native.png')
            videos = [('native_rgb', 'native.mp4')]
            if (source/'propainter/summary.json').exists():
                shutil.copy2(source/'propainter/summary.json', target/'backend_summary.json')
                shutil.copy2(source/'propainter/model_png'/frames[contact_index].name, target/'generated.png')
                videos.append(('propainter/model_png', 'generated.mp4'))
            for relative, name in videos:
                destination = target/name
                if not destination.exists():
                    subprocess.run([ffmpeg, '-v', 'error', '-threads', '2', '-framerate', '10',
                                    '-pattern_type', 'glob', '-i', str(source/relative/'*.png'),
                                    '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
                                    '-threads', '2', '-an', str(destination)], check=True)
            tile = Image.new('RGB', (480, 350), '#152b3d')
            tile.paste(Image.open(frames[contact_index]).convert('RGB'), (0, 30))
            ImageDraw.Draw(tile).text((8, 8), family+' / raw'+frames[contact_index].stem, fill='white')
            panels.append(tile)
            items.append(dict(family=family, frames=31, native_source=str(source),
                              fixed_backend_executed=(target/'backend_summary.json').exists(),
                              formal_admitted=False, quality_accepted=False, human_verdict=None,
                              unknown_camera_domain=family.startswith('ego_'),
                              source_distinct_scene=family == 'vehicle_insertion',
                              collision_frames=sum(bool(x['asset_alignment']['annotated_collision_tracks']) for x in summary['rows']) if family == 'vehicle_insertion' else None))
    assert len({x['family'] for x in items}) == len(items)
    if panels:
        sheet = Image.new('RGB', (1920, 350*((len(panels)+3)//4)), '#152b3d')
        for i, tile in enumerate(panels):
            sheet.paste(tile, (480*(i%4), 350*(i//4)))
        sheet.save(output/'comparison.jpg', quality=95)
    result = dict(context=state['context'], executed_native_families=len(items), families=items,
                  executed_fixed_backend_families=sum(x['fixed_backend_executed'] for x in items),
                  request_count_not_independent_scene_count=True, formal_admitted=0,
                  camera_masks_pending=True, counterfactual_RGB_GT=False,
                  quality_accepted=False, human_verdict=None)
    (output/'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
