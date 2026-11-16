"""固定输入/噪声配对比较，并保留世界 atlas 的轻量索引。"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--dataset', required=True)
    args = p.parse_args()
    root = Path(args.run)
    native_dir = root / 'eval/native_step000000'
    final_dir = root / f'eval/lora_step{args.steps:06d}'
    native = json.loads((native_dir / 'metrics.json').read_text())
    final = json.loads((final_dir / 'metrics.json').read_text())
    for key in ['model','prompt','negative_prompt','inference_seeds','inference_steps','guidance','scheduler','canvas','training_objective','train_ids','eval_ids']:
        assert native['contract'][key] == final['contract'][key], key
    lookup = {(r['id'], r['seed']): r for r in native['rows']}
    pairs = []
    for row in final['rows']:
        before = lookup[(row['id'], row['seed'])]
        assert before['valid_hole_pixels'] == row['valid_hole_pixels']
        assert before['split'] == row['split']
        for key in ['canvas_input','canvas_mask']:
            a = np.array(Image.open(native_dir / f'{row["id"]}_{key}.png'))
            b = np.array(Image.open(final_dir / f'{row["id"]}_{key}.png'))
            assert np.array_equal(a,b), (row['id'], key)
        pairs.append(dict(id=row['id'], split=row['split'], seed=row['seed'], valid_hole_pixels=row['valid_hole_pixels'], native_mae=before['mae'], lora_mae=row['mae'], delta_mae=row['mae']-before['mae'], relative_improvement=1-row['mae']/before['mae'], lora_better=row['mae']<before['mae']))
    summary = {}
    for split in ['train','eval']:
        rows = [r for r in pairs if r['split']==split]
        if not rows:
            continue
        denominator = sum(r['valid_hole_pixels'] for r in rows)
        before = sum(r['native_mae']*r['valid_hole_pixels'] for r in rows)/denominator
        after = sum(r['lora_mae']*r['valid_hole_pixels'] for r in rows)/denominator
        per_tile = {}
        for r in rows:
            per_tile.setdefault(r['id'], []).append(r['delta_mae'])
        summary[split] = dict(world_tiles=len(per_tile), sample_seed_pairs=len(rows), valid_hole_pixel_pairs=denominator, native_mae=before, lora_mae=after, delta_mae=after-before, relative_improvement=1-after/before, better_seed_pairs=int(sum(r['lora_better'] for r in rows)), better_world_tiles=int(sum(np.mean(v)<0 for v in per_tile.values())), median_world_tile_delta_mae=float(np.median([np.mean(v) for v in per_tile.values()])))
    report = dict(terminal_step=args.steps, endpoint_predeclared=True, best_checkpoint_selection=False, equality_checks=dict(model=True, canvas_inputs=True, mask=True, reference=True, context=True, prompt=True, seeds=True, inference_steps=True, scheduler=True), summary=summary, pairs=pairs, limitations=['within-scene spatial holdout; not unseen-scene generalization','pseudo-hole on visible real supported road, not hidden road ground truth','sRGB appearance at reference exposure; not physical PBR','generated unknown support remains a prior and is rejected from pseudo-hole metrics','human_verdict remains null'], human_verdict=None)
    restored_dir=root/'eval/restored_final'
    if (restored_dir/'metrics.json').exists():
        restored=json.loads((restored_dir/'metrics.json').read_text())
        restored_lookup={(r['id'],r['seed']):r for r in restored['rows']}
        assert set(restored_lookup)=={(r['id'],r['seed']) for r in final['rows']}
        exact=all(np.array_equal(np.array(Image.open(final_dir/r['output'])),np.array(Image.open(restored_dir/restored_lookup[(r['id'],r['seed'])]['output']))) for r in final['rows'])
        assert exact,'restored final inference differs'
        report['fresh_process_final_eval_exact']=exact
    (root / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    dataset = json.loads(Path(args.dataset).read_text())
    ids={r['id'] for r in pairs}
    chosen = [s for s in dataset['samples'] if s['split']=='train' and s['id'] in ids][:4] + [s for s in dataset['samples'] if s['split']=='eval' and s['id'] in ids]
    size, header, row_h = 128, 25, 151
    sheet = Image.new('RGB', (size*5, header+row_h*len(chosen)), '#101722')
    draw = ImageDraw.Draw(sheet)
    for j,label in enumerate(['reference','input/context','native (0 updates)',f'LoRA ({args.steps} updates)','target (eval only)']):
        draw.text((j*size+3,6),label,fill='white')
    seed = final['contract']['inference_seeds'][0]
    dsroot = Path(args.dataset).parent
    for i,s in enumerate(chosen):
        name = s['id']
        files = [dsroot/s['reference'],dsroot/s['input'],native_dir/f'{name}_seed{seed}_output.png',final_dir/f'{name}_seed{seed}_output.png',dsroot/s['target']]
        y=header+i*row_h
        for j,f in enumerate(files):
            sheet.paste(Image.open(f).convert('RGB').resize((size,size)),(j*size,y))
        draw.text((3,y+size+3),f'{s["split"]}: {name}',fill='#cbd7e7')
    sheet.save(root / 'native_vs_lora.png')
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2))


if __name__=='__main__':
    main()
