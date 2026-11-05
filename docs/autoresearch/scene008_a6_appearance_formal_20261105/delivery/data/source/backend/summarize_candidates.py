"""Read completed fixed candidates, verify receipts, build a comparison board."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

RUN = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a6_appearance_formal_20261105')
ROOT = RUN / 'candidates_raw106'
ids = [f'prompt{p}_seed{s}' for p in (1, 2) for s in range(4)]
receipts = [json.loads((ROOT / i / 'receipt.json').read_text()) for i in ids]
assert len(receipts) == 8
assert all(r['finite_output'] and r['known_pixels_preserved_exactly'] for r in receipts)
assert all(r['outside_effective_mask_max_abs_rgb_u8'] == 0 for r in receipts)
assert all(r['training_updates'] == 0 and not r['ranking_or_selection_performed'] for r in receipts)
assert all(not r['prompt_encoding']['truncated'] for r in receipts)
assert len({r['model_manifest_sha256'] for r in receipts}) == 1
assert len({json.dumps(r['parameters'], sort_keys=True) for r in receipts}) == 1
smoke = json.loads((RUN / 'backend_smoke/prompt1_seed0/receipt.json').read_text())
replay_exact = (smoke['outputs']['composed.png']['sha256'] == receipts[0]['outputs']['composed.png']['sha256'])
assert replay_exact, 'First formal candidate must exactly reproduce the independent smoke'

board = Image.new('RGB', (4 * 512, 2 * 548 + 52), (18, 18, 18))
draw = ImageDraw.Draw(board)
font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font = ImageFont.truetype(font_path, 20) if Path(font_path).is_file() else ImageFont.load_default()
draw.text((12, 12), 'Frozen SD1.5 + IP-Adapter | BUILD106 a6 only | inferred 2D appearance candidates', fill='white', font=font)
for index, candidate_id in enumerate(ids):
    x, y = (index % 4) * 512, 52 + (index // 4) * 548
    board.paste(Image.open(ROOT / candidate_id / 'composed.png').convert('RGB'), (x, y + 32))
    draw.text((x + 10, y + 5), f'{candidate_id}  |  known RGB exact', fill='white', font=font)
board.save(ROOT / 'candidate_board.png')
summary = {
    'status': 'EIGHT_FIXED_CANDIDATES_QA_PASS',
    'candidate_count': 8,
    'prompts': 2, 'seeds_per_prompt': [0, 1, 2, 3],
    'inference_seconds_total': sum(r['seconds'] for r in receipts),
    'smoke_seconds_separate': smoke['seconds'],
    'max_peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in receipts),
    'max_peak_reserved_bytes': max(r['peak_reserved_bytes'] for r in receipts),
    'all_finite': True, 'all_known_pixels_exact': True,
    'all_outside_effective_mask_exact': True,
    'independent_process_smoke_first_candidate_replay_sha256_exact': replay_exact,
    'prompt_token_counts': {i: r['prompt_encoding']['full_token_count'] for i, r in zip(ids, receipts)},
    'model_training_updates': 0, 'downloads': 0, 'held_rgb_used': False,
    'selection_performed': False,
    'candidate_board': str(ROOT / 'candidate_board.png'),
    'limitations': ['PVD silhouette constrains write mask and composition, not internal diffusion geometry.',
                    'These are inferred 2D appearance candidates, not yet a validated 3D material asset.'],
    'candidates': [{'id': i, 'image': str(ROOT / i / 'composed.png'),
                    'sha256': r['outputs']['composed.png']['sha256'], 'seconds': r['seconds']}
                   for i, r in zip(ids, receipts)],
}
(ROOT / 'qa_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: v for k, v in summary.items() if k != 'candidates'}, indent=2))
