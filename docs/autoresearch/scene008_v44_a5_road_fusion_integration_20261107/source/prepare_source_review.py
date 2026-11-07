"""Read the immutable road delivery; create only this task's review/plan assets."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent.parent
SOURCE=Path('C:/Users/dengyunlong/Documents/Codex/2026-11-05/j-ao-j/outputs/scene008_road_fusion_bake/delivery')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'),parse_constant=lambda value:(_ for _ in ()).throw(ValueError(value)))
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    (ROOT/'figures').mkdir(exist_ok=True)
    source_index_sha=sha(SOURCE/'index.html')
    json_files=[]
    long=Path('\\\\?\\'+str(SOURCE))
    for p in sorted(long.rglob('*.json')):
        value=read(p)
        json_files.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),
            schema=value.get('schema') if isinstance(value,dict) else None,
            status=value.get('status') if isinstance(value,dict) else None))
    assets={}
    arrays={}
    for variant in ('direct','fusion','edgeblend'):
        folder=SOURCE/f'repro/engineer/canonical_{variant}_v1'
        manifest=read(folder/'scene_manifest.json')
        asset=folder/manifest['new_road']['file']
        assert sha(asset)==manifest['new_road']['sha256']
        with np.load(asset,allow_pickle=False) as data:
            arrays[variant]={k:data[k] for k in manifest['new_road']['tuple_fields']}
        assets[variant]=dict(manifest_path=str(folder/'scene_manifest.json'),manifest_sha256=sha(folder/'scene_manifest.json'),
            asset_path=str(asset),asset_sha256=sha(asset),rows=manifest['new_road']['rows'],
            renderer_policy=manifest['renderer_policy'],lineage=manifest['lineage'],
            whole_scene_adoption_claim=manifest.get('adoption_claim'),
            parity_summary={k:v for k,v in read(folder/'parity/receipt.json').items() if k in ('forwards','all_float_exact','all_PNG_byte_exact')})
    geometry={}
    for field in ('render_model_xyz','render_scales_model','render_rotation_wxyz','render_opacity_logit'):
        geometry[field]=dict(A_vs_C_exact=bool(np.array_equal(arrays['direct'][field],arrays['edgeblend'][field])),
                             B_vs_C_exact=bool(np.array_equal(arrays['fusion'][field],arrays['edgeblend'][field])))
        assert all(geometry[field].values())
    pngs=[]
    sizes=Counter()
    for path in sorted(long.rglob('*.png')):
        with Image.open(path) as im:im.verify()
        with Image.open(path) as im:size=im.size
        sizes[str(size)]+=1
        pngs.append(dict(path=str(path),size=list(size),sha256=sha(path)))
    registry=read(SOURCE/'repro/red_review/world_masks49_v1/registry.json')
    assert sha(SOURCE/'repro/red_review/world_masks49_v1/registry.json')=='0af33f63171cc34b404e4a0719495a16a49b993c5aaa36d1a442b1695251e7d8'
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    for state in ('keep','delete'):
        board=Image.new('RGB',(4200,1120),'#f3f7f9');draw=ImageDraw.Draw(board)
        for i,raw in enumerate(range(89,138)):
            x=(i%7)*600;y=(i//7)*160
            with np.load(SOURCE/f'repro/red_review/world_masks49_v1/{raw:03d}.npz',allow_pickle=False) as data:
                mask=data['world_patch_clean_road']
            yy,xx=np.nonzero(mask)
            box=(max(0,int(xx.min())-5),max(0,int(yy.min())-5),min(480,int(xx.max())+6),min(320,int(yy.max())+6)) if xx.size else (0,0,480,320)
            draw.text((x+5,y+3),f'raw{raw} {state.upper()} / original world-mask bbox',font=font,fill='#193b4a')
            for col,(label,path) in enumerate([('Observed',SOURCE/f'frames/observed/{raw}.png'),('AA support',SOURCE/f'frames/baseline/{raw}_{state}.png'),('C material',SOURCE/f'frames/edgeblend/{raw}_{state}.png')]):
                draw.text((x+5+col*198,y+25),label,font=font,fill='#193b4a')
                with Image.open(path) as im:
                    assert im.size==(480,320)
                    crop=im.convert('RGB').crop(box)
                scale=min(186/crop.width,105/crop.height)
                crop=crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale))),Image.Resampling.LANCZOS)
                board.paste(crop,(x+5+col*198+(186-crop.width)//2,y+48+(105-crop.height)//2))
        board.save(ROOT/f'figures/source_C_all49_{state}_road_contact.png')
    summary=read(SOURCE/'repro/red_review/full49_edgeblend_summary_v1.json')
    ab=read(SOURCE/'repro/red_review/full49_review_summary_v1.json')
    review=dict(status='SOURCE_REVIEW_NOT_NEW_INTEGRATION_RESULT',source_report=str(SOURCE/'index.html'),source_report_sha256=source_index_sha,
        recommendation='C_ONLY_AS_APPEARANCE_AND_LATERAL_IMPROVEMENT_CANDIDATE; NOT_A_COMPLETE_SEAM_FIX',
        whole_scene_adopted=False,human_verdict=None,
        source_baseline='Previous AA n2+relativeMip+parent-cellfit 40392-row support, not latest SemanticActor5V44Integration',
        candidate_assets=assets,old_AB_C_geometry_tuple_independently_equal=geometry,
        C_original128=summary['raw128_primary'],C_all49_world_seams=summary['all49_world_seam'],
        C_all49_original_domains=summary['all49_original_domain_metrics'],C_frozen_checks=summary['frozen_checks'],
        AB_all49_world_seams=ab['all49_world_seam'],
        source_scope_limits=['C near-x regresses in both states, in original raw128 six bins and whole49 pooled bins.',
          'Lateral averages improve but many bins regress; unsupported far is null.',
          'C uses strong real-visible-RGB edge anchoring, not isolated Poisson or pure view-independent atlas success.',
          'HELD9 target reads query O for target diagnostics, not native colour conditioning; same-scene exposure remains.',
          'Old C-vs-AA alpha/depth equality cannot be transferred to latest-v44 integration.',
          'Original 6/7 severe wheel misses remain; current material does not repair geometry/opaque first-hit semantics.'],
        review_materials=dict(JSON_files_read=len(json_files),PNG_files_decoded_verified=len(pngs),PNG_sizes=dict(sizes)),
        source_input_manifest=json_files,source_PNG_manifest=pngs)
    assert sha(SOURCE/'index.html')==source_index_sha
    write(ROOT/'evidence/source_road_review.json',review)
    plan=dict(status='PREPARED_AWAIT_NEW_NATIVE_PILOT_RESULTS',main_title='Scene008：保留a5车顶与删车效果的道路融合接入',
        baseline='Latest delivered 49BUILD-fit SemanticActor5V44Integration',variants=['baseline','AA','C'],
        pilot_raws=[94,105,128,137],diagnostic_raws=[130],
        sections=['Current actual decision with no pending values invented','Architecture components',
          'Four-frame native KEEP/DELETE + raw94/105 body + raw128/137 fixed road crops',
          'Latest-vs-AA and latest-vs-C no-regression table; C-vs-AA material increment separate',
          'Original128 six-bin and world near/lateral fixed-bin scores; unsupported far',
          'Raw130 wheel diagnostics separate; original 71 rays preserved',
          'Full49 only after explicit root numerical+native-image gate',
          'Frozen masks, thresholds, source/asset/model guards, compute/retrieval and human null'],
        candidate_data_namespace='evidence/pilot/{baseline,AA,C}/frames/{raw:03d}/{keep,delete}.png and state_float.npz',
        source_review='evidence/source_road_review.json',frozen_contract='contract/independent_evaluation_contract_v1.json',
        no_old_report_or_central_viewer_modified=True,no_GPU_experiment_started=True)
    write(ROOT/'report_plan.json',plan)
    print(json.dumps(dict(status=review['status'],JSON=len(json_files),PNG=len(pngs),geometry=geometry),ensure_ascii=True))


if __name__=='__main__':main()
