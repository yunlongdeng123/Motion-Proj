import json
from pathlib import Path
import shutil
from PIL import Image

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
OUT=Path('/root/autodl-tmp/codex_tmp/horizon_waymo_inventory_20261031_r1')
sources=json.loads((OUT/'canonical_list_mapping.json').read_text())['sources']
cache=ROOT/'scene_search_mirror'
preview_root=cache/'previews';record_root=cache/'records'
output=OUT/'existing_cache_candidates';output.mkdir(exist_ok=False)
summary=[]
counts=dict(mirror_preview_contexts=sum(p.is_dir() for p in preview_root.iterdir()),
    mirror_record_jsons=len(list(record_root.glob('*.json'))),
    scene_search_candidate_contexts=sum(p.is_dir() for p in (ROOT/'scene_search/candidates').iterdir()))
for label,path in [('sorted_val','/root/autodl-tmp/waymo_val_list.txt'),
                   ('drivestudio_train','/root/autodl-tmp/waymo_train_list.txt'),
                   ('dggt_explicit','/root/autodl-tmp/dggt/data/waymo_example_scenes.txt'),
                   ('dggt_val','/root/autodl-tmp/dggt/data/waymo_val_list.txt')]:
    source=next(s for s in sources if s['path']==path)
    for candidate in source['candidate_rows']:
        context=candidate.get('context_id')
        if not context:continue
        record=record_root/(context+'.json');preview=preview_root/context/'camera1.jpg'
        row=dict(list_label=label,scene_number=candidate['scene_number'],context_id=context,
            mirror_record_exists=record.is_file(),mirror_front_preview_exists=preview.is_file(),
            full_raw_data_coverage='not_implied_by_preview_cache')
        if record.is_file():
            data=json.loads(record.read_text())
            verified=(data.get('segment')==context and data.get('length_crc32c')=='passed' and
                data.get('payload_crc32c')=='passed' and data.get('protobuf_context')=='passed')
            row.update(record_source=str(record),source_repository=data.get('repository'),
                pinned_revision=data.get('pinned_revision'),relative_path=data.get('relative_path'),
                expected_raw_bytes=data.get('raw_expected_size_bytes'),first_record_bytes=data.get('first_record_bytes'),
                source_timestamp_micros=data.get('timestamp_micros'),cached_preview_crc_and_context_verified=verified,
                original_record_status=data.get('status'))
            if preview.is_file() and verified:
                folder=output/(label+'_'+candidate['scene_number']);folder.mkdir()
                shutil.copy2(preview,folder/'front_000.jpg')
                row.update(copied_preview=str(folder/'front_000.jpg'),preview_wh=list(Image.open(preview).size),
                    role='CRC_verified_first_original_FRONT_thumbnail_not_native_or_generated_render')
                (folder/'source_receipt.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
        alternative=ROOT/'scene_search/candidates'/context/'cam1_frame000.jpg'
        row['alternative_first_front_path']=str(alternative) if alternative.is_file() else None
        summary.append(row)
(OUT/'preview_cache_inventory.json').write_text(json.dumps(dict(counts=counts,candidates=summary),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(counts=counts,copied=sum('copied_preview' in r for r in summary),
    sorted_val_candidates=[r for r in summary if r['list_label']=='sorted_val']),ensure_ascii=False))
