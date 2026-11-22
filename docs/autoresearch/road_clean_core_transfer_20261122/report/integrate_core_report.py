"""Append real core-only native source and standalone target-stamp evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

from adapt_core_report import HERE,PREFIX,build_source,read,write,sha,need

INDEX=Path(r'C:\Users\dengyunlong\Documents\Codex\2026-11-16\xian\outputs\road_material_three_rounds_20261121\index.html')
BEGIN='<!-- CLEAN_CORE_TRANSFER_BEGIN -->';END='<!-- CLEAN_CORE_TRANSFER_END -->'

def history(raw):
    text=raw.decode('utf8');m=re.search(r'\bconst\s+D\s*=\s*',text);d,n=json.JSONDecoder().raw_decode(text[m.end():]);c=d['catalog']
    counts=[len(c['models']),sum(v.get('batch')!='gaussian_transfer' for v in c['models']),len(c['rows'])];need(counts==[58,55,744],'Original catalog changed')
    result={'counts':counts,'D_literal_sha256':hashlib.sha256(text[m.end():m.end()+n].encode('utf8')).hexdigest(),'blocks':{}}
    for name in ('SOURCE_QUALITY_AUDIT','SOURCE_FIDELITY_CHAIN','SOURCE_MULTIFRAME_PAIR'):
        block=re.search(('<!-- '+name+'_BEGIN -->.*?<!-- '+name+'_END -->').encode('ascii'),raw,re.S);need(block is not None,'Historical report block missing')
        result['blocks'][name]=hashlib.sha256(block[0]).hexdigest()
    return result

def build(source,stamp):
    source,stamp=Path(source).resolve(),Path(stamp).resolve();payload,assets,source_audit=build_source(source)
    receipt=read(stamp/'receipt.json');contract=read(stamp/'contract.json')
    need(receipt['complete'] and receipt['nativeB']==0 and receipt['optimizer_updates']==0,'Target stamp completion/zero-update contract failed')
    need(not receipt['parent_asset_modified'] and not receipt['parent_retirement'] and not receipt['parent_stamp_compositing'],'Independent target stamp changed or composited parent')
    need({v['label'] for v in receipt['variants']}=={v['label'] for v in payload['source_variants']},'Target/source variants differ')
    locator_by_sha={};checks=[]
    def asset(path,relative,wh=None,dedup_locator=False):
        path=Path(path).resolve(strict=True);h=sha(path)
        if wh:
            with Image.open(path) as image:need(list(image.size)==wh,'Target image dimensions differ')
        if dedup_locator and h in locator_by_sha:return locator_by_sha[h]
        rel=(PREFIX/relative).as_posix();assets.append({'source':str(path),'relative':rel,'sha256':h,'bytes':path.stat().st_size})
        if dedup_locator:locator_by_sha[h]=rel
        return rel
    target_variants=[]
    for variant in receipt['variants']:
        label=variant['label'];need(variant['halo_rows']==0 and variant['source_RGB_sigma_opacity_exact'] and variant['internal_pairwise_distances_rigid'],'Target transfer changes core parameters or rigidity')
        need(variant['covariance_reconstruction_max_abs']<1e-12,'Target covariance reconstruction failed')
        base=stamp/label
        with np.load(source/label/'road.npz',allow_pickle=False) as src,np.load(base/'road.npz',allow_pickle=False) as dst:
            for key in ('source_world_row_ids','reference_srgb','gaussian_sigma_m','road_opacity'):
                need(np.array_equal(src[key],dst[key]),'Target core parameter differs: '+key)
        item={k:variant[k] for k in ('label','patch','core_rows','halo_rows','source_RGB_sigma_opacity_exact','internal_pairwise_distances_rigid','covariance_reconstruction_max_abs')}
        item['core_size_m']=[.8,.8] if variant['patch']=='patch_00' else [.6,.6]
        item['component']=asset(base/'road.npz',Path('target')/label/'road.npz')
        item['trace']=asset(base/'transfer_trace.npz',Path('target')/label/'transfer_trace.npz')
        item['provenance']=asset(base/'provenance.json',Path('target')/label/'provenance.json');item['views']=[]
        need({r['raw'] for r in variant['views']}=={86,110,134,179,198},'Target five-view set differs')
        for view in variant['views']:
            raw=view['raw'];folder=base/f'{raw:03d}';box=view['bbox_xyxy'];wh=[box[2]-box[0],box[3]-box[1]]
            with np.load(folder/'target_float.npz',allow_pickle=False) as f:
                alpha=f['alpha'];black=f['target_black'];grey=f['target_grey'];physical=f['physical_core'];score=f['score_mask'];eroded=binary_erosion(physical,iterations=3)
                need(int(physical.sum())==view['physical_core_pixels'] and int(score.sum())==view['opaque_core_pixels'],'Target physical/diagnostic pixel count differs')
                need(np.array_equal(score,eroded&(alpha>=.95)),'Target diagnostic domain differs')
                full_opaque=int((physical&(alpha>=.95)).sum());full_fraction=full_opaque/int(physical.sum())
                need(abs(full_fraction-view['alpha95_fraction'])<1e-10,'Full physical alpha coverage denominator differs')
                rgba_error=float(abs(grey-(black+.32*(1-alpha[:,:,None]))).max());need(rgba_error<1e-6,'Target grey is not actual RGBA')
                effective=black/np.maximum(alpha[:,:,None],1e-8);anomalies=int((np.ptp(effective[score],axis=1)>.15).sum())
                need(anomalies==view['high_chroma_pixels'] and view['reload_RGB_max_abs']==0,'Target chroma count or reload differs')
            row=dict(view);row.pop('output');row.update(crop_wh=wh,eroded_core_pixels=int(eroded.sum()),full_physical_alpha95_pixels=full_opaque,images={})
            for key in ('target_black','target_grey','alpha','metric_mask'):
                row['images'][key]=asset(folder/(key+'.png'),Path('target')/label/f'{raw:03d}'/(key+'.png'),wh)
            row['images']['full_grey']=asset(folder/'target_full_grey.png',Path('target')/label/f'{raw:03d}'/'target_full_grey.png',[1920,1280])
            row['images']['locator']=asset(folder/'parent_footprint_locator.png',Path('target')/label/f'{raw:03d}'/'parent_footprint_locator.png',[1920,1280],True)
            item['views'].append(row);checks.append({'variant':label,'raw':raw,'RGBA_max_abs':rgba_error,'physical_core_pixels':view['physical_core_pixels'],'full_physical_alpha95_pixels':full_opaque,'eroded_core_pixels':int(eroded.sum()),'diagnostic_pixels':view['opaque_core_pixels'],'high_chroma_pixels':anomalies})
        target_variants.append(item)
    payload.update(target_complete=True,target_variants=target_variants,target_contract=contract,target_origin=receipt['target_origin'],target_resources={k:receipt[k] for k in ('nativeF','nativeB','optimizer_updates','new_diffusion_calls')})
    audit_root=HERE.parent/'cpu_audit';aggregate=read(audit_root/'aggregate_cpu_audit.json')
    need(aggregate['complete'] and aggregate['all_required_engineering_checks_passed'],'Independent core/stamp CPU audit failed')
    need(aggregate['whole_component_all_domain_clean'] is False,'Boundary-color observation changed; review actual audit')
    payload['cpu_audit']=aggregate
    payload['evidence'] += [{'name':'单块目标实际原生渲染记录','path':asset(stamp/'receipt.json',Path('evidence/single_stamp_receipt.json'))},{'name':'独立组件零优化合同','path':asset(stamp/'contract.json',Path('evidence/single_stamp_contract.json'))}]
    for name,label in [('aggregate_cpu_audit.json','核心与边界分区CPU审计'),('source_core_cpu_audit.json','源组件独立审计'),('single_stamp_cpu_audit.json','目标组件独立审计')]:
        payload['evidence'].append({'name':label,'path':asset(audit_root/name,Path('evidence')/name)})
    audit={'passed':True,'source_views':source_audit['source_views'],'target_views':len(checks),'default_variant':'source02_original','default_raw':110,'default_source_raw':110,'no_2D_mask_on_colour_images':True,'parent_retirement':False,'parent_stamp_compositing':False,'source_checks':source_audit['checks'],'target_checks':checks}
    return payload,assets,audit

def integrate(index,payload,assets):
    original=index.read_bytes();before=history(original);text=original.decode('utf8');clean=re.sub(re.escape(BEGIN)+r'.*?'+re.escape(END),'',text,flags=re.S)
    literal=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('</','<\\/');block=BEGIN+'\n<script id="clean-core-transfer-data" type="application/json">'+literal+'</script>\n<script>\n'+(HERE/'core_viewer.js').read_text(encoding='utf8')+'\n</script>\n'+END
    position=clean.rfind('</html>');need(position>=0,'Closing HTML tag missing');output=(clean[:position]+block+'\n'+clean[position:]).encode('utf8');need(history(output)==before,'Historical raw blocks changed')
    backup=HERE/'backups'/('index.'+hashlib.sha256(original).hexdigest()[:16]+'.before_core_transfer.html');backup.parent.mkdir(parents=True,exist_ok=True)
    if not backup.exists():shutil.copy2(index,backup)
    for plan in assets:
        src=Path(plan['source']);need(sha(src)==plan['sha256'],'Artifact changed after validation');target=(index.parent/plan['relative']).resolve();need(target.is_relative_to((index.parent/PREFIX).resolve()),'New core asset escaped own directory')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target);need(sha(target)==plan['sha256'],'Copied core artifact hash differs')
    temporary=index.with_name(index.name+'.core-transfer.tmp');temporary.write_bytes(output);os.replace(temporary,index)
    return {'passed':True,'index':str(index),'backup':str(backup),'historical_evidence_raw_bytes_unchanged':True,'before':before,'after':history(index.read_bytes()),'before_html_sha256':hashlib.sha256(original).hexdigest(),'after_html_sha256':sha(index),'assets':len(assets),'default_variant':'source02_original','default_raw':110,'default_source_raw':110}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,default=HERE/'takeback/source_core_probe');parser.add_argument('--stamp',type=Path,default=HERE/'takeback/single_stamp_zero');parser.add_argument('--index',type=Path,default=INDEX);parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    payload,assets,audit=build(args.source,args.stamp);write(HERE/'core_payload.json',payload);write(HERE/'core_asset_plan.json',assets);write(HERE/'adapter_audit.json',audit)
    if args.apply:
        audit=integrate(args.index.resolve(),payload,assets);write(HERE/'integration_audit.json',audit)
    print(json.dumps({'passed':True,'applied':args.apply,'source_views':12,'target_views':20,'assets':len(assets)}))
