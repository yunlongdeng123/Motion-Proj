"""Validate the real unmasked 3D core probe; target stamp is integrated only when provided."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
PREFIX=Path('clean_core_transfer_r4')
LABELS={'source00_original':'源00 · 原资产颜色','source00_camera137':'源00 · 相机校正拟合色','source02_original':'源02 · 原资产颜色','source02_eight400':'源02 · 八帧400步颜色'}

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(value,message):
    if not value:raise ValueError(message)

def build_source(source):
    source=Path(source).resolve();receipt=read(source/'receipt.json');contract=read(source/'contract.json')
    need(receipt['complete'] and receipt['nativeB']==0 and receipt['optimizer_updates']==0 and not receipt['original_asset_modified'],'Source probe completion/zero-update contract failed')
    need(contract['source_geometry_unchanged'] and contract['optimizer_updates']==0 and contract['target_fusion'] is False,'Source geometry/fusion contract failed')
    need({v['label'] for v in receipt['variants']}==set(LABELS),'Expected four actual core variants')
    assets=[];audit=[]
    def asset(path,relative,wh=None):
        path=Path(path).resolve(strict=True)
        if wh:
            with Image.open(path) as im:need(list(im.size)==wh,'Native crop dimensions differ')
        relative=(PREFIX/relative).as_posix();assets.append({'source':str(path),'relative':relative,'sha256':sha(path),'bytes':path.stat().st_size});return relative
    variants=[]
    for variant in receipt['variants']:
        label=variant['label'];rows=1598 if variant['patch']=='patch_00' else 900
        need(variant['core_rows']==rows and variant['halo_rows_exported']==0,'Actual 3D subset includes halo or changes core count')
        folder=source/label
        with np.load(folder/'road.npz',allow_pickle=False) as component:
            need(len(component['source_world_row_ids'])==rows and len(component['reference_srgb'])==rows,'Exported component rows differ')
        item={k:variant[k] for k in ('label','patch','core_rows','halo_rows_exported')};item['display_label']=LABELS[label];item['views']=[]
        item['component']=asset(folder/'road.npz',Path('source')/label/'road.npz')
        for view in variant['views']:
            raw=view['raw'];f=folder/str(raw);box=view['bbox_xyxy'];wh=[box[2]-box[0],box[3]-box[1]]
            with np.load(f/'core_float.npz',allow_pickle=False) as data:
                alpha=data['alpha'];mask=data['score_mask'];physical=data['physical_core'];black=data['core_black'];grey=data['core_grey'];photo=data['photo_pinhole']
                need(list(alpha.shape)==wh[::-1] and int(mask.sum())==view['score_n'] and int(physical.sum())==view['core_n'],'Core/diagnostic pixel denominator differs')
                max_rgba=float(abs(grey-(black+(1-alpha[:,:,None])*.32)).max())
                need(max_rgba<1e-6,'Grey image is not the actual RGBA composite')
                effective=black/np.maximum(alpha[:,:,None],1e-8);threshold=float(np.quantile(np.ptp(photo[mask],axis=1),.999)+.10)
                anomalies=int((np.ptp(effective[mask],axis=1)>threshold).sum())
                need(abs(threshold-view['high_chroma_threshold'])<1e-6 and anomalies==view['high_chroma_pixels'],'Diagnostic chroma count differs')
            need(view['reload_RGB_max_abs']==0,'Actual component reload differs')
            if label.endswith('_original'):need(view['component_vs_parent_core_RGB_max_abs']==0 and anomalies==0,'Original core equivalence or observed high-alpha diagnostic differs')
            row={k:view[k] for k in ('raw','core_n','score_n','high_chroma_threshold','high_chroma_pixels','high_chroma_fraction','reload_RGB_max_abs','component_vs_parent_core_RGB_max_abs','bbox_xyxy')}
            row.update(crop_wh=wh,images={})
            for key in ('photo_pinhole','core_black','core_grey','alpha','metric_mask'):
                row['images'][key]=asset(f/(key+'.png'),Path('source')/label/str(raw)/(key+'.png'),wh)
            item['views'].append(row);audit.append({'variant':label,'raw':raw,'RGBA_composite_max_abs':max_rgba,'diagnostic_mask_n':view['score_n'],'high_chroma_pixels':anomalies,'reload_RGB_max_abs':0})
        variants.append(item)
    evidence=[{'name':'实际3D核心原生渲染记录','path':asset(source/'receipt.json',Path('evidence/source_core_probe_receipt.json'))},{'name':'原生RGBA与诊断域合同','path':asset(source/'contract.json',Path('evidence/source_core_probe_contract.json'))}]
    payload={'schema':'clean_core_transfer_report.v1','source_complete':True,'target_complete':False,'source_variants':variants,'source_contract':contract,'source_resources':{k:receipt[k] for k in ('nativeF','nativeB','optimizer_updates','new_diffusion_calls')},'evidence':evidence,'architecture':asset(HERE/'architecture_components.svg' if (HERE/'architecture_components.svg').exists() else HERE.parent/'architecture_components.svg',Path('architecture_components.svg'))}
    return payload,assets,{'passed':True,'source_views':len(audit),'all_displays_are_actual_native_RGBA_without_2D_mask':True,'original_core_subset_parent_RGB_max_abs':0,'checks':audit}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,default=HERE/'takeback/source_core_probe');args=parser.parse_args()
    payload,assets,audit=build_source(args.source);write(HERE/'source_payload.json',payload);write(HERE/'source_asset_plan.json',assets);write(HERE/'source_adapter_audit.json',audit)
    print(json.dumps({'passed':True,'variants':len(payload['source_variants']),'source_views':audit['source_views'],'assets':len(assets),'HTML_modified':False,'target_waiting':True}))
