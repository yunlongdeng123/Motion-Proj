"""Adapt the actual photo audit and zero-update chain receipts without inventing stages."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_report import HERE, PREFIX, DEFAULT_INDEX, apply, digest, existing_state, write_json


def build(photo_root, probe_root, solver_roots=(), camera_roots=(), static_roots=()):
    photo_root, probe_root = Path(photo_root).resolve(), Path(probe_root).resolve()
    audit = json.loads((photo_root / 'photo_audit.json').read_text(encoding='utf8'))
    decision = json.loads((photo_root / 'reference_decision.json').read_text(encoding='utf8'))
    plan = json.loads((photo_root / 'probe_plan.json').read_text(encoding='utf8'))
    probe = json.loads((probe_root / 'receipt.json').read_text(encoding='utf8'))
    contract = json.loads((probe_root / 'contract.json').read_text(encoding='utf8'))
    if not all(x.get('complete') is True for x in (audit, decision, plan, probe)):
        raise ValueError('Incomplete actual input')
    if (contract['render_wh'] != [1920, 1280] or not contract['geometry_opacity_scales_frozen']
            or probe['original_asset_modified'] is not False or probe['target_migration'] is not False
            or probe['optimizer_updates'] != 0):
        raise ValueError('Initial source-only zero-update contract changed')
    if {(r['patch'], r['raw']) for r in probe['rows']} != {(p, f) for p in ('patch_00','patch_02') for f in (86,110)}:
        raise ValueError('Expected actual initial four controls')
    assets=[]

    def asset(path, relative, wh=None):
        path=Path(path).resolve(strict=True)
        if wh:
            with Image.open(path) as image:
                if list(image.size) != wh:
                    raise ValueError('Native image dimensions differ: '+str(path))
        content=path.read_bytes();relative=(PREFIX / relative).as_posix()
        assets.append({'source':str(path),'relative':relative,'sha256':digest(content),'bytes':len(content)})
        return relative

    records={(r['candidate_id'],r['raw']):r for r in plan['records']}
    photo_rows=[]
    for row in audit['rows']:
        patch,raw=row['candidate_id'],row['raw']
        if patch not in {'patch_00','patch_02'}:
            raise ValueError('Unexpected photo source')
        if (patch,raw) not in records:
            continue  # supplemental photo-only frames have no independently audited halo projection
        item={key:row[key] for key in ('raw','photo_legal','core_size_m','core_metrics','same_physical_0p6m_metrics')}
        item['patch']=patch;item['images']={}
        for mode,key in [('core','core_crop'),('physical','physical0p6m_crop')]:
            crop=row[key];path=Path(crop['path'])
            if crop['pixels_unchanged'] is not True or digest(path.read_bytes()) != crop['sha256']:
                raise ValueError('Photo crop is not its verified original')
            item['images'][mode]={'path':asset(path,Path('photo')/patch/f'{raw:03d}'/(mode+'.png'),crop['native_crop_wh']),
                                  'wh':crop['native_crop_wh'],'bbox_xyxy':crop['bbox_xyxy']}
        # New halo derivatives are exact raw-photo crops; no interpolation or image synthesis.
        original=Path(row['local_original_RGB'])
        uv=records[(patch,raw)]['halo_native_uv']
        box=[max(0,math.floor(min(v[0] for v in uv))),max(0,math.floor(min(v[1] for v in uv))),
             min(1920,math.ceil(max(v[0] for v in uv))+1),min(1280,math.ceil(max(v[1] for v in uv))+1)]
        derived=HERE/'derived_native'/patch/f'{raw:03d}_halo.png';derived.parent.mkdir(parents=True,exist_ok=True)
        with Image.open(original) as image:
            if list(image.size) != [1920,1280]:
                raise ValueError('Photo is not native 1920')
            crop=image.convert('RGB').crop(box);crop.save(derived)
            with Image.open(derived) as saved:
                if crop.tobytes() != saved.convert('RGB').tobytes():
                    raise ValueError('Halo crop changed original pixels')
        wh=[box[2]-box[0],box[3]-box[1]]
        item['images']['halo']={'path':asset(derived,Path('photo')/patch/f'{raw:03d}'/'halo.png',wh),'wh':wh,'bbox_xyxy':box}
        photo_rows.append(item)
    probe_rows=[]
    for row in probe['rows']:
        patch,raw=row['patch'],row['raw'];box=row['bbox_xyxy'];wh=[box[2]-box[0],box[3]-box[1]]
        folder=probe_root/patch/f'{raw:03d}';item=dict(row);item.pop('photo');item.pop('output');item['crop_wh']=wh;item['images']={}
        for key,name in [('photo','observation.png'),('original','original.png'),('direct_sample','direct_sample_render.png'),
                         ('reloaded','reloaded_render.png'),('constant','constant_render.png'),('core_mask','core_mask.png'),
                         ('score_mask','score_mask.png')]:
            item['images'][key]=asset(folder/name,Path('chain_probe')/patch/f'{raw:03d}'/name,wh)
        item['score_mask_sha256']=digest((folder/'score_mask.png').read_bytes())
        if not all(row['non_RGB_fields_exact'].values()) or row['halo_RGB_exact'] is not True or row['alpha_exact'] is not True:
            raise ValueError('Geometry, halo, or alpha freeze failed')
        probe_rows.append(item)
    evidence=[]
    for path,label in [(photo_root/'reference_decision.json','实拍参考选择'),(photo_root/'photo_audit.json','原生实拍审计'),
                       (photo_root/'probe_plan.json','核心与核验边缘投影'),(probe_root/'receipt.json','零更新链路记录'),(probe_root/'contract.json','零更新链路合同')]:
        evidence.append({'name':label,'path':asset(path,Path('evidence')/(str(len(evidence))+'_'+Path(path).name))})
    gallery=photo_root/'same_physical_0p6m_native_1to1.png'
    with Image.open(gallery) as image:gallery_wh=list(image.size)
    solvers=[]
    for solver_root in solver_roots:
        solver_root=Path(solver_root).resolve()
        receipt=json.loads((solver_root/'receipt.json').read_text(encoding='utf8'))
        method=json.loads((solver_root/'contract.json').read_text(encoding='utf8'))
        if (receipt.get('complete') is not True or receipt['nativeB'] != 0 or receipt['gradient_optimizer_updates'] != 0
                or receipt['original_asset_modified'] is not False or receipt['target_migration'] is not False
                or method['geometry_scale_opacity_frozen'] is not True or method['halo_RGB_frozen'] is not True):
            raise ValueError('Formal fixed-colour source contract failed')
        for source in receipt['rows']:
            patch,reference=source['patch'],source['reference']
            if source['source_RGB_noncolor_exact'] is not True or source['halo_RGB_exact'] is not True or source['alpha_exact'] is not True:
                raise ValueError('Formal fixed-colour freeze failed')
            case={key:source[key] for key in ('patch','reference','core_n','score_n','operator_linear_check_max_abs','source_RGB_noncolor_exact','halo_RGB_exact','alpha_exact','solver')}
            case['contract']=method;case['evaluations']=[]
            case['resources']={key:receipt[key] for key in ('nativeF','nativeB','RGB_linear_solves','gradient_optimizer_updates','new_diffusion_calls')}
            case['receipt']=asset(solver_root/'receipt.json',Path('solver')/solver_root.name/'receipt.json')
            case['contract_path']=asset(solver_root/'contract.json',Path('solver')/solver_root.name/'contract.json')
            case['source_packet']=asset(solver_root/patch/'source_fixed_RGB.npz',Path('solver')/solver_root.name/'source_fixed_RGB.npz')
            for ev in source['evaluations']:
                raw=ev['raw'];box=ev['bbox_xyxy'];wh=[box[2]-box[0],box[3]-box[1]]
                item={key:ev[key] for key in ('raw','role','score_n','metrics','bbox_xyxy')};item['crop_wh']=wh;item['images']={}
                for key,name in [('photo','observation.png'),('original','original.png'),('direct_reference','direct_reference.png'),
                                 ('fixed_color_solve','fixed_color_solve.png'),('core_mask','core_mask.png'),('score_mask','score_mask.png')]:
                    item['images'][key]=asset(solver_root/patch/f'{raw:03d}'/name,Path('solver')/solver_root.name/f'{raw:03d}'/name,wh)
                item['score_mask_sha256']=digest((solver_root/patch/f'{raw:03d}'/'score_mask.png').read_bytes())
                case['evaluations'].append(item)
            solvers.append(case)
    camera_controls=[]
    for camera_root in camera_roots:
        camera_root=Path(camera_root).resolve()
        receipt=json.loads((camera_root/'receipt.json').read_text(encoding='utf8'))
        method=json.loads((camera_root/'contract.json').read_text(encoding='utf8'))
        if (receipt.get('complete') is not True or receipt['nativeB'] != 0 or receipt['gradient_optimizer_updates'] != 0
                or receipt['original_asset_modified'] is not False or receipt['target_migration'] is not False
                or receipt['source_non_RGB_fields_exact'] is not True or receipt['halo_RGB_exact'] is not True
                or method['geometry_scale_opacity_frozen'] is not True or method['halo_RGB_frozen'] is not True
                or method['original_images_unchanged'] is not True):
            raise ValueError('Camera-corrected fixed-source contract failed')
        patch,ref=receipt['patch'],receipt['reference'];stage=method['stage']
        case={'patch':patch,'reference':ref,'contract':method,'evaluations':[],
              'resources':{k:receipt[k] for k in ('nativeF','nativeB','RGB_linear_solves','gradient_optimizer_updates','new_diffusion_calls')},
              'receipt':asset(camera_root/'receipt.json',Path('camera')/stage/'receipt.json'),
              'contract_path':asset(camera_root/'contract.json',Path('camera')/stage/'contract.json'),
              'source_packet':asset(camera_root/'source_camera_corrected_RGB.npz',Path('camera')/stage/'source_camera_corrected_RGB.npz')}
        if (camera_root/'recovery_trace.json').exists():
            case['recovery_trace']=json.loads((camera_root/'recovery_trace.json').read_text(encoding='utf8'))
            case['recovery_evidence']=asset(camera_root/'recovery_trace.json',Path('camera')/stage/'recovery_trace.json')
        for ev in receipt['evaluations']:
            raw=ev['raw'];box=ev['bbox_xyxy'];wh=[box[2]-box[0],box[3]-box[1]];folder=camera_root/f'{raw:03d}'
            item={k:ev[k] for k in ('raw','role','metrics','raw_camera_metrics','Brown_inverse_roundtrip_max_px','bbox_xyxy')};item.update(crop_wh=wh,images={})
            for domain in ('raw','pinhole'):
                item['images'][domain]={}
                for key in ('observation','original','uncorrected_solve','camera_corrected_solve'):
                    filename=key+'_'+domain+'.png'
                    item['images'][domain][key]=asset(folder/filename,Path('camera')/stage/f'{raw:03d}'/filename,wh)
                metrics=ev['raw_camera_metrics'] if domain=='raw' else ev['metrics']
                if len({m['score_n'] for m in metrics.values()}) != 1:
                    raise ValueError('Camera comparison changes evaluation mask between source stages')
            with np.load(folder/'camera_float.npz',allow_pickle=False) as floats:
                masks={'raw':floats['raw_mask'],'pinhole':floats['score_mask']}
                for domain,mask in masks.items():
                    dest=HERE/'derived_native'/patch/f'{raw:03d}_{domain}_score.png';dest.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(np.uint8(mask)*255).save(dest)
                    item['images'][domain]['score_mask']=asset(dest,Path('camera')/stage/f'{raw:03d}'/(domain+'_score_mask.png'),wh)
            photo=next(r for r in audit['rows'] if r['candidate_id']==patch and r['raw']==raw)
            with Image.open(photo['local_original_RGB']) as original,Image.open(folder/'observation_raw.png') as displayed:
                if original.convert('RGB').crop(box).tobytes() != displayed.convert('RGB').tobytes():
                    raise ValueError('Raw camera observation PNG changed original photo pixels')
            item['raw_observation_pixels_unchanged']=True
            case['evaluations'].append(item)
        camera_controls.append(case)
    static_controls=[]
    for static_root in static_roots:
        static_root=Path(static_root).resolve()
        receipt=json.loads((static_root/'receipt.json').read_text(encoding='utf8'))
        method=json.loads((static_root/'contract.json').read_text(encoding='utf8'))
        if receipt.get('complete') is not True or receipt['optimizer_updates'] != 0 or receipt['target_migration'] is not False or receipt['original_asset_modified'] is not False:
            raise ValueError('Static diagnostic contract failed')
        static_controls.append({'stage':static_root.name,'contract':method,'rows':receipt['rows'],
                                'geometry':receipt.get('geometry'),'source_rows':receipt.get('source_rows',receipt.get('geometry',{}).get('source_rows')),
                                'resources':{k:receipt[k] for k in ('nativeF','nativeB','optimizer_updates','new_diffusion_calls')},
                                'receipt':asset(static_root/'receipt.json',Path('static')/static_root.name/'receipt.json'),
                                'contract_path':asset(static_root/'contract.json',Path('static')/static_root.name/'contract.json')})
    payload={'schema':'source_fidelity_chain_report.v2','run_id':'road_source_fidelity_20261122_r2','complete':True,
             'render_wh':[1920,1280],'architecture':asset(HERE/'architecture_components.svg',Path('architecture_components.svg')),
             'photo_review':{'decision':decision,'rows':photo_rows,'same_physical_gallery':asset(gallery,Path('photo')/'same_physical_0p6m_native_1to1.png',gallery_wh),'gallery_wh':gallery_wh},
             'chain_probe':{'contract':contract,'rows':probe_rows,'resources':{k:probe[k] for k in ('nativeF','nativeB','optimizer_updates','new_diffusion_calls')}},
             'solvers':sorted(solvers,key=lambda s:(s['patch']!='patch_02',s['reference'])),
             'camera_controls':sorted(camera_controls,key=lambda s:(s['patch']!='patch_02',s['reference'])),
             'static_controls':static_controls,'evidence':evidence}
    return payload,assets


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--photo-audit',type=Path,default=HERE.parent/'photo_audit')
    parser.add_argument('--chain-probe',type=Path,default=HERE/'evidence_probe_initial')
    parser.add_argument('--index',type=Path,default=DEFAULT_INDEX)
    parser.add_argument('--solver',type=Path,action='append',default=[])
    parser.add_argument('--camera',type=Path,action='append',default=[])
    parser.add_argument('--static-control',type=Path,action='append',default=[])
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    payload,assets=build(args.photo_audit,args.chain_probe,args.solver,args.camera,args.static_control)
    write_json(HERE/'actual_payload.json',payload);write_json(HERE/'actual_asset_plan.json',assets)
    audit={'validated':True,'applied':False,'photo_cases':len(payload['photo_review']['rows']),'probe_cases':len(payload['chain_probe']['rows']),'formal_solvers':len(payload['solvers']),'assets':len(assets),'existing':existing_state(args.index.read_text(encoding='utf8'))}
    if args.apply:
        audit=apply(args.index.resolve(),payload,assets)
        write_json(HERE/'integration_audit.json',audit)
    print(json.dumps(audit))
