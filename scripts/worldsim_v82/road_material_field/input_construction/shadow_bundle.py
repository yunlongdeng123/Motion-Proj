"""Lossless CPU export/load of probability-conditioned inputs and write roles."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image
from .shadow_probability import rgb_sha256
from .shadow_composite import compose_shadow_prediction


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_shadow_bundle(bundle,folder):
    folder=Path(folder)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError('use a new directory; existing experiment inputs are immutable')
    folder.mkdir(parents=True,exist_ok=True)
    receipt=bundle['receipt']
    independent=receipt.get('independent_regions',{}).get('mode')=='independent'
    if 'object_shadow' not in receipt and not independent:raise ValueError('this export requires explicit object-shadow probability or independent region mode')
    if independent:
        from .independent_regions import validate_independent_regions
        validate_independent_regions(bundle['masks'],bundle['observation_rgb'].shape[:2])
    observed=bundle['observation_rgb']
    if rgb_sha256(observed)!=receipt['observation_sha256']:raise ValueError('actual observation identity differs')
    paths={}
    for key in ['I_geo','I_mat','geometry_observation','observation_rgb']:
        path=folder/(key+'.png');Image.fromarray(bundle[key]).save(path);paths[key]=path.name
    roles=folder/'roles.npz';np.savez_compressed(roles,**bundle['masks']);paths['roles']=roles.name
    confidence=folder/'shadow_confidence.npz';np.savez_compressed(confidence,**bundle['confidence']);paths['confidence']=confidence.name
    matrix=folder/'native_from_input.npy';np.save(matrix,bundle['native_from_input']);paths['native_from_input']=matrix.name
    record=dict(schema='object_shadow_input_bundle.v1',paths=paths,
                sha256={key:_sha(folder/value) for key,value in paths.items()},receipt=receipt,
                probabilities_saved_without_thresholding=True,GT_paths_present=False,
                raw_decoder_must_be_saved_separately=True,
                compositor='input_construction.shadow_composite.compose_shadow_prediction')
    if independent:
        from .independent_regions import BUNDLE_SCHEMA
        record.update(schema=BUNDLE_SCHEMA,region_mode='independent',completion_scope='authorized_output_only')
    path=folder/'bundle.json';path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    return str(path)


def load_shadow_bundle(manifest):
    manifest=Path(manifest);record=json.loads(manifest.read_text(encoding='utf-8'));folder=manifest.parent
    from .independent_regions import BUNDLE_SCHEMA
    independent=record['schema']==BUNDLE_SCHEMA
    if record['schema']!='object_shadow_input_bundle.v1' and not independent:raise ValueError('wrong probability/independent bundle schema')
    if independent and record.get('region_mode')!='independent':raise ValueError('independent bundle must explicitly declare its region mode')
    for key,path in record['paths'].items():
        if _sha(folder/path)!=record['sha256'][key]:raise ValueError(f'input bundle was changed: {key}')
    result={key:np.asarray(Image.open(folder/record['paths'][key]).convert('RGB')).copy()
            for key in ['I_geo','I_mat','geometry_observation','observation_rgb']}
    with np.load(folder/record['paths']['roles'],allow_pickle=False) as z:result['masks']={key:z[key].copy() for key in z.files}
    with np.load(folder/record['paths']['confidence'],allow_pickle=False) as z:result['confidence']={key:z[key].copy() for key in z.files}
    result['receipt']=record['receipt'];m=result['masks'];f=m['target_unknown'];w=m['W_edit'];k=m['K']
    if any(a.dtype!=bool for a in m.values()):raise ValueError('role maps must remain Boolean; probability has its own file')
    if (w&~f).any() or (w&k).any():raise ValueError('invalid write authority')
    neutral=np.asarray(record['receipt'].get('independent_regions',{}).get('neutral_rgb',[127,127,127]),np.uint8) if independent else 127
    if not np.all(result['I_geo'][f]==neutral):raise ValueError('model RGB contains unknown pixels')
    if independent:
        from .independent_regions import validate_independent_regions
        validate_independent_regions(m,result['observation_rgb'].shape[:2])
        if not np.all(result['I_mat'][m['M_material']]==neutral):raise ValueError('forbidden material RGB was not sanitized')
    if rgb_sha256(result['observation_rgb'])!=record['receipt']['observation_sha256']:raise ValueError('wrong actual observation copy source')
    if ('object_shadow' in record['receipt'] or not independent) and not {'shadow_probability','shadow_uncertainty','shadow_valid'}<=result['confidence'].keys():raise ValueError('soft shadow evidence missing')
    if independent:
        result['native_from_input']=np.load(folder/record['paths']['native_from_input'],allow_pickle=False)
        result['region_mode']='independent'
    return result


def compose_saved_prediction(bundle_manifest,raw_prediction_path,out_dir,*,diagnostic=False):
    bundle=load_shadow_bundle(bundle_manifest)
    raw=np.asarray(Image.open(raw_prediction_path).convert('RGB')).copy()
    composed,receipt=compose_shadow_prediction(bundle['observation_rgb'],raw,bundle['masks'],
        expected_observation_sha256=bundle['receipt']['observation_sha256'],allow_diagnostic_pending=diagnostic,
        region_mode=bundle.get('region_mode','strict'))
    out=Path(out_dir)
    if out.exists() and any(out.iterdir()):raise FileExistsError('preserve old raw and final outputs; use a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    Image.fromarray(raw).save(out/'raw_prediction.png')
    name='unfinished_diagnostic.png' if not receipt['final_eligible'] else 'final.png'
    Image.fromarray(composed).save(out/name)
    record=dict(receipt,bundle_manifest=str(Path(bundle_manifest).resolve()),raw_prediction_source=str(Path(raw_prediction_path).resolve()),output=name)
    (out/'composition.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    return record
