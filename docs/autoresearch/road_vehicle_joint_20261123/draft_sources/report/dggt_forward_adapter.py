"""Read actual DGGT network-control evidence, separate from native joint fitting."""
from pathlib import Path
import hashlib,json

def load_dggt_forward_manifest(manifest_path):
    """Return verified completed evidence, or None before it exists.

    schema: R8.official_DGGT_noDiff_forward.report.v1
    Required: complete, official_network_control=true, checkpoint,
    receipt, images={raw:{channel:{path,sha256}}}, coordinate_domain.
    receipt JSON: network_forward_calls>0, backward_calls=0,
    optimizer_updates=0, diffusion_calls=0; code/checkpoint provenance.
    Prediction and render camera/gauge details remain explicit in the receipt.
    Paths refer to actual local takeback; no native fitter images are accepted.
    """
    path=Path(manifest_path)
    if not path.is_file():return None
    data=json.loads(path.read_text(encoding='utf-8-sig'))
    if not data.get('complete'):return None
    assert data['schema']=='R8.official_DGGT_noDiff_forward.report.v1'
    assert data['official_network_control'] is True
    assert data['coordinate_domain'] and data['checkpoint']['sha256']
    def verify(record):
        target=Path(record['path'])
        if not target.is_absolute():target=path.parent/target
        assert target.is_file(),str(target)
        assert hashlib.sha256(target.read_bytes()).hexdigest()==record['sha256'],str(target)
        return target
    receipt=json.loads(verify(data['receipt']).read_text(encoding='utf-8-sig'))
    assert receipt['network_forward_calls']>0
    assert receipt['backward_calls']==receipt['optimizer_updates']==receipt['diffusion_calls']==0
    assert receipt['checkpoint_sha256']==data['checkpoint']['sha256']
    assert receipt['experiment_kind']=='official_DGGT_network_control_noDiff'
    for channels in data['images'].values():
        for record in channels.values():verify(record)
    return data
