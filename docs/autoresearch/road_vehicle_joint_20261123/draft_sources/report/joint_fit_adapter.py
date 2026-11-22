"""Read a completed, locally taken-back joint-fit report manifest. No rendering."""
from pathlib import Path
import hashlib,json

def load_joint_fit_manifest(manifest_path, *, shared_field_sha256, road_parent_sha256, actor_parent_sha256):
    """Return verified data, or None while no completed manifest exists.

    schema: R9.joint_native_RGB_fit.report.v1
    Required: complete, shared_field_sha256, road_parent_sha256,
    actor_parent_sha256, parameter_contract, receipt, images.
    receipt={path,sha256}; images={raw:{stage:{channel:{path,sha256}}}}.
    Paths are local absolute paths or relative to the manifest directory.
    """
    path=Path(manifest_path)
    if not path.is_file():return None
    data=json.loads(path.read_text(encoding='utf-8-sig'))
    if not data.get('complete'):return None
    assert data['schema']=='R9.joint_native_RGB_fit.report.v1'
    assert data['shared_field_sha256']==shared_field_sha256
    assert data['road_parent_sha256']==road_parent_sha256
    assert data['actor_parent_sha256']==actor_parent_sha256
    contract=data['parameter_contract']
    for key in ['centres_frozen','scale_opacity_frozen','rigid_trajectory_frozen','other_vehicles_protected']:
        assert contract[key] is True,key
    def verify(record):
        target=Path(record['path'])
        if not target.is_absolute():target=path.parent/target
        assert target.is_file(),str(target)
        assert hashlib.sha256(target.read_bytes()).hexdigest()==record['sha256'],str(target)
        return target
    receipt=json.loads(verify(data['receipt']).read_text(encoding='utf-8-sig'))
    assert receipt['complete'] and receipt['optimizer_updates']>0
    assert receipt['shared_field_sha256']==shared_field_sha256
    for raw,stages in data['images'].items():
        assert int(raw) in [86,110,134,179,198]
        for channels in stages.values():
            for record in channels.values():verify(record)
    return data
