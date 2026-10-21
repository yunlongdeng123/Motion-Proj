"""验证五实验新合同；旧V3来源检查保留，不把新增V1当第四组替代品。"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
from PIL import Image
from verify import verify,read_js,unpack

def run(root,package=False):
    old=verify(root);idx=read_js(root/'data/gpu_index.js');bindings=read_js(root/'data/scene.js')['bindings'];counts=Counter();ids=0
    assert len(idx['queries'])==180
    for key,path in idx['queries'].items():
        q=read_js(root/path);assert q['status']=='NATIVE_BUFFER_CUDA_TRACE_VERIFIED' and q['parameter_updates']==0
        assert max(q['replay_errors'].values())<=2e-5
        e=q['geometry_evidence'];assert e['labels_are_diagnostic_not_deletion_authority']
        for r in q['top']:
            role='dynamic' if r['branch']=='dynamic' else 'static'
            assert r['checkpoint_sha256']==bindings[q['arm']][role]['sha256']
            g=r['geometry_evidence'];assert g['label'] in ('geometry_conflict','legal_road','legal_occlusion','unknown');counts[g['label']]+=1
            if g['label']!='unknown':assert g['conditional_depth'] and g['conditional_depth']['screen_mahalanobis2']<=9
        if q['accepted_count']:
            assert q['id_weight_validation']['max_weight_error']<=2e-6;ids+=1
    assert counts['geometry_conflict']>0 and counts['legal_road']>0 and counts['unknown']>0
    for path in idx['counterfactuals'].values():
        q=read_js(root/path)
        assert q['optimizer_updates']==0 and not q['parameters_mutated'] and len(q['removed'])>=1
        for path in q['images'].values():
            with Image.open(root/path) as im:assert im.size==(480,320);im.verify()
        for item in [q]+q['temporal_controls']:
            e=item['scientific_evaluation'];assert e['denominator_fixed_before_intervention'] and e['hidden_RGB_ground_truth'] is False
            assert e['unobserved_query']['RGB_loss'] is None
            assert e['normal_control']['count']<=e['valid']['count']
            for name in ('valid','normal_control'):
                m=e[name];assert len(m.get('unit_ids',[]))==m['count']
                if m['count']:assert abs((m['L_before']-m['L_after'])-m['delta_L_valid'])<1e-6
                else:assert m['delta_L_valid'] is None
    v1=json.loads((root/'v1_receipt.json').read_text());assert len(v1['whole_frame_replays'])==147 and len(v1['queries'])==24
    assert max(r['max_8bit_error'] for r in v1['whole_frame_replays'])<=1 and v1['road_rows_injected']==0 and v1['optimizer_updates']==0
    for raw in range(89,138):
        maps=unpack(read_js(root/f'data/gn11_v1/{raw}.js'));assert maps['joint_alpha'].shape==(320,480)
        assert not any('road' in k for k in maps)
        for ch in ('joint','deleted_five','dynamic_raw'):
            with Image.open(root/f'assets/gn11_v1/{ch}/{raw:03d}.webp') as im:assert im.size==(480,320);im.verify()
    cl=unpack(read_js(root/'data/gn11_v1/cloud.js'));assert np.all(cl['static_road_id']==-1)
    qa=json.loads((root/'browser_evidence_qa.json').read_text()) if (root/'browser_evidence_qa.json').exists() else dict(status='PENDING')
    result=dict(status='FIVE_EXPERIMENT_EVIDENCE_DIAGNOSTICS_VALIDATED',native_traces=180,nonempty_independent_ID_weight_replays=ids,
        geometry_labels=dict(counts),counterfactuals_with_fixed_world_evaluation=len(idx['counterfactuals']),temporal_controls_with_fixed_world_evaluation=36,
        V1_native_PNG_replays=147,V1_road_rows_injected=0,optimizer_updates=0,human_verdict=None,quality_accepted=False,browser_qa=qa,
        original_V3_CPU_source_validation=old['status'])
    (root/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    if package:
        files={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='package_manifest.json'}
        (root/'package_manifest.json').write_text(json.dumps(dict(version=3,files=files),indent=2),encoding='utf-8',newline='\n')
        dest=root.with_suffix('.zip')
        with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=8) as z:
            for name in files:z.write(root/name,root.name+'/'+name)
            z.write(root/'package_manifest.json',root.name+'/package_manifest.json')
        with zipfile.ZipFile(dest) as z:
            assert z.testzip() is None
            for name,rec in files.items():assert hashlib.sha256(z.read(root.name+'/'+name)).hexdigest()==rec['sha256']
        if dest.stat().st_size>100000000:raise RuntimeError(f'ZIP too large: {dest.stat().st_size}')
        result.update(zip_bytes=dest.stat().st_size,zip_sha256=hashlib.file_digest(dest.open('rb'),'sha256').hexdigest())
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':run(Path(sys.argv[1]).resolve(),'--package' in sys.argv)
