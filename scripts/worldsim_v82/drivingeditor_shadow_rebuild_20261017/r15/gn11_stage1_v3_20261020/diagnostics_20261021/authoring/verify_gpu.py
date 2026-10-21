import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image
from verify import verify,read_js

def run(root,package=False):
    result=verify(root);receipt=json.loads((root/'gpu_receipt.json').read_text());views=json.loads((root/'view_receipt.json').read_text())
    assert receipt['status']=='NATIVE_CUDA_TRACE_AND_FULL_RERASTERIZATION_COMPLETE'
    assert receipt['query_count']==116 and receipt['counterfactual_count']==12 and receipt['temporal_control_render_count']==36
    index=read_js(root/'data/gpu_index.js');id_count=0;bindings=read_js(root/'data/scene.js')['bindings']
    for key,path in index['queries'].items():
        q=read_js(root/path);assert q['status']=='NATIVE_BUFFER_CUDA_TRACE_VERIFIED'
        assert q['gpu_used'] and q['no_optimizer'] and q['parameter_updates']==0
        assert max(q['replay_errors'].values())<=2e-5
        if q['saved_final49_replay_errors']:assert max(q['saved_final49_replay_errors'].values())<=3e-5
        if q['accepted_count']:
            assert q['id_weight_validation']['status']=='NATIVE_ONE_HOT_ID_WEIGHT_REPLAY_PASSED'
            assert q['id_weight_validation']['max_weight_error']<=2e-6;id_count+=1
        for r in q['top']:
            component='dynamic' if r['branch']=='dynamic' else 'static'
            assert r['checkpoint_sha256']==bindings[q['arm']][component]['sha256']
            assert r['id'].startswith(q['arm']+':'+r['checkpoint_sha256'][:12]+':')
            assert all(np.isfinite(r[k]).all() for k in ('weight','color_contribution','world_m','sigma_m','cov2'))
    for key,path in index['counterfactuals'].items():
        q=read_js(root/path);assert q['status']=='FULL_NATIVE_RERASTERIZATION_COMPLETE'
        assert 1<=len(q['removed'])<=20 and q['optimizer_updates']==0 and not q['parameters_mutated'] and q['environment_fixed']
        assert q['human_verdict'] is None and not q['quality_accepted'] and len(q['temporal_controls'])==3
        assert len(q['float_asset']['sha256'])==64
        for file in q['images'].values():
            with Image.open(root/file) as im:assert im.size==(480,320);im.verify()
    assert len(views['views'])==25 and len(views['negative_contract_tests'])==3
    for key,path in index['views'].items():
        q=read_js(root/path);assert q['status']=='FULL_NATIVE_VIEW_RENDER_COMPLETE'
        assert q['optimizer_updates']==0 and not q['parameters_mutated'] and not q['novel_view_has_ground_truth']
        assert q['gaussian_count']>=14034
        if q['full_frame_saved_replay_max_error'] is not None:assert q['full_frame_saved_replay_max_error']<=3e-5
        with Image.open(root/q['image']) as im:assert im.size==(480,320);im.verify()
    old=json.loads((root/'validation.json').read_text()) if (root/'validation.json').exists() else {}
    qa_path=root/'browser_gpu_qa.json'
    browser_qa=json.loads(qa_path.read_text()) if qa_path.exists() else old.get('gpu_browser_qa',old.get('browser_qa','PENDING_GPU_UI_QA'))
    result.update(status='NATIVE_CUDA_DIAGNOSTIC_SYSTEM_VALIDATED',gpu_used=True,native_cuda_topk='116_NATIVE_BUFFER_TRACES_REPLAYED',full_counterfactual_rerasterization='12_SELECTED_PLUS36_TEMPORAL_CONTROLS_COMPLETE',
        full_model_views=25,nonempty_traces_with_independent_ID_weight_replay=id_count,additional_optimization_steps=0,
        browser_qa=browser_qa,original_CPU_export_provenance_preserved=True)
    (root/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if package:
        files={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='package_manifest.json'}
        (root/'package_manifest.json').write_text(json.dumps(dict(version=2,files=files),indent=2),encoding='utf-8')
        dest=root.with_suffix('.zip')
        with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=8) as z:
            for name in files:z.write(root/name,root.name+'/'+name)
            z.write(root/'package_manifest.json',root.name+'/package_manifest.json')
        with zipfile.ZipFile(dest) as z:
            assert z.testzip() is None
            for name,info in files.items():assert hashlib.sha256(z.read(root.name+'/'+name)).hexdigest()==info['sha256']
        assert dest.stat().st_size<=100000000
        result['zip_bytes']=dest.stat().st_size;result['zip_sha256']=hashlib.file_digest(dest.open('rb'),'sha256').hexdigest()
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--package',action='store_true');a=p.parse_args();run(a.root.resolve(),a.package)
