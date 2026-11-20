from pathlib import Path
import json,hashlib
Q=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1');G=Q/'gpu_20261119';B=G/'quality_bench_20261120_r1'

def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def row(r):
    out={k:v for k,v in r.items() if k not in ('scales','edges')}
    out['scales']=[]
    for s in r['scales']:
        a=s['tile_aggregate']
        out['scales'].append(dict(sigma_pixels=s['sigma_pixels'],core_pixels=s['core_pixels'],boundary_pixels=s['boundary_pixels'],
            core=s['core'],boundary=s['boundary'],flat=s['flat'],tile_macro=a['macro'],
            tiles=dict(total_fixed=a['total_fixed_tiles'],nonempty_core=a['nonempty_core_tiles'],empty_score=a['empty_score_tiles'],
                scored_no_core=a['scored_but_no_core_tiles'],cosine_NA=a['cosine_undefined_tiles']),
            worst_tiles=a['worst_tiles']))
    out['edges']=r['edges']
    return out

def main():
    dest=B/'delivery';dest.mkdir(exist_ok=True)
    sources={}
    result=dict(schema='road_quality_bench_compact_delivery.v2',primary_metrics_have_no_overall_quality_score=True,
        status='quality not accepted; individual evidence and untested marking shape remain explicit',resolutions={})
    for tag,path in [('coarse480',B/'lowres_r1/summary.json'),('native1920',B/'native_detail_r1/summary.json')]:
        d=read(path);sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        result['resolutions'][tag]=dict(rows=[row(r)if 'scales'in r else r for r in d['rows']],
            seconds=d['seconds'],optimizer_updates=d['optimizer_updates'],native_calls=d['native_calls'],
            full_source_summary=str(path),source_sha256=sources[str(path)])
    extras={
        'gradient':B/'gradient_probe/run_r1/summary.json',
        'native_handoff':B/'secondary_audit/native_r1/results.json',
        'coarse_handoff':B/'secondary_audit/lowres_r1/results.json',
        'frequency_diagnostic':B/'frequency_diagnostic_r1/results.json',
        'native_GPU_receipt':B/'highres/native_completed.json',
        'native_package_qualification':B/'highres/package_qualification.json',
        'native_stream_qualification':B/'highres/stream_480_qualification.json',
        'synthetic_controls':B/'metrics/synthetic_negative_controls.json',
        'teacher_control_qualification':B/'root_code/teacher_control_qualification.json',
        'protocol':B/'protocol.json','metric_freeze':B/'metric_freeze.json',
        'previews':B/'highres/previews/assets_manifest.json','ROIs':B/'highres/previews/roi_manifest.json'}
    for name,path in extras.items():
        if not path.is_file():raise FileNotFoundError(path)
        result[name]=read(path);sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    result['source_summary_sha256']=sources
    write(dest/'report_data.json',result)
    write(dest/'inventory.json',dict(files=sources,bytes=(dest/'report_data.json').stat().st_size))
    print(json.dumps(dict(output=str(dest/'report_data.json'),bytes=(dest/'report_data.json').stat().st_size)))

if __name__=='__main__':main()
