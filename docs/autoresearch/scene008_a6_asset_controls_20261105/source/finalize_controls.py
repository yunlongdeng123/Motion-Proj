"""Integrity/independent-asset QA and finite Git/local delivery preparation."""
import hashlib,json,shutil,sys,tarfile
from pathlib import Path
from html.parser import HTMLParser
import numpy as np
from PIL import Image
from sh_asset import SHAsset

RUN=Path(__file__).resolve().parent.parent;DEST=RUN/'delivery'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def write(path,x):path.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def main():
    page=DEST/'index.html';html=page.read_text()
    if "let filename=String(r).padStart(3,'0');" not in html:
        html=html.replace("let a=document.getElementById('arm').value;", "let a=document.getElementById('arm').value;let filename=String(r).padStart(3,'0');")
        html=html.replace("'frames/'+r+'_objects.jpg'", "'frames/'+filename+'_objects.jpg'")
        html=html.replace("'frames/'+r+'_'+a+'_street.jpg'", "'frames/'+filename+'_'+a+'_street.jpg'")
        page.write_text(html,encoding='utf-8')
    shutil.copyfile(RUN/'source/report_controls.py',DEST/'source/report_controls.py')
    # Preserve the byte-exact executed versions before publishing LF-normalized source.
    archive_path=DEST/'evidence/executed_source_versions.tar.gz'
    with tarfile.open(archive_path,'w:gz') as archive:
        for path in sorted((RUN/'source').iterdir()):
            if path.is_file():archive.add(path,arcname='source/'+path.name)
        archive.add(RUN.parent/'scene008_a6_complete_asset_20261105/source/full_pvd.py',arcname='dependency/full_pvd.py')
    shutil.copyfile(RUN/'source/finalize_controls.py',DEST/'source/finalize_controls.py')
    for name in ('m1_smoke_nondeterministic.log','m2_smoke_binding_failure.log','report_path_failure.log'):
        shutil.copyfile(RUN/name,DEST/'evidence'/name)
    assets={arm:SHAsset.from_geometry(DEST/'assets'/f'{arm}_complete_a6.npz') for arm in ('m1','m2','m3')}
    for asset in assets.values():assert asset.assert_geometry_unchanged()
    a,b=assets['m2'],assets['m3'];assert len(a.xyz)==len(b.xyz)==18432
    for name in ('normal','sigma','thickness','opacity'):
        assert np.array_equal(getattr(a,name).numpy(),getattr(b,name).numpy()),name
    assert np.count_nonzero(a.coefficients[:,1:].numpy())==np.count_nonzero(b.coefficients[:,1:].numpy())==0
    delta=b.xyz.numpy()-a.xyz.numpy();normal=b.normal.numpy();signed=(delta*normal).sum(1)
    tangent=delta-signed[:,None]*normal;assert np.abs(tangent).max()<1e-6
    m3=b.metadata;assert m3['geometry_hash']==b.geometry_hash and m3['provenance_hash']==b.provenance_hash
    provenance=m3['source_provenance'];assert provenance['builder_sha256']==sha(RUN/'source/measured_correction.py')
    frames=provenance['frames'];assert frames['held_frames_read']==[] and len(frames['file_sha256'])==40
    for name,value in frames['file_sha256'].items():assert sha(Path(frames['directory'])/name)==value
    with np.load(RUN/'m2/inputs.npz') as x,np.load(RUN/'m3/inputs.npz') as y:
        for name in ('prior','inferred','target'):assert np.array_equal(x[name],y[name]),name
    assert not any(name in sys.modules for name in ('native_binding','scene_runtime','gaussian_renderer','editor_common'))
    class Parser(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('src','href') and v)
    parser=Parser();parser.feed((DEST/'index.html').read_text())
    for link in parser.links:
        if not link.startswith(('http:','https:','#','data:')):assert (DEST/link).is_file(),link
    photos=list((DEST/'frames').glob('*.jpg'))+list((DEST/'tint').glob('*.png'))
    for path in photos:
        with Image.open(path) as im:im.verify()
    assert len(list((DEST/'frames').glob('*_objects.jpg')))==49
    for arm in assets:assert len(list((DEST/'frames').glob(f'*_{arm}_street.jpg')))==49
    inventory=[]
    for pattern in ('m*/evaluation/frames/*_replay.npz','m*/design*.npz','m*/inputs.npz','m*/checkpoint*.pt','tint_diagnostic/*_exact.npz'):
        for path in sorted(RUN.glob(pattern)):inventory.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
    write(DEST/'evidence/remote_replay_inventory.json',dict(retained_remote=True,autodl='KEEP_ON',files=inventory))
    ledger=dict(formal_steps_per_arm=2000,formal_appearance_steps_total=6000,
      formal_fit_seconds={arm:read(RUN/arm/'fit_receipt.json')['seconds'] for arm in assets},
      all49_evaluation_seconds={arm:read(RUN/arm/'evaluation/metrics.json')['seconds'] for arm in assets},
      extra_smoke_states='32 steps plus two next-step replay calls per attempt; five attempts total including two failed',
      extra_smoke_and_restore_compute_steps=170,extra_M0_optimizer_QC_steps=2000,
      failed_M1='Nondeterministic generic CSR GPU reduction; fixed-tree double accumulator before formal',
      failed_M2='Old runtime import shadowing caught before formal; source binding assertion added',
      geometry_method='Single CPU bounded measured correction, own contract/audit; no numerical threshold search',
      long_run_started=False,autodl='KEEP_ON',shutdown_performed=False,all_original_runs_preserved=True)
    write(DEST/'evidence/execution_ledger.json',ledger)
    qa=dict(status='PASS',standalone_asset_no_native_module_dependency=True,assets={k:dict(rows=len(v.xyz),geometry_hash=v.geometry_hash) for k,v in assets.items()},
      M2_M3_same_prior_target_flags=True,M2_M3_fixed_noncenter_geometry=True,M3_tangent_roundoff_max=float(np.abs(tangent).max()),
      M3_metadata_and_40_BUILD_frame_hashes_verified=True,images_decoded=len(photos),relative_html_links_checked=len(parser.links),
      evaluated_frames_per_arm=49,scientific_quality_accepted=False,
      exact_executed_source_archive_sha256=sha(archive_path),published_source='LF-normalized; byte-exact executed versions in archive')
    write(DEST/'evidence/delivery_QA.json',qa)
    # Normalize generated published text, without changing serialized model/data bytes.
    for path in DEST.rglob('*'):
        if path.is_file() and path.suffix in ('.py','.cu','.json','.csv','.svg','.html','.md','.txt','.log'):
            raw=path.read_bytes()
            if b'\r\n' in raw:path.write_bytes(raw.replace(b'\r\n',b'\n'))
    (DEST/'assets/README.txt').write_text('One complete canonical a6 asset per NPZ; metres, +x forward,+y left,+z up. Load with source/sh_asset.py SHAsset.from_geometry(path); source/full_pvd.py supplies schema/hash helpers only. M1 uses canonical SH3; M2/M3 higher SH are exactlyzero. Apply actor pose once and rotate SH through rows(matrix,camera,pipe,channels=16). No native a6 renderer dependency. colour_inferred and source/child/correction tags distinguish evidence from inferred geometry/material. These are experimental full proposals, not independently verified hidden truth.\n')
    with tarfile.open(RUN/'delivery.tar.gz','w:gz') as archive:archive.add(DEST,arcname='delivery')
    (RUN/'delivery.tar.gz.sha256').write_text(sha(RUN/'delivery.tar.gz')+'  delivery.tar.gz\n')
    print(json.dumps(dict(qa=qa,archive_sha256=sha(RUN/'delivery.tar.gz'))),flush=True)

if __name__=='__main__':main()
