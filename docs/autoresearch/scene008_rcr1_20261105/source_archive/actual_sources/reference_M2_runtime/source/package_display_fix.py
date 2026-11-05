import gzip,hashlib,json,shutil,tarfile
from pathlib import Path
RUN=Path(__file__).resolve().parent.parent
OLD=RUN/'delivery_before_observed_preview_fix';NEW=RUN/'delivery'
DOC=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/docs/autoresearch/scene008_a6_asset_controls_20261105')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    for arm in ('m1','m2','m3'):
        assert sha(OLD/'assets'/f'{arm}_complete_a6.npz')==sha(NEW/'assets'/f'{arm}_complete_a6.npz')
        with gzip.open(OLD/'evidence'/f'{arm}_all49_metrics.json.gz','rt') as a,gzip.open(NEW/'evidence'/f'{arm}_all49_metrics.json.gz','rt') as b:assert json.load(a)==json.load(b)
    files={str(p.relative_to(RUN)):sha(p) for p in sorted(NEW.rglob('*')) if p.is_file()}
    changed=[p for p in sorted(NEW.rglob('*')) if p.is_file() and (not (OLD/p.relative_to(NEW)).exists() or sha(p)!=sha(OLD/p.relative_to(NEW)))]
    manifest=dict(status='PASS',reason='Correct the observed RGB preview normalization only; figures and report packaging updated',
       model_assets_byte_identical=True,all49_numeric_metrics_identical=True,
       remote_full_archive_sha256=sha(RUN/'delivery.tar.gz'),files=files)
    path=RUN/'publication_display_fix_manifest.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
    with tarfile.open(RUN/'display_fix_patch.tar.gz','w:gz') as archive:
        for p in changed:archive.add(p,arcname=str(p.relative_to(RUN)))
        archive.add(path,arcname=path.name)
    (RUN/'display_fix_patch.tar.gz.sha256').write_text(sha(RUN/'display_fix_patch.tar.gz')+'  display_fix_patch.tar.gz\n')
    shutil.copytree(NEW,DOC,dirs_exist_ok=True)
    shutil.copyfile(RUN/'delivery.tar.gz.sha256',DOC/'delivery.tar.gz.sha256')
    shutil.copyfile(path,DOC/path.name)
    with (DOC/'README.md').open('a') as f:f.write('\nThe observed-RGB report preview was corrected to avoid dividing already normalized RGB by255. All three model assets remain byte-identical and all49 numeric metrics are identical to the preceding report. Browser/layout and actual gallery handlers are checked after retrieval.\n')
    print(json.dumps(dict(changed_files=len(changed),patch_bytes=(RUN/'display_fix_patch.tar.gz').stat().st_size,patch_sha256=sha(RUN/'display_fix_patch.tar.gz'))))
if __name__=='__main__':main()
