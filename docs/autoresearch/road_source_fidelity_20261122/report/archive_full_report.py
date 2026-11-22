"""Create and verify the complete historical report companion, kept locally."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.report_root.resolve();dest=args.output.resolve();dest.parent.mkdir(parents=True,exist_ok=True)
    files=sorted(p for p in source.rglob('*') if p.is_file())
    expected={source.name+'/'+p.relative_to(source).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}
    with tarfile.open(dest,'w:gz',compresslevel=1,format=tarfile.PAX_FORMAT) as archive:
        for path in files:
            archive.add(path,arcname=source.name+'/'+path.relative_to(source).as_posix(),recursive=False)
    verified=[]
    with tarfile.open(dest,'r:gz') as archive:
        members=archive.getmembers()
        if len(members)!=len(expected) or {m.name for m in members}!=set(expected):
            raise ValueError('Companion member set differs from complete report')
        for member in members:
            if not member.isfile():
                raise ValueError('Unexpected non-file companion member')
            with archive.extractfile(member) as handle:
                h=hashlib.file_digest(handle,'sha256').hexdigest()
            match=h==expected[member.name]['sha256'] and member.size==expected[member.name]['bytes']
            if not match:
                raise ValueError('Companion member hash mismatch: '+member.name)
            verified.append({'path':member.name,**expected[member.name],'verified':True})
    result={'schema':'source_fidelity_full_report_companion.v1','passed':True,'path':str(dest),
            'filename':dest.name,'sha256':sha(dest),'bytes':dest.stat().st_size,'member_count':len(verified),
            'members_verified':True,'logical_restore_directory':'outputs/'+source.name,
            'archive_extract_parent':'outputs','source_report_directory':str(source),
            'html_sha256':expected[source.name+'/index.html']['sha256'],'files':verified}
    manifest=dest.with_suffix(dest.suffix+'.manifest.json');manifest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    compact={k:v for k,v in result.items() if k!='files'};compact['full_member_manifest_path']=str(manifest)
    (dest.parent/'report_companion_manifest.json').write_text(json.dumps(compact,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(compact))
