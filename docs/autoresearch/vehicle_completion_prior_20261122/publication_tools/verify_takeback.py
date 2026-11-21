"""本地核验取回 zip 本体与所有提交源，不解压、不依赖远端 Git。"""
from pathlib import Path
import argparse
import zipfile
from archive_common import read_json, dump_json, digest, sha, valid_destination


def verify(zip_path, receipt_path, output):
    receipt = read_json(receipt_path)
    zip_path = Path(zip_path)
    if sha(zip_path) != receipt['sha256'] or zip_path.stat().st_size != receipt['bytes']:
        raise ValueError('取回 zip 与远端收据不符')
    sources = receipt['committed_source_sha256']
    expected = {'committed/' + name for name in sources}
    expected |= {'publication/' + name for name in ('publication_receipt.json', 'source_validation.json', 'stage_receipt.json', 'final_verification.json')}
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError('取回 zip 文件清单不符')
        for name, expected_sha in sources.items():
            valid_destination(name, allow_current_docs=True)
            if digest(z.read('committed/' + name)) != expected_sha:
                raise ValueError('取回提交源内容不符：' + name)
        import json
        publication = json.loads(z.read('publication/publication_receipt.json'))
        final = json.loads(z.read('publication/final_verification.json'))
        if not publication['pushed'] or not final['verified'] or publication['commit'] != final['commit']:
            raise ValueError('发布与最终核验收据不一致')
    result = dict(verified=True, zip_sha256=receipt['sha256'], committed_sources=len(sources),
                  commit=publication['commit'], URL=publication['URL'], raw_media_or_models_taken_back=False)
    dump_json(output, result)
    print(result['URL'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('zip')
    parser.add_argument('receipt')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    verify(args.zip, args.receipt, args.output)
