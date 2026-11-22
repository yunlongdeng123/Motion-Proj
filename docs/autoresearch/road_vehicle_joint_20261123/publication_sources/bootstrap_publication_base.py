"""Copy the reviewed R7 isolated-index core; no Git invocation."""
from pathlib import Path
import hashlib,json

def main():
    out=Path(__file__).resolve().parent
    source=out.parents[1]/'road_shared_appearance_20261122_r7/publication/isolated_publication_base.py'
    original=source.read_bytes();text=source.read_text(encoding='utf-8-sig')
    old="if __name__ == '__main__':\n    main()"
    assert old in text
    text=text.replace(old,"if __name__ == '__main__':\n    raise SystemExit('Use publish_road_vehicle_joint.py; no standalone entry.')")
    text=text.replace("{k: result[k] for k in ('inventory_ready'","{k: result.get(k) for k in ('inventory_ready'")
    target=out/'isolated_publication_base.py';target.write_text(text,encoding='utf-8')
    digest=lambda data:hashlib.sha256(data).hexdigest()
    assert source.read_bytes()==original
    record=dict(upstream=str(source),upstream_sha256=digest(original),derived=str(target),derived_sha256=digest(target.read_bytes()),changes=['inventory-only optional-key print guard','disable standalone entry; task wrapper only'],original_shared_file_modified=False,Git_executed=False)
    (out/'isolated_base_derivation.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record))

if __name__=='__main__':main()
