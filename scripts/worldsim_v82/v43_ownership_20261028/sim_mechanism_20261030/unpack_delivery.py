from pathlib import Path
import tarfile,hashlib,json
root=Path(__file__).resolve().parents[2]
source=Path(__file__).with_name('local_delivery.tar.gz')
expected='790002324c5875cc76bb5e722b6dade084c8505565abe63dc4d290c7e08e6ac5'
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
dest=root/'outputs/V43_Ownership_20261027/gpu_v43/sim_mechanism_20261030/evidence'
dest.mkdir(parents=True,exist_ok=True)
with tarfile.open(source,'r:gz') as tar:
 members=tar.getmembers()
 for item in members:
  target=(dest/item.name).resolve()
  assert item.isfile() and target.is_relative_to(dest.resolve())
  target.parent.mkdir(parents=True,exist_ok=True)
  target.write_bytes(tar.extractfile(item).read())
receipt={'sha256':expected,'files':len(members),'bytes':source.stat().st_size,'remote_local_hash_equal':True}
(dest.parent/'transfer_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(receipt)
