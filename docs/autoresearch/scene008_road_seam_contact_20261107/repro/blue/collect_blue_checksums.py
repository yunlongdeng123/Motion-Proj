"""Read-only source/result checksum manifest for local retrieval verification."""
from pathlib import Path
import hashlib,json
RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_road_seam_contact_20261107_r1')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(2**20),b''):h.update(c)
    return h.hexdigest()
paths=list((RUN/'blue/boundary_material_v1').rglob('*'))
for name in ['n2_full49','n2_mip_full49','n2_material_full49','n2_material_pilot','n2_mip_material_pilot']:
    paths += [RUN/f'results/{name}/receipt.json',RUN/f'results/{name}/128/candidate_delete.npz']
paths += [RUN/'diagnostic/layers_v1/128/r0_delete.npz',RUN/'diagnostic/layers_v1/128/observations.npz',RUN/'blue/blue_fitstates_sources_receipts_v1.tgz']
manifest={str(p.relative_to(RUN)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(set(paths)) if p.is_file()}
(RUN/'blue/blue_retrieval_manifest_v1.json').write_text(json.dumps({'schema':'BLUE_RETRIEVAL_SHA256_V1','run':str(RUN),'files':manifest},indent=2),encoding='utf8')
print(json.dumps({'file_count':len(manifest),'archive':manifest['blue/blue_fitstates_sources_receipts_v1.tgz']}))
