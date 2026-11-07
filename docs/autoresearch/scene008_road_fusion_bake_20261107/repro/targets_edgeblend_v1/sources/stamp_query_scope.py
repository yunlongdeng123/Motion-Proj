"""只补query真实RGB角色说明；保留首manifest，不改C数组。"""
from pathlib import Path
import json,hashlib,shutil
RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_road_fusion_bake_20261107_r1');B=RUN/'targets_edgeblend_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=B/'target_manifest.json';backup=B/'target_manifest.pre_query_scope.json'
 if backup.exists():raise FileExistsError('preserve scope stamp')
 d=json.loads(p.read_text());assert d['edgeblend_ready'] is True
 root=next(x for x in [RUN/'root/contract/edgeblend_control_v1.json',RUN/'contract/edgeblend_control_v1.json'] if x.exists())
 shutil.copy2(root,B/'root_posthoc_edgeblend_contract.json');shutil.copy2(p,backup)
 policy=dict(HELD9_target='C_HELD explicitly uses actual query observationRGB O in the deterministic validity-distance blend. It is target construction/diagnosis only; its lower MAE is not a held prediction score.',
   fit='Only BUILD40 target files are used to fit the persistent10098-parent RGB table; HELD9 Ctarget never participates in fitting.',
   native_query='Frozen native RGB rendering is not conditioned on query observationRGB. Existing rt.raw may read/load obs RGB for evaluation/contracts, but scene/raster colour computation uses camera/pose/env/material, not queryO as colour/image input.',
   exposure='Same previously exposed scene; original parents/full199 source history prevents independent generalization claims.',
   control_registration='Posthoc single fixed10px fallback chosen after the hardreplace diagnostic; replaces the unexecuted legacy fallback. No width grid or extra legacy trial.')
 (B/'QUERY_RGB_SCOPE.json').write_text(json.dumps(policy,ensure_ascii=False,indent=2),encoding='utf8')
 d.update(query_RGB_scope=policy,query_scope_sha256=sha(B/'QUERY_RGB_SCOPE.json'),first_manifest_pre_query_scope_sha256=sha(backup),
   root_posthoc_control_contract_sha256=sha(B/'root_posthoc_edgeblend_contract.json'),scope_adapter_source_sha256=sha(__file__),scope_adapter_target_arrays_modified=False)
 p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
 pre=json.loads((B/'target_preflight_0F.json').read_text());pre.update(final_manifest_sha256=sha(p),query_RGB_scope=policy,scope_adapter_target_arrays_modified=False)
 (B/'target_preflight_0F.json').write_text(json.dumps(pre,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(dict(status='FINAL_C_TARGET_LOCK',manifest_sha256=sha(p),selected_target_key=d['selected_target_key'],rows=len(d['rows']),no_array_changes=True)))
if __name__=='__main__':main()
