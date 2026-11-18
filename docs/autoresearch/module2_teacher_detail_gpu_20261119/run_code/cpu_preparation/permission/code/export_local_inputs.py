"""小型local offload源包，复用已导出全图几何；无大UV传输。"""
from teacher_detail_permission import P,Q,N,load,sha
import numpy as np,json
def main():
 out=Q/'permission/final_r1';d=load(out/'teacher_evidence.npz',['common_base_srgb','common_material_valid','observed_H_prior','mH_union','active_world_rows','common_write_active','common_latent_mask','common_reg_support'])
 ids=np.flatnonzero(d['common_write_active']);geo=load(P/'cpu_exact_r1/geometry_kernel.npz',['world_xyz','normal_world','sigma_m'])
 keys=load(N/'provenance.npz',['world_keys'])['world_keys']
 np.savez_compressed(out/'local_compile_inputs.npz',**d,active_local_indices=ids,query_world_rows=d['active_world_rows'][ids],
  query_world_xyz=geo['world_xyz'][ids],normal_world=geo['normal_world'][ids],sigma_m=geo['sigma_m'][ids],world_keys=keys)
 receipt=dict(file='local_compile_inputs.npz',bytes=(out/'local_compile_inputs.npz').stat().st_size,sha256=sha(out/'local_compile_inputs.npz'),
  geometry_file='geometry_for_consumers.npz',geometry_bytes=(out/'geometry_for_consumers.npz').stat().st_size,
  geometry_sha256=sha(out/'geometry_for_consumers.npz'),rows=len(ids),CPU_only=True,native_calls=0,optimizer_updates=0)
 (out/'local_inputs.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()

