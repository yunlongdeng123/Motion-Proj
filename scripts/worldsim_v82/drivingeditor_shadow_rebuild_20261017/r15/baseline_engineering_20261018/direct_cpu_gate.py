from pathlib import Path
import os,sys,unittest,json,ast,hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
E=Path(__file__).resolve().parent
sys.path.insert(0,str(E/'direct_implementation'))
import torch,numpy as np
import formal_support as formal
from types import SimpleNamespace
from restore_contract import assert_exact_state

def main():
 import test_rigid_spatial_topology as spatial
 suite=unittest.defaultTestLoader.loadTestsFromModule(spatial)
 r=unittest.TextTestRunner(verbosity=1).run(suite)
 assert r.wasSuccessful()
 cameras=[SimpleNamespace(image_path='/image/image_0/x.png',image_name=f'0008{89+i}') for i in range(49)]
 s=formal.EpochSampler(0,cameras)
 seen=[s.select(cameras).image_name for _ in range(49)];assert len(set(seen))==49
 for _ in range(17):s.select(cameras)
 cap=s.getstate();other=formal.EpochSampler(999,cameras);other.setstate(cap)
 assert_exact_state(cap,other.getstate())
 assert [s.select(cameras).image_name for _ in range(150)]==[other.select(cameras).image_name for _ in range(150)]
 m=SimpleNamespace(no_time_split=True);cfg=SimpleNamespace(no_time_split=True,densify_until_iter=15000,time_split_frac=.5)
 formal.restore_flags(m,dict(static_no_time_split=False),30000,cfg);assert m.no_time_split is False
 helper=formal.native_static_function();assert callable(helper)
 code=(E/'direct_implementation/train.py').read_text();ast.parse(code)
 assert "'dynamic_alpha': package['dynamic_alpha']" in code
 assert "policy['hybrid_lambda_dynamic_opacity'] = 100.0" in code
 assert 'formal.topology_update' in code and 'sampler.randrange' not in code
 assert not torch.cuda.is_initialized()
 result=dict(status='PASSED',spatial_native_reference_tests=r.testsRun,epoch_sampler_restore_and_future150_exact=True,
   static_temporal_flag_restore=True,actual_train_body_opacity_binding='isolated_raw_union100',
   new_training_updates=0,GPU_executed=False,CUDA_initialized=False,
   source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (E/'direct_implementation').glob('*.py')},
   limitations=['native screen-radius-history reset ordering retained','rigid spatial port, not native PVG temporal split/RNG equivalence','new SGCM disabled in rigid stage; inherited stage1 retained'])
 (E/'direct_CPU_gate.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'}))
if __name__=='__main__':main()
