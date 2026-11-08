"""OSQP仅元数据/API核对；不setup、不factorize、不solve。"""
import inspect
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[key] = '1'
import osqp
import scipy

m = osqp.OSQP()
s = m.ext.OSQPSettings()
if hasattr(m.ext, 'osqp_set_default_settings'):
    m.ext.osqp_set_default_settings(s)
names = ['time_limit', 'max_iter', 'eps_abs', 'eps_rel', 'polishing', 'warm_starting',
         'verbose', 'scaling', 'adaptive_rho', 'adaptive_rho_interval', 'adaptive_rho_fraction',
         'adaptive_rho_tolerance', 'rho', 'rho_is_vec', 'sigma', 'alpha', 'linsys_solver',
         'scaled_termination', 'eps_prim_inf', 'eps_dual_inf', 'check_termination',
         'delta', 'polish_refine_iter']
settings = {name: getattr(s, name) for name in names if hasattr(s, name)}
settings = {k: (int(v) if hasattr(v, 'value') else v) for k, v in settings.items()}
assert osqp.__version__ == '1.0.4' and hasattr(s, 'polishing') and hasattr(s, 'time_limit')
out = dict(status='PASS_OSQP_1_0_4_METADATA_ONLY', osqp_version=osqp.__version__, scipy_version=scipy.__version__,
    algebra=m.algebra, setup_signature=str(inspect.signature(m.setup)),
    solve_signature=str(inspect.signature(m.solve)), warm_start_signature=str(inspect.signature(m.warm_start)),
    actual_settings_defaults=settings, source_file=osqp.__file__, setup_calls=0, QP_calls=0, GPU_forwards=0)
print(json.dumps(out, indent=2))
