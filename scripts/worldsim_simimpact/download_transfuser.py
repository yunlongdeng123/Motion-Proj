"""公开官方LiDAR+RGB策略，用于接通首回波实际被消费的对照。"""
from pathlib import Path
import download_ranged as d
d.ROOT=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1/lidar_policy')
d.ASSETS=d.ROOT/'assets'
d.HOST='https://hf-mirror.com';d.SESSION.trust_env=False
d.FILES=[('transfuser_seed_0.ckpt','autonomousvision/navsim_baselines','transfuser/transfuser_seed_0.ckpt',672447984)]
d.main()
