from pathlib import Path
src=Path('/root/autodl-tmp/envs/nksr-v74/lib/python3.10/site-packages')
dst=Path('/root/autodl-tmp/envs/hugsim-impact/lib/python3.10/site-packages')
for name in ['torch','functorch','torchgen','nvidia','torch-2.4.1+cu118.dist-info']:
 s=src/name;t=dst/name
 if s.exists() and not t.exists():t.symlink_to(s,target_is_directory=s.is_dir());print('linked',name)
