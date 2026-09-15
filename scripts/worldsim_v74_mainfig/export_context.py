import pathlib,json,torch,numpy as np,shutil,tarfile
torch.set_num_threads(1)
P=pathlib.Path('/root/autodl-tmp');R=P/'runs/worldsim_v74_h2/WS-V74-MAINFIG-01/20260915-first-return-r1';out=R/'legacy_context';out.mkdir(exist_ok=True)
D=P/'runs/worldsim_v73/WS-V73-M2-GLOBAL-DATA-01/20260907T180000Z__window-rigid-population-r2'
raw=P/'runs/worldsim_v73/WS-V73-M1-NATIVE-GEOMETRY-ADAPT-01/20260907T161500Z__native-dpt-surround25-dev6-s7301-r3/build_observations.pt'
old=P/'runs/worldsim_v74_h2/WS-V74-H2-REDISCOVERY-01/20260915-cpu-r1'
scenes=torch.load(raw,weights_only=False,map_location='cpu',mmap=True)
for rec,ctx in zip(json.loads((old/'paired/manifest.json').read_text()),json.loads((old/'context/manifest.json').read_text())):
 name=rec['source_selection_method'];s=next(v for v in scenes if v['scene_id']==rec['actor']['scene'])
 c=torch.load(D/rec['actor']['file'],weights_only=False,map_location='cpu');idx=ctx['view_index'];j=list(map(int,c['view_indices'])).index(idx);v=s['views'][idx]
 shutil.copy2(v['image_path'],out/(name+'.jpg'))
 np.savez_compressed(out/(name+'.npz'),camera_from_actor=c['camera_from_actor'][j].numpy(),K=c['intrinsics'][j].numpy(),network_hw=np.asarray(v['image'].shape[-2:]),build=c['points_actor_m'].numpy(),size=c['size_lwh_m'].numpy())
 (out/(name+'.json')).write_text(json.dumps(ctx,indent=2))
with tarfile.open(R/'legacy_context.tar','w') as t:t.add(out,arcname='legacy_context')
print(R/'legacy_context.tar')
