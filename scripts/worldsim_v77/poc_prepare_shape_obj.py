"""Export Hunyuan shape GLBs to Blender-readable OBJ without changing geometry."""
import pathlib
import trimesh

ROOT=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPLICIT-POC-20260926/r1')
for name in ['scene_0230','scene_0255']:
    actor=ROOT/name/'actor'
    mesh=trimesh.load(actor/'shape_untextured.glb',force='mesh')
    mesh.export(actor/'shape_untextured.obj')
    print(name,len(mesh.vertices),len(mesh.faces),mesh.bounds.tolist())
