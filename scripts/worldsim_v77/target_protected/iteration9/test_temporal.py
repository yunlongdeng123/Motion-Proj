import numpy as np
from temporal_metrics import process,cells

def fixtures():
 fs=[];actors=[]
 for i in range(10):
  C=np.eye(4);C[0,3]=i*.1;fs.append({'timestamp':i*100000,'camera_to_world':C.tolist(),'intrinsics_1024':[[100,0,32],[0,100,32],[0,0,1]]})
  actors.append({'translation':[0,0,10],'rotation':[1,0,0,0],'size':[2,4,2]})
 return fs,actors

def test_world_static_is_not_image_static():
 fs,aa=fixtures();hs=[]
 for i in range(10):
  h=np.zeros((64,64),bool);h[20:30,20+i:30+i]=1;hs.append(h)
 r=process(fs,aa,hs);assert r['static_A_moving_ego_image_change']

def test_sweep_and_steady_are_distinct():
 fs,aa=fixtures();b=np.zeros((64,64),bool);b[20:40,10:54]=1
 steady=[];sweep=[]
 for i in range(10):
  h=np.zeros_like(b);h[20:40,10+i*3:25+i*3]=1;sweep.append(h);steady.append(sweep[0])
 assert process(fs,aa,sweep,{'B':[b]*10})['sweep_over_any_B']
 assert not process(fs,aa,steady,{'B':[b]*10})['sweep_over_any_B']

def test_no_other_frame_evidence_when_always_hidden():
 fs,aa=fixtures();fs=[dict(f,camera_to_world=np.eye(4).tolist()) for f in fs];b=np.zeros((64,64),bool);b[26:38,26:38]=1
 r=process(fs,aa,[b]*10,{'B':[b]*10},{'B':aa})
 assert r['protected']['B']['no_geometric_other_frame_support']

def test_revealed_evidence_is_counted_without_changing_GT():
 fs,aa=fixtures();fs=[dict(f,camera_to_world=np.eye(4).tolist()) for f in fs];b=np.zeros((64,64),bool);b[26:38,26:38]=1;hs=[b.copy()]+[np.zeros_like(b)]*9
 r=process(fs,aa,hs,{'B':[b]*10},{'B':aa})
 assert r['protected']['B']['hidden_pixels_with_other_frame_cuboid_cell_support'][0]==1
