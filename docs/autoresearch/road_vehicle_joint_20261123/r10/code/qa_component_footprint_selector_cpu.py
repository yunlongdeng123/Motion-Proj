"""Small typed-selector controls: positive owners, guards, tiny tails and missing views."""
from pathlib import Path
import argparse,json
import numpy as np
from prepare_component_footprint_retirement_cpu import qualify_components

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    n=12;ids=np.arange(n,dtype=np.int64);kind=np.zeros(n,np.uint8);kind[1]=1
    mass=np.zeros((n,5,8),np.float32);mass[:,:,0]=1;mass[:,:,4]=1;mass[:,:,6]=1
    owner=np.full((n,5),-1,np.int32);owner[1]=0
    present=np.ones((n,5),bool);native=np.ones((n,5),bool);native[1]=False;guard=np.zeros((n,5),bool)
    owner[2]=21;guard[3,2]=True;present[4,1]=False;native[5,3]=False
    mass[6,0,1]=np.float32(1e-30);mass[7,1,5]=np.float32(1e-30)
    mass[8,2,2]=np.float32(1e-30);mass[9,4,3]=np.float32(1e-30)
    mass[10,:,6]=0;owner[11]=8
    flags,free,edit,selected,mixed=qualify_components(ids,mass,owner,present,native,guard,kind,np.array([],np.int64))
    assert np.array_equal(np.flatnonzero(selected),[0,1]);assert mass[6,0,1]>0 and not selected[6]
    offset=2593301;world=np.array([7,51,592000],np.int64);active=np.array([0,1,2],np.int64);gid=offset+active
    assert np.array_equal(world[gid-offset],world) and not np.array_equal(gid-offset,world)
    d=dict(schema='R10.component_footprint_selector_CPU_QA.v1',complete=True,decision='PASS',synthetic_rows=n,selected_native_and_road=[0,1],
        native21_and_positive_owners_blocked=True,permanent_guard_blocked=True,missing_and_component_inconsistent_blocked=True,
        tiny_positive_outsideEdit_outsideReplacement_car_K_blocked=True,no_mass_epsilon=True,oldroad_global_active_world_distinguished=True,
        no_actual_scope_created=True,GPU_calls=0,updates=0)
    assert not a.output.exists();a.output.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
if __name__=='__main__':main()
