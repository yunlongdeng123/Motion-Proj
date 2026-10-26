"""Separate appearance-coordinate convention from unchanged edit geometry."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--control',type=Path,required=True);a=p.parse_args()
    summary=json.loads((a.control/'summary.json').read_text());assert summary['SH_orientation_convention']=='object'
    assert summary['adapter_actual_native_max_difference']==0 and summary['master_files_and_tensor_versions_unchanged']
    records=[];sheet=Image.new('RGB',(1440,1050),'#142332');draw=ImageDraw.Draw(sheet)
    for row,name in enumerate(['no_edit','turn_left15','turn_right15']):
        paths=sorted((a.control/name/'float').glob('*.npz'));assert len(paths)==31
        rgbmax=0.;geometry={key:0. for key in ['alpha','dynamic_weight','depth_sum']};maes=[]
        for path in paths:
            with np.load(path) as control,np.load(a.baseline/name/'float'/path.name) as baseline:
                for key in geometry:geometry[key]=max(geometry[key],float(np.abs(control[key]-baseline[key]).max()))
                rgbmax=max(rgbmax,float(np.abs(control['rgb']-baseline['rgb']).max()))
                actor=(control['dynamic_weight'][0]>.05)|(baseline['dynamic_weight'][0]>.05)
                maes.append(float(np.abs(control['rgb']-baseline['rgb']).transpose(1,2,0)[actor].mean()) if actor.any() else 0.)
        assert all(value==0 for value in geometry.values()),geometry
        if name=='no_edit':assert rgbmax<=2e-6
        records.append(dict(case=name,frames=31,unchanged_geometry_channels_max_difference=geometry,rgb_max_difference=rgbmax,actor_domain_rgb_change_MAE_macro=float(np.mean(maes))))
        selected=paths[3*len(paths)//4].stem+'.png'
        left=np.array(Image.open(a.baseline/name/'native_rgb'/selected).convert('RGB'))
        right=np.array(Image.open(a.control/name/'native_rgb'/selected).convert('RGB'))
        delta=np.minimum(np.abs(left.astype(float)-right)*5,255).astype('uint8')
        for col,(label,pixels) in enumerate([('world SH',left),('object SH',right),('RGB difference x5',delta)]):
            sheet.paste(Image.fromarray(pixels),(col*480,row*350+30));draw.text((col*480+5,row*350+8),name+' / '+label+' / raw'+selected[:-4],fill='white')
    sheet.save(a.control/'comparison.jpg',quality=95)
    result=dict(control='SH orientation only',records=records,geometry_unchanged=True,master_unchanged=True,counterfactual_RGB_GT=False,relighting_claim=False,new_training_updates=0,quality_accepted=False,human_verdict=None)
    (a.control/'cpu_control_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
