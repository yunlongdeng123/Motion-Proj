import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

out=Path('outputs/shadow_diagnosis'); gpu=out/'gpu'; focus=out/'focus'
cpu=json.loads((out/'cpu/attribution.json').read_text())
fcpu=json.loads(Path('work/shadow_diagnosis/focus_attribution.json').read_text())
mask=np.asarray(Image.open(focus/'reviewed_artifact_footprint.png'))>0
with np.load(out/'target_observed.npz') as f:
    observed={k:f[k].copy() for k in f.files}
with np.load(gpu/'target_original_Static.npz') as f:
    baseline={k:f[k].copy() for k in f.files}
def metrics(rgb):
    return dict(mean_RGB=rgb[:,mask].mean(1).tolist(),MAE_to_anchor=float(abs(rgb-baseline['RGB'])[:,mask].mean()),MAE_to_GT=float(abs(rgb-observed['gt'])[:,mask].mean()))
times=[]
for fid in ['0008089','0008095','0008101','0008107','0008113','0008125','0008137']:
    rows={}
    arrays={}
    for kind in ['normal','freeze_position_only','freeze_marginal_only','freeze_both']:
        with np.load(gpu/f'fixed_{fid}_{kind}.npz') as f:
            arrays[kind]=f['RGB'].copy();rows[kind]=metrics(arrays[kind])
    rows['normal_vs_freeze_position_MAE']=float(abs(arrays['normal']-arrays['freeze_position_only'])[:,mask].mean())
    rows['normal_vs_freeze_marginal_MAE']=float(abs(arrays['normal']-arrays['freeze_marginal_only'])[:,mask].mean())
    times.append(dict(frame_id=fid,**rows))
anchor=metrics(baseline['RGB'])
summary=dict(footprint_pixels=int(mask.sum()),anchor=anchor,anchor_mean_alpha=float(baseline['alpha'][0,mask].mean()),probe_static_RGB=[baseline['RGB'][:,y,x].tolist() for x,y in [[80,252],[110,251],[145,242]]],probe_depth_expected_center_m=[float(baseline['expected_center_depth_model'][0,y,x]/.1) for x,y in [[80,252],[110,251],[145,242]]],fixed_time=times)
for file in list(gpu.glob('target*.npz'))+list(focus.glob('hide*.npz')):
    with np.load(file) as f: summary[file.stem]=metrics(f['RGB'])
summary['top100_dark_mass']=sum(r['mean_alpha_T'] for r in fcpu['top100_near_black'])
summary['top100_dark_mass_weighted_position_span_m']=sum(r['mean_alpha_T']*r['position_span_m'] for r in fcpu['top100_near_black'])/summary['top100_dark_mass']
summary['top100_dark_mass_marginal_range_gt_05']=sum(r['mean_alpha_T'] for r in fcpu['top100_near_black'] if r['marginal_range'][1]-r['marginal_range'][0]>.5)/summary['top100_dark_mass']
(out/'measured_summary.json').write_text(json.dumps(summary,indent=2))
small={k:v for k,v in summary.items() if k!='fixed_time'}
print(json.dumps(small,indent=2))
print(json.dumps([{'frame_id':r['frame_id'],'normal_anchor_MAE':r['normal']['MAE_to_anchor'],'freeze_position_anchor_MAE':r['freeze_position_only']['MAE_to_anchor'],'freeze_marginal_anchor_MAE':r['freeze_marginal_only']['MAE_to_anchor'],'normal_vs_freeze_position':r['normal_vs_freeze_position_MAE'],'normal_vs_freeze_marginal':r['normal_vs_freeze_marginal_MAE']} for r in times],indent=2))

try: font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
except OSError: font=ImageFont.load_default()
def rgbimg(a): return Image.fromarray(np.rint(np.moveaxis(a.clip(0,1),0,-1)*255).astype('uint8'))
crop=(45,220,170,275)
def sheet(items,path,cols=2,size=(625,275)):
    rows=(len(items)+cols-1)//cols
    result=Image.new('RGB',(cols*size[0],rows*(size[1]+34)), '#102333')
    draw=ImageDraw.Draw(result)
    for i,(label,im) in enumerate(items):
        x=(i%cols)*size[0]; y=(i//cols)*(size[1]+34)
        draw.text((x+8,y+6),label,font=font,fill='white')
        result.paste(im.crop(crop).resize(size,Image.Resampling.NEAREST),(x,y+34))
    result.save(path)
gt=rgbimg(observed['gt']); stat=rgbimg(baseline['RGB'])
overlay=stat.copy();ImageDraw.Draw(overlay).line(fcpu['polygon']+[fcpu['polygon'][0]],fill='#ff665c',width=1)
sheet([('Real GT: target vehicle and road shadow',gt),('Static: vehicle removed, dark patch remains',stat),('Reviewed artifact footprint: 1287 pixels',overlay),('Original GN11: same camera and timestamp',Image.open(gpu/'target_original_GN11_Static.png'))],out/'target_comparison.png')
sheet([(fid+' / '+kind.replace('freeze_','hold '),Image.open(gpu/f'fixed_{fid}_{kind}.png')) for fid in ['0008089','0008101','0008113','0008137'] for kind in ['normal','freeze_position_only','freeze_marginal_only','freeze_both']],out/'temporal_ablation.png',cols=4,size=(400,176))
if (focus/'hide_near_black_SH_max_lt_005_NOT_A_REPAIR.png').exists():
    sheet([('Static baseline',stat),('Temporarily hide all near-black points',Image.open(focus/'hide_near_black_SH_max_lt_005_NOT_A_REPAIR.png')),('Hide near-black centers >0.5m from plane',Image.open(focus/'hide_near_black_center_abs_height_gt_050m_NOT_A_REPAIR.png')),('Hide near-black centers <=0.15m from plane',Image.open(focus/'hide_near_black_center_abs_height_le_015m_NOT_A_REPAIR.png'))],out/'causal_hiding_NOT_A_REPAIR.png')
# Display the actual independent Static alpha on a fixed 0..1 scale.
alpha=baseline['alpha'][0]; alpha_img=Image.fromarray(np.rint(alpha.clip(0,1)*255).astype('uint8')).convert('RGB')
sheet([('Static RGB',stat),('Static alpha (black=0, white=1)',alpha_img)],out/'opacity_check.png')
# A clear two-state crop comparison follows the same plane-ray hypothesis.
items=[]
for k in range(3):
    for fid in ['0008089','0008101','0008113']:
        row=next(r for r in cpu['real_observations'] if r['probe']==k and r['frame_id']==fid)
        # Select tiles from the already exported real49 contact sheets.
        idx=int(fid)-8089; src=Image.open(out/'cpu'/f'real49_road_probe{k}_UNREVIEWED.png')
        tile=src.crop(((idx%7)*100,(idx//7)*120,(idx%7)*100+100,(idx//7)*120+120))
        items.append((f'probe{k} {fid}',tile))
contact=Image.new('RGB',(900,3*380),'#102333');d=ImageDraw.Draw(contact)
for i,(label,im) in enumerate(items):
    x=(i%3)*300;y=(i//3)*380;d.text((x+8,y+5),label,font=font,fill='white');contact.paste(im.resize((300,360),Image.Resampling.NEAREST),(x,y+20))
contact.save(out/'real_road_before_during_after.png')
# Export focused attribution without the large rasterizer group-ID tables.
(focus/'focused_attribution_review.json').write_text(json.dumps({k:v for k,v in fcpu.items() if k!='group_ids'},indent=2))
