from editor_common import *
n=Native();dest=OUT/'c_selected_mechanism'
ck=torch.load(OUT/'edit_smoke_preserve/full199/checkpoint1000_from0.pth',weights_only=False)
road=Road(FULL/'full199_initialization.npz');road.dc.data.copy_(ck['dc']);road.opacity.data.copy_(ck['opacity'])
mult=torch.ones(len(n.static.get_xyz),device='cuda').index_copy(0,ck['ids'],ck['gate'].sigmoid())
results=[]
with torch.no_grad():
    for raw in [102,129,137]:
        c=n.camera(raw);a=n.render(c);b=n.render(c,static_multiplier=mult);out=n.render(c,road,static_multiplier=mult)
        d=np.load(OUT/'crossview'/f'{raw}.npz');valid=d['valid'];target=d['target'];gt=d['rgb'].transpose(2,0,1)
        rgb=[cpu(x['rgb']) for x in [a,b,out]];err=[np.abs(x-gt).mean(0) for x in rgb];con=cpu(out['contribution'])[0]
        domains=[('road_zero',con<1e-6),('road_low',(con>=1e-6)&(con<.05)),('road_visible',con>=.05)]
        row=dict(raw=raw,target_pixels=int(target.sum()),reference_pixels=int(valid.sum()),unknown_pixels=int((target&~valid).sum()),domains={})
        for label,mask in domains:
            mask=mask&valid;row['domains'][label]=dict(pixels=int(mask.sum()),A_MAE=float(err[0][mask].mean()),B_MAE=float(err[1][mask].mean()),C_MAE=float(err[2][mask].mean()),
                C_better_than_B=int(((err[2]<err[1]-1e-4)&mask).sum()),C_worse_than_B=int(((err[2]>err[1]+1e-4)&mask).sum()))
        results.append(row)
        overlay=np.zeros((320,480,3),np.uint8);overlay[target&~valid]=[90,90,90];overlay[valid&(con<1e-6)]=[80,140,235];overlay[valid&(con>=1e-6)&(con<.05)]=[235,195,65];overlay[valid&(con>=.05)]=[45,185,130]
        Image.fromarray(overlay).save(dest/f'road_role_{raw}.png')
        for name,gain in [('static_effect',err[0]-err[1]),('road_effect',err[1]-err[2])]:
            v=np.minimum(np.abs(gain)/.1,1);im=np.zeros((320,480,3));im[:,:,1]=np.where(valid&(gain>0),v,0);im[:,:,0]=np.where(valid&(gain<0),v,0)
            im[target&~valid]=.25;Image.fromarray(np.uint8(im*255)).save(dest/f'{name}_{raw}.png')
assert n.unchanged()
(dest/'spatial_effects.json').write_text(json.dumps(dict(rows=results,updates=0,domain='fixed crossview reference; training/development labels, unknown excluded and counted',interpretation='inference interventions; road can affect learned gates indirectly during joint training even where final road weight is zero'),indent=2))
print(json.dumps(results),flush=True)
