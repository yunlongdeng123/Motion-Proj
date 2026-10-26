"""Fixed-domain coverage and causal road-removal audit; no optimizer updates."""
from editor_common import *
from scipy.ndimage import binary_erosion
sys.path.insert(0,str(C/'tools'))
from v41_dense_geometry import FixedSurface

def main():
    dest=OUT/'coverage_audit';dest.mkdir(exist_ok=True)
    n=Native();m=read('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json')
    rows=[]
    with torch.no_grad():
        for arm,stage in [('window49','init'),('full199','init'),('full199','observed1000')]:
            a=dict(np.load(FULL/f'{arm}_initialization.npz'));road=Road(FULL/f'{arm}_initialization.npz')
            if stage!='init':
                ck=torch.load(OUT/'observed_smoke'/arm/'checkpoint1000_from0.pth',weights_only=False);road.dc.copy_(ck['dc']);road.opacity.copy_(ck['opacity'])
            surface=FixedSurface(m,dict(keys=a['keys'],world=a['world_xyz'],surface_supported=a['surface_supported'],provenance=a['eligible'].astype(np.uint8),base_weight=a['source_valid'],normal_world=a['normal_world']))
            for raw in range(89,138):
                frame=m['frames'][f'0008{raw:03d}'];r=surface.camera(np.array(frame['intrinsic']),np.array(frame['world_to_camera']));sup=r['supported'].reshape(320,480);uid=r['unit_id'].reshape(320,480);active=sup&a['eligible'][np.maximum(uid,0)]
                c=n.camera(raw);direct=render_road(c,road,n.args);edit=n.render(c,road);alpha=cpu(direct['alpha']).squeeze();con=cpu(edit['contribution']).squeeze()
                # Ordered mutually exclusive diagnostic bins. Thresholds are audit conventions, not success criteria.
                state=np.zeros((320,480),np.uint8);state[sup]=1;state[active]=2;state[active&(alpha>=.5)]=3;state[active&(alpha>=.5)&(con>=.05)]=4
                labels=np.load(f'/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v41_20261023_dense_r7/frames/0008{raw:03d}.npz')['actor_labels']
                item=dict(arm=arm,stage=stage,raw=raw,actors={})
                for actor,k in [(37,5),(6,4)]:
                    mask=labels==k;item['actors'][str(actor)]={}
                    for region,sel in [('whole',mask),('core',binary_erosion(mask,iterations=5))]:
                        item['actors'][str(actor)][region]=dict(pixels=int(sel.sum()),bins=np.bincount(state[sel],minlength=5).tolist(),isolated_alpha_ge_half=int((sel&(alpha>=.5)).sum()),actual_contribution_ge_005=int((sel&(con>=.05)).sum()),alpha_mean=float(alpha[sel].mean()) if sel.any() else None,contribution_mean=float(con[sel].mean()) if sel.any() else None)
                if raw in [102,129,137]:
                    np.savez_compressed(dest/f'{arm}_{stage}_{raw}.npz',state=state,alpha=alpha,contribution=con,labels=labels,unit_id=uid)
                    save(dest/f'{arm}_{stage}_{raw}_road.png',direct['rgb']);save(dest/f'{arm}_{stage}_{raw}_edit.png',edit['rgb'])
                    palette=np.array([[235,80,90],[245,180,40],[135,100,225],[50,140,220],[50,215,140]],np.uint8);Image.fromarray(palette[state]).save(dest/f'{arm}_{stage}_{raw}_state.png')
                    item['critical']={f'{x},{y}':dict(unit_id=int(uid[y,x]),state=int(state[y,x]),isolated_alpha=float(alpha[y,x]),actual_contribution=float(con[y,x])) for x,y in [(255,240),(256,219)]}
                rows.append(item)
            print(arm,stage,'done',flush=True)
        interventions=[]
        for name,arm in [('edit_smoke','full199'),('edit_smoke_preserve','full199'),('edit_smoke_preserve','window49')]:
            ck=torch.load(OUT/name/arm/'checkpoint1000_from0.pth',weights_only=False);road=Road(FULL/f'{arm}_initialization.npz');road.dc.copy_(ck['dc']);road.opacity.copy_(ck['opacity']);mult=torch.ones(len(n.static.get_xyz),device='cuda').index_copy(0,ck['ids'],ck['gate'].sigmoid())
            for raw in [102,129,137]:
                c=n.camera(raw);d=np.load(OUT/'crossview'/f'{raw}.npz');valid=torch.tensor(d['valid'],device='cuda');outside=~torch.tensor(d['target'],device='cuda');target=torch.tensor(d['rgb'].transpose(2,0,1),device='cuda');original=n.render(c);only=n.render(c,static_multiplier=mult);combined=n.render(c,road,static_multiplier=mult);direct=render_road(c,road,n.args)
                record=dict(name=name,arm=arm,raw=raw,valid=int(valid.sum()),target=int(d['target'].sum()),original_MAE=float((original['rgb']-target).abs()[:,valid].mean()),gate_only_MAE=float((only['rgb']-target).abs()[:,valid].mean()),gate_road_MAE=float((combined['rgb']-target).abs()[:,valid].mean()),road_increment_mae=float((combined['rgb']-only['rgb']).abs()[:,valid].mean()),outside_change=float((combined['rgb']-original['rgb']).abs()[:,outside].mean()),contribution_mean=float(combined['contribution'][0,valid].mean()),critical={f'{x},{y}':dict(contribution=float(combined['contribution'][0,y,x]),alpha=float(direct['alpha'][0,y,x]),road_rgb_delta=float((combined['rgb']-only['rgb']).abs()[:,y,x].mean())) for x,y in [(255,240),(256,219)]})
                interventions.append(record)
                for label,out in [('gate_only',only),('gate_road',combined),('road',direct)]:save(dest/f'{name}_{arm}_{raw}_{label}.png',out['rgb'])
        totals={}
        for arm,stage in [('window49','init'),('full199','init'),('full199','observed1000')]:
            key=arm+'_'+stage;subset=[r for r in rows if r['arm']==arm and r['stage']==stage];totals[key]={}
            for actor in ['37','6']:
                totals[key][actor]={}
                for region in ['whole','core']:
                    parts=[r['actors'][actor][region] for r in subset];totals[key][actor][region]={k:np.sum([p[k] for p in parts],axis=0).tolist() for k in ['pixels','bins','isolated_alpha_ge_half','actual_contribution_ge_005']}
        assert n.unchanged()
        (dest/'audit.json').write_text(json.dumps(dict(bin_labels=['no_supported_surface','supported_but_ineligible','active_low_isolated_alpha','isolated_alpha_ge_half_but_contribution_lt005','actual_road_visible'],thresholds=dict(alpha=.5,contribution=.05),caveat='A ray-cell qualification and finite Gaussian footprint are different; alpha/contribution also reported independent of cell bins. Not ground truth completion. Evaluation frames overlap development/training.',totals=totals,rows=rows,interventions=interventions,master_unchanged=True,quality_accepted=False,human_verdict=None),indent=2))
        print(json.dumps(dict(totals=totals,interventions=interventions),indent=2),flush=True)
if __name__=='__main__':main()
