"""Read-only, fixed49-view transfer audit of three-view trained edit copy."""
from editor_common import *
from scipy.ndimage import binary_dilation,binary_erosion
import argparse

parser=argparse.ArgumentParser();parser.add_argument('--source',default='edit_smoke_preserve');parser.add_argument('--out',default='edit49');args=parser.parse_args()
dest=OUT/args.out;dest.mkdir(exist_ok=True);n=Native();results=[]
with torch.no_grad():
    for arm in ['window49','full199']:
        ck=torch.load(OUT/args.source/arm/'checkpoint1000_from0.pth',weights_only=False);road=Road(FULL/f'{arm}_initialization.npz');road.dc.copy_(ck['dc']);road.opacity.copy_(ck['opacity']);mult=torch.ones(len(n.static.get_xyz),device='cuda').index_copy(0,ck['ids'],ck['gate'].sigmoid())
        for raw in range(89,138):
            c=n.camera(raw);baseline=n.render(c);edit=n.render(c,road,static_multiplier=mult)
            labels=np.load(f'/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v41_20261023_dense_r7/frames/0008{raw:03d}.npz')['actor_labels'];target=(labels==4)|(labels==5)
            err=cpu((baseline['rgb']-edit['rgb']).abs()).mean(0);con=cpu(edit['contribution']).squeeze();outside=~binary_dilation(target,iterations=2)
            metrics=dict(arm=arm,raw=raw,optimization_frame=raw in [102,129,137],outside_MAE=float(err[outside].mean()),outside_p99=float(np.quantile(err[outside],.99)),outside_fraction_change_gt005=float((err[outside]>.05).mean()),actors={})
            for actor,k in [(37,5),(6,4)]:
                sel=binary_erosion(labels==k,iterations=5);metrics['actors'][str(actor)]=dict(core=int(sel.sum()),visible=int((sel&(con>=.05)).sum()),mean_contribution=float(con[sel].mean()) if sel.any() else None)
            results.append(metrics);save(dest/f'{arm}_{raw}.png',edit['rgb'])
            if arm=='window49':save(dest/f'baseline_{raw}.png',baseline['rgb'])
        print(arm,'done',flush=True)
assert n.unchanged()
(dest/'summary.json').write_text(json.dumps(dict(rows=results,only_three_frames_have_training_crossview_targets=True,other_46_views_are_same_sequence_transfer_checks_not_new_scene_generalization=True,no_clean_hidden_road_PSNR=True,master_unchanged=True,quality_accepted=False,human_verdict=None),indent=2))
