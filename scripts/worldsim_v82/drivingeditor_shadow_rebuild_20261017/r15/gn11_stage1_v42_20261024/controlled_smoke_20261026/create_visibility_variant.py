from pathlib import Path
p=Path(__file__).parent
s=(p/'train_edit_smoke.py').read_text(encoding='utf-8')
s=s.replace("default='edit_smoke'","default='edit_visibility'").replace("default=.001","default=.999")
s=s.replace('start=time.time()',"opening_masks=[torch.tensor(d['valid'],device='cuda') & (torch.tensor(np.load(OUT/'coverage_audit'/f'full199_init_{raw}.npz')['alpha'],device='cuda')>=.5) for raw,d in zip(sources,data)]\n    start=time.time()")
s=s.replace('loss=deleted+preserve+.1*visible+.1*road_loss',"static_only=n.render(c,static_multiplier=multiplier); opening=-torch.log((1-static_only['alpha'][0,opening_masks[i]]).clamp_min(1e-6)).mean(); loss=deleted+preserve+.1*visible+.1*road_loss+.1*opening")
s=s.replace('road=float(road_loss),seconds=', 'road=float(road_loss),opening=float(opening),seconds=')
s=s.replace('del out,direct,loss,multiplier','del out,direct,static_only,loss,multiplier')
s=s.replace("+0.1 dense visible isolated-road L1'", "+0.1 dense visible isolated-road L1 +0.1 negative-log static transmittance on fixed full199 alpha>=.5 AND verified crossview rays; adaptive diagnostic'")
(p/'train_edit_visibility.py').write_text(s,encoding='utf-8')
