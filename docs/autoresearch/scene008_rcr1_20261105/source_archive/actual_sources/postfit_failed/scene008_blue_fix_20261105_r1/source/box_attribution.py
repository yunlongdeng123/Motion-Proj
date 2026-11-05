"""Exact colour attribution for the user's car-flank box and clamp-order control."""
from colour_fit import *
@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);a=p.parse_args();e=Experiment();e.load(Path(a.checkpoint))
    out=ROOT/'boxed_components';out.mkdir(exist_ok=False);records=[]
    for raw in (113,137):
        d=e.rt.raw(raw);scene=e.rt.scene(raw,e.model.asset,delete=True);kw=scene['rows'];directions=kw['means3D']-d['cam'].camera_center;directions/=directions.norm(dim=1,keepdim=True)
        original=eval_sh(e.rt.n.static.active_sh_degree,kw['shs'].transpose(1,2),directions)+.5
        delta=torch.zeros_like(original);delta[:len(e.model.static_dc)]=.1*e.model.static_dc.tanh()
        road=kw['features'][:,4]>0;delta[road]=e.model.road_basis@(.09*e.model.road_world.tanh())
        env=e.model.env(d['cam'].get_world_directions(False).permute(1,2,0)).permute(2,0,1)
        images={};parts={};weights=None
        for name,colors in [('post_clamp',(original.clamp_min(0)+delta).clamp_min(0)),('pre_clamp',(original+delta).clamp_min(0))]:
            features=kw['features'].clone();features[:,6:18]=(features[:,2:6,None]*colors[:,None,:]).reshape(-1,12)
            rendered=raster(d['cam'],e.rt.pipe,e.rt.n.static.active_sh_degree,dict(kw,shs=None,colors_precomp=colors,features=features),e.rt.n.bg)
            images[name]=rendered['rgb']+(1-rendered['alpha'])*env
            parts[name]={key:value for key,value in zip(('a6','a37','road','static'),rendered['extra'][4:16].reshape(4,3,320,480))}
            parts[name]['environment']=(1-rendered['alpha'])*env;weights=rendered['extra'][:4]
            assert (sum(parts[name].values())-images[name]).abs().max()<3e-6
        base=e.model.draw(raw,True,baseline=True)['rgb'];canvas=Image.new('RGB',(1440,696),'white')
        entries=[('M2 baseline',base),('Current correction / after clamp',images['post_clamp']),('Control / before native clamp',images['pre_clamp']),('Road RGB contribution',parts['post_clamp']['road']),('Static RGB contribution',parts['post_clamp']['static']),('Environment RGB contribution',parts['post_clamp']['environment'])]
        for j,(label,value) in enumerate(entries):
            x=j%3*480;y=j//3*348;canvas.paste(image(value),(x,y+28));ImageDraw.Draw(canvas).text((x+6,y+7),f'{ROOT.name.split("_")[-1]} / step{e.step} / raw{raw} / {label}',fill='black')
        canvas.save(out/f'{raw}.png')
        yy,xx=np.indices((320,480));m=torch.tensor((xx>=270)&(xx<313)&(yy>=170)&(yy<225),device='cuda')
        record=dict(raw=raw,window=[270,170,313,225],parts={},pixelmax_post_pre=float((images['post_clamp']-images['pre_clamp']).abs().max()),reobserved_road_pixels=int(d['road_edit'][m].sum()))
        for key,value in parts['post_clamp'].items():record['parts'][key]=dict(meanRGB=cpu(value[:,m].mean(1)).tolist(),blue=float(blue(value[:,m]).mean()))
        record['mean_alphaT']=cpu(weights[:,m].mean(1)).tolist();records.append(record)
    write(out/'metrics.json',dict(rows=records));print(json.dumps(records),flush=True)
if __name__=='__main__':main()
