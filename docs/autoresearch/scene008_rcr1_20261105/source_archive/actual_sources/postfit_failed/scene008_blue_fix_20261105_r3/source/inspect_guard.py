from colour_fit import *
@torch.no_grad()
def main():
    e=Experiment();out=ROOT/'guard_view';out.mkdir(exist_ok=False)
    for raw in (105,113,137):
        d=e.rt.raw(raw);base=e.model.draw(raw,True,baseline=True);mask=e.model.envelope(raw)
        kw=tagged(e.model.asset.rows(d['pose'],d['cam'],e.rt.pipe,16,precomputed=False),'asset')
        solo=raster(d['cam'],e.rt.pipe,3,kw,e.rt.n.bg.new_zeros(3))
        v=base['rgb'].clone();v[0,mask]=.5*v[0,mask]+.5;v[1:,mask]*=.5
        canvas=Image.new('RGB',(1440,348),'white')
        for j,x in enumerate((base['rgb'],v,solo['alpha'].expand(3,-1,-1))):canvas.paste(image(x),(480*j,28))
        canvas.save(out/f'{raw}.png')
        print(raw,int(mask.sum()),int(d['box_candidate'][0].sum()),int((solo['alpha']>1e-5).sum()))
if __name__=='__main__':main()
