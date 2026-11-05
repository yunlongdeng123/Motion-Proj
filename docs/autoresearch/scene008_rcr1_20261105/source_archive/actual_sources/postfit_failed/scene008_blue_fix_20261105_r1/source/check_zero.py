"""Full49, full-image zero-update rendering parity including unobserved pixels."""
from colour_fit import *
@torch.no_grad()
def main():
    e=Experiment();rows=[];out=ROOT/'zero_full49';out.mkdir(exist_ok=False)
    for raw in range(89,138):
        d=e.rt.raw(raw)
        for delete in (False,True):
            base=e.model.draw(raw,delete,baseline=True);new=e.model.draw(raw,delete)
            error=(base['rgb']-new['rgb']).abs();maxabs=float(error.max())
            rows.append(dict(raw=raw,delete=delete,maxabs=maxabs,over_one_byte=int((error>1/255).any(0).sum())))
            if raw==137:
                canvas=Image.new('RGB',(960,348),'white');canvas.paste(image(base['rgb']),(0,28));canvas.paste(image(new['rgb']),(480,28));canvas.save(out/f'137_{delete}.png')
    write(out/'metrics.json',dict(rows=rows,maxabs=max(r['maxabs'] for r in rows)))
    print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
