"""Zero-update control: keep M2 exactly, change only its learned environment."""
from audit import *
@torch.no_grad()
def main():
    out=ROOT/'env_swap_probe';out.mkdir(exist_ok=False);rt=Runtime()
    asset=SHAsset.from_geometry(ASSET/'m2/complete_a6.npz').cuda()
    current=editor_common.B;editor_common.B=C/'scene008_a37_only_B10k_20261103_r1/formal10k'
    old=editor_common.Native();editor_common.B=current
    records=[];yy,xx=np.indices((320,480))
    for raw in (105,113,130):
        d=rt.raw(raw);ref=cpu(d['rgb']);actors=cpu(d['actor_labels'])>0
        regions={name:(xx>=a)&(xx<c)&(yy>=b)&(yy<e)&~actors for name,(a,b,c,e) in BOXES.items()}
        regions['a6']=cpu(d['a6_visible']).astype(bool)
        for delete in (False,True):
            scene=rt.scene(raw,asset,delete=delete);base=rt.render_rows(raw,scene)
            env=old.env(d['cam'].get_world_directions(False).permute(1,2,0)).permute(2,0,1)
            new=rt.render_rows(raw,dict(scene,env=env))
            assert torch.equal(base['alpha'],new['alpha']) and torch.equal(base['depth_sum'],new['depth_sum'])
            record=dict(raw=raw,delete=delete,
                        base={name:stats(cpu(base['rgb']),ref,m) for name,m in regions.items()},
                        swap={name:stats(cpu(new['rgb']),ref,m) for name,m in regions.items()})
            records.append(record)
            canvas=Image.new('RGB',(960,348),'white');canvas.paste(image(cpu(base['rgb'])),(0,28));canvas.paste(image(cpu(new['rgb'])),(480,28))
            ImageDraw.Draw(canvas).text((8,7),'M2 current environment',fill='black');ImageDraw.Draw(canvas).text((488,7),'M2 historical environment control',fill='black')
            canvas.save(out/f'{raw:03d}_{"delete" if delete else "keep"}.jpg',quality=95,subsampling=0)
        rt._data.clear()
    assert rt.modelunchanged() and old.unchanged() and asset.assert_geometry_unchanged()
    (out/'metrics.json').write_text(json.dumps(dict(training_updates=0,rows=records),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(records),flush=True)
if __name__=='__main__':main()
