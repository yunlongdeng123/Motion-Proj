"""Five-view direct surface material review, distinct from Gaussian rendering."""
from pathlib import Path
import argparse,sys,time
import numpy as np
from PIL import Image,ImageDraw

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['r7','r9','geometry','field','parent','masks','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--scope-key',default='candidate_mask');a=p.parse_args()
    assert not a.output.exists();a.output.mkdir(parents=True)
    sys.path.insert(0,str(a.r9/'code'))
    from project_old_raw_candidates import load,sha,write,bilinear,all_four,srgb
    from local_surface_projection import raster_surface
    g=load(a.geometry);f=load(a.field);parent=load(a.parent)
    gains={int(r['raw']):float(r['gain']) for r in __import__('json').loads((a.r7/'exposure_gain.json').read_text())['records']}
    scope=g[a.scope_key].astype(bool);uv=g['uv_global'];rgb=f['atlas_linear']
    extension=f['permission_mask']&~parent['permission_mask']
    assert np.array_equal(rgb[~extension],parent['atlas_linear'][~extension])
    rows=[];panels=[];started=time.time();palette=np.array([[.06,.09,.13],[.12,.85,.85],[.77,.50,.18],[.68,.35,.95],[.90,.38,.62]])
    kinds=f['source_kind'].copy();kinds[(kinds==3)&f['teacher_inferred_field']]=4
    for raw in [86,110,134,179,198]:
        v=load(a.r7/f'projection_cache_v1/{raw:03d}/view_cache.npz');m=load(a.masks/f'{raw:03d}/rectified_masks.npz')
        surface=raster_surface(g['world_xyz'],uv,scope,v['world_to_camera'],v['camera_K'],[1152,768],cell_mask=g.get('cell_asset_mask'))
        mapped=surface['atlas_uv_global'];projected,bounds=bilinear(rgb,mapped)
        available=all_four(f['permission_mask']&(f['source_kind']>0),mapped)
        shown=surface['projection_valid']&bounds&available
        protected=m['protected']|m['actor_visible_proxy']|~m['RGB_valid']
        display=shown&~protected
        final=np.full((768,1152,3),[.06,.09,.13]);final[shown]=srgb(projected[shown]/gains[raw])
        # Context is explicitly a photo-backed surface illustration. It is
        # neither a native Gaussian render nor proof of joint depth ordering.
        context=v['photo_pinhole_srgb'].copy();context[display]=final[display]
        sources,_=bilinear(palette[kinds],mapped);source=np.full_like(final,[.06,.09,.13]);source[shown]=sources[shown]
        out=a.output/f'{raw:03d}';out.mkdir()
        save=lambda name,x:Image.fromarray(np.rint(np.clip(x,0,1)*255).astype(np.uint8)).save(out/name)
        save('direct_surface_material.png',final);save('photo_context_surface_illustration.png',context);save('source_layer.png',source)
        np.savez_compressed(out/'surface_review.npz',**surface,shown=shown,context_display=display,
            unavailable_material=surface['projection_valid']&~(bounds&available))
        rows.append(dict(raw=raw,surface_pixels=int(surface['projection_valid'].sum()),material_pixels=int(shown.sum()),
            unavailable_material_pixels=int((surface['projection_valid']&~(bounds&available)).sum()),
            surface_dark_maxRGB_below40=int((shown&(final.max(-1)<40/255)).sum()),
            photo_context_used_only_for_display=True))
        panel=Image.new('RGB',(1152,394),'#101827');ImageDraw.Draw(panel).text((8,5),f'raw{raw:03d} direct road surface / photo context (not Gaussian joint rendering)',fill='white')
        panel.paste(Image.open(out/'photo_context_surface_illustration.png').crop((0,400,1152,768)),(0,26));panels.append(panel)
    gradients=[]
    for dx,dy in [(1,0),(0,1)]:
        sa=(slice(None,-dy or None),slice(None,-dx or None));sb=(slice(dy,None),slice(dx,None))
        cross=(extension[sa]!=extension[sb])&f['permission_mask'][sa]&f['permission_mask'][sb]
        gradients.extend(np.abs(rgb[sa]-rgb[sb])[cross].reshape(-1).tolist())
    gradients=np.asarray(gradients)
    sheet=Image.new('RGB',(1152,sum(p.height for p in panels)));y=0
    for p in panels:sheet.paste(p,(0,y));y+=p.height
    sheet.save(a.output/'five_view_direct_surface_context.png')
    write(a.output/'receipt.json',dict(schema='R10.direct_surface_material_review.v1',complete=True,
        geometry_sha256=sha(a.geometry),final_atlas_sha256=sha(a.field),parent_sha256=sha(a.parent),views=rows,
        new_extension_texels=int(extension.sum()),parent_RGB_outside_extension_byteexact=True,
        triangle_scope='exact cell_asset_mask when present; context-only cells excluded',
        boundary_channel_abs_difference=dict(samples=len(gradients),median=float(np.median(gradients)) if len(gradients) else None,
            p95=float(np.quantile(gradients,.95)) if len(gradients) else None,max=float(gradients.max()) if len(gradients) else None),
        numeric_acceptance='PENDING independent review',visual_acceptance='PENDING independent and root review',
        native_Gaussian_render=False,CPU_only=True,GPU_forward=0,GPU_backward=0,Gaussian_updates=0,seconds=time.time()-started))
    print('complete five-view surface review')

if __name__=='__main__':main()
