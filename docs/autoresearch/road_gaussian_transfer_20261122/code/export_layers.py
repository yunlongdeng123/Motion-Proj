"""实际新高斯贡献、原资产继承与独立实拍支持；不把来源账本当因果分解。"""
from native import *
from PIL import Image

def png(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.rint(np.clip(value,0,1)*255).astype(np.uint8)).save(path)

def main():
    ctx=open_context(ROOT/'appearance/asset');oldN=len(load(PARENT/'road.npz')['active'])
    new=ctx.active_world_rows>=oldN;ids=ctx.active_world_rows[new]
    old_road=load(PARENT/'road.npz');source_ids=ctx.provenance['transfer_source_world_row_id'][ids]
    source_color=old_road['reference_srgb'][source_ids]
    infer=tensor(new,torch.bool);actual=torch.where(infer[:,None],ctx.base_rgb,torch.zeros_like(ctx.base_rgb))
    raw=torch.zeros_like(ctx.base_rgb);raw[infer]=tensor(source_color)
    result={};folder=ROOT/'layers';folder.mkdir(exist_ok=True)
    try:
        with torch.no_grad():
            for frame in FRAMES:
                full=render(ctx,frame,feature_world=ids);new_rgb=render(ctx,frame,colors=actual)['rgb'];raw_rgb=render(ctx,frame,colors=raw)['rgb']
                alpha=full['alpha'].cpu().numpy();weight=full['new_weight'].cpu().numpy()
                Pnew=new_rgb.cpu().numpy().transpose(1,2,0);Praw=raw_rgb.cpu().numpy().transpose(1,2,0)
                transfer=np.divide(Pnew,weight[:,:,None],out=np.zeros_like(Pnew),where=weight[:,:,None]>1e-8)
                material=np.divide(Praw,weight[:,:,None],out=np.zeros_like(Praw),where=weight[:,:,None]>1e-8)
                fraction=np.clip(np.divide(weight,alpha,out=np.zeros_like(weight),where=alpha>1e-8),0,1)
                color=np.array([.47,.73,1.])*fraction[:,:,None]+np.array([.38,.43,.48])*(1-fraction[:,:,None])
                color=np.where((alpha>0.05)[:,:,None],color,0)
                source=load(C/'road_multisource_20261122_r1/sources/targets'/f'{frame:03d}.npz')
                same_mask=source['same_mask'].copy();same=source['same_rgb'].copy();A=np.zeros_like(same_mask)
                if frame in (86,110):
                    d=load(C/'road_surface_residual_20261121_r2/proposal/domains'/f'{frame:03d}.npz')
                    A=d['appearance_A'];obs=np.asarray(Image.open(C/f'road_common1152_20261121/teacher/raw{frame:03d}/observation_1152.png'),np.float32)/255
                    same[A]=obs[A];same_mask|=A
                same=np.where(same_mask[:,:,None],same,0)
                paths={}
                for name,value in [('same',same),('material',material),('transfer',transfer),('effective',color),('new_weight',weight),('new_fraction',fraction)]:
                    path=folder/f'{frame:03d}_{name}.png';png(path,value);paths[name]=str(path.relative_to(ROOT))
                np.savez_compressed(folder/f'{frame:03d}_contribution.npz',new_rgb=Pnew,new_weight=weight,road_alpha=alpha,new_fraction=fraction,
                    inherited_rgb=full['rgb'].cpu().numpy().transpose(1,2,0)-Pnew)
                result[str(frame)]=dict(same=paths['same'],material=paths['material'],transfer=dict(rgb=paths['transfer'],alpha=paths['new_weight']),effective=paths['effective'],
                    source_contribution_fraction=paths['new_fraction'],same_input_pixels=int(same_mask.sum()),current_A_input_pixels=int(A.sum()),
                    inferred_majority_pixels=int(((fraction>=.5)&(alpha>.05)).sum()),
                    meanings=dict(same='observed input support, not a causal decomposition of inherited Gaussian colors',
                        material='relocated source Gaussian RGB before exposure, evaluated at final geometry/opacity',
                        transfer='actual migrated-kernel color contribution divided by its native unit contribution',
                        effective='blue=migrated native contribution; gray=inherited original asset; no new diffusion'))
        evidence=[]
        for title,path in [('零步验收','transfer/receipt.json'),('接缝拟合','appearance/receipt.json'),('原照核验','source_native/receipt.json'),('运输算子','proposal/receipt.json'),('源块选择','proposal/source_decision.json')]:
            evidence.append(dict(label=title,path=path))
        write(ROOT/'report_inputs.json',dict(source_layers=dict(frames=result),source_texture_rows=[],
            parameter_preservation=dict(source_RGB_exact_zero_step=True,source_sigma_exact_zero_step=True,rigid_covariance=True,
                source_asset_retained=True,individual_RGB_parameters=0),evidence=evidence,new_diffusion_calls=0))
        write(folder/'receipt.json',dict(complete=True,nativeF=15,nativeB=0,optimizer_updates=0,new_diffusion_calls=0,
            operation_source='actual native color linearity and unit-feature contribution; same support separate',frames=list(result)))
        print('SOURCE_LAYERS_READY',flush=True)
    finally:close()
if __name__=='__main__':main()
