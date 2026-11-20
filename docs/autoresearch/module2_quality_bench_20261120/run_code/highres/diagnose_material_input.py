"""Auxiliary fixed-domain material input vs straight-road detail, using frozen metrics."""
from pathlib import Path
import argparse,json,sys,hashlib,time,gc
import numpy as np

Q=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1');G=Q/'gpu_20261119';BENCH=G/'quality_bench_20261120_r1'
DEFAULT=BENCH/'highres'
def read(p):return json.loads(Path(p).read_text())
def load(p,keys=None):
    with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in(z.files if keys is None else keys)}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main(root):
    root=Path(root);out=root/'material_input_aux_r1';out.mkdir(exist_ok=False);started=time.perf_counter()
    sys.path.insert(0,str(BENCH/'metrics'))
    import road_detail_metrics as metrics
    config=metrics.DetailConfig();bindings={str(Path(metrics.__file__)):sha(metrics.__file__),str(Path(__file__)):sha(__file__)}
    rows=[];domains=[];identities=[]
    for raw in [110,86]:
        paths=dict(target=root/f'targets/{raw:03d}.npz',reader=root/f'pixel_reads/{raw:03d}.npz',maps=root/f'maps/{raw:03d}.npz')
        t=load(paths['target'],['T_C','T_S','C_pre_alpha','S_pre_alpha','source_mass'])
        cache=load(paths['reader'],['pixel_ids','source_mass','M01','M11','M11_noH']);m=load(paths['maps'],['pixel_ids','image_hw','surface_hit'])
        ids=cache['pixel_ids'];assert np.array_equal(ids,m['pixel_ids'])and np.array_equal(m['image_hw'],[1280,1920])
        assert len(np.unique(ids))==len(ids)and np.array_equal(ids,np.flatnonzero(m['surface_hit']))
        assert all(cache[k].shape==(len(ids),3)and cache[k].dtype==np.float32 for k in ['M01','M11','M11_noH'])
        mass=np.zeros(1280*1920,np.float64);mass[ids]=cache['source_mass'];mass=mass.reshape(1280,1920)
        assert np.array_equal(mass,t['source_mass'])
        alpha_path=root/f'rendered/M11/{raw:03d}.npz';reference=load(alpha_path,['road_alpha','road_eligible_alphaT']);alpha=reference['road_alpha']
        stable=alpha>=.95;source_ok=mass>0;We=reference['road_eligible_alphaT']
        for p in paths.values():bindings[str(p)]=sha(p)
        identities.append(dict(raw=raw,image_hw=[1280,1920],compact_rows=len(ids),pixel_ids_equal_fresh_geometry=True,
            pixel_ids_are_sorted_global_flat=True,pixel_value_shape=[len(ids),3],pixel_value_dtype='float32',source_mass_exact_to_targets=True))
        for region in ['C','S']:
            score=t[region+'_pre_alpha'];context=score&source_ok&stable
            y,x=np.where(score);pad=int(max(config.pixel_scales)*config.gaussian_truncate+.5)
            y0=max(0,int(y.min())-pad);y1=min(1280,int(y.max())+pad+1);x0=max(0,int(x.min())-pad);x1=min(1920,int(x.max())+pad+1)
            sl=(slice(y0,y1),slice(x0,x1));ctx=context[sl];teacher=np.moveaxis(t['T_'+region][:,y0:y1,x0:x1],0,-1).astype(np.float64)
            domain=dict(raw=raw,region=region,full_prealpha_pixels=int(score.sum()),source_mass_positive_pixels=int((score&source_ok).sum()),
                alpha_ge_0p95_pixels=int((score&stable).sum()),auxiliary_common_pixels=int(context.sum()),
                excluded_source_mass_zero=int((score&~source_ok).sum()),excluded_alpha_below_0p95=int((score&~stable).sum()),
                excluded_both=int((score&~source_ok&~stable).sum()),excluded_any=int((score&~context).sum()),
                computation_bbox_y0_y1_x0_x1=[y0,y1,x0,x1],crop_reason='finite-kernel acceleration only; full score bbox plus maximum Gaussian radius; no nonzero context outside crop',
                native_road_We_over_alpha=dict(mean=float((We[context]/alpha[context]).mean()),
                    exact_zero_pixels=int((context&(We==0)).sum()),p05_p50_p95=np.quantile(We[context]/alpha[context],[.05,.5,.95]).tolist()))
            domains.append(domain)
            teachers={};cores={}
            for sigma in config.pixel_scales:
                teachers[sigma]=metrics._masked_hf(teacher,ctx,sigma,config.gaussian_truncate)[0]
                radius=int(config.gaussian_truncate*sigma+.5);cores[sigma]=ctx&metrics._strict_core(ctx,radius)
            for case in ['M01','M11','M11_noH']:
                packet=root/f'rendered/{case}/{raw:03d}.npz';p=load(packet,['road_rgb','road_alpha']);assert np.array_equal(p['road_alpha'],alpha)
                bindings[str(packet)]=sha(packet)
                field=G/f'formal_r1/{"M11"if case=="M11_noH"else case}/C800/field.pt';bindings[str(field)]=sha(field)
                world=np.zeros((1280*1920,3),np.float32);world[ids]=cache[case];world=world.reshape(1280,1920,3)[sl].astype(np.float64)
                P=np.moveaxis(p['road_rgb'][:,y0:y1,x0:x1],0,-1).astype(np.float64)
                straight=np.divide(P,alpha[sl][...,None],out=np.zeros_like(P),where=stable[sl][...,None])
                for sigma in config.pixel_scales:
                    ht=teachers[sigma];core=cores[sigma]
                    hw=metrics._masked_hf(world,ctx,sigma,config.gaussian_truncate)[0]
                    hr=metrics._masked_hf(straight,ctx,sigma,config.gaussian_truncate)[0]
                    ew,er=hw[core]-ht[core],hr[core]-ht[core];mix=hr[core]-hw[core]
                    ew2=float(np.mean(ew**2))if core.any()else None;er2=float(np.mean(er**2))if core.any()else None
                    mix2=float(np.mean(mix**2))if core.any()else None;cross=float(2*np.mean(ew*mix))if core.any()else None
                    rows.append(dict(raw=raw,region=region,case=case,sigma_native_pixels=sigma,
                        full_prealpha_pixels=domain['full_prealpha_pixels'],common_pixels=domain['auxiliary_common_pixels'],strict_core_pixels=int(core.sum()),
                        partial_context_boundary_pixels=int(ctx.sum()-core.sum()),
                        actual_world_input=metrics._hf_stats(hw,ht,core),rendered_straight_road=metrics._hf_stats(hr,ht,core),
                        actual_world_flat=metrics._flat_stats(hw,ht,core,config.flat_hf_thresholds),rendered_straight_flat=metrics._flat_stats(hr,ht,core,config.flat_hf_thresholds),
                        squared_HF_error_identity=dict(rendered=er2,world_input=ew2,mixing_difference=mix2,cross_term=cross,
                            closure_error=None if er2 is None else er2-ew2-mix2-cross,
                            warning='Signed cross term is essential. These are algebraic components, not nonnegative causal percentages. Mixing difference includes fixed readonly/eligible mixture and alpha normalization.')))
                del p,world,P,straight;gc.collect()
        del t,cache,m;gc.collect()
    result=dict(schema='fixed_domain_material_input_auxiliary.v1',domains=domains,rows=rows,compact_identity=identities,source_SHA256=bindings,
        common_domain='same per frame/region across methods: native prealpha C/S & source_mass>0 & common road_alpha>=.95',
        context='same common domain, frozen metric strict finite Gaussian core; full prealpha and excluded denominators preserved',
        material_input='actual source-normalized pixelbox RGB from pixel_reads before eligible/read-only GS mixture, not unmodulated H alone',
        rendered_input='actual road P divided only at common alpha>=.95; bounded amplification<=1/.95; no use at low alpha',
        primary_native_bench_changed=False,teacher='frozen pseudo raw native C/S, not hidden GT',seconds=time.perf_counter()-started,
        new_native_calls=0,new_backward_calls=0,optimizer_updates=0,GPU_operations=0)
    write(out/'results.json',result)
    compact=[dict(raw=r['raw'],region=r['region'],case=r['case'],sigma=r['sigma_native_pixels'],core=r['strict_core_pixels'],
        world_energy=r['actual_world_input']['HF_energy_ratio'],world_cosine=r['actual_world_input']['HF_cosine'],world_HF_MAE=r['actual_world_input']['HF_MAE'],
        straight_energy=r['rendered_straight_road']['HF_energy_ratio'],straight_cosine=r['rendered_straight_road']['HF_cosine'],straight_HF_MAE=r['rendered_straight_road']['HF_MAE'])for r in rows]
    write(out/'summary.json',compact);print(json.dumps([r for r in compact if r['sigma']==1.]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT);a=p.parse_args();main(a.root)
