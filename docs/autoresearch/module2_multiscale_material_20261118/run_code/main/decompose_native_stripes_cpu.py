"""固定alpha下的解释性乘积二阶差分；仅CPU读取float，不改变训练/质量门。"""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
from qa_world_material_results import P,G,FRAMES,read,write,load,packet


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for b in iter(lambda:stream.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def decomposition(P0,alpha,stencil):
    """At y=center: D2P=c0D2a+a0D2c+(a+-a0)(c+-c0)+(a--a0)(c--c0)."""
    rgb=P0.astype(np.float64);a=alpha.astype(np.float64)
    common=stencil&(a[:-2]>.9)&(a[1:-1]>.9)&(a[2:]>.9)
    yy,xx=np.nonzero(common);y=yy+1
    if not len(y):return dict(frozen_stencil_pixels=int(stencil.sum()),qualified_pixels=0)
    am,a0,ap=[a[y+dy,xx]for dy in(-1,0,1)]
    Pm,Pc,Pp=[rgb[:,y+dy,xx]for dy in(-1,0,1)]
    cm,c0,cp=Pm/am,Pc/a0,Pp/ap
    Da=ap-2*a0+am;Dc=cp-2*c0+cm;D2P=Pp-2*Pc+Pm
    terms=dict(c0_D2alpha=c0*Da,alpha0_D2c=a0*Dc,
        cross_plus=(ap-a0)*(cp-c0),cross_minus=(am-a0)*(cm-c0))
    rebuilt=sum(terms.values());error=float(np.abs(rebuilt-D2P).max())
    if error>1e-12:raise ArithmeticError('float64 product identity reconstruction error>1e-12')
    raw_full=np.abs(rgb[:,2:]-2*rgb[:,1:-1]+rgb[:,:-2])[:,stencil]
    return dict(frozen_stencil_pixels=int(stencil.sum()),qualified_pixels=len(y),
        mask='same frozen vertical3pixel stencil AND all3 native alpha>.9; explanatory subset only',
        RGB_denominator=3*len(y),raw_Ev_full_frozen_mask=float(raw_full.mean()),
        raw_Ev_qualified_mask=float(np.abs(D2P).mean()),effective_color_Ev=float(np.abs(Dc).mean()),
        term_meanabs={name:float(np.abs(value).mean())for name,value in terms.items()},
        reconstruction_max=error,central_alpha_mean=float(a0.mean()),central_effective_color_mean=float(c0.mean()),
        alpha_D2_meanabs=float(np.abs(Da).mean()),effective_color_min=float(min(cm.min(),c0.min(),cp.min())),
        effective_color_max=float(max(cm.max(),c0.max(),cp.max())))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=P/'qa/final_result_manifest.json')
    parser.add_argument('--output',type=Path,default=P/'qa/stripe_product_decomposition.json');args=parser.parse_args()
    manifest=read(args.manifest);stages={s['name']:s for s in manifest['stages']}
    names=['filtered_A0','L800','ABC800','matched_I0'];frequency=load(manifest['frozen_frequency'])
    baseline={raw:packet(stages['filtered_A0'],raw)for raw in FRAMES};rows=[];checks={};missing=[]
    binding={str(args.manifest):sha(args.manifest),manifest['frozen_frequency']:sha(manifest['frozen_frequency'])}
    for name in names:
        if name not in stages:missing.append(name);continue
        for raw in FRAMES:
            try:z=packet(stages[name],raw)
            except FileNotFoundError:missing.append(f'{name}:{raw}');continue
            same=np.array_equal(z['alpha'],baseline[raw]['alpha']);checks[f'{name}:{raw}:alpha_exact']=same
            if not same:raise ValueError('explanatory fixedalpha comparison requires identical full native alpha')
            metric=decomposition(z['P'],z['alpha'],frequency[f'{raw:03d}_vertical'])
            rows.append(dict(stage=name,raw=raw,role='train-source'if raw in(86,110)else'diagnostic_not_blind',
                source=z['path'],**metric));binding[z['path']]=sha(z['path'])
    for raw in FRAMES:
        same=[r['qualified_pixels']for r in rows if r['raw']==raw]
        checks[f'{raw}:same_explanatory_subset']=len(set(same))<=1
    result=dict(schema='native_premult_stripe_product_identity_CPU.v1',GPU_executed=False,native_calls=0,optimizer_updates=0,
        status='failed'if not all(checks.values())else'incomplete'if missing else'pass_identity_only',
        formula='D2P=c0*D2alpha+alpha0*D2c+(alpha_plus-alpha0)*(c_plus-c0)+(alpha_minus-alpha0)*(c_minus-c0)',
        c_definition='c_eff=P/alpha only for diagnostic triplets with all3alpha>.9; never a training target/loss',
        rows=rows,checks=checks,missing=missing,source_sha256=binding,
        interpretation='term mean absolute values are not additive and are not causal percentages; frozenalpha can couple brightness changes to rawEv. Original50%-excess and fit gates remain unchanged on their original masks.',
        human_verdict=None,visual_verdict=None,production_promoted=False)
    write(args.output,result)
    columns=['stage','raw','role','frozen_stencil_pixels','qualified_pixels','raw_Ev_full_frozen_mask','raw_Ev_qualified_mask',
        'effective_color_Ev','c0_D2alpha','alpha0_D2c','cross_plus','cross_minus','reconstruction_max','central_alpha_mean','central_effective_color_mean']
    with args.output.with_suffix('.csv').open('w',encoding='utf-8',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for row in rows:writer.writerow({key:row.get(key,row['term_meanabs'].get(key))for key in columns})
    print(json.dumps(dict(output=str(args.output),status=result['status'],rows=len(rows),missing=missing)))


if __name__=='__main__':main()
