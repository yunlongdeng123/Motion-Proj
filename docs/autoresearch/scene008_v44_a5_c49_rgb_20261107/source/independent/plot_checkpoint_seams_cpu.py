"""Plot actual C-node seam scores separately from training MSE.

--node 0=C_warm0_fixed4560_cpu.json --node 500=C_formal500_fixed4560_cpu.json
--epoch formal500_epoch49_cpu_v1.json may supply the independently read MSE
window table, but never puts MSE onto a seam y-axis. No missing node is invented.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(),parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--node',action='append',required=True)
    p.add_argument('--epoch',action='append',type=Path,default=[]);p.add_argument('--candidate-variant',choices=['C','H'],default='C')
    p.add_argument('--curve-label',default='Complete C support material');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
    nodes={int(k):Path(v) for k,v in (n.split('=',1) for n in a.node)}
    assert len(nodes)==len(a.node);rows=[];bin_rows=[];gates=[];sources=[];scope=None;baseline_signature=None
    for step,path in sorted(nodes.items()):
        data=read(path);sources.append(dict(path=str(path),sha256=sha(path)))
        assert data['contract_sha256']=='4560bb91f4cc3b27b89693acd0918f939d2ac07e9bcd7cbbc6e0574e885516cc'
        assert data['variants_evaluated']==['baseline',a.candidate_variant]
        current_scope=(tuple(data['raw_ids']),tuple(data['diagnostic_raw_ids']))
        if scope is None:scope=current_scope
        assert scope==current_scope,'Never silently mix pilot4 and49-view denominators'
        sig=[(r['state'],r['edge'],r['bins'],r['baseline']) for r in data['pooled_world_seams']]
        if baseline_signature is None:baseline_signature=sig
        assert sig==baseline_signature,'Latest reference changed between nodes'
        for s in ['keep','delete']:
            r=next(v for v in data['records'] if v['variant']==a.candidate_variant and v['raw']==128 and v['state']==s)
            b=next(v for v in data['records'] if v['variant']=='baseline' and v['raw']==128 and v['state']==s)
            rows.append(dict(step=step,state=s,domain='primary128_near6',bins=6,latest=b['primary128_seams']['near_x']['mean'],candidate=r['primary128_seams']['near_x']['mean']))
            for v in data['pooled_world_seams']:
                if v['state']==s:rows.append(dict(step=step,state=s,domain='world_'+v['edge'],bins=v['bins'],latest=v['baseline'],candidate=v['candidate']))
        baseline_records={(r['raw'],r['state']):r for r in data['records'] if r['variant']=='baseline'}
        for r in data['records']:
            if r['variant']!=a.candidate_variant:continue
            b=baseline_records[r['raw'],r['state']]
            items=[('world',edge,'world_seams') for edge in ['near_x','lateral_y']]
            if r['raw']==128:items.append(('primary128','near_x','primary128_seams'))
            for kind,edge,key in items:
                base_bins={v['tangent_bin']:v for v in b[key][edge]['bins']}
                for v in r[key][edge]['bins']:
                    bv=base_bins[v['tangent_bin']]
                    assert (v['minus_count'],v['plus_count'])==(bv['minus_count'],bv['plus_count'])
                    bin_rows.append(dict(step=step,raw=r['raw'],state=r['state'],kind=kind,edge=edge,tangent_bin=v['tangent_bin'],
                        minus_pixels=v['minus_count'],plus_pixels=v['plus_count'],baseline_score=bv['score'],candidate_score=v['score'],delta=v['score']-bv['score'],
                        baseline_signed_R=bv['signed_residual_jump_rgb'][0],baseline_signed_G=bv['signed_residual_jump_rgb'][1],baseline_signed_B=bv['signed_residual_jump_rgb'][2],
                        candidate_signed_R=v['signed_residual_jump_rgb'][0],candidate_signed_G=v['signed_residual_jump_rgb'][1],candidate_signed_B=v['signed_residual_jump_rgb'][2]))
        gates.append(dict(step=step,total=len(data['numeric_gates']),passed=sum(g['passed'] is True for g in data['numeric_gates']),failed=len(data['failed_gates']),unsupported=len(data['unsupported_gates'])))
    for r in rows:
        value,base=r['candidate'],r['latest'];r['delta_vs_latest']=value-base if value is not None else None
        prior=[v for v in rows if v['step']<r['step'] and v['state']==r['state'] and v['domain']==r['domain']]
        previous=max(prior,key=lambda v:v['step']) if prior else None
        r['previous_step']=previous['step'] if previous else None
        r['delta_vs_previous_actual_node']=value-previous['candidate'] if previous and value is not None else None
    with (a.output/'checkpoint_seams.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['step','state','domain','bins','latest','candidate','delta_vs_latest','previous_step','delta_vs_previous_actual_node']);w.writeheader();w.writerows(rows)
    with (a.output/'checkpoint_bins.csv').open('w',newline='') as f:
        fields=['step','raw','state','kind','edge','tangent_bin','minus_pixels','plus_pixels','baseline_score','candidate_score','delta',
            'baseline_signed_R','baseline_signed_G','baseline_signed_B','candidate_signed_R','candidate_signed_G','candidate_signed_B']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(bin_rows)
    epochs=[]
    for path in a.epoch:
        d=read(path);assert d['phase']=='formal','Smoke and formal are separate figures/tables'
        sources.append(dict(path=str(path),sha256=sha(path)))
        epochs.append(dict(step=d['checkpoint_step'],complete_epochs=d['complete_epoch_count'],partial_samples=sum(e['samples'] for e in d['partial_epoch_tail']),
            MSE_window=d['window5_vs_previous5'],material_RGB=d['parent_RGB'],geometry=d['geometry_visibility_vs_Cwarm0']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(11,9),constrained_layout=True)
    for i,(domain,label) in enumerate([('primary128_near6','Original primary128 near (6 bins)'),('world_near_x','World near (fixed equal bins)'),('world_lateral_y','World lateral (fixed equal bins)')]):
        for j,state in enumerate(['keep','delete']):
            ax=axes[i,j];selected=sorted([r for r in rows if r['domain']==domain and r['state']==state],key=lambda r:r['step'])
            if selected and selected[0]['latest'] is not None:ax.axhline(selected[0]['latest'],color='#607d8b',linestyle='--',label='Latest delivered reference')
            ax.plot([r['step'] for r in selected],[r['candidate'] for r in selected],marker='o',color='#b54b25',label=a.curve_label)
            ax.set_title(state.upper()+' / '+label);ax.set_xlabel('New formal material updates; 0 is old C400 warm initializer');ax.set_ylabel('Mean absolute RGB residual jump');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.savefig(a.output/'checkpoint_seams.png',dpi=150);fig.savefig(a.output/'checkpoint_seams.svg');plt.close(fig)
    result={'schema':'C_MATERIAL_NODE_SEAMS_SEPARATE_FROM_MSE_V1','status':'ACTUAL_NODE_COMPARISON_NOT_QUALITY_ADOPTION','scope_raws':list(scope[0]),
        'curve_label':a.curve_label,'candidate_variant':a.candidate_variant,'rows':rows,'per_bin_rows':bin_rows,
        'original_gate_counts':gates,'formal_epoch_window_table':epochs,'source_manifest':sources,
        'missing_nodes_invented':False,'MSE_and_seam_y_axis_mixed':False,'smoke_and_formal_concatenated':False,
        'quality_accepted':False,'human_verdict':None,'limits':['Node0 is an existing C400 warm initializer; x is only the new formal phase counter.',
           'Every recorded view is BUILD/development. Training MSE is a different objective/domain from the fixed seam scores.',
           'Original C4 protection failures do not cancel the explicitly authorized background control, and are not erased by a colour gain.',
           'Geometry/component failures and root actual image review remain separate.']}
    out=a.output.parent/(a.output.name+'.json');out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'output':str(out),'actual_steps':sorted(nodes)}))


if __name__=='__main__':main()
