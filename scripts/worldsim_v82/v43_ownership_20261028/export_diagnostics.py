"""Export semantic observations + exact contribution evidence, never models."""
import argparse,json,base64,zlib,hashlib,tarfile,shutil,time
from pathlib import Path
import numpy as np
from segmentation import ROOT,CONTEXTS,NAMES,write
def pack(arrays):
    data=b'';meta={}
    for k,a in arrays.items():
        a=np.ascontiguousarray(a);v=a.tobytes();meta[k]=dict(offset=len(data),bytes=len(v),dtype=a.dtype.str,shape=list(a.shape));data+=v
    return dict(arrays=meta,data=base64.b64encode(zlib.compress(data,6)).decode())
def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='semantic_delivery_v1');a=p.parse_args();dest=ROOT/a.name;dest.mkdir(exist_ok=False)
    report=dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),contexts=[],semantic_map_version='cityscapes19-plus-unknown-v1',semantic_map=dict(enumerate(NAMES)),no_checkpoint_exported=True,quality_accepted=False,human_verdict=None)
    for ctx in CONTEXTS:
        sem=ROOT/'segformer_b0_attempt3'/ctx;attr=ROOT/'attribution_v2'/ctx;out=dest/ctx;out.mkdir();summary=json.loads((attr/'summary.json').read_text());sizes=summary['branch_sizes'];aggregate=np.load(attr/'aggregate.npz');mass=aggregate['mass'];total=mass[:,21];label=mass[:,:20].argmax(1);best=mass[np.arange(len(mass)),label];purity=best/np.maximum(total,1e-8);label[(total<.01)|(purity<.65)]=0;vehicle=mass[:,14:18].sum(1)/np.maximum(total,1e-8);target=mass[:,20]/np.maximum(total,1e-8);target_support=aggregate['target_support_frames']
        stats=[]
        for part,sl in [('static',slice(0,sizes[0])),('dynamic',slice(sizes[0],sum(sizes)))]:
            stats.append(dict(branch=part,total_rows=int(total[sl].size),observed_rows=int((total[sl]>.01).sum()),class_counts=np.bincount(label[sl],minlength=20).tolist(),target_projection_contributors=int(((total[sl]>.01)&(target[sl]>.5)).sum()),repeated_target_candidate_rows=int(((total[sl]>1)&(target[sl]>.85)&(vehicle[sl]>.65)&(target_support[sl]>=3)).sum()),total_target_alpha_mass=float(mass[sl,20].sum())))
        index=dict(version=report['semantic_map_version'],names=NAMES,branch_sizes=sizes,source='offline frozen SegFormer-B0 Cityscapes + native alpha*T; predictions not GT',frames={},stats=stats,weight_definition=summary['weight_definition'],original_owner_unchanged=True)
        for r in summary['records']:
            raw=r['raw'];d=np.load(attr/('%03d.npz'%raw));s=np.load(sem/('%03d.npz'%raw));packed=pack(dict(ids=d['ids'],class_id=d['class_id'],purity=d['purity'],target_fraction=d['target_fraction'],vehicle_fraction=d['vehicle_fraction'],mass=d['mass'],semantic_image=s['labels'],confidence_image=s['confidence']))
            filename='frame_%03d.js'%raw;(out/filename).write_text('window.GPU_SEMANTIC_FRAMES=window.GPU_SEMANTIC_FRAMES||{};window.GPU_SEMANTIC_FRAMES['+str(raw)+']='+json.dumps(packed,separators=(',',':'))+';')
            index['frames'][str(raw)]=dict(path='gpu_semantics/'+filename,active_rows=len(d['ids']),**r)
        (out/'index.js').write_text('window.GPU_SEMANTIC_INDEX='+json.dumps(index,separators=(',',':'))+';')
        for f in sem.glob('*_overlay.png'):shutil.copy2(str(f),str(out/f.name))
        write(out/'summary.json',dict(index=index,attribution=summary));report['contexts'].append(dict(context=ctx,stats=stats,frames=len(summary['records']),max_closure=max(r['adjoint_closure_relative'] for r in summary['records']),static_target_mass_share=stats[0]['total_target_alpha_mass']/max(sum(x['total_target_alpha_mass'] for x in stats),1e-8)))
    for p in [ROOT/'authorization.json',ROOT/'attribution_dispatch.json',ROOT/'segformer_b0_attempt3/status.json']:
        shutil.copy2(str(p),str(dest/(p.parent.name+'_'+p.name)))
    write(dest/'report.json',report)
    files={str(p.relative_to(dest)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in dest.rglob('*') if p.is_file()};write(dest/'manifest.json',files)
    archive=ROOT/(a.name+'.tar.gz')
    with tarfile.open(str(archive),'w:gz') as t:t.add(str(dest),arcname=a.name)
    receipt=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(files),source_output=str(dest));write(ROOT/(a.name+'_receipt.json'),receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
