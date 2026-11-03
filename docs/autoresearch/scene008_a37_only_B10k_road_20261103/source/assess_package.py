"""最终49帧、固定保车/标线域及RGB开关对照；CPU打包，不新增更新。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,tarfile,shutil,hashlib,subprocess
import numpy as np
from scipy.ndimage import binary_erosion,binary_dilation
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
R=Path(__file__).resolve().parent.parent;C=R.parent;R1=R.with_name(R.name[:-1]+'1');R2=R.with_name(R.name[:-1]+'2')
DENSE=C/'r15_gn11_stage1_v41_20261023_dense_r7/frames'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def add(t,p,arc):t.add(p,arcname=arc,filter=lambda x:None if '__pycache__' in x.name else x)
def aggregate(q,name):
    rr=[x['regions'][name] for x in q['rows'] if x['regions'][name]['samples']];n=sum(x['samples'] for x in rr)
    return dict(samples=n,**{key:sum(x[key]*x['samples'] for x in rr)/n for key in ['before_MAE','after_MAE','native_srgb_change']})
def main():
    assert read(R/'queue_status.json')['stage']=='COMPLETE' and (R/'rgb_off/all49/metrics.json').exists()
    out=R/'review';out.mkdir(exist_ok=False);(out/'frames').mkdir();(out/'boards').mkdir();(out/'source').mkdir()
    qq=[read(p/'evaluation/all49/metrics.json') for p in [R1,R2,R]];q=qq[-1];den={};err={};maximum={};changed={}
    for raw in range(89,138):
        d=np.load(R1/'inputs/views'/f'{raw:03d}.npz');z=np.load(DENSE/f'0008{raw:03d}.npz');labels=d['actor_labels'];yy,xx=np.mgrid[:320,:480]
        selected=(labels>=1)&(labels<=3);other=(z['vehicle_body'][5]>0)&(xx<240)&(labels!=5);left_cars=selected|other
        masks=dict(yellow=d['marking_mask'],left_road=d['valid']&d['domain_a5']&~d['domain_a37']&(labels==0),
            left_cars=left_cars,left_cars_inner=binary_erosion(left_cars,iterations=2),
            outer_seam=binary_dilation(d['repair_mask'],iterations=2)&~d['repair_mask']&d['valid']&(labels==0)&~d['other_vehicle_guard'])
        for label,alias in enumerate([5,26,18,6],1):masks['actor'+str(alias)]=labels==label
        if raw==113:
            red=np.zeros((320,480),bool);red[207:251,223:291]=True;masks['user_red_box']=red
        images={'baseline':R/'evaluation/all49'/f'{raw:03d}_baseline.png','r1':R1/'evaluation/all49'/f'{raw:03d}_repaired.png',
            'r2':R2/'evaluation/all49'/f'{raw:03d}_repaired.png','repaired':R/'evaluation/all49'/f'{raw:03d}_repaired.png','rgb_off':R/'rgb_off/all49'/f'{raw:03d}_repaired.png'}
        arrays={k:np.asarray(Image.open(p).convert('RGB'),np.float32) for k,p in images.items()};base=arrays['baseline']
        for name,mask in masks.items():
            key=name;den[key]=den.get(key,0)+int(mask.sum())
            for method in ['r1','r2','repaired','rgb_off']:
                e=np.abs(arrays[method]-base);k=key+'__'+method
                err[k]=err.get(k,0)+float(e[mask].sum());maximum[k]=max(maximum.get(k,0),float(e[mask].max()) if mask.any() else 0)
                changed[k]=changed.get(k,0)+int((e.max(2)[mask]>1).sum())
        fd=out/'frames'/f'{raw:03d}';fd.mkdir()
        for key in ['baseline','r2','repaired','rgb_off']:shutil.copy2(images[key],fd/(key+'.png'))
        if raw in [89,101,105,113,125,137]:shutil.copy2(images['r1'],fd/'r1.png')
        board=Image.new('RGB',(1440,352),'#eef2f6');draw=ImageDraw.Draw(board);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
        for i,(key,title) in enumerate([('baseline','B10k: delete a37'),('r2','Previous road repair'),('repaired','Final road repair')]):
            draw.text((i*480+12,7),title,fill='#183546',font=font);board.paste(Image.fromarray(arrays[key].astype(np.uint8)),(i*480,32))
        board.save(out/'boards'/f'{raw:03d}.png')
    fixed={name:dict(samples=n,methods={method:dict(MAE_8bit=err[name+'__'+method]/(3*max(n,1)),max_8bit=maximum[name+'__'+method],pixels_changed_over1=changed[name+'__'+method]) for method in ['r1','r2','repaired','rgb_off']}) for name,n in den.items()}
    ff=imageio_ffmpeg.get_ffmpeg_exe();video=out/'all49_comparison.mp4'
    subprocess.run([ff,'-hide_banner','-loglevel','error','-n','-framerate','10','-start_number','89','-i',str(out/'boards/%03d.png'),'-frames:v','49','-c:v','libx264','-crf','17','-pix_fmt','yuv420p',str(video)],check=True)
    it=imageio_ffmpeg.read_frames(str(video));next(it);assert sum(1 for _ in it)==49
    for name,raw,box,scale in [('red_box',113,(215,200,300,258),4),('left_cars',113,(140,150,222,211),4),('left_near_car',105,(0,174,132,315),2)]:
        imgs=[Image.open(out/'frames'/f'{raw:03d}'/(key+'.png')).crop(box).resize(((box[2]-box[0])*scale,(box[3]-box[1])*scale),Image.Resampling.NEAREST) for key in ['baseline','r2','repaired']]
        strip=Image.new('RGB',(imgs[0].width*3,imgs[0].height));[strip.paste(im,(i*im.width,0)) for i,im in enumerate(imgs)];strip.save(out/(name+'_crop.png'))
    shutil.copy2(out/'boards/113.png',out/'preview.png')
    final_agg={name:aggregate(q,name) for name in ['fit','new_dark','original_shadow','yellow','left_road','outside','inferred']}
    summary=dict(task_id='SCENE008-A37-B10K-ROAD-20261103-R3',target=37,parent='single-a37 B10k',final=final_agg,fixed_pixel_domains=fixed,
        original_shadow_stages=[aggregate(x,'original_shadow') for x in qq],
        formal=read(R/'formal/receipt.json'),input=read(R/'inputs/receipt.json'),selection=read(R/'selection.json'),smoke=read(R/'smoke_summary.json'),
        invariants=read(R/'evaluation/invariants.json'),parent_png_max_uint8_difference=q['parent_png_max_uint8_difference'],
        parent_unchanged=q['parent_unchanged'],full49_frames=49,video_frames=49,final_updates=1000,
        total_this_request_updates=3102,formal_across_revisions=3000,smoke_across_revisions=96,restore_updates=6,prior_updates=24240,cumulative_updates=27342,
        failure_ledger_refs=['V82-F01','V82-F06'],failure_ledger_delta='update V82-F06: current-frame ground footprint excludes actor body projection, wrongly preserving ghost',
        all199_BUILD=True,independent_clean_road_GT=False,no_hard_image_route=True,human_verdict=None,quality_accepted=False,GPU_shutdown=False)
    write(out/'result_summary.json',summary);write(R/'result_summary.json',summary)
    for name in ['settings.json','selection.json','smoke_summary.json']:shutil.copy2(R/name,out/name)
    for rel,name in [('inputs/receipt.json','input_receipt.json'),('evaluation/all49/metrics.json','metrics_all49.json'),('evaluation/restore.json','restore.json'),('evaluation/invariants.json','invariants.json'),('rgb_off/all49/metrics.json','rgb_off_metrics.json')]:shutil.copy2(R/rel,out/name)
    shutil.copy2(R/'formal/checkpoint001000.pt',out/'checkpoint001000.pt')
    for p in (R/'source').glob('*.py'):shutil.copy2(p,out/'source'/p.name)
    archive=R/'a37_B10k_road_replay.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as t:
        for name in ['source','inputs','settings.json','formal','smoke','restore_a','restore_b','smoke_summary.json','selection.json','evaluation/invariants.json','evaluation/restore.json','evaluation/all49/metrics.json','rgb_off/all49/metrics.json','result_summary.json']:
            add(t,R/name,'final/'+name)
        for old,label in [(R1,'r1'),(R2,'r2')]:
            for rel in ['source','settings.json','inputs/receipt.json','formal/checkpoint001000.pt','formal/receipt.json','evaluation/all49/metrics.json']:
                add(t,old/rel,'history/'+label+'/'+rel)
    h=hashlib.sha256()
    with archive.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    receipt=dict(file=archive.name,bytes=archive.stat().st_size,sha256=h.hexdigest(),parent_B10k_four_weights_external=True,native_engine_V3_coordinate_bundle_external=True,
        remote_run=str(R),contains_final_inputs_parameters_Adam_RNG_sampler=True)
    write(out/'package_receipt.json',receipt);write(R/'package_receipt.json',receipt)
    with tarfile.open(R/'review_light.tar.gz','w:gz',compresslevel=1) as t:
        for p in out.iterdir():
            if p.name!='boards':add(t,p,'review/'+p.name)
    print(json.dumps(dict(final=final_agg,fixed=fixed,package=receipt)),flush=True)
if __name__=='__main__':main()
