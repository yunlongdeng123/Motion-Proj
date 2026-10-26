"""CPU preservation/backend-burden evidence, never counterfactual quality GT."""
import argparse,json,shutil,subprocess,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_dilation,binary_erosion
from imageio_ffmpeg import get_ffmpeg_exe

def read(p):return json.loads(p.read_text())
def rgb(p):return np.array(Image.open(p).convert('RGB'))

def video(folder,count):
    frames=sorted(folder.glob('*.png'));output=folder.parent/(folder.name+'_h264.mp4')
    if not frames or output.exists():return
    assert len(frames)==count and [int(p.stem) for p in frames]==list(range(int(frames[0].stem),int(frames[0].stem)+count))
    subprocess.run([shutil.which('ffmpeg') or get_ffmpeg_exe(),'-v','error','-nostdin','-framerate','10','-start_number',frames[0].stem,'-i',str(folder/'%03d.png'),'-frames:v',str(count),'-c:v','libx264','-preset','fast','-threads','2','-crf','18','-pix_fmt','yuv420p','-an',str(output)],check=True,timeout=60)

def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True);a=p.parse_args();suite=a.suite;summary=read(suite/'summary.json');dest=suite/'cpu_diagnostics';dest.mkdir(exist_ok=True)
    baseline=suite/'no_edit';cases=sorted(p for p in suite.iterdir() if (p/'native_rgb').exists());records=[]
    source=Path(read(Path(summary['run'])/'effective_config.json')['source_path'])
    # Resolve by the saved native run's campaign, not by an assumed attempt depth.
    campaign=Path(summary['run']).parents[2]
    frame_contract=read(campaign/'motion_edit_suite_v1'/summary['context']/'no_edit.json')['frames'];frame_ids={x['raw']:x['frame_id'] for x in frame_contract}
    branch_rows=[]
    for path in sorted((baseline/'float').glob('*.npz')):
        base=np.load(path);body=np.array(Image.open(source/'dynamic_masks/dynamic_mask_0'/(frame_ids[int(path.stem)]+'.png')).resize((480,320),Image.Resampling.NEAREST))>0;core=binary_erosion(body,iterations=2)
        if not core.any():continue
        dynamic=base['dynamic_weight'][0];static=base['alpha'][0]-dynamic;environment=1-base['alpha'][0]
        branch_rows.append(dict(raw=int(path.stem),body_core_pixels=int(core.sum()),dynamic_weight_mean=float(dynamic[core].mean()),static_weight_mean=float(static[core].mean()),environment_weight_mean=float(environment[core].mean()),static_dominant_fraction=float((static[core]>dynamic[core]).mean())))
    for case in cases:
        rows=[]
        for path in sorted((case/'native_rgb').glob('*.png')):
            source=rgb(baseline/'native_rgb'/path.name);image=rgb(path);mask=np.array(Image.open(case/'mask'/path.name))>0
            out=np.load(case/'float'/(path.stem+'.npz'));base=np.load(baseline/'float'/(path.stem+'.npz'))
            assert np.isfinite(out['rgb']).all() and np.isfinite(out['alpha']).all() and np.isfinite(out['depth_sum']).all()
            # A broad union of original and edited splat support protects the
            # intended edit. The rest is a preservation check, not hole GT.
            support=binary_dilation((base['dynamic_weight'][0]>.001)|(out['dynamic_weight'][0]>.001),iterations=4)
            support|=binary_dilation(mask,iterations=2);outside=~support
            delta=np.abs(image.astype(float)-source)/255
            comparable=not case.name.startswith('camera_')
            rows.append(dict(raw=int(path.stem),generation_pixels=int(mask.sum()),preservation_pixels=int(outside.sum()) if comparable else 0,preservation_MAE=float(delta[outside].mean()) if comparable and outside.any() else None,preservation_max=float(delta[outside].max()) if comparable and outside.any() else None,static_alpha_mean_on_generation_mask=float(out['alpha'][0][mask].mean()) if mask.any() else None))
        backend=read(case/'propainter/summary.json') if (case/'propainter/summary.json').exists() else None
        ma=[x['preservation_MAE'] for x in rows if x['preservation_MAE'] is not None]
        backend_metrics=None
        if backend is not None:
            changes=[x['inside_change_MAE'] for x in backend['rows'] if x['inside_change_MAE'] is not None]
            backend_metrics=dict(frames=backend['frames'],seconds=backend['seconds'],outside_mask_max_byte_difference=max(x['outside_max_difference'] or 0 for x in backend['rows']),inside_change_MAE_macro=float(np.mean(changes)) if changes else None)
        record=dict(case=case.name,frames=len(rows),rows=rows,preservation_MAE_macro=float(np.mean(ma)) if ma else None,generation_pixels_sum=sum(x['generation_pixels'] for x in rows),generation_pixels_mean=float(np.mean([x['generation_pixels'] for x in rows])),backend=backend_metrics)
        records.append(record);video(case/'native_rgb',len(rows))
        if backend:video(case/'propainter/model_png',len(rows))
    samples=[('no_edit',False),('delete_target',False),('delete_target',True),('translate_lateral_1m',False),('translate_lateral_1m',True),('insert_reused_lateral_35m',False),('speed_125',False),('speed_125',True),('turn_left15',False),('turn_left15',True),('camera_lateral_05m',False),('camera_yaw3',False)]
    sheet=Image.new('RGB',(1920,1050),'#142332');draw=ImageDraw.Draw(sheet)
    for index,(name,generated) in enumerate(samples):
        folder=suite/name/('propainter/model_png' if generated else 'native_rgb');frames=sorted(folder.glob('*.png'))
        if not frames:continue
        selected=frames[3*len(frames)//4]
        image=Image.open(selected).convert('RGB');x=(index%4)*480;y=(index//4)*350;sheet.paste(image,(x,y+30));draw.text((x+6,y+7),name+(' / ProPainter' if generated else ' / native3D')+' / raw'+selected.stem,fill='white')
    sheet.save(dest/'comparison.jpg',quality=95)
    if (suite/'continuous/summary.json').exists():
        for folder in (suite/'continuous').glob('*/native_rgb'):video(folder,31)
    branch=dict(rows=branch_rows,body_core_dynamic_weight_macro=float(np.mean([x['dynamic_weight_mean'] for x in branch_rows])),body_core_static_weight_macro=float(np.mean([x['static_weight_mean'] for x in branch_rows])),body_core_static_dominant_fraction_macro=float(np.mean([x['static_dominant_fraction'] for x in branch_rows])),mask_is_prepared_DEVA_not_independent_GT=True)
    result=dict(context=summary['context'],frames=summary['frames'],cases=records,branch_contributions=branch,adapter_native_max_difference=summary['adapter_actual_native_max_difference'],master_unchanged=summary['master_files_and_tensor_versions_unchanged'],continuous_native_executed=summary.get('continuous_sequence_executed',False),counterfactual_RGB_GT=False,metrics_are_preservation_and_backend_burden_not_edit_success=True,geometry_GT=False,contact_frame_selection='same floor(3N/4) index for all panels; center timestamp is the identity anchor for both speed edits',quality_accepted=False,human_verdict=None)
    (dest/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='cases'}))

if __name__=='__main__':main()
