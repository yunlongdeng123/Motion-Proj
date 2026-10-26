"""Actual same-input ProPainter replay, kept separate from ordinary cache reuse."""
import argparse,json,os,shutil,sys
from pathlib import Path
import numpy as np
from PIL import Image
from gpu_slot import exclusive
from resource_control import require_start

def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True);a=p.parse_args();require_start()
    source=a.suite/'continuous/delete_original';dest=a.suite/'continuous/fixed_backend_same_input_replay'
    dest.mkdir(exist_ok=False);case=dest/'same_input';case.mkdir()
    for channel in ['native_rgb','mask']:shutil.copytree(source/channel,case/channel)
    state=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],phase='waiting_GPU',new_training_updates=0,new_case_count=0)
    def update(**kw):
        state.update(kw);tmp=dest/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(dest/'status.json')
    update()
    for channel in ['native_rgb','mask']:
        files=sorted((source/channel).glob('*.png'));assert len(files)==31
        assert all(f.read_bytes()==(case/channel/f.name).read_bytes() for f in files)
    with exclusive():
        require_start();update(phase='actual_same_input_inference')
        import inpaint_baseline
        sys.argv=['inpaint_baseline.py','--root',str(dest),'--case','same_input','--input-kind','native_3d']
        inpaint_baseline.main()
    rows=[]
    for path in sorted((source/'native_rgb').glob('*.png')):
        reference=np.array(Image.open(source/'propainter/model_png'/path.name).convert('RGB'),dtype=np.int16)
        replay=np.array(Image.open(case/'propainter/model_png'/path.name).convert('RGB'),dtype=np.int16)
        mask=np.array(Image.open(source/'mask'/path.name))>0;difference=np.abs(reference-replay)
        rows.append(dict(raw=int(path.stem),mask_pixels=int(mask.sum()),maximum_byte_difference=int(difference.max()),masked_MAE=float(difference[mask].mean()/255) if mask.any() else None))
    result=dict(actual_GPU_inference=True,identical_input_RGB_and_mask_bytes=True,same_backend_seed_and_settings=True,frames=31,rows=rows,maximum_output_byte_difference=max(r['maximum_byte_difference'] for r in rows),macro_masked_MAE=float(np.mean([r['masked_MAE'] for r in rows if r['masked_MAE'] is not None])),ordinary_cache=False,new_case_count=0,new_training_updates=0,counterfactual_RGB_GT=False,quality_accepted=False,human_verdict=None)
    (dest/'summary.json').write_text(json.dumps(result,indent=2));update(phase='complete');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
if __name__=='__main__':main()
