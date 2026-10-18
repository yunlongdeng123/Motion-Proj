"""Export existing30K native evaluation, zero GPU work or training updates."""
from pathlib import Path
import json,os,shutil,tarfile
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2'
import numpy as np
from PIL import Image
E=Path(__file__).resolve().parent
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/runs/scene0008_089_137_hybrid_30000_seed0_GN11_resume20k')
def main():
 assert json.loads((R/'status.json').read_text())['status']=='complete'
 event=json.loads((R/'checkpoint_events/iteration_030000.json').read_text());assert event['status']=='complete' and len(event['frames'])==49
 out=E/'GN11_stage1_full49';out.mkdir(exist_ok=False);rows=[]
 for row in event['frames']:
  fid=Path(row['frame_id']).stem;d=out/fid;d.mkdir()
  for key,target in [('gt','gt'),('composite','full'),('static','static'),('dynamic','dynamic')]:
   src=Path(row['channels'][key]);assert Image.open(src).size==(480,320);shutil.copy2(src,d/(target+'.png'))
  metric=json.loads(Path(row['hybrid_metrics']).read_text())
  with np.load(metric['arrays'],allow_pickle=False) as z:
   for key,target in [('alpha_d','dynamic_alpha'),('alpha_s','static_alpha')]:
    a=z[key];assert a.shape==(1,320,480);Image.fromarray(np.rint(np.clip(a[0],0,1)*255).astype(np.uint8)).save(d/(target+'.png'))
  shutil.copy2(d/'static.png',d/'delete_all.png')
  rows.append(dict(frame_id=fid,source_native_frame=row['frame_id'],native_checkpoint_step=30000))
 assert {x['frame_id'] for x in rows}=={f'0008{f:03d}' for f in range(89,138)}
 receipt=dict(state='COMPLETE_EXISTING_NATIVE_GN11_FIVE_STAGE1_30K_FULL49',frames=49,records=rows,parameter_updates=0,
  source=str(R),stage1_actor_ids=[5,26,18,6,37],full_frame=True,ROI=False,
  delete_all_semantics='native standalone static branch; no physical sun integration exists in this model',
  inputs_are_training_views=True,human_verdict=None)
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2))
 archive=E/'GN11_stage1_full49_pngs.tar.gz'
 with tarfile.open(archive,'w:gz') as t:t.add(out,arcname=out.name)
 print(json.dumps(dict(status=receipt['state'],archive=str(archive),bytes=archive.stat().st_size)))
if __name__=='__main__':main()
