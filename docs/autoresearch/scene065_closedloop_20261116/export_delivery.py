"""Only lightweight 480x320 previews/receipts leave the training host."""
from pathlib import Path
import hashlib,json,tarfile,time
from PIL import Image
import closedloop as t
RUN=t.C/'handoff_coverage_r2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 completed=t.read(RUN/'rendered/completed.json');assert completed['completed'] and completed['frames']==199
 out=RUN/'display_delivery';out.mkdir(exist_ok=False)
 rows=[];total=0
 for raw in range(199):
  row=dict(raw=raw)
  for name in ['new_delete','road_only','static_contribution']:
   src=RUN/'rendered'/f'{raw:03d}'/(name+'.png');dst=out/name/f'{raw:03d}.webp';dst.parent.mkdir(exist_ok=True)
   im=Image.open(src);assert im.size==(480,320);im.save(dst,quality=86,method=6)
   row[name]=dict(path=str(dst.relative_to(out)),sha256=sha(dst));total+=dst.stat().st_size
  rows.append(row)
 manifest=dict(schema='scene065_closedloop_viewer_input.v1',complete=True,run_id='native_closedloop_20261116_r2',
  title='新道路与表示交接',summary_text='最新遮罩补全已写入持久道路场；局部不透明度优化后渲染。车形残留与几何缺口仍存在。',
  baseline_index_sha256='4f41014a95997900cef237fc9ba55b6aae6d27a662fd0dd266851af39158f36d',
  frame_count=199,preview_wh=[480,320],default_raw=179,assets_root='.',old_delete_key='material_delete',
  static_contribution_definition='same_depth_order_rgb_contribution',report_href=None,frames=rows)
 t.write(out/'viewer_manifest.json',manifest)
 t.write(out/'transfer_receipt.json',dict(frames=199,preview_images=597,preview_bytes=total,
    models_or_full_resolution_data_transferred=False,native_results_kept_remote=True,rendered_completion_sha256=sha(RUN/'rendered/completed.json')))
 archive=RUN/'display_delivery.tar'
 with tarfile.open(archive,'w') as tar:tar.add(out,arcname='display_delivery')
 t.write(RUN/'delivery_ready.json',dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,previews=597))
 print('DELIVERY',archive,total,flush=True)

if __name__=='__main__':main()
