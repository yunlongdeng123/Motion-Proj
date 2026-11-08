from pathlib import Path
from html.parser import HTMLParser
import json,subprocess,hashlib
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]/'outputs/waymo_val029_experiment'
class Resources(HTMLParser):
    def __init__(self):super().__init__();self.paths=[]
    def handle_starttag(self,tag,attrs):
        for key,val in attrs:
            if key in ['src','href'] and val and not val.startswith(('#','http:','https:','data:')):self.paths.append(val)
def main():
    p=Resources();p.feed((ROOT/'index.html').read_text(encoding='utf8'));missing=[s for s in p.paths if not (ROOT/s).exists()];assert not missing,missing
    for q in ROOT.glob('*.png'):
        with Image.open(q) as im:im.verify()
    videos=[]
    for name,count in [('full198_comparison.mp4',198),('event119_197_crops.mp4',79)]:
        path=ROOT/'assets/media'/name;r=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,duration,avg_frame_rate','-of','json',str(path)],capture_output=True,check=True,text=True);meta=json.loads(r.stdout)['streams'][0];assert int(meta['nb_frames'])==count,meta;subprocess.run(['ffmpeg','-v','error','-threads','2','-i',str(path),'-f','null','-'],capture_output=True,check=True);videos.append(dict(file=name,decode_all_frames='PASS',**meta))
    recorded=json.loads((ROOT/'evidence_package/evidence/final_all_eval/summary.json').read_text(encoding='utf8'));summary={r['raw']:r for r in recorded['frames']};checks=[]
    for raw in [130,155,160,190]:
        masks=np.array(Image.open(ROOT/'evidence_package/masks'/f'{raw:03d}.png'));observed=np.array(Image.open(ROOT/'assets/frames'/f'{raw:03d}/final_all_eval/observed.png'),np.float32).transpose(2,0,1)/255;folder=ROOT/'evidence_package/key_float/final_all_eval'/f'{raw:03d}';keep=np.load(folder/'keep.npz');delete=np.load(folder/'delete.npz');conservation=max(float(abs(v['group_alpha'].sum(0)-v['alpha'][0]).max()) for v in [keep,delete]);assert conservation<3e-6;assert abs(delete['group_alpha'][0]).max()==0
        for actor,key in [(25,'revealed_white_suv'),(20,'preserved_gray'),(70,'target_red')]:
            mask=masks==actor
            if not mask.any():continue
            mae=float(abs(keep['rgb']-observed)[:,mask].mean());assert abs(mae-summary[raw]['regions'][key]['keep_gt_mae'])<1e-7,(raw,key,mae)
        checks.append(dict(raw=raw,group_conservation_maxabs=conservation,deleted_target_direct_alpha=0,visible_MAE_recomputed='PASS'))
    result=dict(status='PASS_STATIC_RESOURCE_VIDEO_AND_KEY_FLOAT_QA',HTML_resource_links=len(p.paths),missing_resources=missing,videos=videos,key_float_CPU_recompute=checks,browser_visual_preview='blocked by tool file-protocol policy; no workaround attempted',figure_visual_inspection='actual figures inspected with view_image',quality_accepted=False,human_verdict=None);(ROOT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
